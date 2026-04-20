#!/usr/bin/env python3

import argparse
import glob
import os

import fitsio
import numpy as np
import numpy.lib.recfunctions as rfn
from lsst.afw.table import SourceCatalog
from mpi4py import MPI
from xlens.catalog.utils import multiband_shapelets_linear2ell
from xlens.wcs import correct_ellipticity_wcs, jacobian_decomposition


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Process patch masks with MPI."
    )
    parser.add_argument("--field", type=str, required=True, help="field name")
    return parser.parse_args()


# Divide data among MPI ranks
def split_work(data, size, rank):
    return data[rank::size]


# Columns needed from linear modes
linear_moment_cols = [
    "fpfs1_m00", "fpfs1_m20", "fpfs1_m22c", "fpfs1_m22s",
    "fpfs1_m40", "fpfs1_m42c", "fpfs1_m42s", "fpfs1_m44c", "fpfs1_m44s",
    "fpfs1_n00", "fpfs1_n20", "fpfs1_n22c", "fpfs1_n22s",
    "fpfs1_n40", "fpfs1_n42c", "fpfs1_n42s", "fpfs1_n44c", "fpfs1_n44s",
]

# Columns needed from force.fits per band
force_cols_per_band = [
    "flux_gauss2", "dflux_gauss2_dg1", "dflux_gauss2_dg2",
    "flux_gauss2_err",
]

bands = ["r", "i", "z"]


def process_patch(entry):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9

    base_dir = f"{os.environ['s23b_anacal_v2']}/{tract_id}/{patch_id}"

    # Check all required files exist
    for band in bands:
        if not os.path.isfile(
            os.path.join(base_dir, f"fpfs_linear_{band}.fits")
        ):
            return None
    for fname in ["match.fits", "fdfc_sel.fits", "force.fits"]:
        if not os.path.isfile(os.path.join(base_dir, fname)):
            return None

    # Read match and selection
    dd2 = np.array(fitsio.read(os.path.join(base_dir, "match.fits")))
    sel = fitsio.read(os.path.join(base_dir, "fdfc_sel.fits")) > 0
    if np.sum(sel) < 3:
        return None
    dd2 = dd2[sel]
    idx = dd2["index"]

    # Read force.fits and select
    force = np.array(fitsio.read(os.path.join(base_dir, "force.fits")))
    force = force[sel]

    # Build combined catalog with band-prefixed columns
    arrays = []

    # Add object_id
    oid = np.zeros(len(dd2), dtype=[("object_id", "i8")])
    oid["object_id"] = dd2["object_id"]
    arrays.append(oid)

    for band in bands:
        # Read linear modes and select by detection index
        linear = np.array(fitsio.read(
            os.path.join(base_dir, f"fpfs_linear_{band}.fits")
        ))
        linear = linear[idx]

        # Rename fpfs1_* to {band}_fpfs1_*
        cols_to_keep = [c for c in linear_moment_cols if c in linear.dtype.names]
        lin_sub = rfn.repack_fields(linear[cols_to_keep])
        map_dict = {c: f"{band}_{c}" for c in cols_to_keep}
        lin_renamed = rfn.rename_fields(lin_sub, map_dict)
        arrays.append(lin_renamed)

        # Extract force columns for this band, using median flux_gauss2_err
        force_cols = [f"{band}_{c}" for c in force_cols_per_band]
        existing = [c for c in force_cols if c in force.dtype.names]
        if existing:
            force_sub = rfn.repack_fields(force[existing])
            err_col = f"{band}_flux_gauss2_err"
            if err_col in force_sub.dtype.names:
                force_sub[err_col] = np.median(force[err_col])
            arrays.append(force_sub)

    cat = rfn.merge_arrays(arrays, flatten=True)

    # Call multiband_shapelets_linear2ell
    C0 = 0.53
    result = multiband_shapelets_linear2ell(cat, bands=bands, C0=C0)

    # Read CDMatrix from HSC measurement catalog and match by object_id
    cat_dir = f"{os.environ['s23b_meas']}/{tract_id}/{patch_id}/i"
    cat_files = glob.glob(os.path.join(cat_dir, "*.fits"))
    dm_cat = SourceCatalog.readFits(cat_files[0])
    dm_mask = dm_cat["detect_isPrimary"]
    dm_cols = [
        "id",
        "base_LocalWcs_CDMatrix_1_1",
        "base_LocalWcs_CDMatrix_1_2",
        "base_LocalWcs_CDMatrix_2_1",
        "base_LocalWcs_CDMatrix_2_2",
    ]
    dm_arr = rfn.repack_fields(
        dm_cat.asAstropy().as_array()[dm_cols][dm_mask]
    )
    del dm_cat

    # Match by object_id
    _, idx_result, idx_dm = np.intersect1d(
        dd2["object_id"], dm_arr["id"], return_indices=True,
    )
    dm_matched = dm_arr[idx_dm]
    result = result[idx_result]
    matched_oid = np.zeros(len(idx_result), dtype=[("object_id", "i8")])
    matched_oid["object_id"] = dd2["object_id"][idx_result]

    # Compute per-object g1, g2, rho from CDMatrix using jacobian_decomposition
    # CDMatrix is in degrees/pixel (LSST convention, u=East)
    # Convert to GalSim convention (u=West, arcsec/pixel): negate first row, deg->arcsec
    pixel_scale = 0.168
    rad_to_arcsec = (180.0 / np.pi) * 3600.0
    n = len(dm_matched)
    g1 = np.zeros(n)
    g2 = np.zeros(n)
    rho = np.zeros(n)
    for i in range(n):
        jac = np.array([
            [-dm_matched["base_LocalWcs_CDMatrix_1_1"][i] * rad_to_arcsec,
             -dm_matched["base_LocalWcs_CDMatrix_1_2"][i] * rad_to_arcsec],
            [dm_matched["base_LocalWcs_CDMatrix_2_1"][i] * rad_to_arcsec,
             dm_matched["base_LocalWcs_CDMatrix_2_2"][i] * rad_to_arcsec],
        ])
        g1[i], g2[i], rho[i], _ = jacobian_decomposition(jac, pixel_scale)

    # Apply WCS correction to ellipticities
    e1_corr, e2_corr = correct_ellipticity_wcs(
        result, g1, g2, rho, prefix="fpfs1_", flipu=True,
    )
    result["fpfs1_e1"] = e1_corr
    result["fpfs1_e2"] = e2_corr

    # Flip columns with sin(2*theta) or sin(4*theta) dependence for u-flip
    result["fpfs1_de1_dg2"] = -result["fpfs1_de1_dg2"]
    result["fpfs1_de2_dg1"] = -result["fpfs1_de2_dg1"]
    result["fpfs1_dm0_dg2"] = -result["fpfs1_dm0_dg2"]
    result["fpfs1_dm2_dg2"] = -result["fpfs1_dm2_dg2"]

    # Merge object_id with result
    out = rfn.merge_arrays([matched_oid, result], flatten=True)
    return out


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        rootdir = os.environ["s23b"]
        full = fitsio.read(
            f"{rootdir}/tracts_fdfc_v2_final.fits"
        )
        mm = full["field"] == args.field
        selected = full[mm]
    else:
        selected = None

    selected = comm.bcast(selected, root=0)
    my_entries = split_work(selected, size, rank)
    data = []
    for entry in my_entries:
        out = process_patch(entry)
        if out is not None:
            data.append(out)

    data = rfn.stack_arrays(data, usemask=False)
    field = args.field
    out_dir = os.path.join(os.environ['s23b_anacal_v2'], "fields")
    fitsio.write(
        os.path.join(out_dir, f"{field}_multiband_{rank}.fits"),
        data,
    )
    comm.Barrier()

    if rank == 0:
        field = args.field
        out_dir = os.path.join(os.environ['s23b_anacal_v2'], "fields")
        d_all = []
        fnames = glob.glob(
            os.path.join(out_dir, f"{field}_multiband_*.fits")
        )
        for fn in fnames:
            if os.path.isfile(fn):
                d_all.append(fitsio.read(fn))
                os.remove(fn)
        outcome = rfn.stack_arrays(d_all, usemask=False)
        order = np.argsort(outcome["object_id"])
        outcome = outcome[order]
        fitsio.write(
            os.path.join(out_dir, f"{field}_multiband.fits"),
            outcome,
        )
    return


if __name__ == "__main__":
    main()
