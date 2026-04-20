#!/usr/bin/env python3

import argparse
import glob
import os

import fitsio
import numpy as np
import numpy.lib.recfunctions as rfn
import gc
from mpi4py import MPI
from xlens.catalog.utils import _linear_modes_to_derivs
from xlens.wcs import jacobian_decomposition


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


# Output columns per band (same as merge_fpfs.py)
fpfs_moment_cols = [
    "fpfs1_m00", "fpfs1_dm00_dg1", "fpfs1_dm00_dg2",
    "fpfs1_m20", "fpfs1_dm20_dg1", "fpfs1_dm20_dg2",
    "fpfs1_m22c", "fpfs1_dm22c_dg1", "fpfs1_dm22c_dg2",
    "fpfs1_m22s", "fpfs1_dm22s_dg1", "fpfs1_dm22s_dg2",
]

# Columns needed from fpfs_linear_{band}.fits
linear_moment_cols = [
    "fpfs1_m00", "fpfs1_m20", "fpfs1_m22c", "fpfs1_m22s",
    "fpfs1_m40", "fpfs1_m42c", "fpfs1_m42s", "fpfs1_m44c", "fpfs1_m44s",
    "fpfs1_n00", "fpfs1_n20", "fpfs1_n22c", "fpfs1_n22s",
    "fpfs1_n40", "fpfs1_n42c", "fpfs1_n42s", "fpfs1_n44c", "fpfs1_n44s",
]

moment_names = [
    "m00", "m20", "m22c", "m22s",
    "m40", "m42c", "m42s", "m44c", "m44s",
]
noise_names = [
    "n00", "n20", "n22c", "n22s",
    "n40", "n42c", "n42s", "n44c", "n44s",
]

bands = ["r", "i", "z"]


def linear_to_moments(linear, prefix="fpfs1_"):
    """Convert linear modes to moment + derivative columns.

    Returns a structured array with the same 12 columns as fpfs_{band}.fits:
    m00, dm00_dg1, dm00_dg2, m20, dm20_dg1, dm20_dg2,
    m22c, dm22c_dg1, dm22c_dg2, m22s, dm22s_dg1, dm22s_dg2.
    """
    p = prefix
    # Build xx = m - 2*n for derivative computation
    xx = {
        mn: linear[f"{p}{mn}"] - 2.0 * linear[f"{p}{nn}"]
        for mn, nn in zip(moment_names, noise_names)
    }
    d = _linear_modes_to_derivs(xx)

    n = len(linear)
    out = np.zeros(
        n,
        dtype=[(col, np.float64) for col in fpfs_moment_cols],
    )
    out[f"{p}m00"] = linear[f"{p}m00"]
    out[f"{p}dm00_dg1"] = d["dm00_dg1"]
    out[f"{p}dm00_dg2"] = d["dm00_dg2"]
    out[f"{p}m20"] = linear[f"{p}m20"]
    out[f"{p}dm20_dg1"] = d["dm20_dg1"]
    out[f"{p}dm20_dg2"] = d["dm20_dg2"]
    out[f"{p}m22c"] = linear[f"{p}m22c"]
    out[f"{p}dm22c_dg1"] = d["dm22c_dg1"]
    out[f"{p}dm22c_dg2"] = d["dm22c_dg2"]
    out[f"{p}m22s"] = linear[f"{p}m22s"]
    out[f"{p}dm22s_dg1"] = d["dm22s_dg1"]
    out[f"{p}dm22s_dg2"] = d["dm22s_dg2"]
    return out


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
    for fname in ["match.fits", "fdfc_sel.fits"]:
        if not os.path.isfile(os.path.join(base_dir, fname)):
            return None

    # Read match and selection
    dd2 = np.array(fitsio.read(os.path.join(base_dir, "match.fits")))
    sel = fitsio.read(os.path.join(base_dir, "fdfc_sel.fits")) > 0
    if np.sum(sel) < 3:
        return None
    dd2 = dd2[sel]
    idx = dd2["index"]

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

        # Convert linear modes to moment + derivative columns
        fpfs = linear_to_moments(linear)

        # Rename fpfs1_* to {band}_fpfs1_*
        map_dict = {c: f"{band}_{c}" for c in fpfs_moment_cols}
        fpfs_renamed = rfn.rename_fields(fpfs, map_dict)
        arrays.append(fpfs_renamed)

    cat = rfn.merge_arrays(arrays, flatten=True)

    # Read CDMatrix from HSC measurement catalog and match by object_id
    cat_dir = f"{os.environ['s23b_meas']}/{tract_id}/{patch_id}/i"
    cat_files = glob.glob(os.path.join(cat_dir, "*.fits"))
    dm_cat = fitsio.read(cat_files[0])
    dm_cols = [
        "id",
        "base_LocalWcs_CDMatrix_1_1",
        "base_LocalWcs_CDMatrix_1_2",
        "base_LocalWcs_CDMatrix_2_1",
        "base_LocalWcs_CDMatrix_2_2",
    ]
    dm_arr = rfn.repack_fields(
        dm_cat[dm_cols]
    )
    del dm_cat

    # Match by object_id
    _, idx_cat, idx_dm = np.intersect1d(
        cat["object_id"], dm_arr["id"], return_indices=True,
    )
    dm_matched = dm_arr[idx_dm]
    cat = cat[idx_cat]
    pixel_scale = 0.168
    rad_to_arcsec = (180.0 / np.pi) * 3600.0
    n = len(dm_matched)
    wcs_arr = np.zeros(
        n,
        dtype=[
            ("g1_wcs", np.float64),
            ("g2_wcs", np.float64),
            ("rho_wcs", np.float64),
            ("kappa_wcs", np.float64),
        ],
    )
    for i in range(n):
        jac = np.array([
            [-dm_matched["base_LocalWcs_CDMatrix_1_1"][i] * rad_to_arcsec,
             -dm_matched["base_LocalWcs_CDMatrix_1_2"][i] * rad_to_arcsec],
            [dm_matched["base_LocalWcs_CDMatrix_2_1"][i] * rad_to_arcsec,
             dm_matched["base_LocalWcs_CDMatrix_2_2"][i] * rad_to_arcsec],
        ])
        g1, g2, rho, kappa = jacobian_decomposition(jac, pixel_scale)
        wcs_arr["g1_wcs"][i] = g1
        wcs_arr["g2_wcs"][i] = g2
        wcs_arr["rho_wcs"][i] = rho
        wcs_arr["kappa_wcs"][i] = kappa

    # Build object_id array for WCS
    oid_wcs = np.zeros(n, dtype=[("object_id", "i8")])
    oid_wcs["object_id"] = cat["object_id"]
    wcs_out = rfn.merge_arrays([oid_wcs, wcs_arr], flatten=True)

    del wcs_arr
    gc.collect()
    return cat, wcs_out


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
    data_list = []
    wcs_list = []
    for entry in my_entries:
        result = process_patch(entry)
        if result is not None:
            cat_patch, wcs_patch = result
            data_list.append(cat_patch)
            wcs_list.append(wcs_patch)

    data = rfn.stack_arrays(data_list, usemask=False)
    wcs_data = rfn.stack_arrays(wcs_list, usemask=False)
    del data_list, wcs_list
    field = args.field
    out_dir = os.path.join(os.environ['s23b_anacal_v2'], "fields")
    fitsio.write(
        os.path.join(out_dir, f"{field}_multiband0_{rank}.fits"),
        data,
    )
    fitsio.write(
        os.path.join(out_dir, f"{field}_wcs_perturb_{rank}.fits"),
        wcs_data,
    )
    del data, wcs_data
    gc.collect()
    comm.Barrier()

    if rank == 0:
        field = args.field
        out_dir = os.path.join(os.environ['s23b_anacal_v2'], "fields")

        # Merge multiband0
        d_all = []
        fnames = glob.glob(
            os.path.join(out_dir, f"{field}_multiband0_*.fits")
        )
        for fn in fnames:
            if os.path.isfile(fn):
                d_all.append(fitsio.read(fn))
                os.remove(fn)
        outcome = rfn.stack_arrays(d_all, usemask=False)
        order = np.argsort(outcome["object_id"])
        outcome = outcome[order]
        fitsio.write(
            os.path.join(out_dir, f"{field}_multiband0.fits"),
            outcome,
        )
        del d_all, outcome

        # Merge wcs_perturb
        w_all = []
        wnames = glob.glob(
            os.path.join(out_dir, f"{field}_wcs_perturb_*.fits")
        )
        for fn in wnames:
            if os.path.isfile(fn):
                w_all.append(fitsio.read(fn))
                os.remove(fn)
        wcs_outcome = rfn.stack_arrays(w_all, usemask=False)
        wcs_order = np.argsort(wcs_outcome["object_id"])
        wcs_outcome = wcs_outcome[wcs_order]
        fitsio.write(
            os.path.join(out_dir, f"{field}_wcs_perturb.fits"),
            wcs_outcome,
        )
        del w_all, wcs_outcome
    return


if __name__ == "__main__":
    main()
