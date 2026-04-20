#!/usr/bin/env python3

import argparse
import glob
import os

import fitsio
import numpy as np
import numpy.lib.recfunctions as rfn
import gc
from mpi4py import MPI
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


# Columns needed from fpfs_{band}.fits
fpfs_moment_cols = [
    "fpfs1_m00", "fpfs1_dm00_dg1", "fpfs1_dm00_dg2",
    "fpfs1_m20", "fpfs1_dm20_dg1", "fpfs1_dm20_dg2",
    "fpfs1_m22c", "fpfs1_dm22c_dg1", "fpfs1_dm22c_dg2",
    "fpfs1_m22s", "fpfs1_dm22s_dg1", "fpfs1_dm22s_dg2",
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
            os.path.join(base_dir, f"fpfs_{band}.fits")
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
        # Read fpfs moments and select by detection index
        fpfs = np.array(fitsio.read(
            os.path.join(base_dir, f"fpfs_{band}.fits")
        ))
        fpfs = fpfs[idx]

        # Rename fpfs1_* to {band}_fpfs1_*
        cols_to_keep = [c for c in fpfs_moment_cols if c in fpfs.dtype.names]
        fpfs_sub = rfn.repack_fields(fpfs[cols_to_keep])
        map_dict = {c: f"{band}_{c}" for c in cols_to_keep}
        fpfs_renamed = rfn.rename_fields(fpfs_sub, map_dict)
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

    out = rfn.merge_arrays([cat, wcs_arr], flatten=True)
    del cat, wcs_arr
    gc.collect()
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
    del data
    gc.collect()
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
