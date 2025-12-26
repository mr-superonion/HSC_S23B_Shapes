#!/usr/bin/env python3

import argparse
import gc
import glob
import os

import fitsio
import numpy as np
from mpi4py import MPI
from numpy.lib import recfunctions as rfn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process patch masks with MPI (with extra columns from db_color2)."
    )
    parser.add_argument("--field", type=str, required=True, help="field name")
    return parser.parse_args()


def split_work(data, size, rank):
    """Divide a 1D array-like 'data' across MPI ranks."""
    return data[rank::size]


def process_tract(field, tract_id, patch_list):
    # --- Read main catalog from db_color ---
    fname = os.path.join(
        os.environ["s23b"], "db_color", f"{tract_id}.fits"
    )
    data = np.array(fitsio.read(fname))
    # --- Original code1 logic on 'data' ---
    # Flip signs
    data["i_higherordermomentspsf_13"] = -data["i_higherordermomentspsf_13"]
    data["i_higherordermomentspsf_31"] = -data["i_higherordermomentspsf_31"]

    # Keep only patches of interest
    mask = np.isin(data["patch"], patch_list)
    data = rfn.repack_fields(data[mask])

    out_fname = os.path.join(
        os.environ["s23b_anacal_v2"],
        "fields_color",
        f"{field}_{tract_id}.fits",
    )

    out = []
    for patch_db in patch_list:
        patch_x = patch_db // 100
        patch_y = patch_db % 100
        patch_id = patch_x + patch_y * 9
        shape_dir = f"{os.environ['s23b_anacal_v2']}/{tract_id}/{patch_id}"
        sel_fname = os.path.join(shape_dir, "fdfc_sel.fits")
        sel = (fitsio.read(sel_fname) > 0)
        if np.sum(sel) < 3:
            continue
        fname = os.path.join(shape_dir, "match.fits")
        dd = fitsio.read(fname)
        dd = dd[sel]
        sub = data[data["patch"] == patch_db]

        common_ids, sub_idx, _ = np.intersect1d(
            sub["object_id"],
            dd["object_id"],
            return_indices=True,
        )
        assert len(common_ids) == len(dd)
        out.append(sub[sub_idx])
    if out:
        out = rfn.stack_arrays(out, usemask=False)
        fitsio.write(out_fname, out)
    return


def main():
    args = parse_args()
    field = args.field

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rootdir = os.environ["s23b"]
    full = fitsio.read(
        f"{rootdir}/tracts_fdfc_v2_final.fits"
    )

    # Determine tracts for this field on rank 0
    if rank == 0:
        tract_all, idx = np.unique(full["tract"], return_index=True)
        field_list = full[idx]["field"]
        mm = (field_list == field)
        tract_all = tract_all[mm]
    else:
        tract_all = None

    # Broadcast to all ranks
    tract_all = comm.bcast(tract_all, root=0)
    tract_list = split_work(tract_all, size, rank)

    # Per-tract work
    for tract_id in tract_list:
        patch_list = full["patch"][full["tract"] == tract_id]
        process_tract(field, tract_id, patch_list)
        gc.collect()

    comm.Barrier()

    # Rank 0: gather per-tract outputs into one file per field
    if rank == 0:
        out_dir = os.path.join(
            os.environ["s23b_anacal_v2"],
            "fields_color",
        )
        d_all = []
        fnames = glob.glob(os.path.join(out_dir, f"{field}_*.fits"))
        for fn in fnames:
            if os.path.isfile(fn):
                d_all.append(fitsio.read(fn))
                os.remove(fn)

        if len(d_all) == 0:
            raise RuntimeError(
                f"No partial files found for field {field} in {out_dir}"
            )

        outcome = rfn.stack_arrays(d_all, usemask=False)
        order = np.argsort(outcome["object_id"])
        outcome = outcome[order]

        fitsio.write(
            os.path.join(out_dir, f"{field}.fits"),
            outcome,
        )
    return


if __name__ == "__main__":
    main()
