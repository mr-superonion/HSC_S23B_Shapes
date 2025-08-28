#!/usr/bin/env python3

import argparse
import gc
import glob
import os
from tqdm import tqdm

import fitsio
import numpy as np
from mpi4py import MPI
from numpy.lib import recfunctions as rfn


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


def process_tract(field, tract_id, patch_list):
    fname = os.path.join(
        os.environ["s23b"], "db_color", f"{tract_id}.fits"
    )
    data = np.array(fitsio.read(fname))
    data["i_higherordermomentspsf_13"] = -data["i_higherordermomentspsf_13"]
    data["i_higherordermomentspsf_31"] = -data["i_higherordermomentspsf_31"]
    mask = np.isin(data["patch"], patch_list)
    data = rfn.repack_fields(
        data[mask]
    )

    out_fname = os.path.join(
        os.environ["s23b_anacal3"],
        "fields_color",
        f"{field}_{tract_id}.fits",
    )

    out = []
    for patch_db in patch_list:
        patch_x = patch_db // 100
        patch_y = patch_db % 100
        patch_id = patch_x + patch_y * 9
        shape_dir = f"{os.environ['s23b_anacal3']}/{tract_id}/{patch_id}"
        fname = os.path.join(shape_dir, "match.fits")
        dd = fitsio.read(fname)
        dd = dd[dd["wsel"] > 1e-6]
        sub = data[data["patch"] == patch_db]
        # Match on "object_id"
        common_ids, sub_idx, _ = np.intersect1d(
            sub["object_id"], dd["object_id"], return_indices=True
        )
        assert len(common_ids) == len(dd)
        out.append(sub[sub_idx])

    if out:
        out = rfn.stack_arrays(out, usemask=False)
        fitsio.write(
            out_fname,
            out,
        )
    return


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rootdir = os.environ["s23b"]
    full = fitsio.read(
        f"{rootdir}/tracts_fdfc_v1_final.fits"
    )
    field = args.field
    if rank == 0:
        tract_all, idx = np.unique(full["tract"], return_index=True)
        field_list = full[idx]["field"]
        mm = (field_list == field)
        tract_all = tract_all[mm]
    else:
        tract_all = None

    tract_all = comm.bcast(tract_all, root=0)
    tract_list = split_work(tract_all, size, rank)

    pbar = tqdm(total=len(tract_list), desc=f"Rank {rank}", position=rank)
    for tract_id in tract_list:
        patch_list = full["patch"][full["tract"] == tract_id]
        process_tract(field, tract_id, patch_list)
        gc.collect()
        pbar.update(1)
    pbar.close()

    comm.Barrier()
    if rank == 0:
        field = args.field
        out_dir = os.path.join(
            os.environ["s23b_anacal3"],
            "fields_color",
        )
        d_all = []
        fnames = glob.glob(os.path.join(out_dir, f"{field}_*.fits"))
        for fn in fnames:
            if os.path.isfile(fn):
                d_all.append(
                    fitsio.read(fn)
                )
                os.remove(fn)
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
