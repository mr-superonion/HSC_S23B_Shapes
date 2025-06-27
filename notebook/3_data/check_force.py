#!/usr/bin/env python3

import argparse
import os
import glob

import fitsio
from mpi4py import MPI
import gc


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Process patch masks with MPI."
    )
    parser.add_argument(
        "--start", type=int, required=True, help="Start index of datalist."
    )
    parser.add_argument(
        "--end", type=int, required=True, help="End index of datalist."
    )
    return parser.parse_args()


# Divide data among MPI ranks
def split_work(data, size, rank):
    return data[rank::size]


def process_patch(entry):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9
    det_dir = f"{os.environ['s23b_anacal']}/{tract_id}/{patch_id}"
    det_fname = os.path.join(det_dir, "detect.fits")
    try:
        a = fitsio.read(det_fname)
    except Exception:
        print("failed:", entry["index"])

    # for band in ["g", "r", "i", "z", "y"]:
    #     calexp_dir = f"{os.environ['s23b_calexp']}/{tract_id}/{patch_id}/{band}"
    #     if not os.path.isdir(calexp_dir):
    #         print(tract_id, patch_id, band)
    #         return
    #     fnames = glob.glob(os.path.join(calexp_dir, "*.fits"))
    #     if len(fnames) == 0:
    #         print(tract_id, patch_id, band)
    #         return
    #     fname = fnames[0]
    #     if not os.path.isfile(fname):
    #         print(tract_id, patch_id, band)
    #         return
    #     try:
    #         exp = fitsio.read(fname)
    #         exp.shape[0]
    #         del exp
    #     except Exception:
    #         print(tract_id, patch_id, band)


def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        full = fitsio.read("tracts_fdfc_v1_trim6.fits")
        selected = full[args.start: args.end]
    else:
        selected = None

    selected = comm.bcast(selected, root=0)
    my_entries = split_work(selected, size, rank)

    for entry in my_entries:
        process_patch(entry)
        gc.collect()
    return


if __name__ == "__main__":
    main()
