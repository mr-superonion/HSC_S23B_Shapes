#!/usr/bin/env python3

import argparse
import os
import numpy as np

import fitsio
import lsst.afw.image as afwImage
from mpi4py import MPI


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
    sel = False

    bdir = "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/database/"
    basedir = f"{bdir}/s23b-anacal/tracts/{tract_id}/{patch_id}"
    det_fname = os.path.join(basedir, "detect.fits")
    force_fname = os.path.join(basedir, "force.fits")
    if not os.path.isfile(force_fname):
        sel = True
        if os.path.isfile(det_fname):
            os.popen(f"rm {det_fname}")
    return sel


def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        full = fitsio.read(
            "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/tracts_fdfc_v1_trim3.fits"
        )
        selected = full[args.start: args.end]
    else:
        selected = None

    selected = comm.bcast(selected, root=0)
    my_entries = split_work(selected, size, rank)

    indexes = []
    for entry in my_entries:
        valid = process_patch(entry)
        if valid:
            indexes.append(entry["index"])
    indexes = np.array(indexes, dtype=int)

    all_indexes = comm.gather(indexes, root=0)
    if rank == 0:
        merged = np.concatenate([a for a in all_indexes if a.size > 0])
        fitsio.write(
            "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/valid.fits",
            merged,
            clobber=True      # overwrite if the file exists
        )
    return


if __name__ == "__main__":
    main()
