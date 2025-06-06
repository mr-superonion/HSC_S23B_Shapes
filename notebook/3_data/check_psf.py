#!/usr/bin/env python3

import argparse
import glob
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
    image_dir = (
        "/lustre/HSC_DR/hsc_ssp/dr4/s23b/data/s23b_wide/unified/deepCoadd_calexp"
    )
    valid = True
    for bb in ["g", "r", "i", "z", "y"]:
        files = glob.glob(
            os.path.join(image_dir, f"{tract_id}/{patch_id}/{bb}/*")
        )
        if not files:
            return False
        fname = files[0]
        try:
            exposure = afwImage.ExposureF.readFits(fname)
            test = False
            if exposure.hasPsf():
                lsst_psf = exposure.getPsf()
                if lsst_psf is not None:
                    test = True
            del exposure
            valid = valid & test
        except:
            valid = False
    return valid


def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        full = fitsio.read(
            "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/tracts_fdfc_v1_trim2.fits"
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
