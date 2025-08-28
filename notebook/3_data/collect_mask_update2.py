#!/usr/bin/env python3

import argparse
import gc
import os

import anacal
import fitsio
from lsst.skymap.ringsSkyMap import RingsSkyMap, RingsSkyMapConfig
from mpi4py import MPI


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Process patch masks with MPI.",
    )
    parser.add_argument(
        "--start", type=int, required=True, help="Start index of datalist.",
    )
    parser.add_argument(
        "--end", type=int, required=True, help="End index of datalist.",
    )
    return parser.parse_args()


# Divide data among MPI ranks
def split_work(data, size, rank):
    return data[rank::size]


def process_patch(entry, skymap):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9
    patch_info = skymap[tract_id][patch_id]
    bbox = patch_info.getOuterBBox()
    out_dir = f"{os.environ['s23b_mask']}/{tract_id}/{patch_id}"
    out_fname = os.path.join(out_dir, "mask3.fits")
    if os.path.isfile(out_fname):
        return
    msk_fname = os.path.join(out_dir, "mask2.fits")
    mask_array0 = fitsio.read(msk_fname)


    cat_fname = f"{os.environ['s23b']}/gaia/tracts/{tract_id}.fits"
    cat = fitsio.read(cat_fname, columns=["x", "y", "r"])
    cat["x"] = cat["x"] - bbox.getBeginX()
    cat["y"] = cat["y"] - bbox.getBeginY()
    anacal.mask.add_bright_star_mask(mask_array=mask_array0, star_array=cat)
    fitsio.write(out_fname, mask_array0)
    del mask_array0, bbox, patch_info
    return


def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        full = fitsio.read(
            os.path.join(os.environ["s23b"], "tracts_fdfc_v1_final.fits")
        )
        selected = full[args.start : args.end]
    else:
        selected = None

    selected = comm.bcast(selected, root=0)
    my_entries = split_work(selected, size, rank)

    # Set up the configuration
    config = RingsSkyMapConfig()
    config.numRings = 120
    config.projection = "TAN"
    config.tractOverlap = 1.0 / 60  # degrees
    config.pixelScale = 0.168  # arcsec/pixel
    skymap = RingsSkyMap(config)
    for entry in my_entries:
        process_patch(entry, skymap)
        gc.collect()


if __name__ == "__main__":
    main()
