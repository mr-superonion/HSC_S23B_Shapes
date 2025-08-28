#!/usr/bin/env python3

import argparse
import gc
import glob
import os
import numpy as np

import fitsio
import numpy.lib.recfunctions as rfn
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

def process_tract(tract_id, skymap):
    cat_dir = os.path.join(
        os.environ['s23b'],
        "db_random",
    )
    cat_fname = f"{cat_dir}/{tract_id}.fits"
    cat0 = fitsio.read(cat_fname)
    tps = fitsio.read(
        f"{os.environ['s23b']}/tracts_fdfc_v1_final.fits"
    )
    patch_db_list = tps["patch"][tps["tract"] == tract_id]
    cat_all = []
    for patch_db in patch_db_list:
        mm = (cat0["patch"] == patch_db)
        cat = cat0[mm]
        patch_x = patch_db // 100
        patch_y = patch_db % 100
        patch_id = patch_x + patch_y * 9
        patch_info = skymap[tract_id][patch_id]
        wcs = patch_info.getWcs()
        bbox = patch_info.getOuterBBox()
        mask_dir = f"{os.environ['s23b_mask']}/{tract_id}/{patch_id}"
        mask_fname = os.path.join(mask_dir, "mask3.fits")
        bmask = fitsio.read(mask_fname)
        nim_dir = f"{os.environ['s23b_nimg']}/{tract_id}/{patch_id}/i/"
        nim_fname = glob.glob(os.path.join(nim_dir, "*.fits"))[0]
        bmask = (bmask | (fitsio.read(nim_fname) <=2).astype(np.int16))
        x, y = wcs.skyToPixelArray(
            ra=cat["ra"],
            dec=cat["dec"],
            degrees=True,
        )
        x = np.round(x - bbox.getBeginX()).astype(int)
        y = np.round(y - bbox.getBeginY()).astype(int)
        # mask_int = bmask[y, x]
        # cat = rfn.append_fields(
        #     cat, "mask", mask_int, dtypes='i4', usemask=False,
        # )
        cat_all.append(cat[bmask[y, x] == 0])
    cat_all = rfn.stack_arrays(cat_all, usemask=False, asrecarray=False)
    out_fname = f"{cat_dir}/random_masked_{tract_id}.fits"
    fitsio.write(out_fname, cat_all)
    return


def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        rootdir = os.environ["s23b"]
        full = fitsio.read(
            f"{rootdir}/tracts_id.fits"
        )
        selected = full[args.start: args.end]
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
        process_tract(entry, skymap)
        gc.collect()


if __name__ == "__main__":
    main()
