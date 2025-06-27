#!/usr/bin/env python3

import argparse
import gc
import glob
import os

import anacal
import fitsio
import lsst.afw.image as afwimage
import lsst.afw.table as afwtable
import numpy as np
from lsst.skymap.ringsSkyMap import RingsSkyMap, RingsSkyMapConfig
from mpi4py import MPI


badplanes = [
    "BAD",
    "CR",
    "NO_DATA",
    "REJECTED",
    "SAT",
    "UNMASKEDNAN",
]


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
    out_fname = os.path.join(out_dir, "mask2.fits")
    if os.path.isfile(out_fname):
        return
    msk_fname = os.path.join(out_dir, "mask.fits")
    mask_array0 = fitsio.read(msk_fname)

    for band in ["g", "r", "z", "y"]:
        calexp_dir = f"{os.environ['s23b_calexp']}/{tract_id}/{patch_id}/{band}"
        fnames = glob.glob(os.path.join(calexp_dir, "*.fits"))
        fname = fnames[0]
        exposure = afwimage.ExposureF.readFits(fname)
        bitv = exposure.mask.getPlaneBitMask(badplanes)
        mask_array = ((exposure.mask.array & bitv) != 0).astype(np.int16)

        cat_dir = f"{os.environ['s23b_meas']}/{tract_id}/{patch_id}/{band}"
        fnames = glob.glob(os.path.join(cat_dir, "*.fits"))
        fname = fnames[0]
        cat = afwtable.SourceCatalog.readFits(fname)
        snr = (
            cat["base_CircularApertureFlux_3_0_instFlux"]
            / cat["base_CircularApertureFlux_3_0_instFluxErr"]
        )
        mm = (
            (cat["base_PixelFlags_flag_saturated"])
            & (snr > 80)
            & (cat["deblend_nChild"] == 0)
        )
        cat = cat[mm]
        x = cat["base_SdssCentroid_x"] - bbox.getBeginX()
        y = cat["base_SdssCentroid_y"] - bbox.getBeginY()
        r = np.sqrt(cat["base_FootprintArea_value"]) * 1.1
        dtype = np.dtype([("x", float), ("y", float), ("r", float)])
        xy_r = np.zeros(len(x), dtype=dtype)
        xy_r["x"] = x
        xy_r["y"] = y
        xy_r["r"] = r
        anacal.mask.add_bright_star_mask(mask_array=mask_array, star_array=xy_r)
        mask_array0 = (mask_array0 | mask_array).astype(np.int16)
        del xy_r, cat, mm, snr
        del exposure, mask_array
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
            "tracts_fdfc_v1_trim6.fits"
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
