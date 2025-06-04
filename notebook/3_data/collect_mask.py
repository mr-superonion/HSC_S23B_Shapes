#!/usr/bin/env python3

import argparse
import gc
import glob
import os

import anacal
import fitsio
import lsst.afw.image as afwImage
import numpy as np
from lsst.daf.butler import Butler
from mpi4py import MPI


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Process patch masks with MPI.")
    parser.add_argument(
        "--start", type=int, required=True, help="Start index of datalist."
    )
    parser.add_argument("--end", type=int, required=True, help="End index of datalist.")
    return parser.parse_args()


# Divide data among MPI ranks
def split_work(data, size, rank):
    return data[rank::size]


def make_circular_kernel(radius):
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    return mask.astype(np.int16)


def process_patch(entry, skymap):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9
    db_dir = "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/database/"
    outdir = f"{db_dir}/s23b-brightStarMask/tracts_mask/{tract_id}/{patch_id}"
    out_fname = os.path.join(outdir, "mask.fits")
    if os.path.isfile(out_fname):
        return
    os.makedirs(outdir, exist_ok=True)

    patch_info = skymap[tract_id][patch_id]
    wcs = patch_info.getWcs()
    bbox = patch_info.getOuterBBox()

    image_dir = (
        "/lustre/HSC_DR/hsc_ssp/dr4/s23b/data/s23b_wide/unified/deepCoadd_calexp"
    )
    files = glob.glob(os.path.join(image_dir, f"{tract_id}/{patch_id}/i/*"))
    if not files:
        print(os.path.join(image_dir, f"{tract_id}/{patch_id}/i/*"))
        return
    fname = files[0]
    exposure = afwImage.ExposureF.readFits(fname)
    dd = fitsio.read(f"{db_dir}/s23b-brightStarMask/tracts/{tract_id}.fits")
    dd = dd[dd["patch"] == patch_db]
    x, y = wcs.skyToPixelArray(ra=dd["ra"], dec=dd["dec"], degrees=True)
    x = np.array(np.int_(x - bbox.getBeginX()))
    y = np.array(np.int_(y - bbox.getBeginY()))
    valid = (x >= 0) & (x < bbox.getWidth()) & (y >= 0) & (y < bbox.getHeight())
    x = x[valid]
    y = y[valid]

    mask = np.zeros((bbox.getHeight(), bbox.getWidth()), dtype=np.int16)
    mask[y, x] = 1

    bitv = exposure.mask.getPlaneBitMask(["SAT"])
    mask |= ((exposure.mask.array & bitv) != 0).astype(np.int16)

    for radius, threshold in [(40, 2), (20, 0), (20, 2), (10, 2)]:
        kernel = make_circular_kernel(radius)
        mask = anacal.mask.sparse_convolve(mask, kernel)
        mask = (mask > threshold).astype(np.int16)
        del kernel

    fitsio.write(out_fname, mask)
    del exposure, dd, mask
    return


def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        full = fitsio.read(
            "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/tracts_fdfc_v1_trim2.fits"
        )
        selected = full[args.start : args.end]
    else:
        selected = None

    selected = comm.bcast(selected, root=0)
    my_entries = split_work(selected, size, rank)

    obs_repo = "/lustre/work/xiangchong.li/work/hsc_s23b_sim"
    obs_collection = "version1/image"
    skymap_name = "hsc"
    obs_butler = Butler(obs_repo, collections=obs_collection)
    skymap = obs_butler.get("skyMap", skymap=skymap_name)
    for entry in my_entries:
        process_patch(entry, skymap)
        gc.collect()


if __name__ == "__main__":
    main()
