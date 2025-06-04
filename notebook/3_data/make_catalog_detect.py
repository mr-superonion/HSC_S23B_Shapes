#!/usr/bin/env python3

import argparse
import gc
import glob
import os

import fitsio
import lsst.afw.image as afwImage
from lsst.daf.butler import Butler
from mpi4py import MPI
from xlens.process_pipe.anacal_detect import (AnacalDetectPipe,
                                              AnacalDetectPipeConfig)


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

def process_patch(entry, skymap, task):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9
    db_dir = "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/database/"
    outdir = f"{db_dir}/s23b-anacal/tracts/{tract_id}/{patch_id}"
    out_fname = os.path.join(outdir, "detect.fits")
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
    base_dir = "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/database/"
    mask_dir = f"{base_dir}/s23b-brightStarMask/tracts_mask/{tract_id}/{patch_id}"
    mask_fname = os.path.join(mask_dir, "mask.fits")
    bmask = fitsio.read(mask_fname)

    seed = tract_id * 1000 + patch_id
    data = task.anacal.prepare_data(
        exposure=exposure,
        seed=seed,
        noise_corr=None,
        detection=None,
        band=None,
        skyMap=skymap,
        tract=tract_id,
        patch=patch_id,
        star_mask_array=bmask,
    )
    catalog = task.anacal.run(**data)
    del data, exposure, wcs, bbox, bmask
    fitsio.write(out_fname, catalog)
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
    config = AnacalDetectPipeConfig()
    config.anacal.force_size = False
    config.anacal.num_epochs = 8
    config.anacal.num_epochs_deblend = 1
    config.anacal.use_average_psf = False
    config.anacal.badMaskPlanes = [
        "BAD",
        "CR",
        "CROSSTALK",
        "NO_DATA",
        "REJECTED",
        "SAT",
        "SUSPECT",
        "UNMASKEDNAN",
        "SENSOR_EDGE",
        "STREAK",
        "VIGNETTED",
    ]
    task = AnacalDetectPipe(config=config)
    for entry in my_entries:
        process_patch(entry, skymap, task)
        gc.collect()


if __name__ == "__main__":
    main()
