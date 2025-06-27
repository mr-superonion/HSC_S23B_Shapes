#!/usr/bin/env python3

import argparse
import gc
import glob
import numpy as np
import os
from tqdm import tqdm

import fitsio
import lsst.afw.image as afwImage
from lsst.skymap.ringsSkyMap import RingsSkyMap, RingsSkyMapConfig
from mpi4py import MPI
from xlens.process_pipe.fpfs_force import (
    FpfsForcePipe, FpfsForcePipeConfig
)


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
    parser.add_argument(
        "--field", type=str, default="all", required=False, help="field name"
    )
    return parser.parse_args()


# Divide data among MPI ranks
def split_work(data, size, rank):
    return data[rank::size]


def read_files(tract_id, patch_id):
    calexp_dir = f"{os.environ['s23b_calexp']}/{tract_id}/{patch_id}/i"
    exp_fname = glob.glob(os.path.join(calexp_dir, "*.fits"))[0]
    exposure = afwImage.ExposureF.readFits(exp_fname)

    mask_dir = f"{os.environ['s23b_mask']}/{tract_id}/{patch_id}"
    mask_fname = os.path.join(mask_dir, "mask2.fits")
    bmask = fitsio.read(mask_fname)
    nim_dir = f"{os.environ['s23b_nimg']}/{tract_id}/{patch_id}/i"
    nim_fname = glob.glob(os.path.join(nim_dir, "*.fits"))[0]
    bmask = (bmask | (fitsio.read(nim_fname) <=2).astype(np.int16))

    det_dir = f"{os.environ['s23b_anacal']}/{tract_id}/{patch_id}"
    det_fname = os.path.join(det_dir, "detect.fits")
    detection = fitsio.read(det_fname)

    return {
        "exposure": exposure,
        "mask": bmask,
        "detection": detection,
    }


def process_patch(entry, skymap, task, comm, noise_corr):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9
    out_dir = f"{os.environ['s23b_anacal']}/{tract_id}/{patch_id}"
    out_fname = os.path.join(out_dir, "fpfs.fits")
    if os.path.isfile(out_fname):
        return

    patch_info = skymap[tract_id][patch_id]
    wcs = patch_info.getWcs()
    bbox = patch_info.getOuterBBox()
    res = read_files(tract_id, patch_id)
    del wcs, bbox, patch_info

    seed = (tract_id * 1000 + patch_id) * 5
    data = task.fpfs.prepare_data(
        exposure=res["exposure"],
        seed=seed,
        noise_corr=noise_corr,
        detection=res["detection"],
        band=None,
        mask_array=res["mask"],
    )
    catalog = task.fpfs.run(**data)
    del data, res
    fitsio.write(out_fname, catalog)
    del catalog
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
        selected = full[args.start: args.end]
        if args.field != "all":
            sel = (selected["field"] == args.field)
            selected = selected[sel]
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

    config = FpfsForcePipeConfig()
    config.fpfs.do_noise_bias_correction = True
    config.fpfs.use_average_psf = False
    config.fpfs.npix = 64
    config.fpfs.sigma_arcsec1 = 0.5657
    task = FpfsForcePipe(config=config)

    noise_corr = fitsio.read(
        "noise_correlation2.fits"
    )
    # Initialize tqdm progress bar for this rank
    noise_corr = None
    pbar = tqdm(total=len(my_entries), desc=f"Rank {rank}", position=rank)
    for entry in my_entries:
        try:
            process_patch(entry, skymap, task, comm, noise_corr)
            gc.collect()
        except Exception:
            print("failed: ", entry["index"])
        pbar.update(1)
    pbar.close()
    return


if __name__ == "__main__":
    main()
