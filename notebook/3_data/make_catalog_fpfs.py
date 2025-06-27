#!/usr/bin/env python3

import argparse
import gc
import glob
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
        "--field", type=str, required=True, help="field name"
    )
    return parser.parse_args()


# Divide data among MPI ranks
def split_work(data, size, rank):
    return data[rank::size]


def read_files(tract_id, patch_id):
    bdir = "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/database/"
    image_dir = (
        "/lustre/HSC_DR/hsc_ssp/dr4/s23b/data/s23b_wide/unified/deepCoadd_calexp"
    )
    files = glob.glob(os.path.join(image_dir, f"{tract_id}/{patch_id}/i/*"))
    if not files:
        print(f"Canot find image for tract: {tract_id}, patch: {patch_id}")
        return None

    fname = files[0]
    exposure = afwImage.ExposureF.readFits(fname)
    mask_dir = f"{bdir}/s23b-brightStarMask/tracts_mask/{tract_id}/{patch_id}"
    mask_fname = os.path.join(mask_dir, "mask.fits")
    if os.path.isfile(mask_fname):
        bmask = fitsio.read(mask_fname)
    else:
        bmask = None

    det_dir = f"{bdir}/s23b-anacal/tracts/{tract_id}/{patch_id}"
    det_fname = os.path.join(det_dir, "detect.fits")
    if os.path.isfile(mask_fname):
        detection = fitsio.read(det_fname)
    else:
        print(f"cannot find {det_fname}")
        return None

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
    bdir = "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/database/"
    outdir = f"{bdir}/s23b-anacal/tracts/{tract_id}/{patch_id}"
    out_fname = os.path.join(outdir, "fpfs3.fits")
    if os.path.isfile(out_fname):
        print("already has outcome")
        return

    patch_info = skymap[tract_id][patch_id]
    wcs = patch_info.getWcs()
    bbox = patch_info.getOuterBBox()
    res = read_files(tract_id, patch_id)
    del wcs, bbox, patch_info

    if res is not None:
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
        os.makedirs(outdir, exist_ok=True)
        fitsio.write(out_fname, catalog)
        print(tract_id, patch_id, "finished")
        del catalog
    else:
        print(tract_id, patch_id, "cannot finish")
        return
    return


def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        full = fitsio.read(
            "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/tracts_fdfc_v1_trim6.fits"
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
        "/lustre/work/xiangchong.li/superonionIDark/code/image/HSC_S23B_Shapes/notebook/3_data/noise_correlation.fits"
    )
    # Initialize tqdm progress bar for this rank
    noise_corr = None
    pbar = tqdm(total=len(my_entries), desc=f"Rank {rank}", position=rank)
    for entry in my_entries:
        process_patch(entry, skymap, task, comm, noise_corr)
        gc.collect()
        pbar.update(1)
    pbar.close()
    return


if __name__ == "__main__":
    main()
