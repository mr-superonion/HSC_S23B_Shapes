#!/usr/bin/env python3

import argparse
import gc
import glob
import os

import fitsio
import lsst.afw.image as afwImage
from lsst.skymap.ringsSkyMap import RingsSkyMap, RingsSkyMapConfig
from mpi4py import MPI
from xlens.process_pipe.anacal_force import (
    AnacalForcePipe,
    AnacalForcePipeConfig,
)

from numpy.lib import recfunctions as rfn


band_seed = {
    "g": 3,
    "r": 4,
    "i": 0,
    "z": 1,
    "y": 2,
}
colnames = ["flux", "dflux_dg1", "dflux_dg2", "dflux_dj1", "dflux_dj2"]


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


def read_files(tract_id, patch_id, band):
    bdir = "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/database/"
    image_dir = (
        "/lustre/HSC_DR/hsc_ssp/dr4/s23b/data/s23b_wide/unified/deepCoadd_calexp"
    )
    files = glob.glob(
        os.path.join(image_dir, f"{tract_id}/{patch_id}/{band}/*")
    )

    fname = files[0]
    exposure = afwImage.ExposureF.readFits(fname)
    mask_dir = f"{bdir}/s23b-brightStarMask/tracts_mask/{tract_id}/{patch_id}"
    mask_fname = os.path.join(mask_dir, "mask.fits")
    if os.path.isfile(mask_fname):
        bmask = fitsio.read(mask_fname)
    else:
        bmask = None
    return {
        "exposure": exposure,
        "mask": bmask,
    }


def process_patch(entry, skymap, task, noise_corr):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9
    bdir = "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/database/"
    outdir = f"{bdir}/s23b-anacal/tracts/{tract_id}/{patch_id}"
    det_fname = os.path.join(outdir, "detect.fits")
    out_fname = os.path.join(outdir, "force.fits")
    if os.path.isfile(out_fname):
        return None
    if not os.path.isfile(det_fname):
        print(tract_id, patch_id, "cannot find det file")
        return None

    patch_info = skymap[tract_id][patch_id]
    wcs = patch_info.getWcs()
    bbox = patch_info.getOuterBBox()
    try:
        detection = fitsio.read(det_fname)
    except Exception:
        print(tract_id, patch_id, "cannot read det file")
        return None
    catalog = [detection]
    for band in ["g", "r", "i", "z", "y"]:
        res = read_files(tract_id, patch_id, band)
        seed = (tract_id * 1000 + patch_id) * 5 + band_seed[band]
        data = task.anacal.prepare_data(
            exposure=res["exposure"],
            seed=seed,
            noise_corr=noise_corr,
            detection=detection,
            band=band,
            skyMap=skymap,
            tract=tract_id,
            patch=patch_id,
            star_mask_array=res["mask"],
            star_cat=None,
        )
        cat = rfn.repack_fields(task.anacal.run(**data)[colnames])
        del data, seed, res
        map_dict = {name: f"{band}_" + name for name in colnames}
        renamed = rfn.rename_fields(cat, map_dict)
        catalog.append(renamed)
        gc.collect()
    del wcs, bbox

    catalog = rfn.merge_arrays(catalog, flatten=True)
    fitsio.write(out_fname, catalog)

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

    config = AnacalForcePipeConfig()
    config.anacal.force_size = True
    config.anacal.num_epochs = 8
    config.anacal.do_noise_bias_correction = True
    task = AnacalForcePipe(config=config)

    noise_corr = fitsio.read(
        "noise_correlation2.fits"
    )
    for entry in my_entries:
        process_patch(entry, skymap, task, noise_corr)
    return


if __name__ == "__main__":
    main()
