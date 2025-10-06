#!/usr/bin/env python3

import argparse
import gc
import glob
import os

import anacal
import astropy.io.fits as pyfits
import fitsio
import lsst.afw.image as afwImage
import numpy as np
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

def get_vmin_r(exposure, y0, x0, rmax=330, bin_size=30):
    img_arr = exposure.image.array
    mask_arr = exposure.mask.array
    var_arr = exposure.variance.array

    H, W = img_arr.shape

    # clamp the crop to image bounds
    y_min = max(0, int(y0 - rmax))
    y_max = min(H, int(y0 + rmax + 1))
    x_min = max(0, int(x0 - rmax))
    x_max = min(W, int(x0 + rmax + 1))

    # crop
    image = img_arr[y_min:y_max, x_min:x_max]
    mm    = mask_arr[y_min:y_max, x_min:x_max]
    vim   = var_arr[y_min:y_max, x_min:x_max]
    if np.sum(mm == 0) < 5:
        return 0.0, 0.0

    var = np.nanmean(vim[mm == 0])
    # build coordinate grid relative to the point
    y, x = np.indices(image.shape)
    y_center = y0 - y_min
    x_center = x0 - x_min
    r = np.sqrt((x - x_center)**2 + (y - y_center)**2)

    # radial bins and mask
    rbin = (r / bin_size).astype(int)
    valid = (r <= rmax) & (mm == 0)
    if np.sum(valid == 0) < 5:
        return 0.0, 0.0

    # bin means normalized by noise
    sum_pix = np.bincount(rbin[valid], weights=image[valid])
    npixs   = np.bincount(rbin[valid]) + 0.01
    mean = sum_pix / npixs / np.sqrt(var / npixs)

    radii = (np.arange(len(mean)) + 0.5) * bin_size
    if len(mean) <= 1:
        return 0.0, 0.0
    ind = np.argmin(mean)
    return radii[ind], mean[ind]


def process_patch(entry, skymap):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9
    patch_info = skymap[tract_id][patch_id]
    bbox = patch_info.getOuterBBox()
    wcs = patch_info.getWcs()

    msk_fname = os.path.join(
        f"{os.environ['s23b_mask']}/{tract_id}/{patch_id}",
        "mask3.fits",
    )
    mask_array0 = fitsio.read(msk_fname)
    cat_fname = f"{os.environ['s23b']}/low_redshift_gals/gals.fits"
    cat0 = fitsio.read(cat_fname)
    x, y = wcs.skyToPixelArray(
        ra=cat0["ra"],
        dec=cat0["dec"],
        degrees=True,
    )
    mm = np.logical_and.reduce((
        x > bbox.getBeginX(),
        y > bbox.getBeginY(),
        x < bbox.getEndX(),
        y < bbox.getEndY(),
    ))
    x = x[mm] - bbox.getBeginX()
    y = y[mm] - bbox.getBeginY()

    calexp_dir = f"{os.environ['s23b_calexp']}/{tract_id}/{patch_id}/i"
    exp_fname = glob.glob(os.path.join(calexp_dir, "*.fits"))[0]
    exposure = afwImage.ExposureF.readFits(exp_fname)
    keep = []
    r = []
    for i in range(len(x)):
        rr, pp = get_vmin_r(exposure, y[i], x[i])
        if pp < -6:
            keep.append(True)
            r.append(rr * 1.5)
        else:
            keep.append(False)
    if np.sum(keep) > 0:
        out_dir = os.path.join(
            os.environ['s23b'],
            f"low_redshift_gals/{tract_id}/{patch_id}",
        )
        os.makedirs(out_dir, exist_ok=True)
        out_fname = f"{out_dir}/gals.fits"
        if os.path.isfile(out_fname):
            return
        x = x[keep]
        y = y[keep]
        arr = rfn.unstructured_to_structured(
            np.column_stack(
                [np.array(x), np.array(y), np.array(r)]
            ),
            dtype=[("x", "f4"), ("y", "f4"), ("r", "f4")]
        )
        fitsio.write(out_fname, arr)
        anacal.mask.add_bright_star_mask(mask_array=mask_array0, star_array=arr)
        pyfits.writeto(msk_fname, mask_array0, overwrite=True)
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
