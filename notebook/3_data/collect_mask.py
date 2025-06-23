#!/usr/bin/env python3

import argparse
import gc
import glob
import os
import re
import astropy.table as asttable

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
    "CROSSTALK",
    "NO_DATA",
    "REJECTED",
    "SAT",
    "SUSPECT",
    "UNMASKEDNAN",
    "SENSOR_EDGE",
    "STREAK",
    "VIGNETTED",
    "INTRP",
    "EDGE",
    "CLIPPED",
    "INEXACT_PSF",
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


def extract_boxes(filename):
    pattern = re.compile(
        r"box\((?P<ra>\d*\.?\d+),\s*"
        r"(?P<dec>-?\d*\.?\d+),\s*"
        r"(?P<width>\d*\.?\d+)d,\s*"
        r"(?P<height>\d*\.?\d+)d,\s*"
        r"(?P<angle>-?\d*\.?\d+)\)\s*"
        r"# ID: (?P<id>\d+), mag: (?P<mag>\d*\.?\d+)"
    )

    ra_list = []
    dec_list = []
    width_list = []
    height_list = []
    angle_list = []
    id_list = []
    mag_list = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            match = pattern.match(line)
            if match:
                ra_list.append(float(match.group("ra")))
                dec_list.append(float(match.group("dec")))
                width_list.append(float(match.group("width")))
                height_list.append(float(match.group("height")))
                angle_list.append(float(match.group("angle")))
                id_list.append(int(match.group("id")))
                mag_list.append(float(match.group("mag")))

    return asttable.Table(
        [ra_list, dec_list, width_list, height_list, angle_list, id_list, mag_list],
        names=("ra", "dec", "width", "height", "angle", "id", "mag"),
    )


def extract_circles(filename):
    pattern = re.compile(
        r"circle\((?P<ra>\d*\.?\d+),\s*"
        r"(?P<dec>-?\d*\.?\d+),\s*"
        r"(?P<r>\d*\.?\d+)d\)\s*"
        r"# ID: (?P<id>\d+), mag: (?P<mag>\d*\.?\d+)"
    )

    ra_list = []
    dec_list = []
    r_list = []
    id_list = []
    mag_list = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("circle("):
                continue

            match = pattern.match(line)
            if match:
                ra_list.append(float(match.group("ra")))
                dec_list.append(float(match.group("dec")))
                r_list.append(float(match.group("r")))
                id_list.append(int(match.group("id")))
                mag_list.append(float(match.group("mag")))

    out = asttable.Table(
        [ra_list, dec_list, r_list, id_list, mag_list],
        names=("ra", "dec", "r", "id", "mag"),
    )
    out = out[out["r"] != 0.011111]
    return out


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

    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9
    patch_info = skymap[tract_id][patch_id]
    wcs = patch_info.getWcs()
    bbox = patch_info.getOuterBBox()
    image_dir = (
        "/lustre/HSC_DR/hsc_ssp/dr4/s23b/data/s23b_wide/unified/deepCoadd_calexp"
    )
    files = glob.glob(os.path.join(image_dir, f"{tract_id}/{patch_id}/i/*"))
    fname = files[0]
    exposure = afwimage.ExposureF.readFits(fname)
    bitv = exposure.mask.getPlaneBitMask(badplanes)
    mask_array = (
        ((exposure.mask.array & bitv) != 0)
        | (
            exposure.image.array
            < (
                -6.0
                * np.sqrt(
                    np.where(exposure.variance.array < 0, 0, exposure.variance.array)
                )
            )
        )
    ).astype(np.int16)

    mask_dir = "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/database/BrightObjectMasks/"
    mask_fname = os.path.join(
        mask_dir,
        f"{tract_id}/BrightObjectMask-{tract_id}-{patch_x},{patch_y}-HSC-I.reg",
    )

    ddc = extract_circles(mask_fname)
    x, y = wcs.skyToPixelArray(ra=ddc["ra"], dec=ddc["dec"], degrees=True)
    x = np.array(x - bbox.getBeginX(), dtype=float)
    y = np.array(y - bbox.getBeginY(), dtype=float)
    # Angular diameter distance at z
    r = ddc["r"] * 3600 / 0.168

    nx = bbox.getWidth()
    ny = bbox.getHeight()
    msk = (x + r > 0) & (x - r < nx) & (y + r > 0) & (y - r < ny)
    x = x[msk]
    y = y[msk]
    r = r[msk]

    dtype = np.dtype([("x", float), ("y", float), ("r", float)])
    xy_r = np.zeros(len(x), dtype=dtype)
    xy_r["x"] = x
    xy_r["y"] = y
    xy_r["r"] = r
    anacal.mask.add_bright_star_mask(mask_array=mask_array, star_array=xy_r)
    del xy_r, msk, ddc

    cat_dir = "/lustre/HSC_DR/hsc_ssp/dr4/s23b/data/s23b_wide/unified/deepCoadd_meas"
    files = glob.glob(os.path.join(cat_dir, f"{tract_id}/{patch_id}/i/*"))
    cat = afwtable.SourceCatalog.readFits(files[0])
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
    xy_r = np.zeros(len(x), dtype=dtype)
    xy_r["x"] = x
    xy_r["y"] = y
    xy_r["r"] = r
    anacal.mask.add_bright_star_mask(mask_array=mask_array, star_array=xy_r)
    del xy_r, cat, mm

    ddb = extract_boxes(mask_fname)
    x, y = wcs.skyToPixelArray(ra=ddb["ra"], dec=ddb["dec"], degrees=True)
    x = np.array(x - bbox.getBeginX(), dtype=float)
    y = np.array(y - bbox.getBeginY(), dtype=float)
    w = ddb["width"] * 3600 / 0.168
    h = ddb["height"] * 3600 / 0.168
    msk = (
        (x + w / 2 > 0)
        & (x - w / 2 < bbox.getWidth())
        & (y + h / 2 > 0)
        & (y - h / 2 < bbox.getHeight())
    )
    x = x[msk]
    y = y[msk]
    w = w[msk]
    h = h[msk]
    nbox = len(x)
    for i in range(nbox):
        xmin = int(min(max(x[i] - w[i] / 2, 0), nx))
        xmax = int(min(max(x[i] + w[i] / 2, 0), nx))
        ymin = int(min(max(y[i] - h[i] / 2, 0), ny))
        ymax = int(min(max(y[i] + h[i] / 2, 0), ny))
        if (xmin < xmax) & (ymin < ymax):
            mask_array[ymin:ymax, xmin:xmax] = 1
    fitsio.write(out_fname, mask_array)
    del exposure, mask_array
    return


def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        full = fitsio.read(
            "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/tracts_fdfc_v1_trim5.fits"
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
