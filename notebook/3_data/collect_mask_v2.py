#!/usr/bin/env python3

import argparse
import gc
import glob
import os

import anacal
import fitsio
import numpy as np
import numpy.lib.recfunctions as rfn
import lsst.afw.image as afwImage
import lsst.afw.table as afwtable
from lsst.skymap.ringsSkyMap import RingsSkyMap, RingsSkyMapConfig
from mpi4py import MPI

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
badplanes = [
    "BAD",
    "CR",
    "NO_DATA",
    "SAT",
    "UNMASKEDNAN",
]


# ----------------------------------------------------------------------
# Command-line arguments
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build final masks from mask.fits + Gaia + low-z galaxies",
    )
    parser.add_argument(
        "--start", type=int, required=True, help="Start index of datalist.",
    )
    parser.add_argument(
        "--end", type=int, required=True, help="End index of datalist.",
    )
    return parser.parse_args()


def split_work(data, size, rank):
    return data[rank::size]


def get_vmin_r(exposure, y0, x0, rmax=330, bin_size=30):
    """
    Parameters
    ----------
    exposure : lsst.afw.image.ExposureF
    y0, x0   : float
        Center (pixel coordinates within the exposure).
    rmax     : float
        Maximum radius (pixels) to consider.
    bin_size : float
        Radial bin size (pixels).

    Returns
    -------
    r_min : float
        Radius at which the normalized mean is minimum.
    v_min : float
        Minimum normalized mean value.
    """
    img_arr = exposure.image.array
    mask_arr = exposure.mask.array
    var_arr = exposure.variance.array

    H, W = img_arr.shape

    # Clamp crop to image bounds
    y_min = max(0, int(y0 - rmax))
    y_max = min(H, int(y0 + rmax + 1))
    x_min = max(0, int(x0 - rmax))
    x_max = min(W, int(x0 + rmax + 1))

    # Crop
    image = img_arr[y_min:y_max, x_min:x_max]
    mm = mask_arr[y_min:y_max, x_min:x_max]
    vim = var_arr[y_min:y_max, x_min:x_max]

    if np.sum(mm == 0) < 5:
        return 0.0, 0.0

    var = np.nanmean(vim[mm == 0])

    # Build coordinate grid relative to the point
    y, x = np.indices(image.shape)
    y_center = y0 - y_min
    x_center = x0 - x_min
    r = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)

    # Radial bins and mask
    valid = (r <= rmax) & (mm == 0)
    if np.sum(valid) < 5:
        return 0.0, 0.0

    # Bin means normalized by noise
    rbin = (r / bin_size).astype(int)
    sum_pix = np.bincount(rbin[valid], weights=image[valid])
    npixs = np.bincount(rbin[valid]) + 0.01
    mean = sum_pix / npixs / np.sqrt(var / npixs)

    radii = (np.arange(len(mean)) + 0.5) * bin_size
    if len(mean) <= 1:
        return 0.0, 0.0

    ind = np.argmin(mean)
    return radii[ind], mean[ind]


# ----------------------------------------------------------------------
# Core per-patch processing
#   - Start from mask.fits
#   - Add per-band bad-plane + bright saturated sources
#   - Add Gaia mask
#   - Add low-z galaxy mask
#   - Write final mask.fits in s23b_mask_out tree
# ----------------------------------------------------------------------
def process_patch(entry, skymap):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9

    patch_info = skymap[tract_id][patch_id]
    bbox = patch_info.getOuterBBox()
    wcs = patch_info.getWcs()

    # Directories / filenames
    mask_root_in = os.environ["s23b_mask"]
    mask_root_out = os.environ["s23b_mask_v2"]

    in_dir = os.path.join(mask_root_in, f"{tract_id}", f"{patch_id}")
    out_dir = os.path.join(mask_root_out, f"{tract_id}", f"{patch_id}")
    os.makedirs(out_dir, exist_ok=True)

    in_mask_fname = os.path.join(in_dir, "mask.fits")
    out_mask_fname = os.path.join(out_dir, "mask.fits")

    # Skip if already done
    if os.path.isfile(out_mask_fname):
        return

    if not os.path.isfile(in_mask_fname):
        print(
            f"[Tract {tract_id}, patch {patch_id}] not found; skipping."
        )
        return

    mask_array = fitsio.read(in_mask_fname).astype(np.int16)

    # ------------------------------------------------------------------
    # Stage 1: per-band bad-plane mask + bright saturated sources
    #         (this is your first script, but in memory)
    # ------------------------------------------------------------------
    for band in ["g", "r", "z", "y"]:
        calexp_dir = f"{os.environ['s23b_calexp']}/{tract_id}/{patch_id}/{band}"
        fnames = glob.glob(os.path.join(calexp_dir, "*.fits"))
        if len(fnames) == 0:
            continue

        fname = fnames[0]
        exposure = afwImage.ExposureF.readFits(fname)
        bitv = exposure.mask.getPlaneBitMask(badplanes)
        mask_band = ((exposure.mask.array & bitv) != 0).astype(np.int16)

        cat_dir = f"{os.environ['s23b_meas']}/{tract_id}/{patch_id}/{band}"
        cat_files = glob.glob(os.path.join(cat_dir, "*.fits"))
        if len(cat_files) > 0:
            cat_fname = cat_files[0]
            cat = afwtable.SourceCatalog.readFits(cat_fname)
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

            # Add bright-star wings into this band mask
            anacal.mask.add_bright_star_mask(mask_array=mask_band, star_array=xy_r)

            del xy_r, cat, mm, snr, x, y, r

        # OR into main mask
        mask_array = (mask_array | mask_band).astype(np.int16)
        del exposure, mask_band

    # ------------------------------------------------------------------
    # Stage 2: add Gaia bright star mask
    # ------------------------------------------------------------------
    gaia_cat_fname = f"{os.environ['s23b']}/gaia/tracts/{tract_id}.fits"
    if os.path.isfile(gaia_cat_fname):
        gaia_cat = fitsio.read(gaia_cat_fname, columns=["x", "y", "r"])
        # Convert absolute patch coords to local (patch) coords
        gaia_cat["x"] = gaia_cat["x"] - bbox.getBeginX()
        gaia_cat["y"] = gaia_cat["y"] - bbox.getBeginY()
        anacal.mask.add_bright_star_mask(mask_array=mask_array, star_array=gaia_cat)
        del gaia_cat
    else:
        print(f"[Tract {tract_id}] Gaia catalog not found, skipping Gaia mask.")

    # ------------------------------------------------------------------
    # Stage 3: add low-redshift galaxy mask
    # ------------------------------------------------------------------
    lowz_all_fname = f"{os.environ['s23b']}/low_redshift_gals/gals.fits"
    if os.path.isfile(lowz_all_fname):
        cat0 = fitsio.read(lowz_all_fname)
        # Convert sky coords to pixel
        x, y = wcs.skyToPixelArray(
            ra=cat0["ra"],
            dec=cat0["dec"],
            degrees=True,
        )

        mm = np.logical_and.reduce(
            (
                x > bbox.getBeginX(),
                y > bbox.getBeginY(),
                x < bbox.getEndX(),
                y < bbox.getEndY(),
            )
        )

        x = x[mm] - bbox.getBeginX()
        y = y[mm] - bbox.getBeginY()

        # Read i-band exposure for radial profiles
        calexp_dir = f"{os.environ['s23b_calexp']}/{tract_id}/{patch_id}/i"
        exp_candidates = glob.glob(os.path.join(calexp_dir, "*.fits"))
        if len(exp_candidates) == 0:
            print(
                f"[Tract {tract_id}, patch {patch_id}] "
                "No i-band calexp; skipping low-z galaxies."
            )
        else:
            exp_fname = exp_candidates[0]
            exposure = afwImage.ExposureF.readFits(exp_fname)

            keep = []
            r_list = []
            for xi, yi in zip(x, y):
                rr, pp = get_vmin_r(exposure, yi, xi)
                if pp < -8:
                    keep.append(True)
                    r_list.append(rr * 1.2)
                else:
                    keep.append(False)

            keep = np.array(keep, dtype=bool)

            if np.sum(keep) > 0:
                x_keep = x[keep]
                y_keep = y[keep]
                r_keep = np.array(r_list)

                arr = rfn.unstructured_to_structured(
                    np.column_stack([x_keep, y_keep, r_keep]),
                    dtype=[("x", "f4"), ("y", "f4"), ("r", "f4")],
                )

                # Apply these to the mask in memory
                anacal.mask.add_bright_star_mask(
                    mask_array=mask_array, star_array=arr,
                )

                del arr, x_keep, y_keep, r_keep

            del exposure
        del cat0, x, y, mm
    else:
        print(
            "[global] low_redshift gals not found; skipping low-z masking."
        )

    fitsio.write(out_mask_fname, mask_array, clobber=True)
    del mask_array, bbox, patch_info
    return


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Rank 0 reads the list of tracts/patches
    if rank == 0:
        full = fitsio.read(
            os.path.join(os.environ["s23b"], "tracts_fdfc_v1_final.fits")
        )
        selected = full[args.start: args.end]
        del full
    else:
        selected = None

    selected = comm.bcast(selected, root=0)
    my_entries = split_work(selected, size, rank)

    # Set up the sky map configuration (same as your originals)
    config = RingsSkyMapConfig()
    config.numRings = 120
    config.projection = "TAN"
    config.tractOverlap = 1.0 / 60  # degrees
    config.pixelScale = 0.168       # arcsec/pixel
    skymap = RingsSkyMap(config)

    for entry in my_entries:
        process_patch(entry, skymap)
        gc.collect()


if __name__ == "__main__":
    main()
