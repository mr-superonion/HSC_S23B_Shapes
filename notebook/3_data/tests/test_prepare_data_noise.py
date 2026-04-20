#!/usr/bin/env python3
"""Test that prepare_data from FpfsForcePipe and AnacalDetectPipe produce
the same noise_array for the same tract/patch.

This verifies consistency between the noise generation in
make_catalog_8_fpfs_linear_modes_v2.2.py (fpfs path) and
make_catalog_4_force_v2.py (anacal path).
"""

import glob
import os

import fitsio
import lsst.afw.image as afwImage
import numpy as np
from lsst.skymap.ringsSkyMap import RingsSkyMap, RingsSkyMapConfig
from xlens.process_pipe.anacal_detect import (AnacalDetectPipe,
                                              AnacalDetectPipeConfig)
from xlens.process_pipe.fpfs_force import FpfsForcePipe, FpfsForcePipeConfig

band_seed = {
    "g": 3,
    "r": 4,
    "i": 0,
    "z": 1,
    "y": 2,
}


def setup_skymap():
    config = RingsSkyMapConfig()
    config.numRings = 120
    config.projection = "TAN"
    config.tractOverlap = 1.0 / 60
    config.pixelScale = 0.168
    return RingsSkyMap(config)


def read_files(tract_id, patch_id, band):
    calexp_dir = f"{os.environ['s23b_calexp']}/{tract_id}/{patch_id}/{band}"
    exp_fname = glob.glob(os.path.join(calexp_dir, "*.fits"))[0]
    exposure = afwImage.ExposureF.readFits(exp_fname)
    mask_dir = f"{os.environ['s23b_mask_v2']}/{tract_id}/{patch_id}"
    mask_fname = os.path.join(mask_dir, "mask.fits")
    bmask = fitsio.read(mask_fname)
    nim_dir = f"{os.environ['s23b_nimg']}/{tract_id}/{patch_id}/i/"
    nim_fname = glob.glob(os.path.join(nim_dir, "*.fits"))[0]
    bmask = bmask | (fitsio.read(nim_fname) <= 2).astype(np.int16)
    corr_fname = f"{os.environ['s23b_noisecorr']}/{tract_id}.fits"
    noise_corr = fitsio.read(corr_fname)
    return {
        "exposure": exposure,
        "mask": bmask,
        "noise_corr": noise_corr,
    }


def get_one_tract_patch():
    """Read the first available tract/patch from the data list."""
    rootdir = os.environ["s23b"]
    full = fitsio.read(f"{rootdir}/tracts_fdfc_v1_final.fits")
    entry = full[0]
    tract_id = int(entry["tract"])
    patch_db = int(entry["patch"])
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9
    return tract_id, patch_id


def test_noise_array_consistency():
    """Both pipelines must produce identical noise_array for the same input."""
    skymap = setup_skymap()
    tract_id, patch_id = get_one_tract_patch()

    band = "r"
    res = read_files(tract_id, patch_id, band)
    seed = (tract_id * 1000 + patch_id) * 5 + band_seed[band]
    band_use = band  # band != "i", so no None

    # Read detection catalog (needed by fpfs path)
    det_dir = f"{os.environ['s23b_anacal_v2']}/{tract_id}/{patch_id}"
    det_fname = os.path.join(det_dir, "detect.fits")
    detection = fitsio.read(det_fname)

    # --- FpfsForcePipe path (make_catalog_8) ---
    fpfs_config = FpfsForcePipeConfig()
    fpfs_config.fpfs.do_noise_bias_correction = True
    fpfs_config.fpfs.psf_model_type = "object"
    fpfs_config.fpfs.return_only_linear_modes = True
    fpfs_config.fpfs.npix = 64
    fpfs_config.fpfs.sigma_shapelets1 = 0.5657
    fpfs_task = FpfsForcePipe(config=fpfs_config)

    data_fpfs = fpfs_task.fpfs.prepare_data(
        exposure=res["exposure"],
        seed=seed,
        noise_corr=res["noise_corr"],
        detection=detection,
        band=band_use,
        mask_array=res["mask"],
    )

    # --- AnacalDetectPipe path (make_catalog_4) ---
    anacal_config = AnacalDetectPipeConfig()
    anacal_config.anacal.sigma_arcsec = 0.40
    anacal_config.anacal.force_size = True
    anacal_config.anacal.num_epochs = 0
    anacal_config.anacal.do_noise_bias_correction = True
    anacal_config.do_fpfs = False
    anacal_config.fpfs.sigma_shapelets1 = 0.40 * np.sqrt(2.0)
    anacal_task = AnacalDetectPipe(config=anacal_config)

    data_anacal = anacal_task.anacal.prepare_data(
        exposure=res["exposure"],
        seed=seed,
        noise_corr=res["noise_corr"],
        detection=detection,
        band=band_use,
        skyMap=skymap,
        tract=tract_id,
        patch=patch_id,
        mask_array=res["mask"],
    )

    # Check psf_object is not None (psf_model_type="object")
    assert data_fpfs["psf_object"] is not None, "fpfs psf_object is None"

    # Compare noise arrays
    noise_fpfs = data_fpfs["noise_array"]
    noise_anacal = data_anacal["noise_array"]

    assert noise_fpfs is not None, "fpfs noise_array is None"
    assert noise_anacal is not None, "anacal noise_array is None"
    assert noise_fpfs.shape == noise_anacal.shape, (
        f"Shape mismatch: {noise_fpfs.shape} vs {noise_anacal.shape}"
    )
    np.testing.assert_array_equal(
        noise_fpfs,
        noise_anacal,
        err_msg="noise_array differs between fpfs and anacal prepare_data",
    )
    print(
        f"PASSED: noise_array matches for tract={tract_id}, patch={patch_id}, "
        f"band={band}, shape={noise_fpfs.shape}"
    )


if __name__ == "__main__":
    test_noise_array_consistency()
