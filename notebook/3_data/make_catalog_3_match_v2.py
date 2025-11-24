#!/usr/bin/env python3

import argparse
import gc
import glob
import os

import fitsio
import numpy as np
from lsst.afw.image import ExposureF
from lsst.afw.table import SourceCatalog
from lsst.skymap.ringsSkyMap import RingsSkyMap, RingsSkyMapConfig
from mpi4py import MPI
from numpy.lib import recfunctions as rfn
from tqdm import tqdm
from xlens.process_pipe.match import matchPipe, matchPipeConfig

dm_colnames = [
    "id",
    "deblend_nChild",
    "base_SdssCentroid_x",
    "base_SdssCentroid_y",
    "base_Variance_value",
    "base_GaussianFlux_instFlux",
    "base_GaussianFlux_instFluxErr",
    "ext_shapeHSM_HsmPsfMoments_xx",
    "ext_shapeHSM_HsmPsfMoments_yy",
    "ext_shapeHSM_HsmPsfMoments_xy",
    "ext_shapeHSM_HigherOrderMomentsPSF_04",
    "ext_shapeHSM_HigherOrderMomentsPSF_13",
    "ext_shapeHSM_HigherOrderMomentsPSF_22",
    "ext_shapeHSM_HigherOrderMomentsPSF_31",
    "ext_shapeHSM_HigherOrderMomentsPSF_40",
    "base_LocalWcs_CDMatrix_1_1",
    "base_LocalWcs_CDMatrix_1_2",
    "base_LocalWcs_CDMatrix_2_1",
    "base_LocalWcs_CDMatrix_2_2",
]

colnames = [
    "i_id",
    "index",
    "ra",
    "dec",
    "wsel",
    "dwsel_dg1",
    "dwsel_dg2",
    "fpfs_e1",
    "fpfs_de1_dg1",
    "fpfs_de1_dg2",
    "fpfs_e2",
    "fpfs_de2_dg1",
    "fpfs_de2_dg2",
    "fpfs_m0",
    "fpfs_dm0_dg1",
    "fpfs_dm0_dg2",
    "fpfs_m2",
    "fpfs_dm2_dg1",
    "fpfs_dm2_dg2",
    "e1_psf2",
    "e2_psf2",
    "e1_psf4",
    "e2_psf4",
    "fwhm_psf",
    "noise_variance",
    "flux_gauss2",
    "dflux_gauss2_dg1",
    "dflux_gauss2_dg2",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Process patch masks with MPI.")
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


def process_patch(entry, skymap, task):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9

    out_dir = f"{os.environ['s23b_anacal_v2']}/{tract_id}/{patch_id}"
    out_fname = os.path.join(out_dir, "match.fits")
    if os.path.isfile(out_fname):
        return None

    det_fname = os.path.join(out_dir, "detect.fits")
    cat_dir = f"{os.environ['s23b_meas']}/{tract_id}/{patch_id}/i"
    files = glob.glob(os.path.join(cat_dir, "*.fits"))
    cat = SourceCatalog.readFits(files[0])
    mask = cat["detect_isPrimary"]
    cat = rfn.repack_fields(
        cat.asAstropy().as_array()[dm_colnames][mask]
    )
    map_dict = {name: "i_" + name for name in dm_colnames}
    cat = rfn.rename_fields(cat, map_dict)
    del map_dict

    catalog = np.array(fitsio.read(det_fname))
    fpfs_fname = os.path.join(out_dir, "fpfs.fits")
    tmp = np.array(fitsio.read(fpfs_fname))
    catalog["fpfs_e1"] = tmp["fpfs1_e1"]
    catalog["fpfs_de1_dg1"] = tmp["fpfs1_de1_dg1"]
    catalog["fpfs_e2"] = tmp["fpfs1_e2"]
    catalog["fpfs_de2_dg2"] = tmp["fpfs1_de2_dg2"]
    del tmp
    index = np.arange(len(catalog))
    catalog = rfn.append_fields(
        catalog, names="index", data=index, dtypes="i4", usemask=False
    )

    catalog = task.run(
        skyMap=skymap,
        tract=tract_id,
        patch=patch_id,
        catalog=catalog,
        dm_catalog=cat,
        truth_catalog=None,
    ).catalog
    del cat

    pixel_scale = 0.168
    ff = (180 / np.pi) * 3600 / pixel_scale
    j11 = catalog["i_base_LocalWcs_CDMatrix_1_1"] * ff * -1
    j12 = catalog["i_base_LocalWcs_CDMatrix_1_2"] * ff
    j21 = catalog["i_base_LocalWcs_CDMatrix_2_1"] * ff * -1
    j22 = catalog["i_base_LocalWcs_CDMatrix_2_2"] * ff
    kappa = (j11 + j22) / 2.0 - 1
    ff2 = 1.0 / (1 + kappa)
    j11 = j11 * ff2
    j12 = j12 * ff2
    j21 = j21 * ff2
    j22 = j22 * ff2
    rho = (j21 - j12) / 2.0
    g1 = (j11 - j22) / 2.0
    g2 = (j12 + j21) / 2.0
    catalog["fpfs_e1"] = catalog["fpfs_e1"] + g1 * catalog["fpfs_de1_dg1"]
    catalog["fpfs_e2"] = -catalog["fpfs_e2"] + g2 * catalog["fpfs_de2_dg2"]

    psf_mxx = catalog["i_ext_shapeHSM_HsmPsfMoments_xx"] * pixel_scale**2
    psf_myy = catalog["i_ext_shapeHSM_HsmPsfMoments_yy"] * pixel_scale**2
    psf_mxy = catalog["i_ext_shapeHSM_HsmPsfMoments_xy"] * pixel_scale**2
    psf_fwhm = 2.355 * (psf_mxx * psf_myy - psf_mxy**2)**0.25
    e1_psf2 = (psf_mxx - psf_myy) / (psf_mxx + psf_myy)
    e2_psf2 = psf_mxy / (psf_mxx + psf_myy) * -2.0
    e1_psf4 = (
        catalog["i_ext_shapeHSM_HigherOrderMomentsPSF_40"] -
        catalog["i_ext_shapeHSM_HigherOrderMomentsPSF_04"]
    )
    e2_psf4 = -2.0 * (
        catalog["i_ext_shapeHSM_HigherOrderMomentsPSF_31"] +
        catalog["i_ext_shapeHSM_HigherOrderMomentsPSF_13"]
    )
    noise_variance = catalog["i_base_Variance_value"]
    catalog = rfn.append_fields(
        catalog,
        names=[
            "e1_psf2", "e2_psf2", "e1_psf4", "e2_psf4", "fwhm_psf",
            "noise_variance"
        ],
        data=[e1_psf2, e2_psf2, e1_psf4, e2_psf4, psf_fwhm, noise_variance],
        dtypes=["f4"] * 6,
        usemask=False,
    )

    calexp_dir = f"{os.environ['s23b_calexp']}/{tract_id}/{patch_id}/i"
    exp_fname = glob.glob(os.path.join(calexp_dir, "*.fits"))[0]
    bbox = ExposureF.readFits(exp_fname).getBBox()
    begin_x = bbox.beginX
    begin_y = bbox.beginY
    del bbox
    x = np.int_(catalog["x1_det"] / pixel_scale - begin_x)
    y = np.int_(catalog["x2_det"] / pixel_scale - begin_y)
    inputs = {}
    for band in ["g", "r", "i", "z", "y"]:
        nim_dir = f"{os.environ['s23b_nimg']}/{tract_id}/{patch_id}/{band}"
        nim_fname = glob.glob(os.path.join(nim_dir, "*.fits"))[0]
        nimg = fitsio.read(nim_fname)
        inputs[band] = nimg[y, x]
    mask = (
        (inputs["g"] >=2) &
        (inputs["r"] >=2) &
        (inputs["i"] >=3) &
        (inputs["z"] >=2) &
        (inputs["y"] >=2)
    )
    catalog = rfn.repack_fields(
        catalog[colnames][mask]
    )
    catalog = rfn.rename_fields(
        catalog,
        {
            "i_id": "object_id",
            "fpfs_e1": "e1",
            "fpfs_de1_dg1": "de1_dg1",
            "fpfs_de1_dg2": "de1_dg2",
            "fpfs_e2": "e2",
            "fpfs_de2_dg1": "de2_dg1",
            "fpfs_de2_dg2": "de2_dg2",
            "fpfs_m0": "m0",
            "fpfs_dm0_dg1": "dm0_dg1",
            "fpfs_dm0_dg2": "dm0_dg2",
            "fpfs_m2": "m2",
            "fpfs_dm2_dg1": "dm2_dg1",
            "fpfs_dm2_dg2": "dm2_dg2",
            "flux_gauss2": "flux",
            "dflux_gauss2_dg1": "dflux_dg1",
            "dflux_gauss2_dg2": "dflux_dg2",
        }
    )
    fitsio.write(out_fname, catalog)
    print(tract_id, patch_id, "finished")
    return


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        rootdir = os.environ["s23b"]
        full = fitsio.read(
            f"{rootdir}/tracts_fdfc_v1_final.fits"
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

    config = matchPipeConfig()
    task = matchPipe(config=config)

    pbar = tqdm(total=len(my_entries), desc=f"Rank {rank}", position=rank)
    for entry in my_entries:
        process_patch(entry, skymap, task)
        gc.collect()
        pbar.update(1)
    pbar.close()
    return


if __name__ == "__main__":
    main()
