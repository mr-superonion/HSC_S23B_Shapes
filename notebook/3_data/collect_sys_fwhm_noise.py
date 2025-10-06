#!/usr/bin/env python3

import argparse
import gc
import glob
import os

import fitsio
import numpy as np
from lsst.skymap.ringsSkyMap import RingsSkyMap, RingsSkyMapConfig
from mpi4py import MPI
from numpy.lib import recfunctions as rfn
from tqdm import tqdm
from xlens.process_pipe.match import matchPipe, matchPipeConfig

dm_colnames = [
    "deblend_nChild",
    "deblend_peak_center_x",
    "deblend_peak_center_y",
    "base_Variance_value",
    "base_GaussianFlux_instFlux",
    "base_GaussianFlux_instFluxErr",
    "ext_shapeHSM_HsmPsfMoments_xx",
    "ext_shapeHSM_HsmPsfMoments_yy",
    "ext_shapeHSM_HsmPsfMoments_xy",
    "ext_shapeHSM_HigherOrderMomentsPSF_04",
    "ext_shapeHSM_HigherOrderMomentsPSF_13",
    "ext_shapeHSM_HigherOrderMomentsPSF_31",
    "ext_shapeHSM_HigherOrderMomentsPSF_40",
]


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


def process_patch(entry, skymap, task):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9

    out_dir = f"{os.environ['s23b_anacal']}/{tract_id}/{patch_id}"
    out_fname = os.path.join(out_dir, "system.fits")
    if os.path.isfile(out_fname):
        return None
    det_fname = os.path.join(out_dir, "detect.fits")
    anacal_catalog = fitsio.read(det_fname)

    band = "i"
    cat_dir = f"{os.environ['s23b_meas']}/{tract_id}/{patch_id}/{band}"
    files = glob.glob(os.path.join(cat_dir, "*.fits"))
    cat = rfn.repack_fields(fitsio.read(files[0])[dm_colnames])
    map_dict = {name: f"{band}_" + name for name in dm_colnames}
    dm_catalog = rfn.rename_fields(cat, map_dict)
    del cat, map_dict

    catalog = task.run(
        skyMap=skymap,
        tract=tract_id,
        patch=patch_id,
        catalog=anacal_catalog,
        dm_catalog=dm_catalog,
        truth_catalog=None,
    ).catalog
    del dm_catalog, anacal_catalog
    mag = 27.0 - 2.5 * np.log10(catalog["flux"])
    abse = np.sqrt(catalog["fpfs_e1"] ** 2.0 + catalog["fpfs_e2"] ** 2.0)
    mask = (mag < 25.0) & (abse < 0.3)
    catalog = catalog[mask]

    var_val = np.nanmean(catalog["i_base_Variance_value"])
    pixel_scale = 0.168
    psf_mxx = catalog["i_ext_shapeHSM_HsmPsfMoments_xx"] * pixel_scale**2
    psf_myy = catalog["i_ext_shapeHSM_HsmPsfMoments_yy"] * pixel_scale**2
    psf_mxy = catalog["i_ext_shapeHSM_HsmPsfMoments_xy"] * pixel_scale**2
    size_val = np.nanmean(2.355 * (psf_mxx * psf_myy - psf_mxy**2)**0.25)

    e1_psf = np.nanmean((psf_mxx - psf_myy) / (psf_mxx + psf_myy))
    e2_psf = 2.0 * np.nanmean(psf_mxy / (psf_mxx + psf_myy))
    e1_psf4 = np.nanmean(
        catalog["i_ext_shapeHSM_HigherOrderMomentsPSF_40"] -
        catalog["i_ext_shapeHSM_HigherOrderMomentsPSF_04"]
    )
    e2_psf4 = 2.0 * np.nanmean(
        catalog["i_ext_shapeHSM_HigherOrderMomentsPSF_31"] +
        catalog["i_ext_shapeHSM_HigherOrderMomentsPSF_13"]
    )

    e1 = np.sum(catalog["fpfs_e1"] * catalog["wsel"])
    e2 = np.sum(catalog["fpfs_e2"] * catalog["wsel"])
    r1 = np.sum(
        catalog["fpfs_de1_dg1"] * catalog["wsel"] +
        catalog["dwsel_dg1"] * catalog["fpfs_e1"]
    )
    r2 = np.sum(
        catalog["fpfs_de2_dg2"] * catalog["wsel"] +
        catalog["dwsel_dg2"] * catalog["fpfs_e2"]
    )

    out = np.array([
        var_val, e1_psf, e2_psf, e1_psf4, e2_psf4, size_val,
        e1, e2, r1, r2
    ])
    # names=[
    #     "variance", "e1_psf2", "e2_psf2", "e1_psf4", "e2_psf4",
    #     "fwhm", "sum_e1", "sum_e2", "sum_r1", "sum_r2"
    # ]
    fitsio.write(out_fname, out)
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
