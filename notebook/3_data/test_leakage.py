#!/usr/bin/env python3

import argparse
import gc
import glob
import os
import numpy as np
import astropy.table as astTable
from tqdm import tqdm

import fitsio
from lsst.skymap.ringsSkyMap import RingsSkyMap, RingsSkyMapConfig
from mpi4py import MPI
from xlens.process_pipe.match import (
    matchPipe,
    matchPipeConfig,
)

from numpy.lib import recfunctions as rfn

dm_colnames = [
    "deblend_nChild",
    "deblend_blendedness",
    "deblend_peak_center_x",
    "deblend_peak_center_y",
    "base_Blendedness_abs",
    "base_CircularApertureFlux_3_0_instFlux",
    "base_CircularApertureFlux_3_0_instFluxErr",
    "base_GaussianFlux_instFlux",
    "base_GaussianFlux_instFluxErr",
    "base_PsfFlux_instFlux",
    "base_PsfFlux_instFluxErr",
    "base_Variance_value",
    "modelfit_CModel_instFlux",
    "modelfit_CModel_instFluxErr",
    "base_ClassificationExtendedness_value",
    "ext_shapeHSM_HsmPsfMoments_xx",
    "ext_shapeHSM_HsmPsfMoments_yy",
    "ext_shapeHSM_HsmPsfMoments_xy",
    "ext_shapeHSM_HigherOrderMomentsPSF_04",
    "ext_shapeHSM_HigherOrderMomentsPSF_13",
    "ext_shapeHSM_HigherOrderMomentsPSF_31",
    "ext_shapeHSM_HigherOrderMomentsPSF_40",
]
nbins = 5


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


def compute_e_psf_2(catalog, e1, e2, r1, r2, pixel_scale=0.168):
    psf_mxx = catalog["i_ext_shapeHSM_HsmPsfMoments_xx"] * pixel_scale**2
    psf_myy = catalog["i_ext_shapeHSM_HsmPsfMoments_yy"] * pixel_scale**2
    psf_mxy = catalog["i_ext_shapeHSM_HsmPsfMoments_xy"] * pixel_scale**2

    e1_psf = (psf_mxx - psf_myy) / (psf_mxx + psf_myy) / 2
    e2_psf = psf_mxy / (psf_mxx + psf_myy)

    bins = np.linspace(-0.06, 0.06, nbins + 1)
    bc = 0.5 * (bins[:-1] + bins[1:])
    nom1 = np.histogram(e1_psf, weights=e1, bins=bins)[0]
    denom1 = np.histogram(e1_psf, weights=r1, bins=bins)[0]
    nom2 = np.histogram(e2_psf, weights=e2, bins=bins)[0]
    denom2 = np.histogram(e2_psf, weights=r2, bins=bins)[0]
    return bc, nom1, denom1, nom2, denom2


def compute_e_psf_4(catalog, e1, e2, r1, r2):
    e1_psf4 = (
        catalog["i_ext_shapeHSM_HigherOrderMomentsPSF_40"] -
        catalog["i_ext_shapeHSM_HigherOrderMomentsPSF_04"]
    )
    e2_psf4 = 2.0 * (
        catalog["i_ext_shapeHSM_HigherOrderMomentsPSF_31"] +
        catalog["i_ext_shapeHSM_HigherOrderMomentsPSF_13"]
    )

    bins = np.linspace(-0.03, 0.03, nbins + 1)
    e_psf_4 = 0.5 * (bins[:-1] + bins[1:])
    nom3 = np.histogram(e1_psf4, weights=e1, bins=bins)[0]
    denom3 = np.histogram(e1_psf4, weights=r1, bins=bins)[0]
    nom4 = np.histogram(e2_psf4, weights=e2, bins=bins)[0]
    denom4 = np.histogram(e2_psf4, weights=r2, bins=bins)[0]
    return e_psf_4, nom3, denom3, nom4, denom4


def compute_size(catalog, e1, e2, r1, r2, pixel_scale=0.168):
    psf_mxx = catalog["i_ext_shapeHSM_HsmPsfMoments_xx"] * pixel_scale**2
    psf_myy = catalog["i_ext_shapeHSM_HsmPsfMoments_yy"] * pixel_scale**2
    psf_mxy = catalog["i_ext_shapeHSM_HsmPsfMoments_xy"] * pixel_scale**2
    size_val = 2.355 * (psf_mxx * psf_myy - psf_mxy**2)**0.25

    bins = np.linspace(0.45, 0.75, nbins + 1)
    bc = 0.5 * (bins[:-1] + bins[1:])
    nom5 = np.histogram(size_val, weights=e1, bins=bins)[0]
    denom5 = np.histogram(size_val, weights=r1, bins=bins)[0]
    nom6 = np.histogram(size_val, weights=e2, bins=bins)[0]
    denom6 = np.histogram(size_val, weights=r2, bins=bins)[0]
    return bc, nom5, denom5, nom6, denom6


def compute_variance(catalog, e1, e2, r1, r2):
    var_val = catalog["i_base_Variance_value"]
    bins = np.linspace(0.002, 0.008, nbins + 1)
    var = 0.5 * (bins[:-1] + bins[1:])
    nom7 = np.histogram(var_val, weights=e1, bins=bins)[0]
    denom7 = np.histogram(var_val, weights=r1, bins=bins)[0]
    nom8 = np.histogram(var_val, weights=e2, bins=bins)[0]
    denom8 = np.histogram(var_val, weights=r2, bins=bins)[0]
    return var, nom7, denom7, nom8, denom8


def process_patch(entry, skymap, task, comm):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9

    out_dir = f"{os.environ['s23b_anacal']}/{tract_id}/{patch_id}"
    out_fname = os.path.join(out_dir, "leakage.fits")
    if os.path.isfile(out_fname):
        return None
    det_fname = os.path.join(out_dir, "detect.fits")
    anacal_catalog = fitsio.read(det_fname)
    fpfs_fname = os.path.join(out_dir, "fpfs.fits")
    tmp = fitsio.read(fpfs_fname)
    anacal_catalog["fpfs_e1"] = tmp["fpfs1_e1"]
    anacal_catalog["fpfs_de1_dg1"] = tmp["fpfs1_de1_dg1"]
    anacal_catalog["fpfs_e2"] = tmp["fpfs1_e2"]
    anacal_catalog["fpfs_de2_dg2"] = tmp["fpfs1_de2_dg2"]
    del tmp

    band = "i"
    cat_dir = f"{os.environ['s23b_meas']}/{tract_id}/{patch_id}/i"
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
    mask = (catalog["mask_value"] < 10) & (mag < 25.0)
    catalog = catalog[mask]

    e1 = catalog["fpfs_e1"] * catalog["wsel"]
    e2 = catalog["fpfs_e2"] * catalog["wsel"]
    r1 = (
        catalog["fpfs_de1_dg1"] * catalog["wsel"] +
        catalog["dwsel_dg1"] * catalog["fpfs_e1"]
    )
    r2 = (
        catalog["fpfs_de2_dg2"] * catalog["wsel"] +
        catalog["dwsel_dg2"] * catalog["fpfs_e2"]
    )

    e_psf_2, nom1, denom1, nom2, denom2 = compute_e_psf_2(
        catalog, e1, e2, r1, r2)
    e_psf_4, nom3, denom3, nom4, denom4 = compute_e_psf_4(
        catalog, e1, e2, r1, r2)
    size, nom5, denom5, nom6, denom6 = compute_size(
        catalog, e1, e2, r1, r2)
    var, nom7, denom7, nom8, denom8 = compute_variance(
        catalog, e1, e2, r1, r2)

    out = astTable.Table(
        [
            e_psf_2, nom1, denom1, nom2, denom2,
            e_psf_4, nom3, denom3, nom4, denom4,
            size, nom5, denom5, nom6, denom6,
            var, nom7, denom7, nom8, denom8,
        ],
        names=[
            "e_psf_2", "e1_2", "r1_2", "e2_2", "r2_2",
            "e_psf_4", "e1_4", "r1_4", "e2_4", "r2_4",
            "size", "e1_s", "r1_s", "e2_s", "r2_s",
            "var", "e1_v", "r1_v", "e2_v", "r2_v",
        ]
    )
    out.write(out_fname)
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
        process_patch(entry, skymap, task, comm)
        gc.collect()
        pbar.update(1)
    pbar.close()
    return


if __name__ == "__main__":
    main()
