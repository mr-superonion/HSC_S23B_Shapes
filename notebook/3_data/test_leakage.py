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
    "ext_shapeHSM_HigherOrderMomentsPSF_03",
    "ext_shapeHSM_HigherOrderMomentsPSF_12",
    "ext_shapeHSM_HigherOrderMomentsPSF_21",
    "ext_shapeHSM_HigherOrderMomentsPSF_30",
    "ext_shapeHSM_HigherOrderMomentsPSF_04",
    "ext_shapeHSM_HigherOrderMomentsPSF_13",
    "ext_shapeHSM_HigherOrderMomentsPSF_22",
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
        "--field", type=str, required=True, help="field name"
    )
    return parser.parse_args()


# Divide data among MPI ranks
def split_work(data, size, rank):
    return data[rank::size]


def process_patch(entry, skymap, task, comm):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9

    bdir = "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/database/"
    outdir = f"{bdir}/s23b-anacal/tracts/{tract_id}/{patch_id}"
    det_fname = os.path.join(outdir, "detect.fits")
    out_fname = os.path.join(outdir, "leakage.fits")
    if os.path.isfile(out_fname) or (not os.path.isfile(det_fname)):
        return None

    cat_dir = "/lustre/HSC_DR/hsc_ssp/dr4/s23b/data/s23b_wide/unified/deepCoadd_meas"
    band = "i"
    files = glob.glob(os.path.join(cat_dir, f"{tract_id}/{patch_id}/{band}/*"))
    cat = rfn.repack_fields(fitsio.read(files[0])[dm_colnames])
    map_dict = {name: f"{band}_" + name for name in dm_colnames}
    dm_catalog = rfn.rename_fields(cat, map_dict)
    del cat, map_dict
    catalog = task.run(
        skyMap=skymap,
        tract=tract_id,
        patch=patch_id,
        catalog=fitsio.read(det_fname),
        dm_catalog=dm_catalog,
        truth_catalog=None,
    ).catalog
    del dm_catalog
    mag = 27.0 - 2.5 * np.log10(catalog["flux"])
    mask = (catalog["mask_value"] < 20) & (mag < 24.5)
    catalog = catalog[mask]
    psf_mxx = catalog["i_ext_shapeHSM_HsmPsfMoments_xx"]
    psf_myy = catalog["i_ext_shapeHSM_HsmPsfMoments_yy"]
    psf_mxy = catalog["i_ext_shapeHSM_HsmPsfMoments_xy"]
    e1_psf = (psf_mxx - psf_myy) / (psf_mxx + psf_myy) / 2.0
    e2_psf = psf_mxy / (psf_mxx + psf_myy)
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
    bins = np.linspace(-0.1, 0.1, 6)
    nom1 = np.histogram(
        e1_psf, weights=e1, density=False, bins=bins
    )[0]
    denom1 = np.histogram(
        e1_psf, weights=r1, density=False, bins=bins
    )[0]
    nom2 = np.histogram(
        e2_psf, weights=e2, density=False, bins=bins
    )[0]
    denom2 = np.histogram(
        e2_psf, weights=r2, density=False, bins=bins
    )[0]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Add bin_centers as the first column
    out = astTable.Table(
        [bin_centers, nom1, denom1, nom2, denom2],
        names=["bin_center", "e1", "r1", "e2", "r2"]
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
            "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/tracts_fdfc_v1_trim6.fits"
        )
        selected = full[args.start: args.end]
        sel = (selected["field"] == args.field)
        selected = selected[sel]
        print(len(selected))
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
