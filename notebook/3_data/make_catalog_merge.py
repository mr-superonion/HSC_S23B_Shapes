#!/usr/bin/env python3

import argparse
import os
from tqdm import tqdm

import fitsio
from mpi4py import MPI
import numpy.lib.recfunctions as rfn
import glob

colnames = [
    "ra",
    "dec",
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
    "g_flux",
    "g_dflux_dg1",
    "g_dflux_dg2",
    "r_flux",
    "r_dflux_dg1",
    "r_dflux_dg2",
    "i_flux",
    "i_dflux_dg1",
    "i_dflux_dg2",
    "z_flux",
    "z_dflux_dg1",
    "z_dflux_dg2",
    "y_flux",
    "y_dflux_dg1",
    "y_dflux_dg2",
    "g_ext_photometryKron_KronFlux_instFlux",
    "g_ext_photometryKron_KronFlux_instFluxErr",
    "g_modelfit_CModel_instFlux",
    "g_modelfit_CModel_instFluxErr",
    "g_base_PsfFlux_instFlux",
    "g_base_PsfFlux_instFluxErr",
    "r_ext_photometryKron_KronFlux_instFlux",
    "r_ext_photometryKron_KronFlux_instFluxErr",
    "r_modelfit_CModel_instFlux",
    "r_modelfit_CModel_instFluxErr",
    "r_base_PsfFlux_instFlux",
    "r_base_PsfFlux_instFluxErr",
    "i_base_Blendedness_abs",
    "i_base_ClassificationExtendedness_value",
    "i_ext_shapeHSM_HsmPsfMoments_xx",
    "i_ext_shapeHSM_HsmPsfMoments_yy",
    "i_ext_shapeHSM_HsmPsfMoments_xy",
    "i_ext_shapeHSM_HigherOrderMomentsPSF_03",
    "i_ext_shapeHSM_HigherOrderMomentsPSF_12",
    "i_ext_shapeHSM_HigherOrderMomentsPSF_21",
    "i_ext_shapeHSM_HigherOrderMomentsPSF_30",
    "i_ext_shapeHSM_HigherOrderMomentsPSF_04",
    "i_ext_shapeHSM_HigherOrderMomentsPSF_13",
    "i_ext_shapeHSM_HigherOrderMomentsPSF_22",
    "i_ext_shapeHSM_HigherOrderMomentsPSF_31",
    "i_ext_shapeHSM_HigherOrderMomentsPSF_40",
    "i_base_CircularApertureFlux_3_0_instFlux",
    "i_base_CircularApertureFlux_3_0_instFluxErr",
    "i_base_Variance_value",
    "i_ext_photometryKron_KronFlux_instFlux",
    "i_ext_photometryKron_KronFlux_instFluxErr",
    "i_modelfit_CModel_instFlux",
    "i_modelfit_CModel_instFluxErr",
    "i_base_PsfFlux_instFlux",
    "i_base_PsfFlux_instFluxErr",
    "z_ext_photometryKron_KronFlux_instFlux",
    "z_ext_photometryKron_KronFlux_instFluxErr",
    "z_modelfit_CModel_instFlux",
    "z_modelfit_CModel_instFluxErr",
    "z_base_PsfFlux_instFlux",
    "z_base_PsfFlux_instFluxErr",
    "y_ext_photometryKron_KronFlux_instFlux",
    "y_ext_photometryKron_KronFlux_instFluxErr",
    "y_modelfit_CModel_instFlux",
    "y_modelfit_CModel_instFluxErr",
    "y_base_PsfFlux_instFlux",
    "y_base_PsfFlux_instFluxErr",
]


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Process patch masks with MPI."
    )
    parser.add_argument("--field", type=str, required=True, help="field name")
    return parser.parse_args()


# Divide data among MPI ranks
def split_work(data, size, rank):
    return data[rank::size]


def process_patch(entry):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9

    bdir = "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/database/"
    basedir = f"{bdir}/s23b-anacal/tracts/{tract_id}/{patch_id}"
    fname = os.path.join(basedir, "match.fits")
    if os.path.isfile(fname):
        dd = fitsio.read(fname)
        dd = dd[dd["wsel"] > 1e-7]
        dd["fpfs_e1"] = dd["fpfs_e1"] * dd["wsel"]
        dd["fpfs_e2"] = -dd["fpfs_e2"] * dd["wsel"]
        dd["fpfs_de1_dg1"] = (
            dd["fpfs_de1_dg1"] * dd["wsel"] + dd["dwsel_dg1"] * dd["fpfs_e1"]
        )
        dd["fpfs_de1_dg2"] = (
            dd["fpfs_de1_dg2"] * dd["wsel"] + dd["dwsel_dg2"] * dd["fpfs_e1"]
        )
        dd["fpfs_de2_dg1"] = (
            dd["fpfs_de2_dg1"] * dd["wsel"] + dd["dwsel_dg1"] * dd["fpfs_e2"]
        )
        dd["fpfs_de2_dg2"] = (
            dd["fpfs_de2_dg2"] * dd["wsel"] + dd["dwsel_dg2"] * dd["fpfs_e2"]
        )
        dd = dd[colnames]
        return dd
    else:
        return None


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        full = fitsio.read(
            "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/tracts_fdfc_v1_trim4.fits"
        )
        mm = full["field"] == args.field
        selected = full[mm]
    else:
        selected = None

    selected = comm.bcast(selected, root=0)
    my_entries = split_work(selected, size, rank)

    data = []
    pbar = tqdm(total=len(my_entries), desc=f"Rank {rank}", position=rank)
    for entry in my_entries:
        out = process_patch(entry)
        if out is not None:
            if len(out) > 2:
                data.append(out)
        pbar.update(1)
    data = rfn.stack_arrays(data, usemask=False)
    field = args.field
    bdir = "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/database/"
    basedir = f"{bdir}/s23b-anacal/"
    fitsio.write(
        os.path.join(basedir, f"{field}_{rank}.fits"),
        data,
    )
    pbar.close()
    comm.Barrier()

    if rank == 0:
        field = args.field
        bdir = "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/database/"
        basedir = f"{bdir}/s23b-anacal/"
        d_all = []
        fnames = glob.glob(os.path.join(basedir, f"{field}_*.fits"))
        for fn in fnames:
            if os.path.isfile(fn):
                d_all.append(
                    fitsio.read(fn)
                )
                os.popen(f"rm {fn}")
        fitsio.write(
            os.path.join(basedir, f"{field}.fits"),
            rfn.stack_arrays(d_all, usemask=False),
        )
    return


if __name__ == "__main__":
    main()
