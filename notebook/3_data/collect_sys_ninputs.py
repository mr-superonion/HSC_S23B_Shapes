#!/usr/bin/env python3

import argparse
import gc
import glob
import os
import numpy as np
from tqdm import tqdm

import fitsio
from mpi4py import MPI

from lsst.afw.image import ExposureF


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


def process_patch(entry):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9

    out_dir = f"{os.environ['s23b_anacal']}/{tract_id}/{patch_id}"
    out_fname = os.path.join(out_dir, "system2.fits")
    if os.path.isfile(out_fname):
        return None
    det_fname = os.path.join(out_dir, "detect.fits")
    catalog = fitsio.read(det_fname)

    pixel_scale = 0.168
    calexp_dir = f"{os.environ['s23b_calexp']}/{tract_id}/{patch_id}/i"
    exp_fname = glob.glob(os.path.join(calexp_dir, "*.fits"))[0]
    bbox = ExposureF.readFits(exp_fname).getBBox()
    begin_x = bbox.beginX
    begin_y = bbox.beginY
    x = np.int_(catalog["x1_det"] / pixel_scale - begin_x)
    y = np.int_(catalog["x2_det"] / pixel_scale - begin_y)
    del catalog, bbox

    out = []
    for band in ["g", "r", "i", "z", "y"]:
        nim_dir = f"{os.environ['s23b_nimg']}/{tract_id}/{patch_id}/{band}"
        nim_fname = glob.glob(os.path.join(nim_dir, "*.fits"))[0]
        nimg = fitsio.read(nim_fname)
        out.append(np.nanmean(nimg[y, x]))
        del nim_dir, nim_fname, nimg
    # names=[
    #     "g_ninputs",
    #     "r_ninputs",
    #     "i_ninputs",
    #     "z_ninputs",
    #     "y_ninputs",
    # ]
    fitsio.write(out_fname, np.array(out))
    del out
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

    pbar = tqdm(total=len(my_entries), desc=f"Rank {rank}", position=rank)
    for entry in my_entries:
        process_patch(entry)
        gc.collect()
        pbar.update(1)
    pbar.close()
    return


if __name__ == "__main__":
    main()
