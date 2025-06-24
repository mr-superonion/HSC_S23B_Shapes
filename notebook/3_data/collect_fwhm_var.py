#!/usr/bin/env python3

import argparse
import gc
import glob
import os
import numpy as np
from tqdm import tqdm
import lsst.afw.table as afwtable
import astropy.table as asttable

from mpi4py import MPI


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process patch masks with MPI."
    )
    parser.add_argument(
        "--start", type=int, required=True, help="Start index of datalist."
    )
    parser.add_argument(
        "--end", type=int, required=True, help="End index of datalist.")
    return parser.parse_args()


# Divide data among MPI ranks
def split_work(data, size, rank):
    return data[rank::size]


def process_patch(entry, comm):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9
    cat_dir = "/lustre/HSC_DR/hsc_ssp/dr4/s23b/data/s23b_wide/unified/deepCoadd_meas"
    band = "i"
    files = glob.glob(os.path.join(cat_dir, f"{tract_id}/{patch_id}/{band}/*"))
    cat = afwtable.SourceCatalog.readFits(files[0])
    mask = (
        cat["detect_isPrimary"] &
        ~cat["base_PixelFlags_flag_saturated"] &
        ~cat["base_PixelFlags_flag_inexact_psfCenter"]
    )
    cat = cat[mask]
    pixel_scale = 0.168
    psf_mxx = cat["ext_shapeHSM_HsmPsfMoments_xx"] * pixel_scale**2
    psf_myy = cat["ext_shapeHSM_HsmPsfMoments_yy"] * pixel_scale**2
    psf_mxy = cat["ext_shapeHSM_HsmPsfMoments_xy"] * pixel_scale**2
    fwhm = 2.355 * (psf_mxx * psf_myy - psf_mxy**2)**0.25
    var = cat["base_Variance_value"]
    fwhm = np.nanmean(fwhm).astype(float)
    var = np.nanmean(var).astype(float)
    del cat
    index = entry["index"]
    return index, fwhm, var


def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        full = asttable.Table.read(
            "/lustre/work/xiangchong.li/work/hsc_s23b_data/catalogs/tracts_fdfc_v1_trim6.fits"
        )
        selected = full[args.start: args.end]
    else:
        selected = None

    selected = comm.bcast(selected, root=0)
    my_entries = split_work(selected, size, rank)

    results = []
    pbar = tqdm(total=len(my_entries), desc=f"Rank {rank}", position=rank)
    for entry in my_entries:
        result = process_patch(entry, comm)
        results.append(result)
        gc.collect()
        pbar.update(1)
    pbar.close()

    # Gather results at rank 0
    all_results = comm.gather(results, root=0)

    if rank == 0:
        # Flatten list of lists
        flat = [item for sublist in all_results for item in sublist]
        indices, fwhms, vars_ = zip(*flat)
        out = asttable.Table(
            [indices, fwhms, vars_],
            names=["index", "fwhm", "var"],
        )
        out.write("fwhm_var_table.fits", overwrite=True)
    return


if __name__ == "__main__":
    main()
