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


def process_tract(tract_id, skymap, patch_list):
    fname = os.path.join(
        os.environ["s23b"], "db_star", f"{tract_id}.fits"
    )
    data = np.array(fitsio.read(fname))
    mask = np.isin(data["patch"], patch_list)
    data = rfn.repack_fields(
        data[mask]
    )
    out_fname = os.path.join(
        os.environ["s23b"], "db_star", "fields", f"tmp_{tract_id}.fits"
    )
    fitsio.write(out_fname, data)
    # tract_info = skymap[tract_id]
    # wcs = tract_info.getWcs()
    # for patch_db in patch_list:
    #     patch_x = patch_db // 100
    #     patch_y = patch_db % 100
    #     patch_id = patch_x + patch_y * 9

    #     patch_info = tract_info[patch_id]
    #     bbox = patch_info.getOuterBBox()
    return


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    full = fitsio.read(
        "tracts_fdfc_v1_final.fits"
    )
    if rank == 0:
        tract_all, idx = np.unique(full["tract"], return_index=True)
        field_list = full[idx]["field"]
        mm = (field_list == args.field)
        tract_all = tract_all[mm]
    else:
        tract_all = None

    tract_all = comm.bcast(tract_all, root=0)
    tract_list = split_work(tract_all, size, rank)

    # Set up the configuration
    config = RingsSkyMapConfig()
    config.numRings = 120
    config.projection = "TAN"
    config.tractOverlap = 1.0 / 60  # degrees
    config.pixelScale = 0.168  # arcsec/pixel
    skymap = RingsSkyMap(config)

    pbar = tqdm(total=len(tract_list), desc=f"Rank {rank}", position=rank)
    for tract_id in tract_list:
        patch_list = full["patch"][full["tract"] == tract_id]
        process_tract(tract_id, skymap, patch_list)
        gc.collect()
        pbar.update(1)
    pbar.close()

    comm.Barrier()
    if rank == 0:
        field = args.field
        out_dir = os.path.join(
            os.environ["s23b"], "db_star",
        )
        d_all = []
        fnames = glob.glob(os.path.join(out_dir, "fields", "tmp_*.fits"))
        for fn in fnames:
            if os.path.isfile(fn):
                d_all.append(
                    fitsio.read(fn)
                )
                os.remove(fn)
        outcome = rfn.stack_arrays(d_all, usemask=False)
        order = np.argsort(outcome["object_id"])
        outcome = outcome[order]
        fitsio.write(
            os.path.join(out_dir, f"{field}.fits"),
            outcome,
        )
    return


if __name__ == "__main__":
    main()
