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


def process_tract(tract_id, patch_list, field, skymap):
    fname = os.path.join(
        os.environ["s23b"], "db_star", f"{tract_id}.fits"
    )
    data = np.array(fitsio.read(fname))
    fname2 = os.path.join(
        os.environ["s23b"], "db_star2", f"{tract_id}.fits"
    )
    data2 = np.array(fitsio.read(fname2))

    # Verify object_id columns match before masking
    assert np.array_equal(data["object_id"], data2["object_id"]), (
        f"object_id mismatch in tract {tract_id}"
    )

    # Select patches and combine
    mask = np.isin(data["patch"], patch_list)
    data = rfn.repack_fields(data[mask])
    data2 = rfn.repack_fields(data2[mask])

    # Drop duplicate columns from data2 before merging
    cols2 = [c for c in data2.dtype.names if c not in data.dtype.names]
    if cols2:
        data2 = rfn.repack_fields(data2[cols2])
        combined = rfn.merge_arrays([data, data2], flatten=True)
    else:
        combined = data

    combined_fname = os.path.join(
        os.environ["s23b"], "db_star2", f"{tract_id}_combined.fits"
    )
    fitsio.write(combined_fname, combined)

    # Apply pixel mask per patch (same as make_random_catalog_v2.py)
    cat_all = []
    for patch_db in patch_list:
        mm = (combined["patch"] == patch_db)
        cat = combined[mm]
        if len(cat) == 0:
            continue
        patch_x = patch_db // 100
        patch_y = patch_db % 100
        patch_id = patch_x + patch_y * 9
        patch_info = skymap[tract_id][patch_id]
        wcs = patch_info.getWcs()
        bbox = patch_info.getOuterBBox()
        mask_dir = f"{os.environ['s23b_mask_v2']}/{tract_id}/{patch_id}"
        mask_fname = os.path.join(mask_dir, "mask.fits")
        bmask = fitsio.read(mask_fname)
        nim_dir = f"{os.environ['s23b_nimg']}/{tract_id}/{patch_id}/i/"
        nim_fname = glob.glob(os.path.join(nim_dir, "*.fits"))[0]
        bmask = (bmask | (fitsio.read(nim_fname) <= 2).astype(np.int16))
        x, y = wcs.skyToPixelArray(
            ra=cat["i_ra"],
            dec=cat["i_dec"],
            degrees=True,
        )
        x = np.round(x - bbox.getBeginX()).astype(int)
        y = np.round(y - bbox.getBeginY()).astype(int)
        cat_all.append(cat[bmask[y, x] == 0])

    if len(cat_all) == 0:
        return
    cat_all = rfn.stack_arrays(cat_all, usemask=False, asrecarray=False)
    order = np.argsort(cat_all["object_id"])
    cat_all = cat_all[order]

    out_fname = os.path.join(
        os.environ["s23b"], "db_star2", "fields", f"{field}_{tract_id}.fits"
    )
    fitsio.write(out_fname, cat_all)
    return


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rootdir = os.environ["s23b"]
    full = fitsio.read(
        f"{rootdir}/tracts_fdfc_v1_final.fits"
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
        process_tract(tract_id, patch_list, args.field, skymap)
        gc.collect()
        pbar.update(1)
    pbar.close()

    comm.Barrier()
    if rank == 0:
        field = args.field
        out_dir = os.path.join(
            os.environ["s23b"], "db_star2",
        )
        d_all = []
        fnames = glob.glob(os.path.join(out_dir, "fields", f"{field}_*.fits"))
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
            os.path.join(out_dir, "fields", f"{field}.fits"),
            outcome,
        )
    return


if __name__ == "__main__":
    main()
