#!/usr/bin/env python3
"""Measure photo-z point estimates per patch (with shear distortions).

Each entry in tracts_fdfc_v2_final.fits defines one task (tract, patch).
For each task, we select objects by patch from
tracts/{tract}.fits and tracts_color/{tract}.fits (using the 'patch'
column in tracts_color), run flexzboost with 5 distortions, and write
the output to tracts_redshift/{tract}_{patch_id}.fits.
"""

import argparse
import gc
import os
import pickle

import fitsio
import numpy as np
from mpi4py import MPI
from xlens.catalog.redshift import flexzboostEstimator

POINT_KEYS = ("zmode", "z025", "z160", "z500", "z840", "z975", "zbest")

# (suffix, comp, dg) -- suffix "0" is the undistorted version
DISTORTIONS = (
    ("0",  1,  0.00),
    ("1p", 1,  0.01),
    ("1m", 1, -0.01),
    ("2p", 2,  0.01),
    ("2m", 2, -0.01),
)

BASE_DIR = (
    "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/"
    "deepCoadd_anacal_v2"
)
FDFC_PATH = (
    "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/"
    "tracts_fdfc_v2_final.fits"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure photo-z point estimates per patch.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--start", type=int, required=True,
        help="Start index into the fdfc entries.",
    )
    parser.add_argument(
        "--end", type=int, required=True,
        help="End index (exclusive) into the fdfc entries.",
    )
    parser.add_argument(
        "--field", type=str, default="all",
        help="Field name to select (default: all).",
    )
    parser.add_argument(
        "--nmax", type=int, default=0,
        help="Max galaxies per patch (0 = all, for testing).",
    )
    parser.add_argument(
        "--fdfc", type=str, default=FDFC_PATH,
        help=f"Path to fdfc fits file (default: {FDFC_PATH}).",
    )
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print("[warn] Ignoring unknown args:", unknown_args)
    return args


def split_work(data, size, rank):
    return data[rank::size]


def process_patch(entry, zobj, out_dir, nmax=0):
    """Process one (tract, patch) entry."""
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9

    outfname = os.path.join(out_dir, f"{tract_id}_{patch_id}.fits")
    if os.path.isfile(outfname):
        print(f"  Output exists: {outfname}, skipping")
        return

    # Read tract data and select by patch
    tract_path = os.path.join(BASE_DIR, f"tracts/{tract_id}.fits")
    color_path = os.path.join(BASE_DIR, f"tracts_color/{tract_id}.fits")
    ext_path = os.path.join(BASE_DIR, f"tracts_extinction/{tract_id}.fits")

    for fpath in (tract_path, color_path, ext_path):
        if not os.path.isfile(fpath):
            print(f"  Missing: {fpath}, skipping")
            return

    # Use patch column from tracts_color to select rows
    color = fitsio.read(color_path, columns=["patch"])
    patch_mask = color["patch"] == patch_db

    if np.sum(patch_mask) == 0:
        print(f"  No objects for tract={tract_id} patch={patch_db}, skipping")
        return

    rows = np.where(patch_mask)[0]
    if nmax > 0:
        rows = rows[:nmax]

    data_a = fitsio.read(tract_path, rows=rows)
    extinction = fitsio.read(ext_path, rows=rows)

    common_kwargs = dict(
        mag_zero=27.0,
        flux_name="gauss2",
        bands="grizy",
        ref_band="i",
        extinction=extinction,
    )

    n = data_a.shape[0]
    dtype = [("object_id", "i8")] + [
        (f"{key}_{suf}", "f4")
        for suf, _, _ in DISTORTIONS
        for key in POINT_KEYS
    ]
    out = np.empty(n, dtype=dtype)
    out["object_id"] = data_a["object_id"]

    for suf, comp, dg in DISTORTIONS:
        points = zobj.get_z(
            src=data_a,
            comp=comp,
            dg=dg,
            **common_kwargs,
        )
        for key in POINT_KEYS:
            out[f"{key}_{suf}"] = points[key]
        del points
        gc.collect()

    fitsio.write(outfname, out)
    print(f"  Written: {outfname} ({n} objects)")
    del data_a, extinction, out, color
    gc.collect()
    return


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        full = fitsio.read(args.fdfc)
        selected = full[args.start:args.end]
        if args.field != "all":
            sel = selected["field"] == args.field
            selected = selected[sel]
        print(
            f"Processing {len(selected)} entries "
            f"(indices {args.start} to {args.end - 1})"
        )
    else:
        selected = None

    selected = comm.bcast(selected, root=0)
    my_entries = split_work(selected, size, rank)

    # Load photo-z model once per rank
    model_path = os.path.join(BASE_DIR, "model_inform_fzboost_v2.2.pkl")
    with open(model_path, "rb") as f:
        mm = pickle.load(f)
    zobj = flexzboostEstimator(pz_obj=mm)

    out_dir = os.path.join(BASE_DIR, "tracts_redshift")
    os.makedirs(out_dir, exist_ok=True)

    for entry in my_entries:
        tract_id = entry["tract"]
        patch_db = entry["patch"]
        patch_x = patch_db // 100
        patch_y = patch_db % 100
        patch_id = patch_x + patch_y * 9
        if rank == 0:
            print(f"[rank {rank}] tract={tract_id} patch={patch_db} "
                  f"(patch_id={patch_id})")
        process_patch(entry, zobj, out_dir, nmax=args.nmax)
        gc.collect()

    comm.Barrier()
    if rank == 0:
        print("All done.")


if __name__ == "__main__":
    main()
