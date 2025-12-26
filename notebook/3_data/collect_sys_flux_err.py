#!/usr/bin/env python3
import argparse
import gc
import os

import fitsio
import numpy as np
from mpi4py import MPI

OUT_DTYPE = [
    ("index", "i4"),
    ("tract_id", "i4"),
    ("patch_db", "i4"),
    ("patch_id", "i4"),
]
for b in "grizy":
    OUT_DTYPE.extend([
        (f"{b}_flux_gauss0_err", "f8"),
        (f"{b}_flux_gauss2_err", "f8"),
        (f"{b}_flux_gauss4_err", "f8"),
    ])
OUT_DTYPE = np.dtype(OUT_DTYPE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process patches with MPI, gather structured results to rank 0."
    )
    parser.add_argument(
        "--start", type=int, required=True, help="Start index of datalist."
    )
    parser.add_argument(
        "--end", type=int, required=True, help="End index of datalist."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output FITS table path (rank 0 only). Default: $S23B/flux_err.fits",
    )
    return parser.parse_args()


def split_work(data, size, rank):
    return data[rank::size]


def process_patch(entry):
    tract_id = int(entry["tract"])
    patch_db = int(entry["patch"])
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9

    out_dir = f"{os.environ['s23b_anacal_v2']}/{tract_id}/{patch_id}"
    fname = os.path.join(out_dir, "force.fits")
    cat = np.array(fitsio.read(fname))

    row = np.zeros(1, dtype=OUT_DTYPE)
    row["index"] = int(entry["index"])
    row["tract_id"] = tract_id
    row["patch_db"] = patch_db
    row["patch_id"] = patch_id
    for b in "grizy":
        row[f"{b}_flux_gauss0_err"] = np.nanmean(cat[f"{b}_flux_gauss0_err"])
        row[f"{b}_flux_gauss2_err"] = np.nanmean(cat[f"{b}_flux_gauss2_err"])
        row[f"{b}_flux_gauss4_err"] = np.nanmean(cat[f"{b}_flux_gauss4_err"])
    return row


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        rootdir = os.environ["s23b"]
        full = fitsio.read(f"{rootdir}/tracts_fdfc_v1_final.fits")
        selected = full[args.start:args.end]
    else:
        selected = None

    selected = comm.bcast(selected, root=0)
    my_entries = split_work(selected, size, rank)

    # Local structured results
    local_rows = []
    for entry in my_entries:
        row = process_patch(entry)
        if row is not None:
            local_rows.append(row)

    if local_rows:
        local_arr = np.concatenate(local_rows)  # structured, shape (n_local,)
    else:
        local_arr = np.empty(0, dtype=OUT_DTYPE)

    gathered = comm.gather(local_arr, root=0)

    if rank == 0:
        merged = np.concatenate(gathered) if any(a.size for a in gathered) else np.empty(0, dtype=OUT_DTYPE)
        print(f"[rank 0] merged rows = {merged.size}")

        outfname = args.out or f"{os.environ['s23b_anacal_v2']}/flux_err.fits"
        # Write as a FITS *table* (structured array)
        fitsio.write(outfname, merged, clobber=True)
        print(f"[rank 0] wrote: {outfname}")

    gc.collect()


if __name__ == "__main__":
    main()
