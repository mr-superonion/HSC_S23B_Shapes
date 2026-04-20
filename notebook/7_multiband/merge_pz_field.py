#!/usr/bin/env python3
"""Merge per-tract redshift files into per-field files.

For each field in FIELDS, reads the list of tracts from
fields_color/{field}.fits, concatenates the corresponding
tracts_redshift/{tract}.fits files, sorts by object_id, and
writes fields_redshift/{field}.fits.

Parallelized over fields via MPI (one field per rank).
"""

import argparse
import gc
import os

import fitsio
import numpy as np
import numpy.lib.recfunctions as rfn
from mpi4py import MPI

BASE = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/deepCoadd_anacal_v2"
TRACT_DIR = f"{BASE}/tracts_redshift"
COLOR_DIR = f"{BASE}/fields_color"
OUT_DIR = f"{BASE}/fields_redshift"

FIELDS = ("spring1", "spring2", "spring3", "autumn1", "autumn2", "hectomap")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge per-tract redshift files into per-field files."
    )
    parser.add_argument(
        "--field", type=str, default="all",
        help="Single field to process (default: all).",
    )
    return parser.parse_args()


def process_field(field):
    out_path = os.path.join(OUT_DIR, f"{field}.fits")
    if os.path.isfile(out_path):
        print(f"[{field}] already exists, skipping")
        return

    # Get tract IDs for this field from fields_color
    color_path = os.path.join(COLOR_DIR, f"{field}.fits")
    if not os.path.isfile(color_path):
        print(f"[{field}] missing color file: {color_path}")
        return
    color = fitsio.read(color_path, columns=["tract"])
    tracts = sorted(set(int(t) for t in color["tract"]))
    print(f"[{field}] {len(tracts)} tracts")
    del color

    arrays = []
    missing = 0
    for tract_id in tracts:
        tpath = os.path.join(TRACT_DIR, f"{tract_id}.fits")
        if not os.path.isfile(tpath):
            missing += 1
            continue
        arrays.append(fitsio.read(tpath))

    if missing:
        print(f"[{field}] warning: {missing} tracts missing")
    if not arrays:
        print(f"[{field}] no data, skipping")
        return

    merged = rfn.stack_arrays(arrays, usemask=False)
    order = np.argsort(merged["object_id"])
    merged = merged[order]

    os.makedirs(OUT_DIR, exist_ok=True)
    fitsio.write(out_path, merged, clobber=True)
    print(f"[{field}] written: {out_path} ({len(merged)} objects)")

    del merged, arrays
    gc.collect()


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.field != "all":
        fields = [args.field]
    else:
        fields = list(FIELDS)

    # Distribute fields across MPI ranks
    my_fields = fields[rank::size]
    for field in my_fields:
        if rank == 0 or len(fields) > 1:
            print(f"[rank {rank}] processing {field}")
        process_field(field)
        gc.collect()

    comm.Barrier()
    if rank == 0:
        print("All done.")


if __name__ == "__main__":
    main()
