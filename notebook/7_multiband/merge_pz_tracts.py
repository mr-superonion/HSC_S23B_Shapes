#!/usr/bin/env python3
"""Merge per-patch redshift files into per-tract files.

Reads all {tract}_{patch_id}.fits files from tracts_redshift/, groups
by tract, concatenates, sorts by object_id, and writes {tract}.fits
back to the same directory. After writing, the per-patch files are
removed.

Parallelized over tracts via MPI.
"""

import argparse
import gc
import glob
import os
import re

import fitsio
import numpy as np
import numpy.lib.recfunctions as rfn
from mpi4py import MPI

REDSHIFT_DIR = (
    "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/"
    "deepCoadd_anacal_v2/tracts_redshift"
)

PATCH_RE = re.compile(r"^(\d+)_(\d+)\.fits$")


def split_work(data, size, rank):
    return data[rank::size]


def process_tract(tract_id):
    """Concatenate all patches of a tract and write {tract}.fits."""
    pattern = os.path.join(REDSHIFT_DIR, f"{tract_id}_*.fits")
    files = sorted(glob.glob(pattern))
    if not files:
        return

    out_path = os.path.join(REDSHIFT_DIR, f"{tract_id}.fits")
    if os.path.isfile(out_path):
        print(f"  {out_path} already exists, skipping")
        return

    arrays = []
    for fn in files:
        arrays.append(fitsio.read(fn))
    merged = rfn.stack_arrays(arrays, usemask=False)
    order = np.argsort(merged["object_id"])
    merged = merged[order]

    fitsio.write(out_path, merged, clobber=True)
    print(
        f"  Written: {out_path} ({len(merged)} objects"
    )

    del merged, arrays
    # Remove per-patch files after successful write
    for fn in files:
        os.remove(fn)
    gc.collect()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    # Ignore any unknown args passed by submit.sh (--start/--end/--band/...)
    parser.parse_known_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Collect unique tract IDs from per-patch filenames
        tracts = set()
        for fname in os.listdir(REDSHIFT_DIR):
            m = PATCH_RE.match(fname)
            if m:
                tracts.add(int(m.group(1)))
        tracts = sorted(tracts)
        print(f"Found {len(tracts)} tracts with per-patch files")
    else:
        tracts = None

    tracts = comm.bcast(tracts, root=0)
    my_tracts = split_work(tracts, size, rank)

    for tract_id in my_tracts:
        if rank == 0:
            print(f"[rank {rank}] tract {tract_id}")
        process_tract(tract_id)
        gc.collect()

    comm.Barrier()
    if rank == 0:
        print("All done.")


if __name__ == "__main__":
    main()
