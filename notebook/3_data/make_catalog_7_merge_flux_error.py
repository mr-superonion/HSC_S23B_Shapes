#!/usr/bin/env python3

import argparse
import os

import fitsio
import numpy as np
import numpy.lib.recfunctions as rfn
from mpi4py import MPI


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
    directory = os.path.join(
        os.environ['s23b'],
        f"deepCoadd_flux_variance/{tract_id}/{patch_id}"
    )
    fname = os.path.join(directory, "out.fits")
    dd = fitsio.read(fname)
    return dd


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        rootdir = os.environ["s23b"]
        full = fitsio.read(
            f"{rootdir}/tracts_fdfc_v1_final.fits"
        )
        mm = full["field"] == args.field
        selected = full[mm]
    else:
        selected = None

    selected = comm.bcast(selected, root=0)
    my_entries = split_work(selected, size, rank)
    data = []
    for entry in my_entries:
        out = process_patch(entry)
        data.append(out)
    dtype = [
        ("index", "i8"),
        ("var_g", "f8"),
        ("var_r", "f8"),
        ("var_i", "f8"),
        ("var_z", "f8"),
        ("var_y", "f8"),
    ]
    data = np.vstack(data)
    local_struct = rfn.unstructured_to_structured(data, dtype=dtype)
    gathered = comm.gather(local_struct, root=0)

    if rank == 0:
        merged_struct = rfn.stack_arrays(
            gathered, usemask=False, asrecarray=False, autoconvert=True
        )
        order = np.argsort(merged_struct["index"])
        merged_struct = merged_struct[order]
        field = args.field
        out_dir = os.path.join(
            os.environ['s23b'],
            f"deepCoadd_flux_variance/fields/"
        )
        fitsio.write(
            os.path.join(out_dir, f"{field}.fits"),
            merged_struct,
        )
    return


if __name__ == "__main__":
    main()
