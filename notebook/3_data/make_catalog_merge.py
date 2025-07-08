#!/usr/bin/env python3

import argparse
import os
from tqdm import tqdm

import fitsio
from mpi4py import MPI
import numpy.lib.recfunctions as rfn
import glob

colnames = [
    "object_id",
    "ra",
    "dec",
    "wsel",
    "dwsel_dg1",
    "dwsel_dg2",
    "e1",
    "de1_dg1",
    "de1_dg2",
    "e2",
    "de2_dg1",
    "de2_dg2",
    "m0",
    "dm0_dg1",
    "dm0_dg2",
    "m2",
    "dm2_dg1",
    "dm2_dg2",
    "flux",
    "dflux_dg1",
    "dflux_dg2",
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

    base_dir = f"{os.environ['s23b_anacal2']}/{tract_id}/{patch_id}"
    fname = os.path.join(base_dir, "match.fits")
    if os.path.isfile(fname):
        dd = fitsio.read(fname)
        dd = dd[dd["wsel"] > 1e-7]
        dd = rfn.repack_fields(
            dd[colnames]
        )
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
            "tracts_fdfc_v1_final.fits"
        )
        mm = full["field"] == args.field
        selected = full[mm]
        print(len(selected))
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
            print(len(out))
        pbar.update(1)

    data = rfn.stack_arrays(data, usemask=False)
    field = args.field
    base_dir2 = os.environ['s23b_anacal2']
    fitsio.write(
        os.path.join(base_dir2, f"{field}_{rank}.fits"),
        data,
    )
    pbar.close()
    comm.Barrier()

    if rank == 0:
        field = args.field
        base_dir2 = os.environ['s23b_anacal2']
        d_all = []
        fnames = glob.glob(os.path.join(base_dir2, f"{field}_*.fits"))
        for fn in fnames:
            if os.path.isfile(fn):
                d_all.append(
                    fitsio.read(fn)
                )
                os.popen(f"rm {fn}")
        fitsio.write(
            os.path.join(base_dir2, f"{field}.fits"),
            rfn.stack_arrays(d_all, usemask=False),
        )
    return


if __name__ == "__main__":
    main()
