#!/usr/bin/env python3

import argparse
import glob
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

    base_dir = f"{os.environ['s23b_anacal_v2']}/{tract_id}/{patch_id}"
    fname = os.path.join(base_dir, "fpfs_linear.fits")
    fname2 = os.path.join(base_dir, "match.fits")
    sel_fname = os.path.join(base_dir, "fdfc_sel.fits")
    if os.path.isfile(fname):
        dd = np.array(fitsio.read(fname))
        dd2 = np.array(fitsio.read(fname2))
        sel = (fitsio.read(sel_fname) > 0)
        if np.sum(sel) < 3:
            return None
        dd2 = dd2[sel]
        dd = dd[dd2["index"]]
        C = 0.53
        m00 = dd["fpfs1_m00"]
        m40 = dd["fpfs1_m40"]
        m22c, m22s = dd["fpfs1_m22c"], dd["fpfs1_m22s"]
        m44c, m44s = dd["fpfs1_m44c"], dd["fpfs1_m44s"]
        n00 = dd["fpfs1_n00"]
        n40 = dd["fpfs1_n40"]
        n22c, n22s = dd["fpfs1_n22c"], dd["fpfs1_n22s"]
        n44c, n44s = dd["fpfs1_n44c"], dd["fpfs1_n44s"]
        m00_r = m00 - 2.0 * n00
        m40_r = m40 - 2.0 * n40
        m22c_r = m22c - 2.0 * n22c
        m22s_r = m22s - 2.0 * n22s
        m44c_r = m44c - 2.0 * n44c
        m44s_r = m44s - 2.0 * n44s
        rt2 = np.sqrt(2.0)
        e44c_r = m44c_r / (m00 + C)
        e44s_r = m44s_r / (m00 + C)
        e22c_r = m22c_r / (m00 + C)
        e22s_r = m22s_r / (m00 + C)
        R_r = 1.0 / rt2 * (m00_r - m40_r) / (m00 + C)
        out = np.empty(len(dd), dtype=[
            ("object_id",  "i8"),
            ("e22c",  "f8"),
            ("e22s",  "f8"),
            ("e44c",  "f8"),
            ("e44s",  "f8"),
            ("R",  "f8"),
        ])
        out["object_id"]  = dd2["object_id"]
        out["e22c"]  = e22c_r
        out["e22s"]  = -e22s_r
        out["e44c"]  = e44c_r
        out["e44s"]  = -e44s_r
        out["R"]  = R_r
        return out
    else:
        return None


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        rootdir = os.environ["s23b"]
        full = fitsio.read(
            f"{rootdir}/tracts_fdfc_v2_final.fits"
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
        if out is not None:
            data.append(out)

    data = rfn.stack_arrays(data, usemask=False)
    field = args.field
    out_dir = os.path.join(os.environ['s23b_anacal_v2'], "fields")
    fitsio.write(
        os.path.join(out_dir, f"{field}_{rank}.fits"),
        data,
    )
    comm.Barrier()

    if rank == 0:
        field = args.field
        out_dir = os.path.join(os.environ['s23b_anacal_v2'], "fields")
        d_all = []
        fnames = glob.glob(os.path.join(out_dir, f"{field}_*.fits"))
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
            os.path.join(out_dir, f"{field}_response.fits"),
            outcome,
        )
    return


if __name__ == "__main__":
    main()
