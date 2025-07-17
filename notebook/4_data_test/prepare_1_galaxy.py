#!/usr/bin/env python3

import os
import argparse

import fitsio
import numpy as np


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Process patch masks with MPI."
    )
    parser.add_argument(
        "--field", type=str, default="all", required=False, help="field name"
    )
    return parser.parse_args()

def select_data(d, sel):
    outcome = d[sel]
    order = np.argsort(outcome["object_id"])
    outcome = outcome[order]
    return outcome


def main():
    args = parse_args()
    field = args.field
    rootdir = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/deepCoadd_anacal2"
    outdir = f"{rootdir}/tracts/"
    os.makedirs(outdir, exist_ok=True)
    fname = f"{rootdir}/fields/{field}.fits"
    data = fitsio.read(fname)

    rootdir2 = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/db_color"
    outdir2 = f"{rootdir2}/tracts/"
    os.makedirs(outdir2, exist_ok=True)
    fname2 = f"{rootdir2}/fields/{field}.fits"
    data2 = fitsio.read(fname2)

    tract = data2["tract"]
    tlist = np.unique(tract)
    for tt in tlist:
        if os.path.isfile(f"{outdir}/{tt}.fits"):
            print(tt)
        else:
            sel = (tract == tt)
            out1 = select_data(data, sel)
            fitsio.write(f"{outdir}/{tt}.fits", out1)
            del out1
            out2 = select_data(data2, sel)
            fitsio.write(f"{outdir2}/{tt}.fits", out2)
            del out2
    return


if __name__ == "__main__":
    main()
