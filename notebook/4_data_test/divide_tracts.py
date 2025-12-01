#!/usr/bin/env python3
import argparse
import os

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
    rootdir = os.environ['s23b_anacal_v2']
    outdir = f"{rootdir}/tracts/"
    os.makedirs(outdir, exist_ok=True)
    fname = f"{rootdir}/fields/{field}.fits"
    data = fitsio.read(fname)

    outdir2 = f"{rootdir}/tracts_color/"
    os.makedirs(outdir2, exist_ok=True)
    fname2 = f"{rootdir}/fields_color/{field}.fits"
    data2 = fitsio.read(fname2)

    tract = data2["tract"]
    tlist = np.unique(tract)
    for tt in tlist:
        sel = (tract == tt)
        if not os.path.isfile(f"{outdir}/{tt}.fits"):
            out1 = select_data(data, sel)
            fitsio.write(f"{outdir}/{tt}.fits", out1)
            del out1
        if not os.path.isfile(f"{outdir2}/{tt}.fits"):
            out2 = select_data(data2, sel)
            fitsio.write(f"{outdir2}/{tt}.fits", out2)
            del out2
    return


if __name__ == "__main__":
    main()
