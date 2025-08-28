#!/usr/bin/env python3
import os
import argparse
import healpy as hp

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
    rootdir = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/deepCoadd_anacal3"
    outdir = f"{rootdir}/healpix/"
    os.makedirs(outdir, exist_ok=True)
    fname = f"{rootdir}/fields/{field}.fits"
    data = np.array(fitsio.read(fname))
    ra = data['ra']
    dec = data['dec']
    theta = np.deg2rad(90.0 - dec)
    phi = np.deg2rad(ra)
    # Convert to HEALPix pixel indices (in NESTED ordering)
    pix = hp.ang2pix(128, theta, phi, nest=True)
    plist = np.unique(pix)
    for tt in plist:
        if os.path.isfile(f"{outdir}/{tt}.fits"):
            print(tt)
        else:
            sel = (pix == tt)
            out1 = select_data(data, sel)
            fitsio.write(f"{outdir}/{tt}.fits", out1)
            del out1
    return


if __name__ == "__main__":
    main()
