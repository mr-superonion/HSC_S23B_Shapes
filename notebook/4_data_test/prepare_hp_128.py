#!/usr/bin/env python3

import os
import argparse

import fitsio
import numpy as np
import healpy as hp


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Process patch masks with MPI."
    )
    parser.add_argument(
        "--field", type=str, default="all", required=False, help="field name"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    field = args.field
    NSIDE = 128
    rootdir = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/deepCoadd_anacal2"
    outdir = f"{rootdir}/hpix_{NSIDE}"
    os.makedirs(outdir, exist_ok=True)
    fname = f"{rootdir}/fields/{field}.fits"
    data = fitsio.read(fname)
    ra = data['ra']
    dec = data['dec']
    theta = np.deg2rad(90.0 - dec)
    phi = np.deg2rad(ra)
    # Convert to HEALPix pixel indices (in NESTED ordering)
    pix = hp.ang2pix(NSIDE, theta, phi, nest=True)
    hp_list = np.unique(pix)
    for pp in hp_list:
        outfname = f"{outdir}/{pp}_{field}.fits"
        fitsio.write(outfname, data[pix == pp])
    return


if __name__ == "__main__":
    main()
