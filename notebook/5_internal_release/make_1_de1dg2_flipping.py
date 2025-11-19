#!/usr/bin/env python3
import argparse

import numpy as np
import astropy.io.fits as pyfits


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Process patch masks with MPI."
    )
    parser.add_argument(
        "--field",
        type=str,
        default="hectomap",
        required=False,
        help="field name",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    field = args.field
    rootdir = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/deepCoadd_anacal3"
    fname = f"{rootdir}/fields/{field}.fits"

    with pyfits.open(fname, mode="update", memmap=False) as hdul:
        hdu = hdul[1]
        data = hdu.data
        data["de1_dg2"] = -data["de1_dg2"]
        data["de2_dg1"] = -data["de2_dg1"]
        hdul.flush()
    return


if __name__ == "__main__":
    main()
