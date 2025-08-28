#!/usr/bin/env python3
import argparse

import numpy as np
import healpy as hp
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

def select_data(d, sel):
    outcome = d[sel]
    order = np.argsort(outcome["object_id"])
    outcome = outcome[order]
    return outcome


def main():
    args = parse_args()
    field = args.field
    rootdir = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/deepCoadd_anacal3"

    hmask = hp.read_map(
        "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/fdfc_hp_window.fits",
        nest=True, dtype=bool
    )

    NSIDE = 1024
    fname = f"{rootdir}/fields/{field}.fits"
    dd = pyfits.getdata(fname)
    pix = hp.ang2pix(
        NSIDE,
        np.deg2rad(90.0 - dd["dec"]),
        np.deg2rad(dd["ra"]),
        nest=True
    )
    mm = hmask[pix]
    print(np.sum(mm) / len(mm))
    pyfits.writeto(fname, dd[mm], overwrite=True)
    del dd, pix

    fname = f"{rootdir}/fields_color/{field}.fits"
    dd = pyfits.getdata(fname)
    pyfits.writeto(fname, dd[mm], overwrite=True)
    return


if __name__ == "__main__":
    main()
