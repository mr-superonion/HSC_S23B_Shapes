#!/usr/bin/env python3
import argparse

import fitsio
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

    fname = f"{rootdir}/fields_color/{field}.fits"
    dd0 = np.array(fitsio.read(fname))
    mm = (
        (~np.isnan(dd0["g_variance_value"])) &
        (~np.isnan(dd0["r_variance_value"])) &
        (~np.isnan(dd0["i_variance_value"])) &
        (~np.isnan(dd0["z_variance_value"])) &
        (~np.isnan(dd0["y_variance_value"]))
    )
    data = dd0[mm]

    nside = 1024
    npix = hp.nside2npix(nside)
    ra = data['i_ra']
    dec = data['i_dec']
    theta = np.deg2rad(90.0 - dec)
    phi = np.deg2rad(ra)
    pix = hp.ang2pix(nside, theta, phi, nest=True)

    maps = {}
    num_map = np.bincount(pix, minlength=npix)

    for band in ["g", "r", "i", "z", "y"]:
        m = np.bincount(
            pix, weights=data[f"{band}_variance_value"], minlength=npix
        )
        m = m / (num_map + 0.001)
        maps[f"{band}"] = m
    print(maps.keys())

    ra = dd0['i_ra']
    dec = dd0['i_dec']
    theta = np.deg2rad(90.0 - dec)
    phi = np.deg2rad(ra)
    pix = hp.ang2pix(nside, theta, phi, nest=True)

    for band in ["g", "r", "i", "z", "y"]:
        sel = np.isnan(dd0[f"{band}_variance_value"])
        inds = pix[sel]
        dd0[f"{band}_variance_value"][sel] = maps[f"{band}"][inds]
    pyfits.writeto(fname, dd0, overwrite=True)

    return


if __name__ == "__main__":
    main()
