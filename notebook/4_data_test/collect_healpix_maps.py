#!/usr/bin/env python3
import argparse
import os

import fitsio
import healpy as hp
import numpy as np
import numpy.lib.recfunctions as rfn

field_int_map = {
    "spring1": 1,
    "spring2": 1,
    "spring3": 1,
    "autumn1": 2,
    "autumn2": 2,
    "hectomap": 4,
}

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

    fname = f"{rootdir}/fields/{field}.fits"
    data = np.array(fitsio.read(fname))
    mag = 27.0 - 2.5 * np.log10(data["flux"])
    abse2 = data["e1"] ** 2.0 + data["e2"] ** 2.0
    mm = (mag < 24.5) & (abse2 < 0.09)

    data = data[mm]
    r1 = (
        data["de1_dg1"] * data["wsel"] +
        data["dwsel_dg1"] * data["e1"]
    )
    r2 = (
        data["de2_dg2"] * data["wsel"] +
        data["dwsel_dg2"] * data["e2"]
    )

    fname2 = f"{rootdir}/fields_color/{field}.fits"
    data2 = fitsio.read(fname2)
    data2 = data2[mm]

    variance = data2["i_variance_value"]
    psf_mxx = data2["i_hsmpsfmoments_shape11"]
    psf_myy = data2["i_hsmpsfmoments_shape22"]
    psf_mxy = data2["i_hsmpsfmoments_shape12"]
    fwhm = 2.355 * (psf_mxx * psf_myy - psf_mxy**2)**0.25
    print(np.nanmean(fwhm))

    nside = 1024
    npix = hp.nside2npix(nside)
    ra = data['ra']
    dec = data['dec']
    theta = np.deg2rad(90.0 - dec)
    phi = np.deg2rad(ra)
    pix = hp.ang2pix(nside, theta, phi, nest=True)

    num_map = np.bincount(pix, minlength=npix)
    mask_map = (num_map>0).astype(int) * field_int_map[field]
    res_map = np.bincount(pix, weights=(r1+r2)/2.0, minlength=npix)
    var_map = np.bincount(pix, weights=variance, minlength=npix)
    fwhm_map = np.bincount(pix, weights=fwhm, minlength=npix)

    outfname = f"{os.environ['s23b_anacal3']}/tests/gal_maps_{field}.fits"
    dtype = np.dtype([
        ("num",  "i4"),
        ("mask", "i4"),
        ("response", "f8"),
        ("variance", "f4"),
        ("fwhm","f4"),
    ])
    outcome = rfn.unstructured_to_structured(
        np.column_stack([num_map, mask_map, res_map, var_map, fwhm_map]),
        dtype=dtype,
        copy=False
    )
    fitsio.write(outfname, outcome)
    return


if __name__ == "__main__":
    main()
