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
    rootdir = os.environ['s23b']
    fname = f"{rootdir}/db_star/fields/{field}.fits"
    data = np.array(fitsio.read(fname))
    snr = data["i_psfflux_flux"] / data["i_psfflux_fluxerr"]
    mask = (data["i_calib_psf_reserved"]) & (snr > 200.0)
    data = data[mask]
    psf_mxx = data["i_hsmpsfmoments_shape11"]
    psf_myy = data["i_hsmpsfmoments_shape22"]
    T_psf = (psf_mxx + psf_myy)

    star_mxx = data["i_hsmsourcemoments_shape11"]
    star_myy = data["i_hsmsourcemoments_shape22"]
    T_star = (star_mxx + star_myy)
    dsize = (T_star - T_psf) / T_psf
    print(len(data))
    print(np.max(np.abs(dsize)))

    nside = 1024
    npix = hp.nside2npix(nside)
    ra = data['i_ra']
    dec = data['i_dec']
    theta = np.deg2rad(90.0 - dec)
    phi = np.deg2rad(ra)
    pix = hp.ang2pix(nside, theta, phi, nest=True)

    num_map = np.bincount(pix, minlength=npix)
    mask_map = (num_map>0).astype(int) * field_int_map[field]
    dsize_map = np.bincount(pix, weights=dsize, minlength=npix)

    outfname = f"{os.environ['s23b_anacal_v2']}/galmaps/star_maps_{field}.fits"
    dtype = np.dtype([
        ("num",  "i4"),
        ("mask", "i4"),
        ("dsize","f4"),
    ])
    outcome = rfn.unstructured_to_structured(
        np.column_stack([
            num_map, mask_map, dsize_map
        ]),
        dtype=dtype,
        copy=False
    )
    fitsio.write(outfname, outcome)
    return


if __name__ == "__main__":
    main()
