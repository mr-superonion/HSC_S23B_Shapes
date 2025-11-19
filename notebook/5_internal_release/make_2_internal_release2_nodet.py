#!/usr/bin/env python3
import argparse

import fitsio
import numpy as np
import numpy.lib.recfunctions as rfn
from scipy.spatial import cKDTree

colname1 = [
 'object_id',
 'ra',
 'dec',
 'wsel',
 'dwsel_dg1',
 'dwsel_dg2',
 'e1',
 'de1_dg1',
 'de1_dg2',
 'e2',
 'de2_dg1',
 'de2_dg2',
 'flux',
 'dflux_dg1',
 'dflux_dg2',
 'g_flux',
 'g_dflux_dg1',
 'g_dflux_dg2',
 'r_flux',
 'r_dflux_dg1',
 'r_dflux_dg2',
 'i_flux',
 'i_dflux_dg1',
 'i_dflux_dg2',
 'z_flux',
 'z_dflux_dg1',
 'z_dflux_dg2',
 'y_flux',
 'y_dflux_dg1',
 'y_dflux_dg2'
]


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
    outdir = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b_shape/"

    fname = f"{rootdir}/fields/{field}.fits"
    dd = np.array(fitsio.read(fname, columns=colname1))
    r1 = dd["de1_dg1"] * dd["wsel"]
    # dd["dwsel_dg1"] * dd["e1"]
    r2 = dd["de2_dg2"] * dd["wsel"]
    # dd["dwsel_dg2"] * dd["e2"]
    out = np.zeros(len(dd), dtype=[("object_id", "i8"), ("response", "f8")])
    out["object_id"] = dd["object_id"]
    out["response"] = (r1 + r2) / 2.0
    fitsio.write(f"{outdir}/.response2/{field}.fits", out)
    return


if __name__ == "__main__":
    main()
