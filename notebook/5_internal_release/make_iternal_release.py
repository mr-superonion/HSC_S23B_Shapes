#!/usr/bin/env python3
import argparse

import numpy as np
import fitsio
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

cnd1 = [
 'wsel',
 'dwsel_dg1',
 'dwsel_dg2',
 'de1_dg1',
 'de1_dg2',
 'de2_dg1',
 'de2_dg2',
]

colname2 = [
 'object_id',
 'tract',
 'patch',
 'g_cmodel_mag',
 'g_cmodel_magerr',
 'r_cmodel_mag',
 'r_cmodel_magerr',
 'i_cmodel_mag',
 'i_cmodel_magerr',
 'z_cmodel_mag',
 'z_cmodel_magerr',
 'y_cmodel_mag',
 'y_cmodel_magerr',
 'a_g',
 'a_r',
 'a_i',
 'a_z',
 'a_y',
 'i_sdssshape_shape11',
 'i_sdssshape_shape12',
 'i_sdssshape_shape22',
 'i_hsmpsfmoments_shape11',
 'i_hsmpsfmoments_shape22',
 'i_hsmpsfmoments_shape12',
 'i_higherordermomentspsf_04',
 'i_higherordermomentspsf_13',
 'i_higherordermomentspsf_22',
 'i_higherordermomentspsf_31',
 'i_higherordermomentspsf_40',
 'g_variance_value',
 'r_variance_value',
 'i_variance_value',
 'z_variance_value',
 'y_variance_value',
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

def select_data(d, sel):
    outcome = d[sel]
    order = np.argsort(outcome["object_id"])
    outcome = outcome[order]
    return outcome


def main():
    args = parse_args()
    field = args.field
    rootdir = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/deepCoadd_anacal3"
    outdir = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b_release/"
    tplist = np.array(fitsio.read(
        "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/tracts_fdfc_v1_final.fits"
    ))

    fname = f"{rootdir}/fields/{field}.fits"
    dd = np.array(fitsio.read(fname, columns=colname1))
    fname = f"{rootdir}/fields_color/{field}.fits"
    dd2 = np.array(fitsio.read(fname, columns=colname2))
    assert np.sum(np.abs(dd2["object_id"] - dd["object_id"])) == 0
    r1 = (
        dd["de1_dg1"] * dd["wsel"] +
        dd["dwsel_dg1"] * dd["e1"]
    )
    r2 = (
        dd["de2_dg2"] * dd["wsel"] +
        dd["dwsel_dg2"] * dd["e2"]
    )
    out = np.zeros(len(dd), dtype=[("object_id", "i8"), ("response", "f8")])
    out["object_id"] = dd["object_id"]
    out["response"] = (r1 + r2) / 2.0
    fitsio.write(f"{outdir}/.response/{field}.fits", out)
    dd["e1"] = dd["wsel"] * dd["e1"]
    dd["e2"] = dd["wsel"] * dd["e2"]
    dd = rfn.drop_fields(dd, cnd1, usemask=False)
    fitsio.write(f"{outdir}/anacal_{field}.fits", dd)

    tp1 = dd2["tract"].astype(int) * 1000 + dd2["patch"].astype(int)
    tp2 = (
        tplist["tract"].astype(int) * 1000 +
        tplist["patch"].astype(int)
    )
    assert (len(np.unique(tp2)) == len(tp2))
    tp1 = np.column_stack((tp1, np.zeros_like(tp1)))
    tp2 = np.column_stack((tp2, np.zeros_like(tp2)))
    tree = cKDTree(tp2)

    dist, ind = tree.query(tp1, distance_upper_bound=0.5)
    dd2["a_g"] = dd2["a_g"] + tplist["g_mag_offset"][ind]
    dd2["a_r"] = dd2["a_r"] + tplist["r_mag_offset"][ind]
    dd2["a_i"] = dd2["a_i"] + tplist["i_mag_offset"][ind]
    dd2["a_z"] = dd2["a_z"] + tplist["z_mag_offset"][ind]
    dd2["a_y"] = dd2["a_y"] + tplist["y_mag_offset"][ind]
    fitsio.write(f"{outdir}/dm_{field}.fits", dd2)
    return


if __name__ == "__main__":
    main()
