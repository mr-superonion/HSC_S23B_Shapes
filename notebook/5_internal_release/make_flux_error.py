#!/usr/bin/env python3
import argparse
import os

import fitsio
import numpy as np
from scipy.spatial import cKDTree


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
    indir = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b_shape/"
    dd = fitsio.read(f"{indir}/dm_{field}.fits")
    ndata = len(dd)
    err_dir = os.path.join(
        os.environ['s23b'],
        f"deepCoadd_flux_variance/fields/"
    )
    errs = fitsio.read(f"{err_dir}/{field}.fits")
    tplist = np.array(fitsio.read(
        "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/tracts_fdfc_v1_final.fits"
    ))[errs["index"]]
    tp1 = dd["tract"].astype(int) * 1000 + dd["patch"].astype(int)
    tp2 = (
        tplist["tract"].astype(int) * 1000 +
        tplist["patch"].astype(int)
    )
    assert (len(np.unique(tp2)) == len(tp2))
    tp1 = np.column_stack((tp1, np.zeros_like(tp1)))
    tp2 = np.column_stack((tp2, np.zeros_like(tp2)))
    tree = cKDTree(tp2)
    _, ind = tree.query(tp1, distance_upper_bound=0.5)

    dtype = [
        ("object_id", "i8"),
        ("g_flux_err", "f8"),
        ("r_flux_err", "f8"),
        ("i_flux_err", "f8"),
        ("z_flux_err", "f8"),
        ("y_flux_err", "f8"),
    ]
    out = np.zeros(ndata, dtype=dtype)
    out["object_id"] = dd["object_id"]
    for b in "grizy":
        out[f"{b}_flux_err"] = np.sqrt(
            dd[f"{b}_variance_value"] * errs[f"var_{b}"][ind]
        )

    outdir = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b_others/"
    fitsio.write(f"{outdir}/fluxerr_{field}.fits", out)
    return


if __name__ == "__main__":
    main()
