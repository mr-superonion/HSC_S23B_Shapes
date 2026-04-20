#!/usr/bin/env python3
"""Prepare per-field response catalogs with optimal weighting.

For each field:
  1. Read fields_multiband/{field}.fits (de1/de2 derivatives, m0, wsel,
     response_denoised).
  2. Read fields_bin/{field}_imag25.0.fits (bin column, 5 bins).
  3. Compute optimal weight wopt = (A*ln(m0) + B) / 13.3 and
     w_total = wsel * wopt.
  4. Compute per-object weighted response components:
        response  = (de1_dg1 + de2_dg2) / 2 * wsel * wopt
        R_4c      = (de1_dg1 - de2_dg2) / 2 * wsel * wopt
        R_4s      = (de1_dg2 + de2_dg1) / 2 * wsel * wopt
  5. For each tomographic bin b in {0, 1, 2, 3, 4}:
        <resp>_b      = mean(response[bin == b])
        <resp_den>_b  = mean(response_denoised * wsel * wopt [bin == b])
        factor_b      = <resp>_b / <resp_den>_b
     and rescale response_denoised[bin == b] *= factor_b so that the
     bin-averaged denoised response matches the raw weighted one.
     Objects with bin == -1 are left unchanged.
  6. Write fields_multiband/{field}_response.fits with columns:
        object_id, response, R_4c, R_4s, response_denoised, wopt

Parallelized over fields via MPI (one field per rank).
"""

import argparse
import gc
import os

import fitsio
import numpy as np
from mpi4py import MPI

BASE = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/deepCoadd_anacal_v2"
MB_DIR = f"{BASE}/fields_multiband"
BIN_DIR = f"{BASE}/fields_bin"

FIELDS = ("spring1", "spring2", "spring3", "autumn1", "autumn2", "hectomap")

N_BINS = 5  # bins 0, 1, 2, 3, 4

# Default optimal weight parameters (same as test_0_stats_weight_bins.py)
A_DEFAULT = 4.11
B_DEFAULT = 4.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare per-field weighted response.fits files."
    )
    parser.add_argument(
        "--field", type=str, default="all",
        help="Single field to process (default: all).",
    )
    return parser.parse_args()


def process_field(field, A_wopt, B_wopt):
    mb_path = os.path.join(MB_DIR, f"{field}.fits")
    bin_path = os.path.join(BIN_DIR, f"{field}_imag25.0.fits")
    out_path = os.path.join(MB_DIR, f"{field}_response.fits")

    if os.path.isfile(out_path):
        print(f"[{field}] already exists, skipping")
        return
    if not (os.path.isfile(mb_path) and os.path.isfile(bin_path)):
        print(f"[{field}] missing inputs, skipping")
        return

    mb_cols = [
        "object_id",
        "e1", "e2",
        "de1_dg1", "de1_dg2", "de2_dg1", "de2_dg2",
        "m0", "wsel",
        "response_denoised",
    ]
    d = fitsio.read(mb_path, columns=mb_cols)
    n = len(d)
    print(f"[{field}] {n} objects")

    b = fitsio.read(bin_path, columns=["object_id", "bin"])
    assert np.array_equal(b["object_id"], d["object_id"]), (
        f"{field}: object_id mismatch between multiband and bin"
    )
    bin_col = b["bin"]
    del b

    # Optimal weight
    m0 = d["m0"].astype(np.float64)
    wopt = (A_wopt * np.log(np.clip(m0, 1e-30, None)) + B_wopt) / 13.3
    wsel = d["wsel"].astype(np.float64)
    w_total = wsel * wopt

    # Per-object weighted response components
    de1_dg1 = d["de1_dg1"].astype(np.float64)
    de1_dg2 = d["de1_dg2"].astype(np.float64)
    de2_dg1 = d["de2_dg1"].astype(np.float64)
    de2_dg2 = d["de2_dg2"].astype(np.float64)

    e1 = d["e1"].astype(np.float64)
    e2 = d["e2"].astype(np.float64)
    we1 = e1 * w_total
    we2 = e2 * w_total

    response = 0.5 * (de1_dg1 + de2_dg2) * w_total
    R_4c = 0.5 * (de1_dg1 - de2_dg2) * w_total
    R_4s = 0.5 * (de1_dg2 + de2_dg1) * w_total

    response_denoised = d["response_denoised"].astype(np.float64).copy()
    response_denoised = response_denoised * w_total

    # Per-bin rescaling of response_denoised so that
    # mean(response_denoised * w_total) matches mean(response)
    for ib in range(N_BINS):
        sel = bin_col == ib
        n_sel = int(np.sum(sel))
        if n_sel == 0:
            print(f"[{field}]   bin {ib}: empty, skipping rescale")
            continue
        mean_r = float(np.mean(response[sel]))
        mean_rd = float(np.mean(response_denoised[sel]))
        if abs(mean_rd) < 1e-30:
            print(
                f"[{field}]   bin {ib}: mean_rd ~ 0, skipping rescale"
            )
            continue
        factor = mean_r / mean_rd
        response_denoised[sel] *= factor
        print(
            f"[{field}]   bin {ib}: n={n_sel}, "
            f"<R*w>={mean_r:.6f}, <R_den*w>={mean_rd:.6f}, "
            f"factor={factor:.6f}"
        )

    out = np.empty(n, dtype=[
        ("object_id", "i8"),
        ("e1", "f8"),
        ("e2", "f8"),
        ("response", "f8"),
        ("R_4c", "f8"),
        ("R_4s", "f8"),
        ("response_denoised", "f8"),
    ])
    out["object_id"] = d["object_id"]
    out["e1"] = we1
    out["e2"] = we2
    out["response"] = response
    out["R_4c"] = R_4c
    out["R_4s"] = R_4s
    out["response_denoised"] = response_denoised

    fitsio.write(out_path, out, clobber=True)
    print(f"[{field}] written: {out_path}")

    del d, bin_col, we1, we2, response, R_4c, R_4s, response_denoised, out
    gc.collect()


def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.field != "all":
        fields = [args.field]
    else:
        fields = list(FIELDS)

    my_fields = fields[rank::size]
    for field in my_fields:
        if rank == 0 or len(fields) > 1:
            print(f"[rank {rank}] processing {field}")
        process_field(field, A_DEFAULT, B_DEFAULT)
        gc.collect()

    comm.Barrier()
    if rank == 0:
        print("All done.")


if __name__ == "__main__":
    main()
