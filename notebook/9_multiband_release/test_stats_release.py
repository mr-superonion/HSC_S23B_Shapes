#!/usr/bin/env python3
"""Per-bin shear statistics from the released catalog.

Reads from /gpfs02/work/xiangchong.li/work/hsc_data/catalog_v2.5/:
  - s23b_shape/anacal_{field}.fits       (e1, e2)
  - s23b_shape/.response/{field}.fits    (response, response_denoised)
  - s23b_selection/{field}_imag25.0.fits (bin + sel/weight bias in header)

Accumulates per-bin sums across all 6 fields, then prints
mean(e) / (R_shear + R_sel + R_weight) per bin.

5 bins: (0.3,0.6], (0.6,0.9], (0.9,1.2], (1.2,1.5], (1.5,1.8].
"""

import gc

import fitsio
import numpy as np

FIELDS = ("spring1", "spring2", "spring3", "autumn1", "autumn2", "hectomap")
BASE = "/gpfs02/work/xiangchong.li/work/hsc_data/catalog_v2.5"
SHAPE_DIR = f"{BASE}/s23b_shape"
RESP_DIR = f"{BASE}/s23b_shape/.response"
SEL_DIR = f"{BASE}/s23b_selection"

Z_BINS = [(0.3, 0.6), (0.6, 0.9), (0.9, 1.2), (1.2, 1.5), (1.5, 1.8)]
NBINS = len(Z_BINS)

IMAG = 25.0

# Accumulators (length NBINS)
N_cut = np.zeros(NBINS, dtype=np.int64)
sum_e1 = np.zeros(NBINS)
sum_e2 = np.zeros(NBINS)
sum_e1sq = np.zeros(NBINS)
sum_e2sq = np.zeros(NBINS)
sum_resp = np.zeros(NBINS)

# Read sel_bias and weight_bias from header of first bin file
sel_path_0 = f"{SEL_DIR}/{FIELDS[0]}_imag{IMAG:.1f}.fits"
hdr = fitsio.read_header(sel_path_0, ext=1)
sel_bias = np.array([hdr.get(f"SELB_{ib}", 0.0) for ib in range(NBINS)])
weight_bias = np.array([hdr.get(f"WGTB_{ib}", 0.0) for ib in range(NBINS)])

for field in FIELDS:
    shape_path = f"{SHAPE_DIR}/anacal_{field}.fits"
    resp_path = f"{RESP_DIR}/{field}.fits"
    sel_path = f"{SEL_DIR}/{field}_imag{IMAG:.1f}.fits"

    s = fitsio.read(shape_path, columns=["object_id", "e1", "e2"])
    r = fitsio.read(resp_path, columns=["object_id", "response"])
    b = fitsio.read(sel_path, columns=["object_id", "bin"])

    assert np.array_equal(s["object_id"], r["object_id"]), (
        f"{field}: object_id mismatch between shape and response"
    )
    assert np.array_equal(s["object_id"], b["object_id"]), (
        f"{field}: object_id mismatch between shape and selection"
    )
    bin_col = b["bin"]

    e1 = s["e1"].astype(np.float64)
    e2 = s["e2"].astype(np.float64)
    resp = r["response"].astype(np.float64)

    for ib in range(NBINS):
        sel = bin_col == ib
        nc = int(np.sum(sel))
        if nc == 0:
            continue
        N_cut[ib] += nc
        sum_e1[ib] += np.sum(e1[sel])
        sum_e2[ib] += np.sum(e2[sel])
        sum_e1sq[ib] += np.sum(e1[sel] ** 2)
        sum_e2sq[ib] += np.sum(e2[sel] ** 2)
        sum_resp[ib] += np.sum(resp[sel])

    print(f"[{field}] done, N_cut per bin: {N_cut}")
    del s, r, b, bin_col
    gc.collect()

# ---- Per-bin combined statistics ----
print(f"\n=== Stats per z-bin (all fields, imag={IMAG:.1f}) ===")

for ib, (z_lo, z_hi) in enumerate(Z_BINS):
    nc = N_cut[ib]
    if nc == 0:
        print(f"\nBin {ib} ({z_lo:.1f}, {z_hi:.1f}]: EMPTY")
        continue

    mean_e1 = sum_e1[ib] / nc
    mean_e2 = sum_e2[ib] / nc
    R_shear = sum_resp[ib] / nc
    R_sel = sel_bias[ib]
    R_wb = weight_bias[ib]
    R_total = R_shear + R_sel + R_wb

    var_e1 = sum_e1sq[ib] / nc - mean_e1 ** 2
    var_e2 = sum_e2sq[ib] / nc - mean_e2 ** 2
    std_e1 = np.sqrt(max(var_e1, 0.0))
    std_e2 = np.sqrt(max(var_e2, 0.0))

    print(f"\nBin {ib} ({z_lo:.1f}, {z_hi:.1f}]:")
    print(f"  N_cut: {nc}")
    print(f"  {'mean(e1) / R_total':<28} {mean_e1 / R_total:>18.6f}")
    print(f"  {'mean(e2) / R_total':<28} {mean_e2 / R_total:>18.6f}")
    print(f"  {'R_shear':<28} {R_shear:>18.6f}")
    print(f"  {'R_sel':<28} {R_sel:>18.6f}")
    print(f"  {'R_weight_bias':<28} {R_wb:>18.6f}")
    print(f"  {'R_total':<28} {R_total:>18.6f}")
    print(f"  {'std_e1 / R_total':<28} {std_e1 / R_total:>18.6f}")
    print(f"  {'std_e2 / R_total':<28} {std_e2 / R_total:>18.6f}")
