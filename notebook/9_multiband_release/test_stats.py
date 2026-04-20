#!/usr/bin/env python3
"""Per-bin shear statistics from pre-computed response and bin files.

Reads {field}_response.fits (we1, we2, response) and
{field}_imag25.0.fits (bin assignments + sel/weight bias in header)
for all 6 fields. Accumulates per-bin sums, then prints
mean(we) / (R_shear + R_sel + R_weight) per bin.

5 bins: (0.3,0.6], (0.6,0.9], (0.9,1.2], (1.2,1.5], (1.5,1.8].
"""

import gc

import fitsio
import numpy as np

FIELDS = ("spring1", "spring2", "spring3", "autumn1", "autumn2", "hectomap")
BASE = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/deepCoadd_anacal_v2"
MB_DIR = f"{BASE}/fields_multiband"
BIN_DIR = f"{BASE}/fields_bin"

Z_BINS = [(0.3, 0.6), (0.6, 0.9), (0.9, 1.2), (1.2, 1.5), (1.5, 1.8)]
NBINS = len(Z_BINS)

IMAG = 25.0

# Accumulators (length NBINS)
N_cut = np.zeros(NBINS, dtype=np.int64)
sum_we1 = np.zeros(NBINS)
sum_we2 = np.zeros(NBINS)
sum_we1sq = np.zeros(NBINS)
sum_we2sq = np.zeros(NBINS)
sum_resp = np.zeros(NBINS)

# Read sel_bias and weight_bias from header of first bin file
bin_path_0 = f"{BIN_DIR}/{FIELDS[0]}_imag{IMAG:.1f}.fits"
hdr = fitsio.read_header(bin_path_0, ext=1)
sel_bias = np.array([hdr.get(f"SELB_{ib}", 0.0) for ib in range(NBINS)])
weight_bias = np.array([hdr.get(f"WGTB_{ib}", 0.0) for ib in range(NBINS)])

for field in FIELDS:
    resp_path = f"{MB_DIR}/{field}_response.fits"
    bin_path = f"{BIN_DIR}/{field}_imag{IMAG:.1f}.fits"

    r = fitsio.read(resp_path)
    b = fitsio.read(bin_path, columns=["object_id", "bin"])
    assert np.array_equal(r["object_id"], b["object_id"]), (
        f"{field}: object_id mismatch between response and bin"
    )
    bin_col = b["bin"]

    we1 = r["e1"].astype(np.float64)
    we2 = r["e2"].astype(np.float64)
    resp = r["response"].astype(np.float64)

    for ib in range(NBINS):
        sel = bin_col == ib
        nc = int(np.sum(sel))
        if nc == 0:
            continue
        N_cut[ib] += nc
        sum_we1[ib] += np.sum(we1[sel])
        sum_we2[ib] += np.sum(we2[sel])
        sum_we1sq[ib] += np.sum(we1[sel] ** 2)
        sum_we2sq[ib] += np.sum(we2[sel] ** 2)
        sum_resp[ib] += np.sum(resp[sel])

    print(f"[{field}] done, N_cut per bin: {N_cut}")
    del r, b, bin_col
    gc.collect()

# ---- Per-bin combined statistics ----
print(f"\n=== Stats per z-bin (all fields, imag={IMAG:.1f}) ===")

for ib, (z_lo, z_hi) in enumerate(Z_BINS):
    nc = N_cut[ib]
    if nc == 0:
        print(f"\nBin {ib} ({z_lo:.1f}, {z_hi:.1f}]: EMPTY")
        continue

    mean_we1 = sum_we1[ib] / nc
    mean_we2 = sum_we2[ib] / nc
    R_shear = sum_resp[ib] / nc
    R_sel = sel_bias[ib]
    R_wb = weight_bias[ib]
    R_total = R_shear + R_sel + R_wb

    var_we1 = sum_we1sq[ib] / nc - mean_we1 ** 2
    var_we2 = sum_we2sq[ib] / nc - mean_we2 ** 2
    std_we1 = np.sqrt(max(var_we1, 0.0))
    std_we2 = np.sqrt(max(var_we2, 0.0))

    print(f"\nBin {ib} ({z_lo:.1f}, {z_hi:.1f}]:")
    print(f"  N_cut: {nc}")
    print(f"  {'mean(we1) / R_total':<28} {mean_we1 / R_total:>18.6f}")
    print(f"  {'mean(we2) / R_total':<28} {mean_we2 / R_total:>18.6f}")
    print(f"  {'R_shear':<28} {R_shear:>18.6f}")
    print(f"  {'R_sel':<28} {R_sel:>18.6f}")
    print(f"  {'R_weight_bias':<28} {R_wb:>18.6f}")
    print(f"  {'R_total':<28} {R_total:>18.6f}")
    print(f"  {'std_we1 / R_total':<28} {std_we1 / R_total:>18.6f}")
    print(f"  {'std_we2 / R_total':<28} {std_we2 / R_total:>18.6f}")
