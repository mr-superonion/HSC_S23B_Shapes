#!/usr/bin/env python3
"""Global stats with optimal weight wopt = A * ln(m00) + B,
accumulated over all 6 fields.

Loops over each field, reads fields_multiband + fields_redshift,
accumulates weighted sums (with chain-rule response for wopt),
then computes the combined statistics."""

import argparse

import gc
import fitsio
import numpy as np

from selection import get_cut, MAG_CUTS_MULTIBAND, DG

FIELDS = ("spring1", "spring2", "spring3", "autumn1", "autumn2", "hectomap")
BASE = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/deepCoadd_anacal_v2"

parser = argparse.ArgumentParser()
parser.add_argument("--emax", type=float, default=0.3, help="max |e| for cut")
parser.add_argument(
    "--imag", type=float, default=MAG_CUTS_MULTIBAND["i"],
    help="i-band magnitude cut",
)
parser.add_argument(
    "--zkey", type=str, default="zbest",
    choices=["zbest", "zmode"],
    help="photo-z point estimate",
)
parser.add_argument("--A", type=float, default=4.11, help="wopt slope")
parser.add_argument("--B", type=float, default=4.0, help="wopt offset")
args = parser.parse_args()
zkey = args.zkey
A_wopt = args.A
B_wopt = args.B

mag_cuts = dict(MAG_CUTS_MULTIBAND)
mag_cuts["i"] = args.imag

# Accumulators
N_cut = 0
sum_we1 = 0.0
sum_we2 = 0.0
sum_we1sq = 0.0
sum_we2sq = 0.0
sum_R11 = 0.0     # chain-rule shear response
sum_R22 = 0.0
sum_w = 0.0
# Selection response numerators (w_total * e, over ALL objects)
sum_wp1 = 0.0
sum_wm1 = 0.0
sum_wp2 = 0.0
sum_wm2 = 0.0

dg = DG

for field in FIELDS:
    d = fitsio.read(f"{BASE}/fields_multiband/{field}.fits")
    zbin = fitsio.read(f"{BASE}/fields_redshift/{field}.fits")
    ext = fitsio.read(f"{BASE}/fields_extinction/{field}.fits")

    common_kwargs = dict(
        zbin=zbin, zkey=zkey, ext=ext, emax=args.emax, mag_cuts=mag_cuts,
    )

    # Optimal weight
    m00 = d["m0"].astype(np.float64)
    wopt = (A_wopt * np.log(np.clip(m00, 1e-30, None)) + B_wopt) / 13.3
    dwopt_dm0 = A_wopt / m00 / 13.3
    wsel = d["wsel"].astype(np.float64)
    w_total = wsel * wopt

    n = len(d)
    cut = get_cut(d, comp=1, dg_eff=0.0, **common_kwargs)
    nc = int(np.sum(cut))
    print(f"[{field}] {nc} / {n}")

    e1c = d["e1"][cut].astype(np.float64)
    e2c = d["e2"][cut].astype(np.float64)
    wsel_c = wsel[cut]
    wopt_c = wopt[cut]
    w_total_c = w_total[cut]
    dwopt_dm0_c = dwopt_dm0[cut]

    de11 = d["de1_dg1"][cut].astype(np.float64)
    de22 = d["de2_dg2"][cut].astype(np.float64)
    dw1 = d["dwsel_dg1"][cut].astype(np.float64)
    dw2 = d["dwsel_dg2"][cut].astype(np.float64)
    dm0_dg1 = d["dm0_dg1"][cut].astype(np.float64)
    dm0_dg2 = d["dm0_dg2"][cut].astype(np.float64)

    dwopt_dg1 = dwopt_dm0_c * dm0_dg1
    dwopt_dg2 = dwopt_dm0_c * dm0_dg2

    we1 = w_total_c * e1c
    we2 = w_total_c * e2c

    N_cut += nc
    sum_we1 += np.sum(we1)
    sum_we2 += np.sum(we2)
    sum_we1sq += np.sum(we1 ** 2)
    sum_we2sq += np.sum(we2 ** 2)

    # Chain-rule shear response sums
    sum_R11 += np.sum(
        wsel_c * wopt_c * de11
        + dw1 * wopt_c * e1c
        + wsel_c * dwopt_dg1 * e1c
    )
    sum_R22 += np.sum(
        wsel_c * wopt_c * de22
        + dw2 * wopt_c * e2c
        + wsel_c * dwopt_dg2 * e2c
    )
    sum_w += np.sum(w_total_c)

    # Selection response sums (full sample, using w_total)
    e1_all = d["e1"].astype(np.float64)
    e2_all = d["e2"].astype(np.float64)

    cut_p1 = get_cut(d, comp=1, dg_eff=+dg, **common_kwargs)
    cut_m1 = get_cut(d, comp=1, dg_eff=-dg, **common_kwargs)
    cut_p2 = get_cut(d, comp=2, dg_eff=+dg, **common_kwargs)
    cut_m2 = get_cut(d, comp=2, dg_eff=-dg, **common_kwargs)

    sum_wp1 += np.sum(w_total[cut_p1] * e1_all[cut_p1])
    sum_wm1 += np.sum(w_total[cut_m1] * e1_all[cut_m1])
    sum_wp2 += np.sum(w_total[cut_p2] * e2_all[cut_p2])
    sum_wm2 += np.sum(w_total[cut_m2] * e2_all[cut_m2])

    del d, zbin, ext, cut_p1, cut_m1, cut_p2, cut_m2
    gc.collect()

# ---- Combined statistics ----
mean_we1 = sum_we1 / N_cut
mean_we2 = sum_we2 / N_cut
R11 = sum_R11 / N_cut
R22 = sum_R22 / N_cut
r_sel_1 = (sum_wp1 - sum_wm1) / (2.0 * dg) / N_cut
r_sel_2 = (sum_wp2 - sum_wm2) / (2.0 * dg) / N_cut

R11_total = R11 + r_sel_1
R22_total = R22 + r_sel_2
R_avg = 0.5 * (R11_total + R22_total)

var_we1 = sum_we1sq / N_cut - mean_we1 ** 2
var_we2 = sum_we2sq / N_cut - mean_we2 ** 2
std_we1 = np.sqrt(var_we1)
std_we2 = np.sqrt(var_we2)

print(
    f"\n=== Weighted stats (all fields, "
    f"wopt = {A_wopt}*ln(m0)+{B_wopt}, zkey={zkey}) ==="
)
print(f"N_cut: {N_cut}, sum(w): {sum_w:.2f}")
print(f"\n{'stat':<30} {'value':>18}")
print("-" * 49)
print(f"{'mean(w*e1)/R11_total':<30} {mean_we1/R11_total:>18.6f}")
print(f"{'mean(w*e2)/R22_total':<30} {mean_we2/R22_total:>18.6f}")
print(f"{'R11 (shear)':<30} {R11:>18.6f}")
print(f"{'R22 (shear)':<30} {R22:>18.6f}")
print(f"{'R_sel_1':<30} {r_sel_1:>18.6f}")
print(f"{'R_sel_2':<30} {r_sel_2:>18.6f}")
print(f"{'R11_total':<30} {R11_total:>18.6f}")
print(f"{'R22_total':<30} {R22_total:>18.6f}")
print(f"{'std_e1 / R_avg':<30} {std_we1 / R_avg:>18.6f}")
print(f"{'std_e2 / R_avg':<30} {std_we2 / R_avg:>18.6f}")
