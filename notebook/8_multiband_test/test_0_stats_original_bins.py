#!/usr/bin/env python3
"""Global stats for the original (i-band) catalog,
accumulated over all 6 fields, in 4 redshift bins.

Bins: (0.3,0.6], (0.6,0.9], (0.9,1.2], (1.2,1.5].
All accumulators are length-4 arrays (one element per z-bin).
After looping over fields, prints per-bin statistics."""

import argparse
import gc

import fitsio
import numpy as np

from selection import get_cut, MAG_CUTS_ORIGINAL, DG

FIELDS = ("spring1", "spring2", "spring3", "autumn1", "autumn2", "hectomap")
BASE = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/deepCoadd_anacal_v2"

# Redshift bins: (z_lo, z_hi]
Z_BINS = [(0.3, 0.6), (0.6, 0.9), (0.9, 1.2), (1.2, 1.5)]
NBINS = len(Z_BINS)

parser = argparse.ArgumentParser()
parser.add_argument("--emax", type=float, default=0.3, help="max |e| for cut")
parser.add_argument(
    "--imag", type=float, default=MAG_CUTS_ORIGINAL["i"],
    help="i-band magnitude cut",
)
parser.add_argument(
    "--zkey", type=str, default="zbest",
    choices=["zbest", "zmode"],
    help="photo-z point estimate",
)
args = parser.parse_args()
zkey = args.zkey

mag_cuts = dict(MAG_CUTS_ORIGINAL)
mag_cuts["i"] = args.imag

# Accumulators (length NBINS)
N_cut = np.zeros(NBINS, dtype=np.int64)
sum_we1 = np.zeros(NBINS)
sum_we2 = np.zeros(NBINS)
sum_we1sq = np.zeros(NBINS)
sum_we2sq = np.zeros(NBINS)
sum_R11 = np.zeros(NBINS)
sum_R22 = np.zeros(NBINS)
sum_w = np.zeros(NBINS)
sum_wp1 = np.zeros(NBINS)
sum_wm1 = np.zeros(NBINS)
sum_wp2 = np.zeros(NBINS)
sum_wm2 = np.zeros(NBINS)
# Weight bias: sum(dw*e) per component
sum_wb1 = np.zeros(NBINS)
sum_wb2 = np.zeros(NBINS)

dg = DG

for field in FIELDS:
    d = fitsio.read(f"{BASE}/fields/{field}.fits")
    zbin = fitsio.read(f"{BASE}/fields_redshift/{field}.fits")
    ext = fitsio.read(f"{BASE}/fields_extinction/{field}.fits")

    # Get the photo-z for bin assignment (undistorted)
    z0 = zbin[f"{zkey}_0"].astype(np.float64)

    # Base selection (no z-cut -- we apply z-bin manually)
    common_kwargs_noz = dict(
        zbin=None, ext=ext, emax=args.emax, combined_mag_cut=None, mag_cuts=mag_cuts,
    )
    base_cut = get_cut(d, comp=1, dg_eff=0.0, **common_kwargs_noz)

    # Perturbed cuts for selection response
    base_cut_p1 = get_cut(d, comp=1, dg_eff=+dg, **common_kwargs_noz)
    base_cut_m1 = get_cut(d, comp=1, dg_eff=-dg, **common_kwargs_noz)
    base_cut_p2 = get_cut(d, comp=2, dg_eff=+dg, **common_kwargs_noz)
    base_cut_m2 = get_cut(d, comp=2, dg_eff=-dg, **common_kwargs_noz)

    # Perturbed photo-z columns
    z_1p = zbin[f"{zkey}_1p"].astype(np.float64)
    z_1m = zbin[f"{zkey}_1m"].astype(np.float64)
    z_2p = zbin[f"{zkey}_2p"].astype(np.float64)
    z_2m = zbin[f"{zkey}_2m"].astype(np.float64)

    # Pre-compute per-object arrays
    e1_all = d["e1"].astype(np.float64)
    e2_all = d["e2"].astype(np.float64)
    wsel = d["wsel"].astype(np.float64)

    for ib, (z_lo, z_hi) in enumerate(Z_BINS):
        # Fiducial z-bin mask
        zcut = (z0 > z_lo) & (z0 <= z_hi)
        cut = base_cut & zcut
        nc = int(np.sum(cut))
        if nc == 0:
            continue

        e1c = e1_all[cut]
        e2c = e2_all[cut]
        wsel_c = wsel[cut]

        de11 = d["de1_dg1"][cut].astype(np.float64)
        de22 = d["de2_dg2"][cut].astype(np.float64)
        dw1 = d["dwsel_dg1"][cut].astype(np.float64)
        dw2 = d["dwsel_dg2"][cut].astype(np.float64)

        we1 = wsel_c * e1c
        we2 = wsel_c * e2c

        N_cut[ib] += nc
        sum_we1[ib] += np.sum(we1)
        sum_we2[ib] += np.sum(we2)
        sum_we1sq[ib] += np.sum(we1 ** 2)
        sum_we2sq[ib] += np.sum(we2 ** 2)
        sum_w[ib] += np.sum(wsel_c)

        sum_R11[ib] += np.sum(wsel_c * de11 + dw1 * e1c)
        sum_R22[ib] += np.sum(wsel_c * de22 + dw2 * e2c)

        # Weight bias: the dw*e terms in the response
        sum_wb1[ib] += np.sum(dw1 * e1c)
        sum_wb2[ib] += np.sum(dw2 * e2c)

        # Selection response: perturbed base cut AND perturbed z-bin
        zcut_1p = (z_1p > z_lo) & (z_1p <= z_hi)
        zcut_1m = (z_1m > z_lo) & (z_1m <= z_hi)
        zcut_2p = (z_2p > z_lo) & (z_2p <= z_hi)
        zcut_2m = (z_2m > z_lo) & (z_2m <= z_hi)

        cp1 = base_cut_p1 & zcut_1p
        cm1 = base_cut_m1 & zcut_1m
        cp2 = base_cut_p2 & zcut_2p
        cm2 = base_cut_m2 & zcut_2m

        sum_wp1[ib] += np.sum(wsel[cp1] * e1_all[cp1])
        sum_wm1[ib] += np.sum(wsel[cm1] * e1_all[cm1])
        sum_wp2[ib] += np.sum(wsel[cp2] * e2_all[cp2])
        sum_wm2[ib] += np.sum(wsel[cm2] * e2_all[cm2])

    print(f"[{field}] done, N_cut per bin: {N_cut}")
    del d, zbin, ext
    gc.collect()

# ---- Per-bin combined statistics ----
print(
    f"\n=== Stats per z-bin (all fields, original catalog, zkey={zkey}) ==="
)

for ib, (z_lo, z_hi) in enumerate(Z_BINS):
    nc = N_cut[ib]
    if nc == 0:
        print(f"\nBin {ib} ({z_lo:.1f}, {z_hi:.1f}]: EMPTY")
        continue

    mean_we1 = sum_we1[ib] / nc
    mean_we2 = sum_we2[ib] / nc
    R11 = sum_R11[ib] / nc
    R22 = sum_R22[ib] / nc
    r_sel_1 = (sum_wp1[ib] - sum_wm1[ib]) / (2.0 * dg) / nc
    r_sel_2 = (sum_wp2[ib] - sum_wm2[ib]) / (2.0 * dg) / nc
    r_sel = 0.5 * (r_sel_1 + r_sel_2)
    # Weight bias: averaged over 2 components
    wb1 = sum_wb1[ib] / nc
    wb2 = sum_wb2[ib] / nc
    wb = 0.5 * (wb1 + wb2)

    R11_total = R11 + r_sel_1
    R22_total = R22 + r_sel_2
    R_avg = 0.5 * (R11_total + R22_total)

    var_we1 = sum_we1sq[ib] / nc - mean_we1 ** 2
    var_we2 = sum_we2sq[ib] / nc - mean_we2 ** 2
    std_we1 = np.sqrt(max(var_we1, 0.0))
    std_we2 = np.sqrt(max(var_we2, 0.0))

    print(f"\nBin {ib} ({z_lo:.1f}, {z_hi:.1f}]:")
    print(f"  N_cut: {nc}, sum(w): {sum_w[ib]:.2f}")
    print(f"  {'mean(w*e1)/R11_total':<28} {mean_we1/R11_total:>18.6f}")
    print(f"  {'mean(w*e2)/R22_total':<28} {mean_we2/R22_total:>18.6f}")
    print(f"  {'R11 (shear)':<28} {R11:>18.6f}")
    print(f"  {'R22 (shear)':<28} {R22:>18.6f}")
    print(f"  {'R_sel (avg)':<28} {r_sel:>18.6f}")
    print(f"  {'R_weight_bias (avg)':<28} {wb:>18.6f}")
    print(f"  {'R11_total':<28} {R11_total:>18.6f}")
    print(f"  {'R22_total':<28} {R22_total:>18.6f}")
    print(f"  {'std_e1 / R_avg':<28} {std_we1 / R_avg:>18.6f}")
    print(f"  {'std_e2 / R_avg':<28} {std_we2 / R_avg:>18.6f}")
