#!/usr/bin/env python3

import argparse
import csv
import os

import fitsio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--field", type=str, default="spring1", help="field name")
parser.add_argument("--emax", type=float, default=0.3, help="max |e| for cut")
args = parser.parse_args()
emax_sq = args.emax ** 2
field = args.field

base = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/deepCoadd_anacal_v2/fields"
d1 = fitsio.read(f"{base}/{field}.fits")
d2 = fitsio.read(f"{base}/{field}_multiband.fits")
color = fitsio.read(
    "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/"
    f"deepCoadd_anacal_v2/fields_color/{field}.fits"
)

# Apply magnitude and ellipticity cut
bright = (
    ((27 - 2.5 * np.log10(d1["g_flux_gauss2"])) < 27.0)
    & ((27 - 2.5 * np.log10(d1["r_flux_gauss2"])) < 26.0)
    & ((27 - 2.5 * np.log10(d1["i_flux_gauss2"])) < 24.6)
    & ((27 - 2.5 * np.log10(d1["z_flux_gauss2"])) < 25.0)
    & ((27 - 2.5 * np.log10(d1["y_flux_gauss2"])) < 25.5)
    & (d2["fpfs1_e1"] ** 2 + d2["fpfs1_e2"] ** 2 < emax_sq)
    & (d1["m2"] / d1["m0"] > 0.05)
)
print(f"Before cut: {len(d1)}, after cut: {np.sum(bright)}")
d1 = d1[bright]
d2 = d2[bright]
color = color[bright]

tracts = color["tract"]
patches = color["patch"]

tp = np.unique(np.column_stack([tracts, patches]), axis=0)
print(f"Unique tract-patch pairs: {len(tp)}")

out_rows = []
for t, p in tp:
    mask = (tracts == t) & (patches == p)
    n = np.sum(mask)
    if n < 3:
        continue
    p1 = d1[mask]
    p2 = d2[mask]
    row = {
        "tract": int(t),
        "patch": int(p),
        "n_obj": n,
        "std_e1": f"{np.std(p1['e1']):.6f}",
        "std_fpfs1_e1": f"{np.std(p2['fpfs1_e1']):.6f}",
        "std_e2": f"{np.std(p1['e2']):.6f}",
        "std_fpfs1_e2": f"{np.std(p2['fpfs1_e2']):.6f}",
        "mean_de1_dg1": f"{np.mean(p1['de1_dg1']):.6f}",
        "mean_fpfs1_de1_dg1": f"{np.mean(p2['fpfs1_de1_dg1']):.6f}",
        "mean_de1_dg2": f"{np.mean(p1['de1_dg2']):.6f}",
        "mean_fpfs1_de1_dg2": f"{np.mean(p2['fpfs1_de1_dg2']):.6f}",
        "mean_de2_dg1": f"{np.mean(p1['de2_dg1']):.6f}",
        "mean_fpfs1_de2_dg1": f"{np.mean(p2['fpfs1_de2_dg1']):.6f}",
        "mean_de2_dg2": f"{np.mean(p1['de2_dg2']):.6f}",
        "mean_fpfs1_de2_dg2": f"{np.mean(p2['fpfs1_de2_dg2']):.6f}",
    }
    out_rows.append(row)
    print(p, np.std(p1['e1']), np.std(p2['fpfs1_e1']))

out_fname = f"{base}/{field}_patch_comparison.csv"
fields = list(out_rows[0].keys())
with open(out_fname, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(out_rows)

print(f"\nWritten {len(out_rows)} patches to {out_fname}")
for row in out_rows[:3]:
    print(row)
