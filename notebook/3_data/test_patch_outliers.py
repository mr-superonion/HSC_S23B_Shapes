#!/usr/bin/env python3

import argparse
import csv

import fitsio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--field", type=str, default="spring1", help="field name")
args = parser.parse_args()
field = args.field

base = (
    "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/"
    "deepCoadd_anacal_v2/fields"
)

fname = f"{base}/{field}_patch_comparison.csv"
with open(fname) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total patches: {len(rows)}")

bad = []
for r in rows:
    e1_bad = float(r["std_e1"]) < float(r["std_fpfs1_e1"])
    e2_bad = float(r["std_e2"]) < float(r["std_fpfs1_e2"])
    if e1_bad or e2_bad:
        bad.append(r)

print(
    f"Patches with std(e1)<std(fpfs1_e1) or "
    f"std(e2)<std(fpfs1_e2): {len(bad)}\n"
)
print(
    f"{'tract':>6} {'patch':>6} {'n_obj':>6} "
    f"{'std_e1':>10} {'std_fpfs1_e1':>13} "
    f"{'std_e2':>10} {'std_fpfs1_e2':>13} {'flag':>8}"
)
print("-" * 80)
for r in bad:
    e1_flag = "*" if float(r["std_e1"]) < float(r["std_fpfs1_e1"]) else ""
    e2_flag = "*" if float(r["std_e2"]) < float(r["std_fpfs1_e2"]) else ""
    flag = f"e1{e1_flag} e2{e2_flag}"
    print(
        f"{r['tract']:>6} {r['patch']:>6} {r['n_obj']:>6} "
        f"{r['std_e1']:>10} {r['std_fpfs1_e1']:>13} "
        f"{r['std_e2']:>10} {r['std_fpfs1_e2']:>13} {flag:>8}"
    )

# Good patches: remove the bad ones
good = []
for r in rows:
    e1_bad = float(r["std_e1"]) < float(r["std_fpfs1_e1"])
    e2_bad = float(r["std_e2"]) < float(r["std_fpfs1_e2"])
    if not (e1_bad or e2_bad):
        good.append(r)

print(f"\n=== Good patches: {len(good)} / {len(rows)} ===")
float_cols = [
    "std_e1", "std_fpfs1_e1", "std_e2", "std_fpfs1_e2",
    "mean_de1_dg1", "mean_fpfs1_de1_dg1",
    "mean_de1_dg2", "mean_fpfs1_de1_dg2",
    "mean_de2_dg1", "mean_fpfs1_de2_dg1",
    "mean_de2_dg2", "mean_fpfs1_de2_dg2",
]
print(f"{'column':<25} {'mean':>12} {'std':>12}")
print("-" * 50)
for col in float_cols:
    vals = np.array([float(r[col]) for r in good])
    print(f"{col:<25} {np.mean(vals):>12.6f} {np.std(vals):>12.6f}")
