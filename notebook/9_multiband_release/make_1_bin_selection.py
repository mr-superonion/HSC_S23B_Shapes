#!/usr/bin/env python3
"""Build per-field tomographic bin selection files (fiducial + perturbed).

For each field, reads fields_multiband/{field}.fits (shapes/fluxes +
derivatives), fields_redshift/{field}.fits (photo-z + perturbed photo-z
columns) and fields_extinction/{field}.fits. Uses selection.get_cut with
extinction correction.

5 tomographic bins on zmode:
  0.3 <  zmode <= 0.6  -> bin 0
  0.6 <  zmode <= 0.9  -> bin 1
  0.9 <  zmode <= 1.2  -> bin 2
  1.2 <  zmode <= 1.5  -> bin 3
  1.5 <  zmode <= 1.8  -> bin 4
  otherwise            -> -1

Five bin columns are written, each from a different observable
perturbation (all shear-dependent quantities perturbed by dg_eff along
the given component; extinction is unchanged):

  bin     : fiducial      (dg_eff = 0,   zmode_0)
  bin_1p  : comp=1, dg=+0.01           (zmode_1p)
  bin_1m  : comp=1, dg=-0.01           (zmode_1m)
  bin_2p  : comp=2, dg=+0.01           (zmode_2p)
  bin_2m  : comp=2, dg=-0.01           (zmode_2m)

Selection bias and weight bias per bin are read from the output of
test_0_stats_weight_bins.py and stored in the FITS header.

Output: fields_bin/{field}_imag{imag}_emax{emax}.fits.
Parallelized over fields via MPI.
"""

import argparse
import gc
import os
import sys

import fitsio
import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "8_multiband_test"))
from selection import get_cut, MAG_CUTS_MULTIBAND, DG

BASE = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/deepCoadd_anacal_v2"
MB_DIR = f"{BASE}/fields_multiband"
Z_DIR = f"{BASE}/fields_redshift"
EXT_DIR = f"{BASE}/fields_extinction"
OUT_DIR = f"{BASE}/fields_bin"

FIELDS = ("spring1", "spring2", "spring3", "autumn1", "autumn2", "hectomap")

# --- tomographic bins on zmode ---
BIN_EDGES = [(0.3, 0.6), (0.6, 0.9), (0.9, 1.2), (1.2, 1.5), (1.5, 1.8)]
NBINS = len(BIN_EDGES)

dg = DG

# Five perturbation variants: (output_suffix, comp, dg_eff, zbest_suffix)
VARIANTS = [
    ("",    1,  0.0, "0"),
    ("_1p", 1, +dg, "1p"),
    ("_1m", 1, -dg, "1m"),
    ("_2p", 2, +dg, "2p"),
    ("_2m", 2, -dg, "2m"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build tomographic bin selection per field."
    )
    parser.add_argument(
        "--field", type=str, default="all",
        help="Single field to process (default: all).",
    )
    parser.add_argument(
        "--imag", type=float, default=MAG_CUTS_MULTIBAND["i"],
        help="i-band magnitude cut",
    )
    return parser.parse_args()


def assign_bin(sel, zmode):
    bin_arr = np.full(len(zmode), -1, dtype=np.int8)
    for ibin, (z_lo, z_hi) in enumerate(BIN_EDGES):
        bin_mask = sel & (zmode > z_lo) & (zmode <= z_hi)
        bin_arr[bin_mask] = ibin
    return bin_arr


def process_field(field, mag_cuts, imag, emax=0.4):
    out_path = os.path.join(
        OUT_DIR, f"{field}_imag{imag:.1f}.fits"
    )
    if os.path.isfile(out_path):
        print(f"[{field}] already exists, skipping")
        return

    mb_path = os.path.join(MB_DIR, f"{field}.fits")
    z_path = os.path.join(Z_DIR, f"{field}.fits")
    ext_path = os.path.join(EXT_DIR, f"{field}.fits")
    if not (os.path.isfile(mb_path) and os.path.isfile(z_path)
            and os.path.isfile(ext_path)):
        print(f"[{field}] missing inputs, skipping")
        return

    d = fitsio.read(mb_path)
    n = len(d)
    print(f"[{field}] {n} objects")

    # Redshift file
    z_cols = ["object_id",
              "zbest_0", "zbest_1p", "zbest_1m", "zbest_2p", "zbest_2m"]
    zb = fitsio.read(z_path, columns=z_cols)
    assert np.array_equal(zb["object_id"], d["object_id"]), (
        f"{field}: object_id mismatch between multiband and redshift"
    )

    # Extinction file
    ext = fitsio.read(ext_path)
    assert np.array_equal(ext["object_id"], d["object_id"]), (
        f"{field}: object_id mismatch between multiband and extinction"
    )

    # Common kwargs for get_cut (no z-cut -- we assign bins manually)
    common_kwargs = dict(
        ext=ext, emax=emax, mag_cuts=mag_cuts, zbin=None,
    )

    # Compute each variant's bin column
    dtype = [("object_id", "i8")]
    for suf, _, _, _ in VARIANTS:
        dtype.append((f"bin{suf}", "i1"))
    out = np.empty(n, dtype=dtype)
    out["object_id"] = d["object_id"]

    for suf, comp, dg_eff, z_suffix in VARIANTS:
        sel = get_cut(d, comp=comp, dg_eff=dg_eff, **common_kwargs)
        zbest = zb[f"zbest_{z_suffix}"]
        bin_arr = assign_bin(sel, zbest)
        out[f"bin{suf}"] = bin_arr

        name = f"bin{suf}"
        counts = ", ".join(
            f"{ib}:{np.sum(bin_arr == ib)}"
            for ib in [-1, 0, 1, 2, 3, 4]
        )
        print(f"[{field}] {name}: {counts}")

    # Read selection bias and weight bias from test_0 output
    bias_path = f"{BASE}/bias_per_zbin_imag{imag:.1f}.fits"
    header = {}
    if os.path.isfile(bias_path):
        bias = fitsio.read(bias_path)
        for ib in range(NBINS):
            header[f"SELB_{ib}"] = float(bias["sel_bias"][ib])
            header[f"WGTB_{ib}"] = float(bias["weight_bias"][ib])
        print(f"[{field}] bias loaded from {bias_path}")
    else:
        print(f"[{field}] WARNING: bias file not found: {bias_path}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with fitsio.FITS(out_path, "rw", clobber=True) as fits:
        fits.write(out, header=header)
    print(f"[{field}] written: {out_path}")

    del d, zb, ext, out
    gc.collect()


def main():
    args = parse_args()
    mag_cuts = dict(MAG_CUTS_MULTIBAND)
    mag_cuts["i"] = args.imag

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
        process_field(field, mag_cuts, args.imag)
        gc.collect()

    comm.Barrier()
    if rank == 0:
        print("All done.")


if __name__ == "__main__":
    main()
