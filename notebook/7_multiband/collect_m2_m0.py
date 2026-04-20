#!/usr/bin/env python3
"""Collect combined m0, m2, shear response, and selection weight.

Reads fields_multiband/{field}.fits (plain-name columns) and
fields_redshift/{field}.fits, applies selection via the shared
``selection.get_cut`` with emax=0.4. Writes TWO output files:

  - m2_m0_collected_zbest.fits  (photo-z cut on zbest_0)
  - m2_m0_collected_zmode.fits  (photo-z cut on zmode_0)

Both contain per-object: m0, m2, wsel, resp, dw_resp, e1, e2.

The output is read by 2_wsel.ipynb and show_m2_m0_grid.ipynb.
"""

import os
import sys

import fitsio
import numpy as np

# Import the shared selection function
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "8_multiband_test",
))
from selection import get_cut, MAG_CUTS_MULTIBAND, COMBINED_MAG_CUT

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
FIELDS = ["spring1", "spring2", "spring3", "autumn1", "autumn2", "hectomap"]
EMAX = 0.4

BASE = (
    "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/"
    "deepCoadd_anacal_v2"
)
MB_DIR = f"{BASE}/fields_multiband"
Z_DIR = f"{BASE}/fields_redshift"
OUT_DIR = BASE

ZKEYS = ("zbest", "zmode")


def process_field(field, zkey):
    """Load one field, apply cuts, return per-object arrays."""
    print(f"[{field}/{zkey}] loading...")
    d = fitsio.read(f"{MB_DIR}/{field}.fits")
    zbin = fitsio.read(f"{Z_DIR}/{field}.fits")
    assert np.array_equal(zbin["object_id"], d["object_id"])

    # Apply selection via the shared function
    mask = get_cut(
        d, comp=1, dg_eff=0.0,
        zbin=zbin, zkey=zkey,
        emax=EMAX,
    )
    n_cut = int(np.sum(mask))
    print(f"[{field}/{zkey}] after cuts: {n_cut} / {len(d)}")

    w = d["wsel"][mask]
    e1 = d["e1"][mask]
    e2 = d["e2"][mask]
    m0 = d["m0"][mask]
    m2 = d["m2"][mask]
    resp = 0.5 * (d["de1_dg1"][mask] + d["de2_dg2"][mask])
    dw_resp = 0.5 * (
        e1 * d["dwsel_dg1"][mask]
        + e2 * d["dwsel_dg2"][mask]
    )

    return {
        "m0": m0.astype(np.float32),
        "m2": m2.astype(np.float32),
        "wsel": w.astype(np.float32),
        "resp": resp.astype(np.float32),
        "dw_resp": dw_resp.astype(np.float32),
        "e1": e1.astype(np.float32),
        "e2": e2.astype(np.float32),
    }


def main():
    keys = ("m0", "m2", "wsel", "resp", "dw_resp", "e1", "e2")
    dtype = [(k, "f4") for k in keys]

    for zkey in ZKEYS:
        parts = [process_field(field, zkey) for field in FIELDS]

        combined = {k: np.concatenate([p[k] for p in parts]) for k in keys}
        n_total = len(combined["m0"])
        print(f"[{zkey}] total collected: {n_total} objects")

        out = np.empty(n_total, dtype=dtype)
        for k in keys:
            out[k] = combined[k]

        out_path = f"{OUT_DIR}/m2_m0_collected_{zkey}.fits"
        fitsio.write(out_path, out, clobber=True)
        print(f"Written: {out_path}")
        del parts, combined, out


if __name__ == "__main__":
    main()
