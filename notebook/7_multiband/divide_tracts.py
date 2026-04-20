#!/usr/bin/env python3
"""Divide a field into per-tract catalogs after multiband combination.

For each field, this script:
  1. Reads {field}_multiband.fits and combines r/i/z moments with SNR^2 weights.
  2. Computes ellipticities and their derivatives via _moments_to_ell.
  3. Applies WCS correction using {field}_wcs_perturb.fits (flipu=True)
     and sign-flips the u-dependent derivatives.
  4. Reads {field}.fits to get wsel, dwsel_dg1, dwsel_dg2, per-band fluxes
     and their shear derivatives.
  5. Builds a combined riz flux_gauss2 (+ derivatives).
  6. Writes {tract}.fits in tracts_multiband/ with the full combined
     catalog. Also writes {tract}.fits in tracts_color/ (unchanged
     per-tract slice of fields_color).
"""

import argparse
import gc
import os

import fitsio
import numpy as np
from xlens.catalog.utils import _moments_to_ell
from xlens.wcs import correct_ellipticity_wcs

# --- configuration ---
C0 = 0.53
BANDS_MULTIBAND = ("r", "i", "z")
WEIGHTS = {"r": 0.2215, "i": 0.5593, "z": 0.2192}
BANDS_FLUX = ("g", "r", "i", "z", "y")

MOMENT_NAMES = (
    "m00", "dm00_dg1", "dm00_dg2",
    "m20", "dm20_dg1", "dm20_dg2",
    "m22c", "dm22c_dg1", "dm22c_dg2",
    "m22s", "dm22s_dg1", "dm22s_dg2",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a field into per-tract multiband catalogs."
    )
    parser.add_argument(
        "--field", type=str, required=True, help="field name"
    )
    return parser.parse_args()


def build_combined(d1, d2, wcs):
    """Combine multiband moments, compute ellipticities, WCS correct."""
    # Weighted multiband moments
    avg = {}
    for mn in MOMENT_NAMES:
        avg[mn] = sum(
            WEIGHTS[b] * d2[f"{b}_fpfs1_{mn}"] for b in BANDS_MULTIBAND
        )
    nobj = len(d2)

    ell = _moments_to_ell(
        nobj, C0, "fpfs1_",
        avg["m00"], avg["m20"], avg["m22c"], avg["m22s"],
        avg["dm00_dg1"], avg["dm00_dg2"],
        avg["dm20_dg1"], avg["dm20_dg2"],
        avg["dm22c_dg1"], avg["dm22c_dg2"],
        avg["dm22s_dg1"], avg["dm22s_dg2"],
    )
    del avg

    # WCS correction (flipu=True) + sign flips on u-dependent derivatives
    e1_corr, e2_corr = correct_ellipticity_wcs(
        ell, wcs["g1_wcs"], wcs["g2_wcs"], wcs["rho_wcs"],
        prefix="fpfs1_", flipu=True,
    )
    ell["fpfs1_e1"] = e1_corr
    ell["fpfs1_e2"] = e2_corr
    ell["fpfs1_de1_dg2"] = -ell["fpfs1_de1_dg2"]
    ell["fpfs1_de2_dg1"] = -ell["fpfs1_de2_dg1"]
    ell["fpfs1_dm0_dg2"] = -ell["fpfs1_dm0_dg2"]
    ell["fpfs1_dm2_dg2"] = -ell["fpfs1_dm2_dg2"]

    # Combined riz flux (+ shear derivatives)
    flux_gauss2 = sum(
        WEIGHTS[b] * d1[f"{b}_flux_gauss2"] for b in BANDS_MULTIBAND
    )
    dflux_gauss2_dg1 = sum(
        WEIGHTS[b] * d1[f"{b}_dflux_gauss2_dg1"] for b in BANDS_MULTIBAND
    )
    dflux_gauss2_dg2 = sum(
        WEIGHTS[b] * d1[f"{b}_dflux_gauss2_dg2"] for b in BANDS_MULTIBAND
    )

    # Build structured output array
    dtype = [
        ("object_id", "i8"),
        ("ra", "f8"),
        ("dec", "f8"),
        ("wsel", "f8"),
        ("dwsel_dg1", "f8"),
        ("dwsel_dg2", "f8"),
        ("e1", "f8"),
        ("e2", "f8"),
        ("de1_dg1", "f8"),
        ("de1_dg2", "f8"),
        ("de2_dg1", "f8"),
        ("de2_dg2", "f8"),
        ("m0", "f8"),
        ("dm0_dg1", "f8"),
        ("dm0_dg2", "f8"),
        ("m2", "f8"),
        ("dm2_dg1", "f8"),
        ("dm2_dg2", "f8"),
        ("flux_gauss2", "f8"),
        ("dflux_gauss2_dg1", "f8"),
        ("dflux_gauss2_dg2", "f8"),
    ]
    # Per-band fluxes (kept for per-band magnitude cuts)
    for b in BANDS_FLUX:
        dtype.append((f"{b}_flux_gauss2", "f8"))
        dtype.append((f"{b}_dflux_gauss2_dg1", "f8"))
        dtype.append((f"{b}_dflux_gauss2_dg2", "f8"))

    out = np.empty(nobj, dtype=dtype)
    out["object_id"] = d1["object_id"]
    out["ra"] = d1["ra"]
    out["dec"] = d1["dec"]
    out["wsel"] = d1["wsel"]
    out["dwsel_dg1"] = d1["dwsel_dg1"]
    out["dwsel_dg2"] = d1["dwsel_dg2"]

    out["e1"] = ell["fpfs1_e1"]
    out["e2"] = ell["fpfs1_e2"]
    out["de1_dg1"] = ell["fpfs1_de1_dg1"]
    out["de1_dg2"] = ell["fpfs1_de1_dg2"]
    out["de2_dg1"] = ell["fpfs1_de2_dg1"]
    out["de2_dg2"] = ell["fpfs1_de2_dg2"]
    out["m0"] = ell["fpfs1_m0"]
    out["dm0_dg1"] = ell["fpfs1_dm0_dg1"]
    out["dm0_dg2"] = ell["fpfs1_dm0_dg2"]
    out["m2"] = ell["fpfs1_m2"]
    out["dm2_dg1"] = ell["fpfs1_dm2_dg1"]
    out["dm2_dg2"] = ell["fpfs1_dm2_dg2"]

    out["flux_gauss2"] = flux_gauss2
    out["dflux_gauss2_dg1"] = dflux_gauss2_dg1
    out["dflux_gauss2_dg2"] = dflux_gauss2_dg2
    for b in BANDS_FLUX:
        out[f"{b}_flux_gauss2"] = d1[f"{b}_flux_gauss2"]
        out[f"{b}_dflux_gauss2_dg1"] = d1[f"{b}_dflux_gauss2_dg1"]
        out[f"{b}_dflux_gauss2_dg2"] = d1[f"{b}_dflux_gauss2_dg2"]

    del ell
    return out


def select_data(d, sel):
    outcome = d[sel]
    order = np.argsort(outcome["object_id"])
    return outcome[order]


def main():
    args = parse_args()
    field = args.field
    rootdir = os.environ["s23b_anacal_v2"]

    field_fname = f"{rootdir}/fields/{field}.fits"
    multi_fname = f"{rootdir}/fields/{field}_multiband.fits"
    wcs_fname = f"{rootdir}/fields/{field}_wcs_perturb.fits"
    color_fname = f"{rootdir}/fields_color/{field}.fits"

    print(f"[{field}] reading inputs...")
    d1 = fitsio.read(field_fname)
    d2 = fitsio.read(multi_fname)
    wcs = fitsio.read(wcs_fname)
    d_color = fitsio.read(color_fname)

    n = len(d1)
    print(f"[{field}] {n} objects")
    assert len(d2) == n, "field and multiband row counts differ"
    assert len(wcs) == n, "field and wcs_perturb row counts differ"
    assert len(d_color) == n, "field and color row counts differ"

    print(f"[{field}] combining multiband moments and computing ell...")
    combined = build_combined(d1, d2, wcs)
    del d1, d2, wcs
    gc.collect()

    out_dir_multi = f"{rootdir}/tracts_multiband"
    out_dir_color = f"{rootdir}/tracts_color"
    os.makedirs(out_dir_multi, exist_ok=True)
    os.makedirs(out_dir_color, exist_ok=True)

    tract_col = d_color["tract"]
    tracts = np.unique(tract_col)
    print(f"[{field}] {len(tracts)} tracts")

    for tt in tracts:
        sel = tract_col == tt
        mpath = f"{out_dir_multi}/{tt}.fits"
        if not os.path.isfile(mpath):
            out1 = select_data(combined, sel)
            fitsio.write(mpath, out1, clobber=True)
            del out1
        gc.collect()

    print(f"[{field}] done.")


if __name__ == "__main__":
    main()
