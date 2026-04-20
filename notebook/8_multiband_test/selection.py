"""Common selection helpers for the multiband tests.

All tests apply essentially the same cut set:

  - per-band magnitude cuts (g/r/i/z/y)
  - [optional] combined riz magnitude cut (SNR^2-weighted combined flux)
  - |e|^2 < emax^2
  - m2/m0 > trace_min
  - [optional] photo-z cut  z_min < z < z_max

``get_cut`` returns a boolean mask, optionally with observables perturbed
along shear component ``comp`` by ``dg_eff``. Set ``dg_eff = 0.0`` for the
fiducial selection (undistorted). ``zbin=None`` skips the photo-z cut.
"""

import numpy as np

# ---- defaults (match get_global_stats / test_0_stats.py) ----
MAG_ZERO = 27.0
MAG_CUTS_MULTIBAND = {"g": 27.0, "r": 27.0, "i": 25.8, "z": 26.0, "y": 26.0}
MAG_CUTS_ORIGINAL = {"g": 27.0, "r": 26.0, "i": 24.6, "z": 25.0, "y": 25.5}
COMBINED_MAG_CUT = 25.8
TRACE_MIN = 0.05
EMAX_DEFAULT = 0.3
DG = 0.01
Z_MIN = 0.3
Z_MAX = 1.5

RIZ_BANDS = ("r", "i", "z")
RIZ_WEIGHTS = {"r": 0.2215, "i": 0.5593, "z": 0.2192}


def get_mag(flux, mag_zero=MAG_ZERO):
    """Flux -> magnitude with a floor of 40 for non-positive flux."""
    flux = np.asarray(flux)
    mag = np.full(len(flux), 40.0, dtype=np.float64)
    pos = flux > 0
    mag[pos] = mag_zero - 2.5 * np.log10(flux[pos])
    return mag


def _z_column(zbin, zkey, comp, dg_eff, perturb_z):
    if (not perturb_z) or dg_eff == 0.0:
        return zbin[f"{zkey}_0"]
    suffix = "p" if dg_eff > 0 else "m"
    return zbin[f"{zkey}_{comp}{suffix}"]


def get_cut(
    d,
    *,
    comp=1,
    dg_eff=0.0,
    ext=None,
    zbin=None,
    zkey="zbest",
    perturb_z=True,
    emax=EMAX_DEFAULT,
    mag_cuts=MAG_CUTS_MULTIBAND,
    combined_mag_cut=COMBINED_MAG_CUT,
    trace_min=TRACE_MIN,
    z_min=Z_MIN,
    z_max=Z_MAX,
):
    """Boolean mask for the multiband (or original) selection.

    Parameters
    ----------
    d : structured array or dict-like
        Per-object catalog with plain-name columns. Required always:
            e1, e2, m0, m2, {g,r,i,z,y}_flux_gauss2
        Required when ``dg_eff != 0``:
            de1_dg{comp}, de2_dg{comp},
            dm0_dg{comp}, dm2_dg{comp},
            {b}_dflux_gauss2_dg{comp}
        Required when ``combined_mag_cut is not None``:
            flux_gauss2  (and dflux_gauss2_dg{comp} if dg_eff != 0)
    comp : {1, 2}
        Shear component to perturb.
    dg_eff : float
        Perturbation (``+DG``, ``-DG`` or ``0.0`` for fiducial).
    ext : structured array or None
        Extinction catalog with columns a_g, a_r, a_i, a_z, a_y.
        If None, no extinction correction is applied.
    zbin : structured array or None
        Photo-z catalog. If None, the z-cut is skipped.
    zkey : {"zbest", "zmode"}
    perturb_z : bool
        If False, the z-cut always uses ``{zkey}_0`` regardless of
        ``dg_eff`` (use this to exclude the photo-z from the selection
        response).
    emax : float
    mag_cuts : dict
        Per-band mag limits. Defaults to ``MAG_CUTS_MULTIBAND``; pass
        ``MAG_CUTS_ORIGINAL`` for the i-band-only analysis.
    combined_mag_cut : float or None
        Combined riz magnitude cut on ``d["flux_gauss2"]``. Pass
        ``None`` to skip the combined cut (e.g. original i-band
        analysis).
    trace_min : float
    z_min, z_max : float
    """
    emax_sq = emax ** 2

    e1 = d["e1"]
    e2 = d["e2"]
    if dg_eff == 0.0:
        esq = e1 ** 2 + e2 ** 2
        m0 = d["m0"]
        m2 = d["m2"]
    else:
        de1 = d[f"de1_dg{comp}"]
        de2 = d[f"de2_dg{comp}"]
        esq = e1 ** 2 + e2 ** 2 + 2.0 * dg_eff * (e1 * de1 + e2 * de2)
        m0 = d["m0"] + dg_eff * d[f"dm0_dg{comp}"]
        m2 = d["m2"] + dg_eff * d[f"dm2_dg{comp}"]

    mask = (esq < emax_sq) & ((m2 / m0) > trace_min)

    # Per-band magnitude cuts
    for b in "grizy":
        flux = d[f"{b}_flux_gauss2"]
        if dg_eff != 0.0:
            flux = flux + dg_eff * d[f"{b}_dflux_gauss2_dg{comp}"]
        mag_b = get_mag(flux)
        if ext is not None:
            mag_b = mag_b - ext[f"a_{b}"]
        mask &= mag_b < mag_cuts[b]

    # Combined riz magnitude cut (skip if combined_mag_cut is None)
    if combined_mag_cut is not None:
        flux_riz = d["flux_gauss2"]
        if dg_eff != 0.0:
            flux_riz = flux_riz + dg_eff * d[f"dflux_gauss2_dg{comp}"]
        combined_mag = get_mag(flux_riz)
        if ext is not None:
            a_riz = sum(RIZ_WEIGHTS[b] * ext[f"a_{b}"] for b in RIZ_BANDS)
            combined_mag = combined_mag - a_riz
        mask &= combined_mag < combined_mag_cut

    # Photo-z cut (optional)
    if zbin is not None:
        zcol = _z_column(zbin, zkey, comp, dg_eff, perturb_z)
        mask &= (zcol > z_min) & (zcol < z_max)

    return mask


def sel_response(d, comp, zbin=None, zkey="zbest", perturb_z=True, dg=DG,
                 **kwargs):
    """Selection response for ``comp`` via centred finite difference.

    Returns ``(mean(w*e)[cut_p] - mean(w*e)[cut_m]) / (2 dg)``.
    """
    e = d[f"e{comp}"]
    w = d["wsel"]
    cut_p = get_cut(d, comp=comp, dg_eff=+dg, zbin=zbin, zkey=zkey,
                    perturb_z=perturb_z, **kwargs)
    cut_m = get_cut(d, comp=comp, dg_eff=-dg, zbin=zbin, zkey=zkey,
                    perturb_z=perturb_z, **kwargs)
    return (np.mean(w[cut_p] * e[cut_p])
            - np.mean(w[cut_m] * e[cut_m])) / (2.0 * dg)
