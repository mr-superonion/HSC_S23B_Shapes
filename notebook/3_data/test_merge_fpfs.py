#!/usr/bin/env python3
"""Test that the merged catalog columns work with
multiband_shapelets2ell and correct_ellipticity_wcs.
Runs on a single patch (first entry in tracts_fdfc_v2_final.fits).
"""

import glob
import os

import fitsio
import numpy as np
import numpy.lib.recfunctions as rfn
from lsst.afw.table import SourceCatalog
from xlens.catalog.utils import multiband_shapelets2ell
from xlens.wcs import correct_ellipticity_wcs, jacobian_decomposition

# --- reproduce process_patch for one entry ---
rootdir = os.environ["s23b"]
full = fitsio.read(f"{rootdir}/tracts_fdfc_v2_final.fits")
entry = full[0]
tract_id = int(entry["tract"])
patch_db = int(entry["patch"])
patch_x = patch_db // 100
patch_y = patch_db % 100
patch_id = patch_x + patch_y * 9
print(f"tract={tract_id}, patch_id={patch_id}")

base_dir = f"{os.environ['s23b_anacal_v2']}/{tract_id}/{patch_id}"
bands = ["r", "i", "z"]

fpfs_moment_cols = [
    "fpfs1_m00", "fpfs1_dm00_dg1", "fpfs1_dm00_dg2",
    "fpfs1_m20", "fpfs1_dm20_dg1", "fpfs1_dm20_dg2",
    "fpfs1_m22c", "fpfs1_dm22c_dg1", "fpfs1_dm22c_dg2",
    "fpfs1_m22s", "fpfs1_dm22s_dg1", "fpfs1_dm22s_dg2",
    "fpfs1_m42c", "fpfs1_dm42c_dg1", "fpfs1_dm42c_dg2",
    "fpfs1_m42s", "fpfs1_dm42s_dg1", "fpfs1_dm42s_dg2",
]
force_cols_per_band = [
    "flux_gauss2", "dflux_gauss2_dg1", "dflux_gauss2_dg2",
    "flux_gauss2_err",
]

dd2 = np.array(fitsio.read(os.path.join(base_dir, "match.fits")))
sel = fitsio.read(os.path.join(base_dir, "fdfc_sel.fits")) > 0
dd2 = dd2[sel]
idx = dd2["index"]

force = np.array(fitsio.read(os.path.join(base_dir, "force.fits")))
force = force[sel]

arrays = []
oid = np.zeros(len(dd2), dtype=[("object_id", "i8")])
oid["object_id"] = dd2["object_id"]
arrays.append(oid)

for band in bands:
    fpfs = np.array(fitsio.read(
        os.path.join(base_dir, f"fpfs_{band}.fits")
    ))
    fpfs = fpfs[idx]
    cols_to_keep = [c for c in fpfs_moment_cols if c in fpfs.dtype.names]
    fpfs_sub = rfn.repack_fields(fpfs[cols_to_keep])
    map_dict = {c: f"{band}_{c}" for c in cols_to_keep}
    fpfs_renamed = rfn.rename_fields(fpfs_sub, map_dict)
    arrays.append(fpfs_renamed)

    force_cols = [f"{band}_{c}" for c in force_cols_per_band]
    existing = [c for c in force_cols if c in force.dtype.names]
    if existing:
        force_sub = rfn.repack_fields(force[existing])
        err_col = f"{band}_flux_gauss2_err"
        if err_col in force_sub.dtype.names:
            force_sub[err_col] = np.median(force[err_col])
        arrays.append(force_sub)

cat = rfn.merge_arrays(arrays, flatten=True)
print(f"Merged catalog: {len(cat)} objects, {len(cat.dtype.names)} columns")

# --- WCS columns ---
cat_dir = f"{os.environ['s23b_meas']}/{tract_id}/{patch_id}/i"
cat_files = glob.glob(os.path.join(cat_dir, "*.fits"))
dm_cat = SourceCatalog.readFits(cat_files[0])
dm_mask = dm_cat["detect_isPrimary"]
dm_cols = [
    "id",
    "base_LocalWcs_CDMatrix_1_1",
    "base_LocalWcs_CDMatrix_1_2",
    "base_LocalWcs_CDMatrix_2_1",
    "base_LocalWcs_CDMatrix_2_2",
]
dm_arr = rfn.repack_fields(
    dm_cat.asAstropy().as_array()[dm_cols][dm_mask]
)
del dm_cat

_, idx_cat, idx_dm = np.intersect1d(
    cat["object_id"], dm_arr["id"], return_indices=True,
)
dm_matched = dm_arr[idx_dm]
cat = cat[idx_cat]

pixel_scale = 0.168
rad_to_arcsec = (180.0 / np.pi) * 3600.0
n = len(dm_matched)
wcs_arr = np.zeros(
    n,
    dtype=[
        ("g1_wcs", np.float64),
        ("g2_wcs", np.float64),
        ("rho_wcs", np.float64),
        ("kappa_wcs", np.float64),
    ],
)
for i in range(n):
    jac = np.array([
        [-dm_matched["base_LocalWcs_CDMatrix_1_1"][i] * rad_to_arcsec,
         -dm_matched["base_LocalWcs_CDMatrix_1_2"][i] * rad_to_arcsec],
        [dm_matched["base_LocalWcs_CDMatrix_2_1"][i] * rad_to_arcsec,
         dm_matched["base_LocalWcs_CDMatrix_2_2"][i] * rad_to_arcsec],
    ])
    g1, g2, rho, kappa = jacobian_decomposition(jac, pixel_scale)
    wcs_arr["g1_wcs"][i] = g1
    wcs_arr["g2_wcs"][i] = g2
    wcs_arr["rho_wcs"][i] = rho
    wcs_arr["kappa_wcs"][i] = kappa

cat = rfn.merge_arrays([cat, wcs_arr], flatten=True)
print(f"After WCS merge: {len(cat)} objects, {len(cat.dtype.names)} columns")

# === Step 1: multiband_shapelets2ell ===
C0 = 0.53
result = multiband_shapelets2ell(cat, bands=bands, C0=C0)
print("\nAfter multiband_shapelets2ell:")
print("Result columns:", result.dtype.names)
print(f"e1 range: [{result['fpfs1_e1'].min():.6f}, {result['fpfs1_e1'].max():.6f}]")
print(f"e2 range: [{result['fpfs1_e2'].min():.6f}, {result['fpfs1_e2'].max():.6f}]")
print(f"m0 range: [{result['fpfs1_m0'].min():.6f}, {result['fpfs1_m0'].max():.6f}]")
print(f"m2 range: [{result['fpfs1_m2'].min():.6f}, {result['fpfs1_m2'].max():.6f}]")

# === Step 2: correct_ellipticity_wcs ===
e1_corr, e2_corr = correct_ellipticity_wcs(
    result,
    cat["g1_wcs"],
    cat["g2_wcs"],
    cat["rho_wcs"],
    prefix="fpfs1_",
    flipu=True,
)
result["fpfs1_e1"] = e1_corr
result["fpfs1_e2"] = e2_corr

result["fpfs1_de1_dg2"] = -result["fpfs1_de1_dg2"]
result["fpfs1_de2_dg1"] = -result["fpfs1_de2_dg1"]
result["fpfs1_dm0_dg2"] = -result["fpfs1_dm0_dg2"]
result["fpfs1_dm2_dg2"] = -result["fpfs1_dm2_dg2"]

print("\nAfter WCS correction:")
print(f"e1 range: [{result['fpfs1_e1'].min():.6f}, {result['fpfs1_e1'].max():.6f}]")
print(f"e2 range: [{result['fpfs1_e2'].min():.6f}, {result['fpfs1_e2'].max():.6f}]")
print(f"mean e1: {np.mean(result['fpfs1_e1']):.6f}")
print(f"mean e2: {np.mean(result['fpfs1_e2']):.6f}")

print("\nPASSED")
