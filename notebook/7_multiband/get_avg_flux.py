#!/usr/bin/env python3

import fitsio
import numpy as np

bands = ["r", "i", "z"]

base = (
    "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/"
    "deepCoadd_anacal_v2/fields"
)
fields = ["spring1", "spring2", "spring3", "autumn1", "autumn2", "hectomap"]

total_n = 0
total_flux = {b: 0.0 for b in bands}
total_err = {b: 0.0 for b in bands}

for field in fields:
    d1 = fitsio.read(f"{base}/{field}.fits")

    # Photo-z selection
    zbin = fitsio.read(
        f"/gpfs02/work/xiangchong.li/work/hsc_data/s23b/"
        f"deepCoadd_anacal_v2/fields_redshift/{field}.fits"
    )
    zbest = zbin["zbest_0"]
    zmask = (zbest > 0.3) & (zbest < 1.5)
    d1 = d1[zmask]

    cut = (
        ((27 - 2.5 * np.log10(np.clip(d1["g_flux_gauss2"], 1e-20, None))) < 27.0)
        & ((27 - 2.5 * np.log10(np.clip(d1["r_flux_gauss2"], 1e-20, None))) < 26.0)
        & ((27 - 2.5 * np.log10(np.clip(d1["i_flux_gauss2"], 1e-20, None))) < 25.5)
        & ((27 - 2.5 * np.log10(np.clip(d1["z_flux_gauss2"], 1e-20, None))) < 25.0)
        & ((27 - 2.5 * np.log10(np.clip(d1["y_flux_gauss2"], 1e-20, None))) < 25.5)
    )
    n = np.sum(cut)
    total_n += n
    for b in bands:
        total_flux[b] += np.sum(d1[cut][f"{b}_flux_gauss2"])
        total_err[b] += np.sum(d1[cut][f"{b}_flux_gauss2_err"])
    print(f"{field}: {n} objects after cut")
    del d1

print(f"\nTotal objects: {total_n}")
print(f"\n{'band':<6} {'avg_flux_gauss2':>18} {'avg_flux_gauss2_err':>22}")
print("-" * 48)
for b in bands:
    print(f"{b:<6} {total_flux[b] / total_n:>18.6f} {total_err[b] / total_n:>22.6f}")
