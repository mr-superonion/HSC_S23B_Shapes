import os

import astropy.table as astTable
import fitsio
import numpy as np

version = 1

tract_list = np.sort(
    astTable.Table.read(
        "/work/xiangchong.li/work/hsc_s23b_sim/catalogs/tracts.csv"
    )["tract"]
)
gal_dir = "/work/xiangchong.li/work/hsc_s23b_sim/catalogs/database/s23b-galaxy/tracts/"
band_list = ["g", "r", "i", "z", "y"]
if version == 0:
    min_list = [4, 4, 4, 6, 6]
    min_5per = 2
elif version == 1:
    min_list = [4, 4, 4, 4, 4]
    min_5per = 2
elif version == 2:
    min_list = [4, 4, 4, 6, 6]
    min_5per = 1
elif version == 3:
    min_list = [4, 4, 4, 4, 4]
    min_5per = 1
else:
    raise ValueError("version wrong")

output = []
for tract in tract_list:
    gal_fname = os.path.join(gal_dir, f"{tract}.fits")
    data = astTable.Table.read(gal_fname)
    patch_list = np.unique(data["patch"])
    for patch in patch_list:
        tmp = data[data["patch"] == patch]
        select = True
        for band, input_min in zip(band_list, min_list):
            input_ave = np.round(np.average(tmp[f"{band}_inputcount_value"]))
            input_5per = np.percentile(tmp[f"{band}_inputcount_value"], 5)
            select = (
                select & (input_ave >= input_min) & (input_5per >= min_5per)
            )
            if ~select:
                break
        if select:
            output.append((tract, patch))

res = np.array(output, dtype=[("tract", "i4"), ("patch", "i4")])
fitsio.write(
    "/work/xiangchong.li/work/hsc_s23b_sim/catalogs/tracts_fdfc_v%d.fits" % (
        version
    ),
    res,
)
