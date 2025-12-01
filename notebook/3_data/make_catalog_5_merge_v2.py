#!/usr/bin/env python3

import argparse
import glob
import os

import fitsio
import healpy as hp
import numpy as np
import numpy.lib.recfunctions as rfn
import smatch
from lsst.skymap.ringsSkyMap import RingsSkyMap, RingsSkyMapConfig
from mpi4py import MPI

colnames = [
    "object_id",
    "ra",
    "dec",
    "wsel",
    "dwsel_dg1",
    "dwsel_dg2",
    "e1",
    "de1_dg1",
    "de1_dg2",
    "e2",
    "de2_dg1",
    "de2_dg2",
    "m0",
    "dm0_dg1",
    "dm0_dg2",
    "m2",
    "dm2_dg1",
    "dm2_dg2",
]


colnames2 = []
for b in "grizy":
    for t in "24":
        colnames2.append(f"{b}_flux_gauss{t}")
        colnames2.append(f"{b}_dflux_gauss{t}_dg1")
        colnames2.append(f"{b}_dflux_gauss{t}_dg2")
        colnames2.append(f"{b}_flux_gauss{t}_err")


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Process patch masks with MPI."
    )
    parser.add_argument("--field", type=str, required=True, help="field name")
    return parser.parse_args()


# Divide data among MPI ranks
def split_work(data, size, rank):
    return data[rank::size]


def process_patch(entry, hmask, stars, skymap):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9

    patch_info = skymap[tract_id][patch_id]
    wcs = patch_info.getWcs()
    bbox = patch_info.getOuterBBox()
    x, y = wcs.skyToPixelArray(
        ra=stars["ra"],
        dec=stars["dec"],
        degrees=True,
    )
    mm = (
        (x >= bbox.getBeginX()) &
        (y >= bbox.getBeginY()) &
        (x <  bbox.getEndX())   &
        (y <  bbox.getEndY())
    )
    ss = stars[mm]

    base_dir = f"{os.environ['s23b_anacal_v2']}/{tract_id}/{patch_id}"
    fname = os.path.join(base_dir, "match.fits")
    fname2 = os.path.join(base_dir, "force.fits")
    sel_fname = os.path.join(base_dir, "fdfc_sel.fits")
    radius_deg = 2.0 / 3600.0 # degrees
    NSIDE = 1024
    if os.path.isfile(fname) and os.path.isfile(fname2):
        dd = np.array(fitsio.read(fname))
        matches = smatch.match(
            ra1=ss["ra"], dec1=ss["dec"],
            radius1=radius_deg,
            ra2=dd["ra"], dec2=dd["dec"],
            maxmatch=15,
        )
        mstar = np.ones(len(dd), dtype=bool)
        i2 = matches["i2"]
        i2_valid = i2[i2 >= 0]
        if i2_valid.size > 0:
            mstar[np.unique(i2_valid)] = False

        if not os.path.isfile(sel_fname):
            if hmask is not None:
                pix = hp.ang2pix(
                    NSIDE,
                    np.deg2rad(90.0 - dd["dec"]),
                    np.deg2rad(dd["ra"]),
                    nest=True
                )
                sel = (dd["wsel"] > 1e-6) & (mstar) & (hmask[pix])
            else:
                sel = (dd["wsel"] > 1e-6) & (mstar)
            fitsio.write(sel_fname, sel.astype(int))
        else:
            sel = (fitsio.read(sel_fname) > 0)
        if np.sum(sel) < 3:
            return None

        dd = dd[sel]
        dd["dflux_dg2"] = -dd["dflux_dg2"]
        dd["dwsel_dg2"] = -dd["dwsel_dg2"]
        dd["dm0_dg2"] = -dd["dm0_dg2"]
        dd["dm2_dg2"] = -dd["dm2_dg2"]
        dd["de1_dg2"] = -dd["de1_dg2"]
        dd["de2_dg1"] = -dd["de2_dg1"]
        dd = rfn.repack_fields(
            dd[colnames]
        )

        dd2 = fitsio.read(fname2, columns=colnames2)
        new_dtype = [(name, "f4") for name in dd2.dtype.names]
        dd2 = dd2.astype(new_dtype)
        dd2 = dd2[sel]
        for b in "grizy":
            for t in "24":
                dd2[f"{b}_dflux_gauss{t}_dg2"] = -dd2[f"{b}_dflux_gauss{t}_dg2"]
        dd2 = rfn.repack_fields(dd2)
        return rfn.merge_arrays([dd, dd2], usemask=False, flatten=True)
    else:
        return None


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        rootdir = os.environ["s23b"]
        full = fitsio.read(
            f"{rootdir}/tracts_fdfc_v1_final.fits"
        )
        mm = full["field"] == args.field
        selected = full[mm]
        hpfname = f"{rootdir}/fdfc_hp_window_v2.fits"
        if os.path.isfile(hpfname):
            hmask = hp.read_map(
                hpfname,
                nest=True, dtype=bool,
            )
        else:
            hmask = None
        stars = np.array(fitsio.read(f"{rootdir}/gaia/stars.fits"))
        stars = stars[(stars["g_mag"]<21.0) & (stars["g_mag"]>15.0)]
    else:
        selected = None
        hmask = None
        stars = None

    # Set up the configuration
    config = RingsSkyMapConfig()
    config.numRings = 120
    config.projection = "TAN"
    config.tractOverlap = 1.0 / 60  # degrees
    config.pixelScale = 0.168  # arcsec/pixel
    skymap = RingsSkyMap(config)

    selected = comm.bcast(selected, root=0)
    hmask = comm.bcast(hmask, root=0)
    stars = comm.bcast(stars, root=0)
    my_entries = split_work(selected, size, rank)

    data = []
    for entry in my_entries:
        out = process_patch(entry, hmask, stars, skymap)
        if out is not None:
            data.append(out)

    data = rfn.stack_arrays(data, usemask=False)
    field = args.field
    out_dir = os.path.join(os.environ['s23b_anacal_v2'], "fields")
    fitsio.write(
        os.path.join(out_dir, f"{field}_{rank}.fits"),
        data,
    )
    comm.Barrier()

    if rank == 0:
        field = args.field
        out_dir = os.path.join(os.environ['s23b_anacal_v2'], "fields")
        d_all = []
        fnames = glob.glob(os.path.join(out_dir, f"{field}_*.fits"))
        for fn in fnames:
            if os.path.isfile(fn):
                d_all.append(
                    fitsio.read(fn)
                )
                os.remove(fn)
        outcome = rfn.stack_arrays(d_all, usemask=False)
        order = np.argsort(outcome["object_id"])
        outcome = outcome[order]
        fitsio.write(
            os.path.join(out_dir, f"{field}.fits"),
            outcome,
        )
    return


if __name__ == "__main__":
    main()
