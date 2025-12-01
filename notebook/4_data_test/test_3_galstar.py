#!/usr/bin/env python3

import argparse
import os

import fitsio
import numpy as np
import numpy.lib.recfunctions as rfn
import treecorr
from mpi4py import MPI
import healpy as hp


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Process patch masks with MPI."
    )
    parser.add_argument(
        "--start", type=int, required=True, help="Start index of datalist."
    )
    parser.add_argument(
        "--end", type=int, required=True, help="End index of datalist."
    )
    return parser.parse_args()


# Divide data among MPI ranks
def split_work(data, size, rank):
    return data[rank::size]

def prepare_catalogs():
    data = []
    field_list = [
        "spring1", "spring2", "spring3", "autumn1", "autumn2", "hectomap"
    ]
    for field in field_list:
        fname = f"{os.environ['s23b_db_star']}/fields/{field}.fits"
        data.append(fitsio.read(fname))
    data = rfn.stack_arrays(data, usemask=False)
    snr = data["i_psfflux_flux"] / data["i_psfflux_fluxerr"]
    ra = data["i_ra"]
    dec = data["i_dec"]
    psf_mxx = data["i_hsmpsfmoments_shape11"]
    psf_myy = data["i_hsmpsfmoments_shape22"]
    psf_mxy = data["i_hsmpsfmoments_shape12"]

    e1p2 = (psf_mxx - psf_myy) / (psf_mxx + psf_myy)
    e2p2 = psf_mxy / (psf_mxx + psf_myy) * 2.0

    star_mxx = data["i_hsmsourcemoments_shape11"]
    star_myy = data["i_hsmsourcemoments_shape22"]
    star_mxy = data["i_hsmsourcemoments_shape12"]

    e1s2 = (star_mxx - star_myy) / (star_mxx + star_myy)
    e2s2 = star_mxy / (star_mxx + star_myy) * 2.0

    e1p4 = (
        data["i_higherordermomentspsf_40"] -
        data["i_higherordermomentspsf_04"]
    )
    e2p4 = 2.0 * (
        data["i_higherordermomentspsf_31"] +
        data["i_higherordermomentspsf_13"]
    )

    e1s4 = (
        data["i_higherordermomentssource_40"] -
        data["i_higherordermomentssource_04"]
    )
    e2s4 = 2.0 * (
        data["i_higherordermomentssource_31"] +
        data["i_higherordermomentssource_13"]
    )

    NSIDE = 1024
    hpfname = f"{os.environ['s23b']}/fdfc_hp_window_updated.fits"
    hmask = hp.read_map(
        hpfname,
        nest=True, dtype=bool,
    )
    pix = hp.ang2pix(
        NSIDE,
        np.deg2rad(90.0 - dec),
        np.deg2rad(ra),
        nest=True
    )
    msk = (
        (~np.isnan(e1p2))
        & (~np.isnan(e1s2))
        & (~np.isnan(e1p4))
        & (~np.isnan(e1s4))
        & (~np.isnan(e2p2))
        & (~np.isnan(e2s2))
        & (~np.isnan(e2p4))
        & (~np.isnan(e2s4))
        & (data["i_calib_psf_reserved"])
        & (snr>200.0)
        # & hmask[pix]
    )

    ra = ra[msk]
    dec = dec[msk]
    e1p2 = e1p2[msk]
    e2p2 = e2p2[msk]
    e1p4 = e1p4[msk]
    e2p4 = e2p4[msk]

    e1s2 = e1s2[msk]
    e2s2 = e2s2[msk]
    e1s4 = e1s4[msk]
    e2s4 = e2s4[msk]

    catP2 = treecorr.Catalog(
        g1=e1p2, g2=-e2p2,
        ra=ra, dec=dec,
        ra_units="deg",
        dec_units="deg"
    )
    catQ2 = treecorr.Catalog(
        g1=e1p2 - e1s2,
        g2=-(e2p2 - e2s2),
        ra=ra,
        dec=dec,
        ra_units="deg",
        dec_units="deg",
    )

    catP4 = treecorr.Catalog(
        g1=e1p4, g2=-e2p4,
        ra=ra, dec=dec,
        ra_units="deg",
        dec_units="deg"
    )
    catQ4 = treecorr.Catalog(
        g1=e1p4 - e1s4,
        g2=-(e2p4 - e2s4),
        ra=ra,
        dec=dec,
        ra_units="deg",
        dec_units="deg",
    )
    return {
        "P2": catP2,
        "P4": catP4,
        "Q2": catQ2,
        "Q4": catQ4,
    }

def get_shape(fname):
    catalog = np.array(fitsio.read(fname))
    mag = 27.0 - 2.5 * np.log10(catalog["i_flux_gauss4"])
    abse2 = catalog["e1"] ** 2.0 + catalog["e2"] ** 2.0
    NSIDE = 1024
    hpfname = f"{os.environ['s23b']}/fdfc_hp_window_updated.fits"
    hmask = hp.read_map(
        hpfname,
        nest=True, dtype=bool,
    )
    pix = hp.ang2pix(
        NSIDE,
        np.deg2rad(90.0 - catalog["dec"]),
        np.deg2rad(catalog["ra"]),
        nest=True
    )
    mask = (
        (mag < 24.5) &
        (abse2 < 0.09) &
        hmask[pix]
    )
    catalog = catalog[mask]
    if len(catalog) < 2:
        return None, None
    e1 = catalog["e1"] * catalog["wsel"]
    e2 = catalog["e2"] * catalog["wsel"]

    r1 = (
        catalog["de1_dg1"] * catalog["wsel"] +
        catalog["dwsel_dg1"] * catalog["e1"]
    )
    r2 = (
        catalog["de2_dg2"] * catalog["wsel"] +
        catalog["dwsel_dg2"] * catalog["e2"]
    )
    response = (r1 + r2) / 2.0
    cate = treecorr.Catalog(
        g1=e1,
        g2=-e2,
        ra=catalog["ra"],
        dec=catalog["dec"],
        ra_units="deg",
        dec_units="deg",
    )
    catk = treecorr.Catalog(
        k=response,
        ra=catalog["ra"],
        dec=catalog["dec"],
        ra_units="deg",
        dec_units="deg",
    )
    return cate, catk

def process_tract(tract_id):
    fname = f"{os.environ['s23b_anacal_v2']}/tracts/{tract_id}.fits"
    cate, catk = get_shape(fname)
    if cate is None:
        return None
    nbins = 12
    catalogs = prepare_catalogs()
    dd =[]
    for kk in ["P2", "P4", "Q2", "Q4"]:
        cor1 = treecorr.GGCorrelation(
            nbins=nbins, min_sep=0.25, max_sep=360.0, sep_units="arcmin"
        )
        cor2 = treecorr.NKCorrelation(
            nbins=nbins, min_sep=0.25, max_sep=360.0, sep_units="arcmin"
        )
        cor1.process(catalogs[kk], cate)
        dd.append(cor1.xip * cor1.npairs)
        if kk =="Q4":
            cor2.process(catalogs[kk], catk)
            dd.append(cor2.xi * cor1.npairs)
    return np.stack(dd)


def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        full = fitsio.read(
            f"{os.environ['s23b']}/tracts_id_v2.fits"
        )
        selected = full[args.start: args.end]
    else:
        selected = None

    selected = comm.bcast(selected, root=0)
    my_entries = split_work(selected, size, rank)
    outcome = []
    for tract_id in my_entries:
        tout = process_tract(tract_id)
        if tout is not None:
            outcome.append(tout)

    gathered_results = comm.gather(outcome, root=0)
    if rank == 0:
        flat = [arr for sublist in gathered_results for arr in sublist]
        gathered = np.stack(flat)
        outfname = f"{os.environ['s23b_anacal_v2']}/tests/psfstar.fits"
        fitsio.write(outfname, gathered)
    comm.Barrier()
    return


if __name__ == "__main__":
    main()
