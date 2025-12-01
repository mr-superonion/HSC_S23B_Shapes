#!/usr/bin/env python3

import argparse
import os

import fitsio
import numpy as np
from mpi4py import MPI
import healpy as hp


def get_shape(catalog):
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
    return {
        "e1": e1,
        "e2": e2,
        "r1": r1,
        "r2": r2,
    }

def test_e_psf2_bin(catalog, shape):
    nbins = 5
    psf_mxx = catalog["i_hsmpsfmoments_shape11"]
    psf_myy = catalog["i_hsmpsfmoments_shape22"]
    psf_mxy = catalog["i_hsmpsfmoments_shape12"]

    e1_psf = (psf_mxx - psf_myy) / (psf_mxx + psf_myy)
    e2_psf = psf_mxy / (psf_mxx + psf_myy) * 2.0

    bins = np.linspace(-0.06, 0.06, nbins + 1)
    bc = 0.5 * (bins[:-1] + bins[1:])
    nom1 = np.histogram(e1_psf, weights=shape["e1"], bins=bins)[0]
    denom1 = np.histogram(e1_psf, weights=shape["r1"], bins=bins)[0]
    nom2 = np.histogram(e2_psf, weights=shape["e2"], bins=bins)[0]
    denom2 = np.histogram(e2_psf, weights=shape["r2"], bins=bins)[0]
    return np.stack([bc, nom1, denom1, nom2, denom2])


def test_e_psf4_bin(catalog, shape):
    nbins = 5
    e1_psf4 = (
        catalog["i_higherordermomentspsf_40"] -
        catalog["i_higherordermomentspsf_04"]
    )
    e2_psf4 = 2.0 * (
        catalog["i_higherordermomentspsf_31"] +
        catalog["i_higherordermomentspsf_13"]
    )

    bins = np.linspace(-0.02, 0.02, nbins + 1)
    bc = 0.5 * (bins[:-1] + bins[1:])
    nom3 = np.histogram(e1_psf4, weights=shape["e1"], bins=bins)[0]
    denom3 = np.histogram(e1_psf4, weights=shape["r1"], bins=bins)[0]
    nom4 = np.histogram(e2_psf4, weights=shape["e2"], bins=bins)[0]
    denom4 = np.histogram(e2_psf4, weights=shape["r2"], bins=bins)[0]
    return np.stack([bc, nom3, denom3, nom4, denom4])


def test_size_bin(catalog, shape):
    nbins = 5
    psf_mxx = catalog["i_hsmpsfmoments_shape11"]
    psf_myy = catalog["i_hsmpsfmoments_shape22"]
    psf_mxy = catalog["i_hsmpsfmoments_shape12"]
    size_val = 2.355 * (psf_mxx * psf_myy - psf_mxy**2)**0.25

    bins = np.linspace(0.5, 0.7, nbins + 1)
    bc = 0.5 * (bins[:-1] + bins[1:])
    nom5 = np.histogram(size_val, weights=shape["e1"], bins=bins)[0]
    denom5 = np.histogram(size_val, weights=shape["r1"], bins=bins)[0]
    nom6 = np.histogram(size_val, weights=shape["e2"], bins=bins)[0]
    denom6 = np.histogram(size_val, weights=shape["r2"], bins=bins)[0]
    return np.stack([bc, nom5, denom5, nom6, denom6])


def test_variance_bin(catalog, shape, band="i"):
    nbins = 5
    var_val = catalog[f"{band}_variance_value"]
    bins = np.linspace(0.002, 0.007, nbins + 1)
    bc = 0.5 * (bins[:-1] + bins[1:])
    nom7 = np.histogram(var_val, weights=shape["e1"], bins=bins)[0]
    denom7 = np.histogram(var_val, weights=shape["r1"], bins=bins)[0]
    nom8 = np.histogram(var_val, weights=shape["e2"], bins=bins)[0]
    denom8 = np.histogram(var_val, weights=shape["r2"], bins=bins)[0]
    return np.stack([bc, nom7, denom7, nom8, denom8])

def test_color_bin(data, shape, ctype="gr"):
    nbins = 5
    if ctype=="gr":
        mag_g = 27 - 2.5*np.log10(data["g_flux_gauss2"])
        mag_r = 27 - 2.5*np.log10(data["r_flux_gauss2"])
        c = mag_g - mag_r
        cmin = -0.5
        cmax = 2.0
    elif ctype=="ri":
        mag_r = 27 - 2.5*np.log10(data["r_flux_gauss2"])
        mag_i = 27 - 2.5*np.log10(data["i_flux_gauss2"])
        c = mag_r - mag_i
        cmin = -0.2
        cmax = 1.5
    elif ctype=="iz":
        mag_i = 27 - 2.5*np.log10(data["i_flux_gauss2"])
        mag_z = 27 - 2.5*np.log10(data["z_flux_gauss2"])
        c = mag_i - mag_z
        cmin = -0.4
        cmax = 1.2
    elif ctype=="zy":
        mag_z = 27 - 2.5*np.log10(data["z_flux_gauss2"])
        mag_y = 27 - 2.5*np.log10(data["y_flux_gauss2"])
        c = mag_z - mag_y
        cmin = -1.0
        cmax = 1.0
    else:
        raise ValueError("ctype not support")
    bins = np.linspace(cmin, cmax, nbins + 1)
    bc = 0.5 * (bins[:-1] + bins[1:])
    nom7 = np.histogram(c, weights=shape["e1"], bins=bins)[0]
    denom7 = np.histogram(c, weights=shape["r1"], bins=bins)[0]
    nom8 = np.histogram(c, weights=shape["e2"], bins=bins)[0]
    denom8 = np.histogram(c, weights=shape["r2"], bins=bins)[0]
    return np.stack([bc, nom7, denom7, nom8, denom8])


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


def process_tract(tract_id):
    fname1 = f"{os.environ['s23b_anacal_v2']}/tracts/{tract_id}.fits"
    data = fitsio.read(fname1)
    mag = 27.0 - 2.5 * np.log10(data["i_flux_gauss2"])
    abse2 = data["e1"] ** 2.0 + data["e2"] ** 2.0
    NSIDE = 1024
    hpfname = f"{os.environ['s23b']}/fdfc_hp_window_updated.fits"
    hmask = hp.read_map(
        hpfname,
        nest=True, dtype=bool,
    )
    pix = hp.ang2pix(
        NSIDE,
        np.deg2rad(90.0 - data["dec"]),
        np.deg2rad(data["ra"]),
        nest=True
    )
    mask = (
        (mag < 24.5)
        & (abse2 < 0.09)
        # & hmask[pix]
    )
    data = data[mask]
    fname2 = f"{os.environ['s23b_anacal_v2']}/tracts_color/{tract_id}.fits"
    color = fitsio.read(fname2)
    color = color[mask]

    shape = get_shape(data)
    psf2 = test_e_psf2_bin(color, shape)
    psf4 = test_e_psf4_bin(color, shape)
    var = test_variance_bin(color, shape)
    size = test_size_bin(color, shape)
    gr = test_color_bin(data, shape, "gr")
    ri = test_color_bin(data, shape, "ri")
    iz = test_color_bin(data, shape, "iz")
    zy = test_color_bin(data, shape, "zy")
    return {
        "psf2": psf2,
        "psf4": psf4,
        "var": var,
        "size": size,
        "gr": gr,
        "ri": ri,
        "iz": iz,
        "zy": zy,
    }


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
    outcome = {
        "psf2": [],
        "psf4": [],
        "var": [],
        "size": [],
        "gr": [],
        "ri": [],
        "iz": [],
        "zy": [],
    }
    test_names = ["psf2", "psf4", "var", "size", "gr", "ri", "iz", "zy"]

    for tract_id in my_entries:
        out = process_tract(tract_id)
        for tt in test_names:
            outcome[tt].append(out[tt])

    gathered = {}
    for tt in test_names:
        per_rank = comm.gather(outcome[tt], root=0)
        if rank == 0:
            flat = [arr for sublist in per_rank for arr in sublist]
            gathered[tt] = np.stack(flat)
    if rank == 0:
        for tt in test_names:
            outfname = f"{os.environ['s23b_anacal_v2']}/tests/{tt}_stack.fits"
            fitsio.write(outfname, gathered[tt])
    comm.Barrier()
    return


if __name__ == "__main__":
    main()
