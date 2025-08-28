#!/usr/bin/env python3

import argparse
import os
import treecorr
import numpy as np
import healpy as hp

import fitsio
from mpi4py import MPI


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
        "res": (r1 + r2) / 2.0,
        "ra": catalog["ra"],
        "dec": catalog["dec"],
    }


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

def compute_visit(shape, visits, mask):
    cate = treecorr.Catalog(
        g1=shape["e1"][mask],
        g2=-shape["e2"][mask],
        ra=shape["ra"][mask],
        dec=shape["dec"][mask],
        ra_units="deg",
        dec_units="deg",
    )
    catk = treecorr.Catalog(
        k=shape["res"][mask],
        ra=shape["ra"][mask],
        dec=shape["dec"][mask],
        ra_units="deg",
        dec_units="deg",
    )
    cor1 = treecorr.NGCorrelation(
        nbins=1, min_sep=9.48, max_sep=13.36, sep_units="arcmin"
    )
    cor2 = treecorr.NKCorrelation(
        nbins=1, min_sep=9.48, max_sep=13.36, sep_units="arcmin"
    )
    cat0 = treecorr.Catalog(
        ra=visits["ra"],
        dec=visits["dec"],
        ra_units="deg",
        dec_units="deg",
    )
    cor1.process(cat0, cate)
    cor2.process(cat0, catk)
    return np.array([
        cor1.xi[0],
        cor1.xi_im[0],
        cor2.xi[0],
        cor2.npairs[0],
    ])


def process_tract(pid, visits):
    fname1 = f"{os.environ['s23b_anacal3']}/healpix/{pid}.fits"
    data = fitsio.read(fname1)
    mag = 27.0 - 2.5 * np.log10(data["flux"])
    abse2 = data["e1"] ** 2.0 + data["e2"] ** 2.0
    mask = (
        (mag < 24.5) &
        (abse2 < 0.09)
    )
    data = data[mask]
    shape = get_shape(data)
    ra = data['ra']
    dec = data['dec']
    theta = np.deg2rad(90.0 - dec)
    phi = np.deg2rad(ra)
    # Convert to HEALPix pixel indices (in NESTED ordering)
    pix = hp.ang2pix(1024, theta, phi, nest=True)
    pixlist = np.unique(pix)
    del data
    out = []
    for ipix in pixlist:
        mm = (pix == ipix)
        out.append(compute_visit(shape, visits, mm))
    return out


def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        rootdir = os.environ["s23b"]
        full = fitsio.read(
            f"{rootdir}/healpix_list.fits"
        )
        selected = full[args.start: args.end]
        visits = fitsio.read(f"{rootdir}/visits.fits", columns=["ra", "dec"])
    else:
        selected = None
        visits = None

    selected = comm.bcast(selected, root=0)
    visits = comm.bcast(visits, root=0)
    my_entries = split_work(selected, size, rank)

    # Initialize tqdm progress bar for this rank
    outcome = []
    for tract_id in my_entries:
        out = process_tract(tract_id, visits)
        for _ in out:
            outcome.append(_)

    per_rank = comm.gather(outcome, root=0)
    if rank == 0:
        flat = [arr for sublist in per_rank for arr in sublist]
        gathered = np.stack(flat)
        outfname = f"{os.environ['s23b_anacal3']}/NG_visit_recheck.fits"
        fitsio.write(outfname, gathered)
    comm.Barrier()
    return


if __name__ == "__main__":
    main()
