#!/usr/bin/env python3

import argparse
import os
import treecorr
import numpy as np
from tqdm import tqdm

import fitsio
from mpi4py import MPI
import astropy.io.ascii as pyascii


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

def compute_cluster(cate, catk, clusters):
    cor1 = treecorr.NGCorrelation(
        nbins=20, min_sep=0.25, max_sep=360.0, sep_units="arcmin"
    )
    cor2 = treecorr.NKCorrelation(
        nbins=20, min_sep=0.25, max_sep=360.0, sep_units="arcmin"
    )
    cat0 = treecorr.Catalog(
        ra=clusters["ra"],
        dec=clusters["dec"],
        ra_units="deg",
        dec_units="deg",
    )
    cor1.process(cat0, cate)
    cor2.process(cat0, catk)
    return np.array([
        cor1.meanr,
        cor1.xi,
        cor1.xi_im,
        cor2.xi,
    ])

def compute_tract(cate, catk, tracts):
    cor1 = treecorr.NGCorrelation(
        nbins=20, min_sep=0.25, max_sep=360.0, sep_units="arcmin"
    )
    cor2 = treecorr.NKCorrelation(
        nbins=20, min_sep=0.25, max_sep=360.0, sep_units="arcmin"
    )
    cat0 = treecorr.Catalog(
        ra=tracts["ra"],
        dec=tracts["dec"],
        ra_units="deg",
        dec_units="deg",
    )
    cor1.process(cat0, cate)
    cor2.process(cat0, catk)
    return np.array([
        cor1.meanr,
        cor1.xi,
        cor1.xi_im,
        cor2.xi,
    ])


def compute_visit(cate, catk, visits):
    cor1 = treecorr.NGCorrelation(
        nbins=20, min_sep=0.25, max_sep=360.0, sep_units="arcmin"
    )
    cor2 = treecorr.NKCorrelation(
        nbins=20, min_sep=0.25, max_sep=360.0, sep_units="arcmin"
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
        cor1.meanr,
        cor1.xi,
        cor1.xi_im,
        cor2.xi,
    ])

def process_tract(tract_id, clusters, tracts, visits):
    fname1 = f"{os.environ['s23b_anacal2']}/tracts/{tract_id}.fits"
    data = fitsio.read(fname1)
    mag = 27.0 - 2.5 * np.log10(data["flux"])
    abse2 = data["e1"] ** 2.0 + data["e2"] ** 2.0
    mask = (
        (mag < 25.0) &
        (abse2 < 0.09) &
        (np.abs(data["dwsel_dg1"]) < 3000) &
        (np.abs(data["dwsel_dg2"]) < 3000)
    )
    data = data[mask]
    shape = get_shape(data)
    del data
    cate = treecorr.Catalog(
        g1=shape["e1"],
        g2=-shape["e2"],
        ra=shape["ra"],
        dec=shape["dec"],
        ra_units="deg",
        dec_units="deg",
    )
    catk = treecorr.Catalog(
        k=shape["res"],
        ra=shape["ra"],
        dec=shape["dec"],
        ra_units="deg",
        dec_units="deg",
    )

    return {
        "cluster": compute_cluster(cate, catk, clusters),
        "tract": compute_tract(cate, catk, tracts),
        "visit": compute_visit(cate, catk, visits),
    }


def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        full = fitsio.read(
            "tracts.fits"
        )
        selected = full[args.start: args.end]
        rootdir = os.environ["s23b"]
        clusters = pyascii.read(f"{rootdir}/camira/camira_s23b_wide_v3.csv")
        clusters = clusters[(clusters["N_mem"] > 12) & (clusters["z_cl"] < 1.5)]
        tracts = fitsio.read(f"{rootdir}/tracts.fits", columns=["ra", "dec"])
        visits = fitsio.read(f"{rootdir}/visits.fits", columns=["ra", "dec"])
    else:
        selected = None
        clusters = None
        tracts = None
        visits = None

    selected = comm.bcast(selected, root=0)
    clusters = comm.bcast(clusters, root=0)
    tracts = comm.bcast(tracts, root=0)
    visits = comm.bcast(visits, root=0)

    my_entries = split_work(selected, size, rank)
    outcome = {
        "cluster": [],
        "visit": [],
        "tract": [],
    }
    test_names = ["cluster", "visit", "tract"]

    # Initialize tqdm progress bar for this rank
    pbar = tqdm(total=len(my_entries), desc=f"Rank {rank}", position=rank)
    for tract_id in my_entries:
        out = process_tract(tract_id, clusters, tracts, visits)
        for tt in test_names:
            outcome[tt].append(out[tt])
        pbar.update(1)
    pbar.close()

    gathered = {}
    for tt in test_names:
        per_rank = comm.gather(outcome[tt], root=0)
        if rank == 0:
            flat = [arr for sublist in per_rank for arr in sublist]
            gathered[tt] = np.stack(flat)
    if rank == 0:
        for tt in test_names:
            outfname = f"{os.environ['s23b_anacal2']}/NG_{tt}.fits"
            fitsio.write(outfname, gathered[tt])
    comm.Barrier()
    return


if __name__ == "__main__":
    main()
