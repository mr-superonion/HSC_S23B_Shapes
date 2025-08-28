#!/usr/bin/env python3

import argparse
import os
import treecorr
import numpy as np
from tqdm import tqdm

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
    cate = treecorr.Catalog(
        g1=e1,
        g2=-e2,
        ra=catalog["ra"],
        dec=catalog["dec"],
        ra_units="deg",
        dec_units="deg",
    )
    catk = treecorr.Catalog(
        k=(r1 + r2) / 2.0,
        ra=catalog["ra"],
        dec=catalog["dec"],
        ra_units="deg",
        dec_units="deg",
    )
    return cate, catk


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

def compute_corr(cate, catk, points):
    cor1 = treecorr.NGCorrelation(
        nbins=20, min_sep=0.25, max_sep=360.0, sep_units="arcmin"
    )
    cor2 = treecorr.NKCorrelation(
        nbins=20, min_sep=0.25, max_sep=360.0, sep_units="arcmin"
    )
    cat0 = treecorr.Catalog(
        ra=points["ra"],
        dec=points["dec"],
        ra_units="deg",
        dec_units="deg",
    )
    cor1.process(cat0, cate)
    cor2.process(cat0, catk)
    return np.array([
        cor1.rnom,
        cor1.xi * cor1.npairs,
        cor1.xi_im * cor1.npairs,
        cor2.xi * cor1.npairs,
    ])

def process_tract(tract_id, bright, median, faint):
    fname1 = f"{os.environ['s23b_anacal3']}/tracts/{tract_id}.fits"
    data = fitsio.read(fname1)
    mag = 27.0 - 2.5 * np.log10(data["flux"])
    abse2 = data["e1"] ** 2.0 + data["e2"] ** 2.0
    mask = (
        (mag < 24.5) &
        (abse2 < 0.09)
        # (np.abs(data["dwsel_dg1"]) < 3000) &
        # (np.abs(data["dwsel_dg2"]) < 3000)
    )
    data = data[mask]
    cate, catk = get_shape(data)
    del data

    return {
        "gaia_bright": compute_corr(cate, catk, bright),
        "gaia_median": compute_corr(cate, catk, median),
        "gaia_faint": compute_corr(cate, catk, faint),
    }

def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        rootdir = os.environ["s23b"]
        full = fitsio.read(
            f"{rootdir}/tracts_id.fits"
        )
        selected = full[args.start: args.end]
        bdir = os.path.join(rootdir, "gaia")
        gaia = fitsio.read(f"{bdir}/stars.fits", columns=["ra", "dec", "g_mag"])
        mag = gaia["g_mag"]
        bright = gaia[(mag<11)]
        median = gaia[(mag<14) & (mag>=11)]
        faint = gaia[(mag<17) & (mag>=14)]
    else:
        selected = None
        bright = None
        median = None
        faint = None

    selected = comm.bcast(selected, root=0)
    bright = comm.bcast(bright, root=0)
    median = comm.bcast(median, root=0)
    faint = comm.bcast(faint, root=0)

    my_entries = split_work(selected, size, rank)
    outcome = {
        "gaia_bright": [],
        "gaia_median": [],
        "gaia_faint": [],
    }
    test_names = list(outcome.keys())

    # Initialize tqdm progress bar for this rank
    pbar = tqdm(total=len(my_entries), desc=f"Rank {rank}", position=rank)
    for tract_id in my_entries:
        out = process_tract(tract_id, bright, median, faint)
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
            outfname = f"{os.environ['s23b_anacal3']}/tests/NG_{tt}.fits"
            fitsio.write(outfname, gathered[tt])
    comm.Barrier()
    return


if __name__ == "__main__":
    main()
