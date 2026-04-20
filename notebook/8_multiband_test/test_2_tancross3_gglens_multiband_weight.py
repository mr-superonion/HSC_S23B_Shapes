#!/usr/bin/env python3

import argparse
import os

import fitsio

from selection import get_cut, MAG_CUTS_MULTIBAND
import numpy as np
import treecorr
from mpi4py import MPI
import healpy as hp


def get_shape(iband, mb, wopt, dwopt_dm0):
    w_total = iband["wsel"] * wopt
    dwopt_dg1 = dwopt_dm0 * mb["dm0_dg1"]
    dwopt_dg2 = dwopt_dm0 * mb["dm0_dg2"]
    e1 = mb["e1"] * w_total
    e2 = mb["e2"] * w_total
    r1 = (
        mb["de1_dg1"] * w_total +
        iband["dwsel_dg1"] * wopt * mb["e1"] +
        iband["wsel"] * dwopt_dg1 * mb["e1"]
    )
    r2 = (
        mb["de2_dg2"] * w_total +
        iband["dwsel_dg2"] * wopt * mb["e2"] +
        iband["wsel"] * dwopt_dg2 * mb["e2"]
    )
    cate = treecorr.Catalog(
        g1=e1,
        g2=-e2,
        ra=iband["ra"],
        dec=iband["dec"],
        ra_units="deg",
        dec_units="deg",
    )
    catk = treecorr.Catalog(
        k=(r1 + r2) / 2.0,
        ra=iband["ra"],
        dec=iband["dec"],
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
    parser.add_argument(
        "--emax", type=float, default=0.3, help="max |e| for cut."
    )
    parser.add_argument(
        "--imag", type=float, default=MAG_CUTS_MULTIBAND["i"],
        help="i-band magnitude cut",
    )
    parser.add_argument(
        "--A", type=float, default=4.11, help="wopt slope",
    )
    parser.add_argument(
        "--B", type=float, default=4.0, help="wopt offset",
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

def process_tract(tract_id, emax_sq, mag_cuts, A_wopt, B_wopt, lowz, cmass1, cmass2):
    fname2 = f"{os.environ['s23b_anacal_v2']}/tracts_multiband/{tract_id}.fits"
    mb = fitsio.read(fname2)
    iband = mb
    zbin = fitsio.read(
        f"{os.environ['s23b_anacal_v2']}/tracts_redshift/{tract_id}.fits",
        columns=["object_id", "zbest_0"],
    )
    ext = fitsio.read(
        f"{os.environ['s23b_anacal_v2']}/tracts_extinction/{tract_id}.fits",
    )
    # Optimal weight
    m00 = mb["m0"].astype(np.float64)
    wopt = ( A_wopt * np.log(np.clip(m00, 1e-30, None)) + B_wopt) / 13.3
    dwopt_dm0 = A_wopt / m00 / 13.3

    mask = get_cut(mb, comp=1, dg_eff=0.0, ext=ext, zbin=zbin, zkey="zbest", emax=np.sqrt(emax_sq), mag_cuts=mag_cuts)
    iband = iband[mask]
    mb = mb[mask]
    if len(iband) < 2:
        return {
            "lowz": None,
            "cmass1": None,
            "cmass2": None,
        }
    cate, catk = get_shape(iband, mb, wopt[mask], dwopt_dm0[mask])
    del iband, mb

    return {
        "lowz": compute_corr(cate, catk, lowz),
        "cmass1": compute_corr(cate, catk, cmass1),
        "cmass2": compute_corr(cate, catk, cmass2),
    }


def main():
    args = parse_args()
    mag_cuts_local = dict(MAG_CUTS_MULTIBAND)
    mag_cuts_local["i"] = args.imag

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        rootdir = os.environ["s23b"]
        full = fitsio.read(
            f"{rootdir}/tracts_id_v2.fits"
        )
        selected = full[args.start: args.end]
        bdir = os.path.join(rootdir, "boss_dr11")
        lowz = fitsio.read(f"{bdir}/lowz.fits", columns=["ra", "dec"])
        cmass1 = fitsio.read(f"{bdir}/cmass1.fits", columns=["ra", "dec"])
        cmass2 = fitsio.read(f"{bdir}/cmass2.fits", columns=["ra", "dec"])
    else:
        selected = None
        lowz = None
        cmass1 = None
        cmass2 = None

    selected = comm.bcast(selected, root=0)
    lowz = comm.bcast(lowz, root=0)
    cmass1 = comm.bcast(cmass1, root=0)
    cmass2 = comm.bcast(cmass2, root=0)

    my_entries = split_work(selected, size, rank)
    outcome = {
        "lowz": [],
        "cmass1": [],
        "cmass2": [],
    }
    test_names = list(outcome.keys())

    for tract_id in my_entries:
        out = process_tract(tract_id, args.emax ** 2, mag_cuts_local, args.A, args.B, lowz, cmass1, cmass2)
        if out["lowz"] is not None:
            for tt in test_names:
                outcome[tt].append(out[tt])

    gathered = {}
    for tt in test_names:
        per_rank = comm.gather(outcome[tt], root=0)
        if rank == 0:
            flat = [arr for sublist in per_rank for arr in sublist]
            gathered[tt] = np.stack(flat)
    if rank == 0:
        outdir = f"{os.environ['s23b_anacal_v2']}/tests_multiband_weight/imag{args.imag:.1f}_emax{args.emax:.2f}"
        os.makedirs(outdir, exist_ok=True)
        for tt in test_names:
            outfname = f"{outdir}/NG_{tt}.fits"
            fitsio.write(outfname, gathered[tt])
    comm.Barrier()
    return


if __name__ == "__main__":
    main()
