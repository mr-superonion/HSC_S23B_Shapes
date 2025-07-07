#!/usr/bin/env python3

import argparse
import dask.dataframe as dd
import gc
import os
import numpy as np
import pandas as pd
import astropy.table as astTable
from tqdm import tqdm

import fitsio
from mpi4py import MPI

nbins = 5


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
        "--field", type=str, default="all", required=False, help="field name"
    )
    return parser.parse_args()


# Divide data among MPI ranks
def split_work(data, size, rank):
    return data[rank::size]


def compute_e_psf_2(catalog, e1, e2, r1, r2):
    e1_psf = catalog["e1_psf2"]
    e2_psf = catalog["e2_psf2"]

    bins = np.linspace(-0.06, 0.06, nbins + 1)
    bc = 0.5 * (bins[:-1] + bins[1:])
    nom1 = np.histogram(e1_psf, weights=e1, bins=bins)[0]
    denom1 = np.histogram(e1_psf, weights=r1, bins=bins)[0]
    nom2 = np.histogram(e2_psf, weights=e2, bins=bins)[0]
    denom2 = np.histogram(e2_psf, weights=r2, bins=bins)[0]
    return bc, nom1, denom1, nom2, denom2


def compute_e_psf_4(catalog, e1, e2, r1, r2):
    e1_psf = catalog["e1_psf4"]
    e2_psf = catalog["e2_psf4"]
    bins = np.linspace(-0.02, 0.02, nbins + 1)
    bc = 0.5 * (bins[:-1] + bins[1:])
    nom3 = np.histogram(e1_psf, weights=e1, bins=bins)[0]
    denom3 = np.histogram(e1_psf, weights=r1, bins=bins)[0]
    nom4 = np.histogram(e2_psf, weights=e2, bins=bins)[0]
    denom4 = np.histogram(e2_psf, weights=r2, bins=bins)[0]
    return bc, nom3, denom3, nom4, denom4


def compute_size(catalog, e1, e2, r1, r2):
    size_val = catalog["fwhm_psf"]
    bins = np.linspace(0.5, 0.7, nbins + 1)
    bc = 0.5 * (bins[:-1] + bins[1:])
    nom5 = np.histogram(size_val, weights=e1, bins=bins)[0]
    denom5 = np.histogram(size_val, weights=r1, bins=bins)[0]
    nom6 = np.histogram(size_val, weights=e2, bins=bins)[0]
    denom6 = np.histogram(size_val, weights=r2, bins=bins)[0]
    return bc, nom5, denom5, nom6, denom6


def compute_variance(catalog, e1, e2, r1, r2):
    var_val = catalog["noise_variance"]
    bins = np.linspace(0.002, 0.007, nbins + 1)
    var = 0.5 * (bins[:-1] + bins[1:])
    nom7 = np.histogram(var_val, weights=e1, bins=bins)[0]
    denom7 = np.histogram(var_val, weights=r1, bins=bins)[0]
    nom8 = np.histogram(var_val, weights=e2, bins=bins)[0]
    denom8 = np.histogram(var_val, weights=r2, bins=bins)[0]
    return var, nom7, denom7, nom8, denom8


def process_patch(entry):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9

    out_dir1 = f"{os.environ['s23b_anacal']}/{tract_id}/{patch_id}"
    out_fname1 = os.path.join(out_dir1, "leakage2.fits")

    match_dir = f"{os.environ['s23b_anacal']}/{tract_id}/{patch_id}"
    fname = os.path.join(match_dir, "match.fits")
    catalog = np.array(fitsio.read(fname))
    mag = 27.0 - 2.5 * np.log10(catalog["flux"])
    abse = np.sqrt(catalog["e1"] ** 2.0 + catalog["e2"] ** 2.0)
    mask = (
        (mag < 25.0) &
        (abse < 0.3)
    )
    catalog = catalog[mask]

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
    if not os.path.isfile(out_fname1):
        e_psf_2, nom1, denom1, nom2, denom2 = compute_e_psf_2(
            catalog, e1, e2, r1, r2)
        e_psf_4, nom3, denom3, nom4, denom4 = compute_e_psf_4(
            catalog, e1, e2, r1, r2)
        size, nom5, denom5, nom6, denom6 = compute_size(
            catalog, e1, e2, r1, r2)
        var, nom7, denom7, nom8, denom8 = compute_variance(
            catalog, e1, e2, r1, r2)

        out = astTable.Table(
            [
                e_psf_2, nom1, denom1, nom2, denom2,
                e_psf_4, nom3, denom3, nom4, denom4,
                size, nom5, denom5, nom6, denom6,
                var, nom7, denom7, nom8, denom8,
            ],
            names=[
                "e_psf_2", "e1_2", "r1_2", "e2_2", "r2_2",
                "e_psf_4", "e1_4", "r1_4", "e2_4", "r2_4",
                "size", "e1_s", "r1_s", "e2_s", "r2_s",
                "var", "e1_v", "r1_v", "e2_v", "r2_v",
            ]
        )
        out.write(out_fname1)

    out_dir2 = f"{os.environ['s23b_anacal']}/summary"
    sum_e1 = np.sum(e1)
    sum_e2 = np.sum(e2)
    sum_r1 = np.sum(r1)
    sum_r2 = np.sum(r2)
    sum_w = np.sum(catalog["wsel"])
    summary_df = pd.DataFrame([{
        "tract": tract_id,
        "patch": patch_id,
        "sum_e1": sum_e1,
        "sum_e2": sum_e2,
        "sum_r1": sum_r1,
        "sum_r2": sum_r2,
        "sum_w": sum_w,
    }])
    summary_ddf = dd.from_pandas(summary_df, npartitions=1)
    summary_ddf.to_parquet(
        out_dir2,
        partition_on=["tract", "patch"],
        compression="zstd",
        write_index=False
    )
    return


def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        full = fitsio.read(
            "./tracts_fdfc_v1_trim6.fits"
        )
        selected = full[args.start: args.end]
        if args.field != "all":
            sel = (selected["field"] == args.field)
            selected = selected[sel]
    else:
        selected = None

    selected = comm.bcast(selected, root=0)
    my_entries = split_work(selected, size, rank)

    pbar = tqdm(total=len(my_entries), desc=f"Rank {rank}", position=rank)
    for entry in my_entries:
        process_patch(entry)
        gc.collect()
        pbar.update(1)
    pbar.close()
    return


if __name__ == "__main__":
    main()
