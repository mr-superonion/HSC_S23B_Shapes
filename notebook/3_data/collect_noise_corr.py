#!/usr/bin/env python3

import argparse
import gc
import glob
import os
from tqdm import tqdm

import fitsio
import numpy as np
import lsst.afw.image as afwImage
from mpi4py import MPI


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
        "--field", type=str, default="all", required=False, help="field name"
    )
    return parser.parse_args()


# Divide data among MPI ranks
def split_work(data, size, rank):
    return data[rank::size]

def process_patch(entry):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9
    out_dir = f"{os.environ['s23b_noisecorr']}/{tract_id}/{patch_id}"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_fname = os.path.join(out_dir, "noise_correlation.fits")
    if os.path.isfile(out_fname):
        return None
    calexp_dir = f"{os.environ['s23b_calexp']}/{tract_id}/{patch_id}/i"
    exp_fname = glob.glob(os.path.join(calexp_dir, "*.fits"))[0]
    exposure = afwImage.ExposureF.readFits(exp_fname)
    mask_dir = f"{os.environ['s23b_mask']}/{tract_id}/{patch_id}"
    mask_fname = os.path.join(mask_dir, "mask2.fits")
    bmask = fitsio.read(mask_fname)
    nim_dir = f"{os.environ['s23b_nimg']}/{tract_id}/{patch_id}/i"
    nim_fname = glob.glob(os.path.join(nim_dir, "*.fits"))[0]
    bmask = (bmask | (fitsio.read(nim_fname) <=2).astype(np.int16))

    noise_array = np.asarray(
        exposure.image.array,
        dtype=np.float32,
    )[500:3500, 500:3500]

    window_array = np.asarray(
        (bmask == 0) & (exposure.mask.array == 0) &
        (exposure.image.array ** 2.0 < exposure.variance.array * 10),
        dtype=np.float32,
    )[500:3500, 500:3500]
    del exposure, bmask
    noise_array[~window_array.astype(bool)] = 0.0
    pad_width = ((10, 10), (10, 10))  # ((top, bottom), (left, right))
    noise_array = np.pad(
        noise_array,
        pad_width=pad_width,
        mode="constant",
        constant_values=0.0,
    )
    window_array = np.pad(
        window_array,
        pad_width=pad_width,
        mode="constant",
        constant_values=0.0,
    )

    npix = 49
    ny, nx = window_array.shape
    npixl = int(npix // 2)
    npixr = int(npix // 2 + 1)
    noise_corr = np.fft.fftshift(
        np.fft.ifft2(np.abs(np.fft.fft2(noise_array)) ** 2.0)
    ).real[
        ny // 2 - npixl : ny // 2 + npixr, nx // 2 - npixl : nx // 2 + npixr
    ]
    window_corr = np.fft.fftshift(
        np.fft.ifft2(np.abs(np.fft.fft2(window_array)) ** 2.0)
    ).real[
        ny // 2 - npixl : ny // 2 + npixr, nx // 2 - npixl : nx // 2 + npixr
    ]
    del noise_array, window_array
    noise_corr = noise_corr / window_corr
    fitsio.write(out_fname, noise_corr)
    return


def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        full = fitsio.read(
            "tracts_fdfc_v1_final.fits"
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
