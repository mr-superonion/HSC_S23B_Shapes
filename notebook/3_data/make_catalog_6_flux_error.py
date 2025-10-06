#!/usr/bin/env python3

import argparse
import gc
import glob
import os

import fitsio
import lsst.afw.image as afwImage
import numpy as np
import xlens
from mpi4py import MPI


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


def read_files(tract_id, patch_id, band):
    calexp_dir = f"{os.environ['s23b_calexp']}/{tract_id}/{patch_id}/{band}"
    exp_fname = glob.glob(os.path.join(calexp_dir, "*.fits"))[0]
    exposure = afwImage.ExposureF.readFits(exp_fname)
    pixel_scale = float(exposure.getWcs().getPixelScale().asArcseconds())
    lsst_bbox = exposure.getBBox()
    lsst_psf = exposure.getPsf()
    psf_array = xlens.utils.image.get_psf_array(
        lsst_psf=lsst_psf,
        lsst_bbox=lsst_bbox,
        npix=64,
        dg=250,
        lsst_mask=exposure.mask,
    )
    del exposure
    corr_fname = f"{os.environ['s23b_noisecorr']}/{tract_id}.fits"
    noise_corr = fitsio.read(corr_fname)
    return {
        "psf_array": psf_array,
        "pixel_scale": pixel_scale,
        "noise_corr": noise_corr,
    }

def gaussian_flux_variance(
    psf, sigma, sigma_arcsec,
    pixel_scale=1.0, eps=1e-6,
    noise_corr=None,
):
    ny, nx = psf.shape
    fx = np.fft.fftfreq(nx, d=1.0)
    fy = np.fft.fftfreq(ny, d=1.0)
    kx, ky = np.meshgrid(2*np.pi*fx, 2*np.pi*fy)
    k2 = kx**2 + ky**2
    psf = psf / psf.sum()
    P = np.fft.fft2(np.fft.ifftshift(psf))
    sigma_pix = np.sqrt(sigma**2.0 + sigma_arcsec**2.0) / pixel_scale
    sigma_pix2 = sigma_arcsec / pixel_scale
    T = np.exp(-0.5 * sigma_pix2**2 * k2)
    denom = P.copy()
    tiny = eps * np.abs(P[0,0])
    denom[np.abs(denom) < tiny] = tiny
    H = T / denom
    x0, y0 = nx // 2, ny // 2
    W = np.exp(-0.5 * sigma_pix**2 * k2)
    W = W * np.exp(-1j * (kx*x0 + ky*y0))
    ff = 4.0 * np.pi * sigma_pix**2
    if noise_corr is not None:
        noise_corr = np.pad(noise_corr, (8, 7))
        noise_pow = np.fft.fft2(np.fft.ifftshift(noise_corr)).real
    else:
        noise_pow = 1.0
    var_flux = np.sum(np.abs(W * H)**2 * noise_pow) * (ff**2) / (nx * ny)
    return var_flux


def process_patch(entry):
    tract_id = entry["tract"]
    patch_db = entry["patch"]
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    patch_id = patch_x + patch_y * 9
    out = [int(entry["index"])]
    for band in ["g", "r", "i", "z", "y"]:
        res = read_files(tract_id, patch_id, band)
        flux_variance = gaussian_flux_variance(
            psf=res["psf_array"],
            sigma=0.3, sigma_arcsec=0.4,
            pixel_scale=res["pixel_scale"],
            noise_corr=res["noise_corr"],
        )
        out.append(flux_variance)
        del res
        gc.collect()
    out_dir = os.path.join(
        os.environ['s23b'],
        f"deepCoadd_flux_variance/{tract_id}/{patch_id}"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_fname = os.path.join(out_dir, "out.fits")
    fitsio.write(out_fname, np.array(out))
    return


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
        selected = full[args.start: args.end]
        if args.field != "all":
            sel = (selected["field"] == args.field)
            selected = selected[sel]
    else:
        selected = None

    selected = comm.bcast(selected, root=0)
    my_entries = split_work(selected, size, rank)

    for entry in my_entries:
        process_patch(entry)
    return


if __name__ == "__main__":
    main()
