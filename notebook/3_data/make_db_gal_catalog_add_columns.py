#!/usr/bin/env python3

import os

import astropy.io.fits as pyfits
import fitsio
import numpy as np
from mpi4py import MPI
from numpy.lib import recfunctions as rfn


# Divide data among MPI ranks
def split_work(data, size, rank):
    return data[rank::size]

def process_tract(tract_id):
    fname = os.path.join(
        os.environ["s23b"], "db_color", f"{tract_id}.fits"
    )
    cnames = [
        "g_variance_value", "r_variance_value", "z_variance_value",
        "y_variance_value", "i_sdssshape_shape11", "i_sdssshape_shape12",
        "i_sdssshape_shape22",
    ]
    data = np.array(fitsio.read(fname))
    fname2 = os.path.join(
        os.environ["s23b"], "db_color2", f"{tract_id}.fits"
    )
    data2 = np.array(fitsio.read(fname2, columns=cnames))
    c = rfn.merge_arrays([data, data2], flatten=True, asrecarray=False)
    outname = os.path.join(
        os.environ["s23b"], "db_color3", f"{tract_id}.fits"
    )
    pyfits.writeto(outname, c, overwrite=True)
    return


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rootdir = os.environ["s23b"]
    full = fitsio.read(
        f"{rootdir}/tracts_fdfc_v1_final.fits"
    )
    if rank == 0:
        tract_all = np.unique(full["tract"])
    else:
        tract_all = None

    tract_all = comm.bcast(tract_all, root=0)
    tract_list = split_work(tract_all, size, rank)
    for tract_id in tract_list:
        process_tract(tract_id)
    return


if __name__ == "__main__":
    main()
