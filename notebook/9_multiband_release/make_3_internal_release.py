#!/usr/bin/env python3
"""Make the internal release.

Reads ra/dec from fields_multiband/{field}.fits and shapes/response from
{field}_response.fits. Outputs two files per field under $OUTDIR:
  - .response/{field}.fits  : (object_id, response, response_denoised)
  - anacal_{field}.fits     : (object_id, ra, dec, e1, e2, R_4c, R_4s)

Parallelized over fields via MPI (one field per rank).
"""

import argparse
import gc
import os

import fitsio
import numpy as np
import numpy.lib.recfunctions as rfn
from mpi4py import MPI

ROOTDIR = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/deepCoadd_anacal_v2"
OUTDIR = "/gpfs02/work/xiangchong.li/work/hsc_data/catalog_v2.5/s23b_shape"

FIELDS = ("spring1", "spring2", "spring3", "autumn1", "autumn2", "hectomap")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make the internal multiband release."
    )
    parser.add_argument(
        "--field",
        type=str,
        default="all",
        required=False,
        help="field name (default: all)",
    )
    return parser.parse_args()


def process_field(field):
    print(f"[{field}] processing")
    os.makedirs(f"{OUTDIR}/.response", exist_ok=True)

    # Read ra, dec from multiband catalog
    mb_path = f"{ROOTDIR}/fields_multiband/{field}.fits"
    mb = fitsio.read(mb_path, columns=["object_id", "ra", "dec"])

    # Read response catalog (object_id, e1, e2, R_4c, R_4s, response, response_denoised)
    resp_path = f"{ROOTDIR}/fields_multiband/{field}_response.fits"
    dd = np.array(fitsio.read(resp_path))
    assert np.array_equal(mb["object_id"], dd["object_id"]), (
        f"{field}: object_id mismatch between multiband and response"
    )
    print(f"[{field}] {len(dd)} objects")

    # --- .response/{field}.fits ---
    out = np.empty(
        len(dd),
        dtype=[
            ("object_id", "i8"),
            ("response", "f8"),
            ("response_denoised", "f8"),
        ],
    )
    out["object_id"] = dd["object_id"]
    out["response"] = dd["response"]
    out["response_denoised"] = dd["response_denoised"]
    fitsio.write(f"{OUTDIR}/.response/{field}.fits", out, clobber=True)
    print(f"[{field}] wrote {OUTDIR}/.response/{field}.fits")

    # --- anacal_{field}.fits ---
    # Combine ra/dec with e1, e2, R_4c, R_4s (drop response/response_denoised)
    anacal = rfn.drop_fields(
        dd, ["response", "response_denoised"], usemask=False,
    )
    anacal = rfn.append_fields(
        anacal,
        names=["ra", "dec"],
        data=[mb["ra"], mb["dec"]],
        dtypes=["f8", "f8"],
        usemask=False,
    )
    fitsio.write(f"{OUTDIR}/anacal_{field}.fits", anacal, clobber=True)
    print(f"[{field}] wrote {OUTDIR}/anacal_{field}.fits")

    del mb, dd, out, anacal
    gc.collect()


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.field != "all":
        fields = [args.field]
    else:
        fields = list(FIELDS)

    my_fields = fields[rank::size]
    for field in my_fields:
        if rank == 0 or len(fields) > 1:
            print(f"[rank {rank}] processing {field}")
        process_field(field)
        gc.collect()

    comm.Barrier()
    if rank == 0:
        print("All done.")


if __name__ == "__main__":
    main()
