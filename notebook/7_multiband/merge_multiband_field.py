#!/usr/bin/env python3
"""Merge per-tract multiband files into per-field multiband files.

For each field in FIELDS, reads the list of tracts from
fields_color/{field}.fits, concatenates the corresponding
tracts_multiband/{tract}.fits files, sorts by object_id, computes a
per-object ``response_model`` column using a pickled
``predict_response`` model (SVR for m0 < M0_SVR_MAX, the object's own
measured response otherwise), and writes fields_multiband/{field}.fits.

Parallelized over fields via MPI (one field per rank).
"""

import argparse
import gc
import os
import pickle

import fitsio
import numpy as np
import numpy.lib.recfunctions as rfn
from mpi4py import MPI

BASE = "/gpfs02/work/xiangchong.li/work/hsc_data/s23b/deepCoadd_anacal_v2"
TRACT_DIR = f"{BASE}/tracts_multiband"
COLOR_DIR = f"{BASE}/fields_color"
OUT_DIR = f"{BASE}/fields_multiband"
MODEL_PATH = "./response_model.pkl"

FIELDS = ("spring1", "spring2", "spring3", "autumn1", "autumn2", "hectomap")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-tract multiband files into per-field files and "
            "attach a response_model column."
        )
    )
    parser.add_argument(
        "--field", type=str, default="all",
        help="Single field to process (default: all).",
    )
    return parser.parse_args()


def load_predict_response(pkl_path):
    """Load the pickled SVR + scaler and return a callable.

    Returns a function with signature ``predict_response(m0, trace, response)``
    that mirrors the one defined in show_m2_m0_grid.ipynb.
    """
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    svr = payload["svr"]
    scaler = payload["scaler"]
    m0_svr_max = payload["m0_svr_max"]
    trace_min = payload["trace_min"]
    trace_max = payload["trace_max"]

    def predict_response(m0_val, trace_val, response):
        m0_val = np.asarray(m0_val)
        trace_val = np.asarray(trace_val)
        response = np.asarray(response)
        out = np.array(response, dtype=np.float64, copy=True)

        svr_mask = (
            (m0_val > 1e-5)
            & (m0_val < m0_svr_max)
            & (trace_val > trace_min)
            & (trace_val < trace_max)
        )
        if np.any(svr_mask):
            X_s = scaler.transform(
                np.column_stack([
                    np.log10(m0_val[svr_mask]),
                    np.log10(trace_val[svr_mask]),
                ])
            )
            out[svr_mask] = svr.predict(X_s)
        return out

    return predict_response


def process_field(field, predict_response):
    out_path = os.path.join(OUT_DIR, f"{field}.fits")
    if os.path.isfile(out_path):
        print(f"[{field}] already exists, skipping")
        return

    color_path = os.path.join(COLOR_DIR, f"{field}.fits")
    if not os.path.isfile(color_path):
        print(f"[{field}] missing color file: {color_path}")
        return
    color = fitsio.read(color_path, columns=["tract"])
    tracts = sorted(set(int(t) for t in color["tract"]))
    print(f"[{field}] {len(tracts)} tracts")
    del color

    arrays = []
    missing = 0
    for tract_id in tracts:
        tpath = os.path.join(TRACT_DIR, f"{tract_id}.fits")
        if not os.path.isfile(tpath):
            missing += 1
            continue
        arrays.append(fitsio.read(tpath))

    if missing:
        print(f"[{field}] warning: {missing} tracts missing")
    if not arrays:
        print(f"[{field}] no data, skipping")
        return

    merged = rfn.stack_arrays(arrays, usemask=False)
    del arrays
    order = np.argsort(merged["object_id"])
    merged = merged[order]

    # Compute response_model via predict_response(m0, trace, response)
    m0 = merged["m0"].astype(np.float64)
    m2 = merged["m2"].astype(np.float64)
    trace = m2 / m0
    response = 0.5 * (
        merged["de1_dg1"].astype(np.float64)
        + merged["de2_dg2"].astype(np.float64)
    )
    response_model = predict_response(m0, trace, response).astype(np.float32)

    # Append the response_model column
    merged = rfn.append_fields(
        merged, "response_denoised", response_model,
        dtypes=np.float64, usemask=False,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    fitsio.write(out_path, merged, clobber=True)
    print(f"[{field}] written: {out_path} ({len(merged)} objects)")

    del merged, response_model, response, m0, m2, trace
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

    if rank == 0:
        print(f"Loading predict_response from {MODEL_PATH}")
    predict_response = load_predict_response(MODEL_PATH)

    my_fields = fields[rank::size]
    for field in my_fields:
        if rank == 0 or len(fields) > 1:
            print(f"[rank {rank}] processing {field}")
        process_field(field, predict_response)
        gc.collect()

    comm.Barrier()
    if rank == 0:
        print("All done.")


if __name__ == "__main__":
    main()
