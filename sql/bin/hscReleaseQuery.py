#!/usr/bin/env python
# Based on
# https://hsc-gitlab.mtk.nao.ac.jp/ssp-software/data-access-tools/-/blob/master/dr3/catalogQuery/hscSspQuery3.py
import os
import json
import time
import getpass
import argparse
import urllib.request
import urllib.error
import urllib.parse
import astropy.io.ascii as ascii
from astropy.table import Table


version = 20190924.1
vv = "-gals2"
release_year = "s23b"
release_version = "dr4"
prefix = "database/%s%s/tracts" % (release_year, vv)
api_url = "https://hscdata.mtk.nao.ac.jp/datasearch/api/catalog_jobs/"


if not os.path.exists(prefix):
    os.makedirs(prefix)


class QueryError(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "sql-file",
        type=argparse.FileType("r"),
        help="SQL file",
    )
    parser.add_argument(
        "--user", "-u", required=True, help="specify your STARS account"
    )
    parser.add_argument(
        "--tracts", "-t", required=True, help="specify tract list"
    )
    hsc_obj = HscQuery(parser=parser)
    hsc_obj.run()
    return


class HscQuery():
    def __init__(self, parser):
        args = parser.parse_args()
        self.sql = args.__dict__["sql-file"].read()
        self.credential = {
            "account_name": args.user,
            "password": get_password(),
        }
        self.tractname = args.tracts

    def run(self):
        tracts = ascii.read(self.tractname)["tract"]
        for tt in tracts:
            outfname = "%s.fits" % str(tt)
            outfname = os.path.join(prefix, outfname)
            if not os.path.isfile(outfname):
                print("Tract: %s" % tt)
                self.download_tract(tt)

    def download_tract(self, tname):
        job = None
        sql_use = self.sql.replace("{$tract}", str(tname))
        outfname = "%s.fits" % str(tname)
        outfname = os.path.join(prefix, outfname)
        if os.path.exists(outfname):
            print("already have output")
            return
        print("querying data")
        job = submit_job(self.credential, sql_use, "fits")
        print("blocking data")
        block_until_finish(self.credential, job["id"])
        print("downloading data")
        outcome = open(outfname, "w")
        buffer = outcome.buffer
        download(self.credential, job["id"], buffer)
        print("closing output file")
        buffer.close()
        outcome.close()
        del buffer
        del outcome
        delete_job(self.credential, job["id"])
        print("removing '_isnull'")
        table = Table.read(outfname, format='fits')

        columns_to_remove = [
            col for col in table.colnames if col.endswith('_isnull')
        ]
        table.remove_columns(columns_to_remove)
        table.write(outfname, format='fits', overwrite=True)
        del table
        return


def http_json_post(url, data):
    data["clientVersion"] = version
    post_data = json.dumps(data)
    return http_post(url, post_data, {"Content-type": "application/json"})


def http_post(url, post_data, headers):
    default_headers = {
        "User-Agent": "Mozilla/5.0",  # Pretend to be a browser
        "Accept": "application/json",  # Explicitly accept JSON
        "Content-Type": "application/json",  # Needed for HSC API
    }
    # Merge headers: default values will be overridden by input if needed
    default_headers.update(headers)

    req = urllib.request.Request(url, post_data.encode("utf-8"), headers=default_headers)
    try:
        return urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        print(f"HTTPError {e.code}: {e.reason}")
        print("Headers sent:", req.header_items())
        print("Payload sent:", post_data)
        raise


def submit_job(credential, sql, out_format):
    url = api_url + "submit"
    catalog_job = {
        "sql": sql,
        "out_format": out_format,
        "include_metainfo_to_body": True,
        "release_version": release_version,
    }
    post_data = {
        "credential": credential,
        "catalog_job": catalog_job,
        "nomail": True,
        "skip_syntax_check": False,
    }
    res = http_json_post(url, post_data)
    job = json.load(res)
    return job


def job_status(credential, job_id):
    url = api_url + "status"
    post_data = {"credential": credential, "id": job_id}
    res = http_json_post(url, post_data)
    job = json.load(res)
    return job


def block_until_finish(credential, job_id):
    max_interval = 0.5 * 60  # sec.
    interval = 1
    while True:
        time.sleep(interval)
        job = job_status(credential, job_id)
        if job["status"] == "error":
            raise QueryError("query error: " + job["error"])
        if job["status"] == "done":
            break
        interval *= 2
        if interval > max_interval:
            interval = max_interval
    print("blocking over")
    return


def download(credential, job_id, out):
    url = api_url + 'download'
    post_data = {'credential': credential, 'id': job_id}
    res = http_json_post(url, post_data)
    buff_size = 64 * 1 << 10  # 64k
    while True:
        buf = res.read(buff_size)
        out.write(buf)
        if len(buf) < buff_size:
            break


def delete_job(credential, job_id):
    url = api_url + "delete"
    post_data = {"credential": credential, "id": job_id}
    http_json_post(url, post_data)
    return


def get_password():
    password_from_envvar = os.environ.get("HSC_SSP_CAS_PASSWORD", "")
    if password_from_envvar != "":
        return password_from_envvar
    else:
        return getpass.getpass("password? ")


if __name__ == "__main__":
    main()
