#!/usr/bin/env python
import getpass
import os
import sys
import time

import astropy.io.fits as pyfits
import gnupg
import numpy as np
from astropy.table import Table


# NOTE: Make sure it is the gpg directory
# And the gpg contains the private key
gpgfile = os.path.join(os.environ["HOME"], ".gnupg")
gpg = gnupg.GPG(gnupghome=gpgfile)

testing = False
if not testing:
    blinder = "Jim Bosch"
    hsc_mail = "hscblinder@gmail.com"

    user = "Raul Teixeira"
    user_mail = "raul.teixeira@duke.edu"
    # user = "Shreya Mukherjee"
    # user_mail = "shreya.mukherjee@iucaa.in"
    # user = "Ryo Terasawa"
    # user_mail = "ryo.terasawa@ipmu.jp"
else:
    blinder = "xiangchong"
    hsc_mail = "xiangchong.li@ipmu.jp"
    user = "xiangchong"
    user_mail = "xiangchong.li@ipmu.jp"

field_names = [
    "spring1",
    "spring2",
    "spring3",
    "autumn1",
    "autumn2",
    "hectomap",
]


def PGP_to_FITS(message):
    """
    Turns a PGP message into a string that can be stored in a FITS
    header, by removing the header and footer of the message as well
    as any \n characters
    """
    s = ""
    for st in message.split("\n"):
        if len(st) > 0 and "PGP MESSAGE" not in st and "GnuPG" not in st:
            s += st
    return s


def FITS_to_PGP(message):
    """
    Turns a string stored into an FITS comment back into a proper
    PGP message
    """
    s = "-----BEGIN PGP MESSAGE-----\n\n"
    s += message
    s += "\n-----END PGP MESSAGE-----\n"
    return s


def generate_dm2_values(rand_gen):
    if rand_gen < 1.0 / 3.0:
        m_blind = [1.0 / 1.05**2 - 1, 1.0 / 1.05 - 1, 0.00]
    elif rand_gen >= 1.0 / 3.0 and rand_gen < 2.0 / 3.0:
        m_blind = [1.0 / 1.05 - 1, 0.00, 0.05]
    elif rand_gen >= 2.0 / 3.0:
        m_blind = [0.00, 0.05, 1.05 ** 2 - 1]
    else:
        raise ValueError("wrong dm2 generator")
    return np.array(m_blind)


def generate_dm1_values(n_blind_cat, dm_range):
    m_blind = []
    for _ in range(n_blind_cat):
        m_blind.append(dm_range * (2.0 * np.random.rand() - 1))
    return m_blind


def encrypt():
    # paramters
    n_blind_cat = 3
    dm_range = 0.10

    s23b = os.getenv("s23b")
    if not s23b:
        raise RuntimeError("Environment variable s23b is not set.")
    root_dir = f"{s23b}/../"
    # directory
    if testing:
        outdir = os.path.join(
            root_dir,
            "s23b_response",
            "%s-%dblind-test/" % (user.replace(" ", "_"), n_blind_cat),
        )
    else:
        outdir = os.path.join(
            root_dir,
            "s23b_response",
            "%s-%dblind/" % (user.replace(" ", "_"), n_blind_cat),
        )
    os.makedirs(outdir, exist_ok=True)

    colname = "response"

    # get the key ids
    keys = gpg.list_keys()
    keyid1 = None
    keyid2 = None
    for k in keys:
        if user_mail in k["uids"][0]:
            keyid1 = k["fingerprint"]
        if hsc_mail in k["uids"][0]:
            keyid2 = k["fingerprint"]

    # generate dm1 lists
    assert keyid1 is not None, "Key fingerprint not found for %s" % (user_mail)
    dm1_list = generate_dm1_values(n_blind_cat, dm_range)
    # write the dm1 into header
    meta = {}
    meta.update({"ncat": n_blind_cat})
    meta.update({"user": user})
    meta.update({"uid": keyid1})

    if n_blind_cat == 3:
        # generate dm2 lists
        assert keyid2 is not None, \
            "Key fingerprint not found for %s" % (hsc_mail)
        # blinding seed
        seed = int(time.time())
        np.random.seed(seed)
        seedtmp = gpg.encrypt("%d" % seed, keyid2).data.decode("utf8")
        # assert '=c1l =c1l =c1l =' not in seedtmp
        seed_cry = PGP_to_FITS(seedtmp)
        rand_gen = np.random.rand()
        dm2_list = generate_dm2_values(rand_gen)
        # write the dm2 into header
        meta.update({"blinder": blinder})
        meta.update({"bid": keyid2})
        meta.update({"seed": seed_cry})
        if testing:
            print("Random seed %d" % seed)
    else:
        dm2_list = np.zeros(n_blind_cat)

    for ib in range(n_blind_cat):
        # dm1
        dmtmp = gpg.encrypt("%.6f" % dm1_list[ib], keyid1).data.decode("utf8")
        # assert '=c1l =c1l =c1l =' not in dmtmp
        dm1_cry = PGP_to_FITS(dmtmp)
        meta.update({"dm1c%s" % ib: dm1_cry})
        # dm2
        if not np.all(np.abs(dm2_list) < 1.0e-4):
            dmtmp = gpg.encrypt(
                "%.6f" % dm2_list[ib],
                keyid2
            ).data.decode("utf8")
            dm2_cry = PGP_to_FITS(dmtmp)
            meta.update({"dm2c%s" % ib: dm2_cry})

    for fieldname in field_names:
        input_fname = os.path.join(
            root_dir,
            "s23b_shape",
            f".response/{fieldname}.fits",
        )
        data = Table.read(input_fname)
        col_res = data[colname].copy()
        col_id = data["object_id"].copy()
        for ib in range(n_blind_cat):
            dmp1 = (1.0 + dm2_list[ib]) * (1.0 + dm1_list[ib])
            if testing and fieldname == "hectomap":
                print("cat%d: (1 + dm1) = %.6f" % (ib, 1 + dm1_list[ib]))
                print("cat%d: (1 + dm2) = %.6f" % (ib, 1 + dm2_list[ib]))
                print("cat%d: (1 + dm1)(1 + dm2) = %.6f" % (ib, dmp1))
            table_out = Table([col_id, col_res / dmp1])
            table_out.meta.update(meta)
            outfname = os.path.join(outdir, "%s_%d.fits" % (fieldname, ib))
            if os.path.isfile(outfname):
                print("Already have the blinded catalog m; overwriting")
            table_out.write(outfname, overwrite=True)
            del table_out
        del data
    return


def decrypt(header):
    n_blind_cat = header["ncat"]
    keys = gpg.list_keys(secret=True)
    private_fingerprints = [k["fingerprint"] for k in keys]
    keyid1 = header["uid"]
    if keyid1 in private_fingerprints:
        print("Decrypt User Blinding:")
        pwd = getpass.getpass(prompt="Password for dm1: ")
        for ib in range(n_blind_cat):
            dm1_cry = FITS_to_PGP(header["dm1c%s" % ib])
            dec_s1 = gpg.decrypt(dm1_cry, passphrase=pwd)
            assert dec_s1.ok, "Problem during decryption: %s" % dec_s1.stderr
            dm = np.float64(dec_s1.data)
            print("cat%d: (1+dm1) = %.6f" % (ib, 1.0+dm))
        print(
            "Please multiply response by (1+dm1) for each catalog, respectively"
        )
    else:
        print("Cannot find user's fingerprint to decrypt dm1!!")

    if n_blind_cat != 3:
        print("Do not have HSC level blinding")
        print("cat0: (1+dm2) = 1")
        print("Nothing needs to be done for HSC level unblinding")
        return
    keyid2 = header["bid"]
    if keyid2 in private_fingerprints:
        print("\nDecrypt HSC Blinding:")
        pwd = getpass.getpass(prompt="Password for dm2: ")

        seed_cry = FITS_to_PGP(header["seed"])
        dec_s2 = gpg.decrypt(seed_cry, passphrase=pwd)
        assert dec_s2.ok, "Problem during decryption: %s" % dec_s2.stderr
        seed = int(dec_s2.data)
        print("Random seed is: %d" % seed)
        for ib in range(n_blind_cat):
            dm2_cry = FITS_to_PGP(header["dm2c%s" % ib])
            dec_s2 = gpg.decrypt(dm2_cry, passphrase=pwd)
            assert dec_s2.ok, "Problem during decryption: %s" % dec_s2.stderr
            dm = np.float64(dec_s2.data)
            print("cat%d: (1+dm2) = %.6f" % (ib, 1.0+dm))
    else:
        print("Cannot find hsc's fingerprint for dm2!!")
    return


def main(argv):
    if len(argv) < 2:
        print("Usage: script.py [encrypt|decrypt] [fits_file]")
        return

    if argv[1] == "encrypt":
        if len(argv) == 2:
            encrypt()
        else:
            print("Wrong number of inputs")
    elif argv[1] == "decrypt":
        if len(argv) == 3:
            if os.path.isfile(argv[2]):
                decrypt(pyfits.getheader(argv[2], 1))
            else:
                print("Cannot find the fits file %s" % argv[2])
        elif len(argv) < 3:
            print("Please input the relative directory of a fits file")
        elif len(argv) > 3:
            print("Wrong number of inputs")
    else:
        print("Do not support argument %s" % argv[1])
        print("Please choose from (encrypt / decrypt)")
        return


if __name__ == "__main__":
    main(sys.argv)
