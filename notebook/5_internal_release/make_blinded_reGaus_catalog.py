#!/usr/bin/env python
import os
import sys
import time
import gnupg
import getpass

import numpy as np
import astropy.io.fits as pyfits
from astropy.table import Table

s23b = os.getenv("s23b")
if not s23b:
    raise RuntimeError("Environment variable s23b is not set.")
rootDir = f"{s23b}/s23b_release/"

# NOTE: Make sure it is the gpg directory
# And the gpg contains the private key
gpgfile = os.path.join(os.environ["HOME"], ".gnupg")
gpg = gnupg.GPG(gnupghome=gpgfile)

testing = True
if not testing:
    blinder = "Jim Bosch"
    hsc_mail = "hscblinder@gmail.com"
    user = "Jingjing"
    user_mail = "jingssrs1989@gmail.com"
    """
    user        =   'divya'
    user_mail   =   'divyar@iucaa.in'
    user        =   'mkwiecie'
    user_mail   =   'mkwiecie@ucsc.edu'
    user        =   'tqzhang'
    user_mail   =   'tianqinz@andrew.cmu.ed'
    user        =   'nobuhiro2'
    user_mail   =   'okabe@hiroshima-u.ac.jp'
    user        =   'kfchen'
    user_mail   =   'kfchen@mit.edu'
    user        =   'amitk'
    user_mail   =   'amitk@iucaa.in'
    user        =   'noriaki'
    user_mail   =   'nakasawa.noriaki.a9@s.mail.nagoya-u.ac.jp'
    user        =   'Tam'
    user_mail   =   'sutieng8110@gmail.com'
    user        =   'RoohiDalal'
    user_mail   =   'roohidalal@gmail.com'
    user        =   'KenOsato'
    user_mail   =   'ken.osato@yukawa.kyoto-u.ac.jp'
    user        =   'masato'
    user_mail   =   'masato.shirasaki@gmail.com'
    user        =   'masamune'
    user_mail   =   'masamune.oguri@ipmu.jp'
    user        =   'wenting'
    user_mail   =   'wenting.wang@sjtu.edu.cn'
    user        =   'Andres Plazas'     #
    user_mail   =   'plazasmalagon@gmail.com'
    user        =   'wentao'
    user_mail   =   'wentao.luo@ipmu.jp'
    user        =   'surhud'
    user_mail   =   'surhudkicp@gmail.com'
    user        =   'INonChiu'
    user_mail   =   'inchiu@asiaa.sinica.edu.tw'
    user        =   'lalitwadee'
    user_mail   =   'kawinwanichakij@g.ecc.u-tokyo.ac.jp'
    user        =   'Tomomi'
    user_mail   =   'sunayama.tomomi@d.mbox.nagoya-u.ac.jp'
    user        =   'xiangchong'
    user_mail   =   'xiangchong.li@ipmu.jp'
    user        =   'MasatoGfarm'
    user_mail   =   'masato.shirasaki@nao.ac.jp'
    user        =   'hironao2'
    user_mail   =   'miyatake@kmi.nagoya-u.ac.jp'
    """
else:
    """
    user        =   'surhud'
    user_mail   =   'surhudkicp@gmail.com'
    blinder      =   'xiangchong'
    hsc_mail    =   'xiangchong.li@ipmu.jp'
    """
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
        m_blind = [-0.10, -0.05, 0.00]
    elif rand_gen >= 1.0 / 3.0 and rand_gen < 2.0 / 3.0:
        m_blind = [-0.05, 0.00, 0.05]
    elif rand_gen >= 2.0 / 3.0:
        m_blind = [0.00, 0.05, 0.10]
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
    n_blind_cat = 1
    dm_range = 0.10

    # directory
    if testing:
        outdir = os.path.join(
            rootDir,
            "%s-%dblind-test/" % (user.replace(" ", "_"), n_blind_cat),
        )
    else:
        outdir = os.path.join(
            rootDir,
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
    dm1List = generate_dm1_values(n_blind_cat, dm_range)
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
        # assert '=c1l=c1l=c1l=' not in seedtmp
        seed_cry = PGP_to_FITS(seedtmp)
        rand_gen = np.random.rand()
        dm2List = generate_dm2_values(rand_gen)
        # write the dm2 into header
        meta.update({"blinder": blinder})
        meta.update({"bid": keyid2})
        meta.update({"seed": seed_cry})
        if testing:
            print("Random seed %d" % seed)
    else:
        dm2List = np.zeros(n_blind_cat)

    for ib in range(n_blind_cat):
        # dm1
        dmtmp = gpg.encrypt("%.5f" % dm1List[ib], keyid1).data.decode("utf8")
        # assert '=c1l=c1l=c1l=' not in dmtmp
        dm1_cry = PGP_to_FITS(dmtmp)
        # print(keyid1)
        # print(gpg.encrypt('abc', keyid1))
        # print(dmtmp,dm1_cry)
        meta.update({"dm1c%s" % ib: dm1_cry})
        # dm2
        if not np.all(np.abs(dm2List) < 1.0e-4):
            dmtmp = gpg.encrypt(
                "%.5f" % dm2List[ib],
                keyid2
            ).data.decode("utf8")
            dm2_cry = PGP_to_FITS(dmtmp)
            meta.update({"dm2c%s" % ib: dm2_cry})

    for fieldname in field_names:
        input_fname = os.path.join(
            rootDir,
            f".response/{fieldname}.fits",
        )
        data = Table.read(input_fname)
        columnIn = data[colname].copy()
        colId = data["object_id"].copy()
        for ib in range(n_blind_cat):
            dmp1 = (1.0 + dm2List[ib]) * (1.0 + dm1List[ib])
            if testing and fieldname == "VVDS":
                print("(1 + dm1) cat%d: %.5f" % (ib, 1 + dm1List[ib]))
                print("(1 + dm2) cat%d: %.5f" % (ib, 1 + dm2List[ib]))
                print("(1 + dm1)(1 + dm2) cat%d:  %.5f" % (ib, dmp1))
            table_out = Table([colId, columnIn * dmp1])
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
            dm = float(dec_s1.data)
            print("dm1-cat%d: %.5f" % (ib, dm))
    else:
        print("Cannot find user's  fingerprint to decrypt dm1!!")

    if n_blind_cat != 3:
        print("Do not have HSC level blinding")
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
            dm = float(dec_s2.data)
            print("dm2-cat%d: %.5f" % (ib, dm))
    else:
        print("Cannot find hsc's  fingerprint for dm2!!")
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
