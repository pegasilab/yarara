#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 16:34:29 2019

@author: cretignier

"""

from __future__ import print_function

import platform

import matplotlib

if platform.system() == "Linux":
    matplotlib.use("Agg", force=True)
else:
    matplotlib.use("Qt5Agg", force=True)

import pickle
import sys

import pandas as pd
from astropy.io import fits

pickle.DEFAULT_PROTOCOL = 3

# =============================================================================
# FUNCTIONS LIBRARY
# =============================================================================

h_planck = 6.626e-34
c_lum = 299.792e6
k_boltz = 1.3806e-23

sys_python = sys.version[0]
protocol_pickle = "auto"
voice_name = ["Victoria", "Daniel", None][0]  # voice pitch of the auditive feedback

if protocol_pickle == "auto":
    if sys_python == "3":
        protocol_pick = 3
    else:
        protocol_pick = 2
else:
    protocol_pick = int(protocol_pickle)


def open_pickle(filename):
    if filename.split(".")[-1] == "p":
        a = pd.read_pickle(filename)
        return a
    elif filename.split(".")[-1] == "fits":
        data = fits.getdata(filename)
        header = fits.getheader(filename)
        return data, header


def save_pickle(filename, output, header=None):
    if filename.split(".")[-1] == "p":
        pickle.dump(output, open(filename, "wb"), protocol=protocol_pick)
    if filename.split(".")[-1] == "fits":  # for futur work
        pass


def try_field(dico, field):
    try:
        a = dico[field]
        return a
    except:
        return None
