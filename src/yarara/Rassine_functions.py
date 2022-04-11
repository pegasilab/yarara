#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 16:34:29 2019

@author: cretignier

"""

from __future__ import print_function

import platform

import matplotlib

from .io import open_pickle, pickle_dump, save_pickle

if platform.system() == "Linux":
    matplotlib.use("Agg", force=True)
else:
    matplotlib.use("Qt5Agg", force=True)

import pickle
import sys

# =============================================================================
# FUNCTIONS LIBRARY
# =============================================================================


sys_python = sys.version[0]


# replace dico.get
def try_field(dico, field):
    try:
        a = dico[field]
        return a
    except:
        return None
