from __future__ import annotations

import glob as glob
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from numpy import float64, int64, ndarray
from pandas.core.frame import DataFrame
from tqdm import tqdm

from ... import io, util
from ...analysis import tableXY

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_exploding_pickle(self: spec_time_series) -> None:
    self.import_table()
    file_test = self.import_spectrum()

    sub_dico = util.string_contained_in(list(file_test.keys()), "matching")[1]

    sub_dico_to_ventile = []
    sub_dico_to_delete = []
    for sb in sub_dico:
        if "continuum_linear" in file_test[sb].keys():
            sub_dico_to_ventile.append(sb)
        else:
            sub_dico_to_delete.append(sb)

    files = np.array(self.table["filename"])

    c = -1
    for sb in sub_dico_to_ventile:
        c += 1
        print(
            "\n [INFO] Venting sub_dico %s, number of dico remaining : %.0f \n"
            % (sb, len(sub_dico_to_ventile) - c)
        )
        continua = []
        for f in tqdm(files):
            file = pd.read_pickle(f)
            continua.append(file[sb]["continuum_linear"])
            if (sb != "matching_anchors") & (sb != "matching_diff"):
                del file[sb]
            io.pickle_dump(file, open(f, "wb"))
        continua = np.array(continua)
        fname = self.dir_root + "WORKSPACE/CONTINUUM/Continuum_%s.npy" % (sb)
        np.save(fname, continua)

    c = -1
    for sb in sub_dico_to_delete:
        c += 1
        print(
            "\n [INFO] Deleting sub_dico %s, number of dico remaining : %.0f \n"
            % (sb, len(sub_dico_to_delete) - c)
        )
        for f in tqdm(files):
            file = pd.read_pickle(f)
            del file[sb]
            io.pickle_dump(file, open(f, "wb"))

    kw = list(util.string_contained_in(list(file_test.keys()), "flux")[1])
    kw2 = list(util.string_contained_in(list(file_test.keys()), "continuum")[1])

    c = -1
    for sb in kw + kw2:
        c += 1
        print("\n [INFO] Venting key word %s \n" % (sb))
        flux = []
        for f in tqdm(files):
            file = pd.read_pickle(f)
            flux.append(file[sb])
            io.pickle_dump(file, open(f, "wb"))
        flux = np.array(flux)
        fname = self.dir_root + "WORKSPACE/FLUX/Flux_%s.npy" % (sb)
        np.save(fname, flux)


# endregion
