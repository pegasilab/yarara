from __future__ import annotations

import glob as glob
import logging
import os
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...io import pickle_dump

if TYPE_CHECKING:
    from .. import spec_time_series


def suppress_time_RV(self: spec_time_series, liste: NDArray[np.bool_]):

    self.import_ccf()

    if np.any(liste):
        mask = list(self.table_ccf.keys())[1:]
        try:
            for m in mask:
                for d in self.table_ccf[m].keys():
                    self.table_ccf[m][d]["table"] = self.table_ccf[m][d]["table"][
                        ~liste
                    ].reset_index(drop=True)

            pickle_dump(self.table_ccf, open(self.directory + "Analyse_ccf.p", "wb"))

            print(" [INFO] CCF table modifed")
        except:
            print(" [ERROR] CCF cannot be modified")


def suppress_time_spectra(
    self: spec_time_series,
    mask: NDArray[np.bool_],
    supress: bool = False,
    name_ext: str = "temp",
) -> None:
    """
    Supress spectra according to time

    Parameters
    ----------

    jdb or num are both inclusive in lower and upper limit
    snr_cutoff : treshold value of the cutoff below which spectra are removed
    supress : True/False to delete of simply hide the files

    """

    self.import_table()
    jdb = self.table["jdb"]
    name = self.table["filename"]
    directory = "/".join(np.array(name)[0].split("/")[:-1])

    if np.any(mask):
        idx = np.arange(len(jdb))[mask]
        logging.info("Following spectrum indices will be supressed : ", idx)
        logging.info("Number of spectrum supressed : %.0f \n" % (sum(mask)))
        maps = glob.glob(self.dir_root + "CORRECTION_MAP/*.p")
        if len(maps):
            for names in maps:
                correction_map = pd.read_pickle(names)
                correction_map["correction_map"] = np.delete(
                    correction_map["correction_map"], idx, axis=0
                )
                pickle_dump(correction_map, open(names, "wb"))
                print("%s modified" % (names.split("/")[-1]))

        name = name[mask]

        files_to_process = np.sort(np.array(name))
        for j in files_to_process:
            if supress:
                print("File deleted : %s " % (j))
                os.system("rm " + j)
            else:
                new_name = name_ext + "_" + j.split("/")[-1]
                print("File renamed : %s " % (directory + "/" + new_name))
                os.system("mv " + j + " " + directory + "/" + new_name)

        os.system("rm " + directory + "/Analyse_summary.p")
        os.system("rm " + directory + "/Analyse_summary.csv")
        self.yarara_analyse_summary()

        self.suppress_time_RV(mask)
