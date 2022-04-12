import glob as glob
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..my_rassine_tools import spec_time_series

    # =============================================================================
    # IMPORT ALL RASSINE DICTIONNARY
    # =============================================================================


def import_rassine_output(self: spec_time_series, return_name=False, kw1=None, kw2=None):
    """
    Import all the RASSINE dictionnaries in a list

    Parameters
    ----------
    return_name : True/False to also return the filenames

    Returns
    -------
    Return the list containing all thedictionnary

    """

    directory = self.directory

    files = glob.glob(directory + "RASSI*.p")
    if len(files) <= 1:  # 1 when merged directory
        print("No RASSINE file found in the directory : %s" % (directory))
        if return_name:
            return [], []
        else:
            return []
    else:
        files = np.sort(files)
        file = []
        for i, j in enumerate(files):
            self.debug = j
            file.append(pd.read_pickle(j))

            if kw1 is not None:
                file[-1] = file[-1][kw1]

            if kw2 is not None:
                file[-1] = file[-1][kw2]

        if return_name:
            return file, files
        else:
            return file


# =============================================================================
# IMPORT SUMMARY TABLE
# =============================================================================


def import_star_info(self: spec_time_series):
    self.star_info = pd.read_pickle(
        self.dir_root + "STAR_INFO/Stellar_info_" + self.starname + ".p"
    )


def import_table(self: spec_time_series):
    self.table = pd.read_pickle(self.directory + "Analyse_summary.p")


def import_material(self: spec_time_series):
    self.material = pd.read_pickle(self.directory + "Analyse_material.p")
