from __future__ import annotations

import glob as glob
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union, overload

import numpy as np
import pandas as pd
from numpy import float64, int64, ndarray
from pandas.core.frame import DataFrame
from tqdm import tqdm

from ... import iofun, util
from ...analysis import tableXY

if TYPE_CHECKING:
    from .. import spec_time_series


def import_dico_tree(self: spec_time_series) -> None:
    self.import_info_reduction()
    file_test = self.info_reduction
    kw = list(file_test.keys())
    kw_kept = []
    kw_chain = []
    for k in kw:
        if len(k.split("matching_")) == 2:
            kw_kept.append(k)
    kw_kept = np.array(kw_kept)

    info = []
    for n in kw_kept:
        try:
            s = file_test[n]["step"]
            dico = file_test[n]["sub_dico_used"]
            info.append([n, s, dico])
        except:
            pass
    info = pd.DataFrame(info, columns=["dico", "step", "dico_used"])
    self.dico_tree = info.sort_values(by="step")


def import_dico_chain(self: spec_time_series, last_dico: str):
    self.import_info_reduction()
    test_file = self.info_reduction

    kw = list(test_file.keys())
    kw_kept = []
    kw_chain = []
    for k in kw:
        if len(k.split("matching_")) == 2:
            kw_kept.append(k)
    kw_kept = np.array(kw_kept)

    while last_dico != "matching_anchors":
        next_dico = test_file[last_dico]["sub_dico_used"]
        kw_chain.append(last_dico)
        last_dico = next_dico

    chain = np.array(kw_chain)
    chain = pd.DataFrame([[n, test_file[n]["step"]] for n in chain], columns=["dico", "step"])
    chain_sorted = np.array(chain.sort_values(by="step")["dico"])[::-1]

    self.dico_chain = chain_sorted


def yarara_add_step_dico(
    self: spec_time_series,
    sub_dico: str,
    step: int,
    sub_dico_used: str,
    chain: bool = False,
) -> None:
    """Add the step kw for the dico chain, if chain is set to True numbered the full chain"""
    self.import_table()
    self.import_info_reduction()
    self.info_reduction[sub_dico] = {"sub_dico_used": sub_dico_used, "step": step, "valid": True}
    self.update_info_reduction()
