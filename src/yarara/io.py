"""
This modules does XXX
"""

import os
import pickle
from io import BufferedWriter
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from astropy.io import fits
from numpy import float64, ndarray, str_
from pandas.core.frame import DataFrame

from . import pickle_protocol_version


def touch_pickle(filename: str) -> Dict[Union[str, str_], Dict[str, Any]]:
    if not os.path.exists(filename):
        pickle_dump({}, open(filename, "wb"))
        return {}
    else:
        return pd.read_pickle(filename)


def open_pickle(filename):
    if filename.split(".")[-1] == "p":
        a = pd.read_pickle(filename)
        return a
    elif filename.split(".")[-1] == "fits":
        data = fits.getdata(filename)
        header = fits.getheader(filename)
        return data, header


def save_pickle(
    filename: Union[str_, str],
    output: Union[
        Dict[
            str,
            Union[
                ndarray,
                Dict[str, ndarray],
                Dict[str, Union[str, int, float64, float, bool, List[str]]],
                Dict[str, Union[ndarray, Dict[str, Optional[Union[str, int, float]]]]],
                Dict[str, Union[Dict[str, Union[str, int, bool]], ndarray]],
            ],
        ],
        Dict[
            str,
            Union[
                ndarray,
                Dict[str, ndarray],
                Dict[str, Union[str, int, float64, float, bool, List[str]]],
                Dict[str, Union[ndarray, Dict[str, Optional[Union[str, int, float]]]]],
                Dict[str, Union[Dict[str, Union[str, int, bool]], ndarray]],
                Dict[str, Union[ndarray, bool, float64]],
                Dict[str, float64],
            ],
        ],
        DataFrame,
    ],
    header: None = None,
) -> None:
    if filename.split(".")[-1] == "p":
        pickle.dump(output, open(filename, "wb"), protocol=pickle_protocol_version)
    if filename.split(".")[-1] == "fits":  # for futur work
        pass


# remove "protocol" parameter
def pickle_dump(obj: Any, obj_file: BufferedWriter, protocol: None = None) -> None:
    if protocol is None:
        protocol = pickle_protocol_version
    pickle.dump(obj, obj_file, protocol=protocol)
