from typing import TypedDict, Union

import numpy as np
import numpy.typing as npt
from beartype import beartype
from beartype.vale import Is, IsAttr, IsEqual
from typing_extensions import Annotated

Vec_f8 = Annotated[npt.NDArray[np.float64], IsAttr["ndim", IsEqual[1]]]
Vec_f4 = Annotated[npt.NDArray[np.float64], IsAttr["ndim", IsEqual[1]]]
Vec_f = Union[Vec_f8, Vec_f4]
Vec_bool = Annotated[npt.NDArray[np.bool_], IsAttr["ndim", IsEqual[1]]]
Mat_f8 = Annotated[npt.NDArray[np.float64], IsAttr["ndim", IsEqual[2]]]
Mat_f = Union[Mat_f8, Mat_f4]
Vec_f8_W = Vec_f8
Vec_f8_N = Vec_f8


class ModelTelluric(TypedDict):
    """
    Telluric model

    File Material/model_telluric.p
    """

    wave: Vec_f8  #: wavelength

    flux_norm: Vec_f8  #: flux


#: wavelength_left wavelength_right weight
MaskCCF = Annotated[npt.NDArray[np.float64], Is[lambda x: x.shape[1] == 3]]


class Contamination_HARPN(TypedDict):
    wave: Vec_f8  #: Wavelength
    contam: Vec_bool  #: If it is contamined
    contam_1: Vec_bool  #: Unused to remove
    contam_2: Vec_bool  #: Unused to remove
    contam_backup: Vec_bool  #: Unused to remove


Float = Union[float, np.float64]


class InstrumentGhost(TypedDict):
    jdb: Float  #: unused
    berv: Float  #: unused
    wave: Mat_f  #: Size n_pixels x n_orders
    stitching: Mat_f  #: Size n_pixels x n_orders, values are 0 or 1
    ghost_a: Mat_f  #: Size n_pixels x n_orders
    ghost_b: Mat_f  #: Size n_pixels x n_orders
