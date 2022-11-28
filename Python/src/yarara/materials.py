import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import TypedDict

from . import Float


class FixedString(TypedDict):
    fixed: str


class FixedFloat(TypedDict):
    fixed: Float


class FixedInt(TypedDict):
    fixed: int


class SIMBADEntry(TypedDict):
    Name: str
    Simbad_name: FixedString
    Sp_type: FixedString


class Contam_HARPN(TypedDict):
    #: Wavelengths in angstroms
    wave: NDArray[np.float32]
    #: Tells which wavelenghts are contaminated (True = contaminated)
    contam: NDArray[np.bool_]
    #: Tells which wavelenghts are contaminated (True = contaminated)
    contam_1: NDArray[np.bool_]
    #: Tells which wavelenghts are contaminated (True = contaminated)
    contam_2: NDArray[np.bool_]
    #: Tells which wavelenghts are contaminated (True = contaminated)
    contam_backup: NDArray[np.bool_]


class Telluric_spectrum(TypedDict):
    #: Wavelengths in angstroms
    wave: NDArray[np.float64]
    #: Telluric spectrum in normalised flux units
    flux_norm: NDArray[np.float64]


class Table_stellar_model(TypedDict):
    #: Table of Mass/Temperature/Radius from Gray+09 for dwarfs stars
    V: pd.DataFrame
    #: Table of Mass/Temperature/Radius from Gray+09 for evolved stars
    IV: pd.DataFrame


class Ghost_HARPN(TypedDict):
    #: JDB time of the reference wavelength solution
    jdb: float
    #: Not used anymore
    berv: float
    #: 2D matrix of the reference wavelength solution
    wave: NDArray[np.float32]
    #: 2D matrix of the stitching contamination (fiber A)
    stitching: NDArray
    #: 2D matrix of the ghost contamination (fiber A)
    ghost_a: NDArray
    #: 2D matrix of the ghost contamination (fiber B)
    ghost_b: NDArray
