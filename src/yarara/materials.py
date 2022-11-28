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
    #: Starname
    Name: str
    #: Starname in simbad query
    Simbad_name: FixedString
    #: Stellar spectral type
    Sp_type: FixedString
    #: Right ascension coordinate
    Ra: FixedString
    #: Declination coordinate
    Dec: FixedString
    #: Proper motion ascension
    Pma: FixedFloat
    #: Proper motion declination
    Pmd: FixedFloat
    #: Systemic RV in km/s
    RV_sys: FixedFloat
    #: Stellar mass in solar mass
    Mstar: FixedFloat
    #: Stellar radius in solar radius
    Rstar: FixedFloat
    #: U band magnitude
    Umag: FixedFloat
    #: B band magnitude
    Bmag: FixedFloat
    #: V band magnitude
    Vmag: FixedFloat
    #: R band magnitude
    Rmag: FixedFloat
    #: UB color index
    UB: FixedFloat
    #: BV color index
    BV: FixedFloat
    #: VR color index
    VR: FixedFloat
    #: Stellar distance in parsec
    Dist_pc: FixedFloat
    #: Stellar effective temperature in Kelvin
    Teff: FixedInt
    #: Stellar surface gravity in cgs
    Log_g: FixedFloat
    #: Stellar metallicity
    FeH: FixedFloat
    #: Stellar vsini in km/s
    Vsini: FixedFloat
    #: Stellar microturbulence+macro in km/s
    Vmicro: FixedFloat
    #: Stellar rotationnal period
    Prot: FixedInt
    #: Stellar vsini in km/s
    Pmag: FixedInt
    #: FWHM of the CCF in km/s
    FWHM: FixedFloat
    #: Contrast of the CCF in km/s
    Contrast: FixedFloat
    #: Width of the CCF comb in wavelength index
    CCF_delta: FixedInt
    #: Name of the stellar template in the ATLAS database
    stellar_template: FixedString


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
