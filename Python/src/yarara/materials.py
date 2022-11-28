import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypedDict


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
