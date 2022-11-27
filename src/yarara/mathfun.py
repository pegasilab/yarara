"""
This modules does XXX
"""

from typing import Union

import numpy as np
from astropy.modeling.models import Voigt1D
from numpy import float64, ndarray


def parabole(x, a, b, c):
    return a + b * x + c * x**2


def voigt(x, amp, cen, wid, wid2):
    func = Voigt1D(x_0=cen, amplitude_L=2, fwhm_L=wid2, fwhm_G=wid)(x)
    return 1 + amp * func / func.max()


def sinus(
    x: ndarray,
    amp: Union[float64, int],
    period: Union[float, float64],
    phase: Union[float64, int],
    a: int,
    b: int,
    c: Union[float64, int],
) -> ndarray:
    return amp * np.sin(x * 2 * np.pi / period + phase) + a * x**2 + b * x + c


def gaussian(
    x: ndarray,
    cen: float64,
    amp: float64,
    offset: float64,
    wid: float64,
) -> ndarray:
    return amp * np.exp(-((x - cen) ** 2) / (2 * wid**2)) + offset


def lorentzian(x, amp, cen, offset, wid):
    return amp * wid**2 / ((x - cen) ** 2 + wid**2) + offset
