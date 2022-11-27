"""
This modules does XXX
"""
from typing import Literal, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import signal
from scipy.signal import savgol_filter
from scipy.stats import norm
from numpy import ndarray


def smooth2d(
    y: ndarray, box_pts: int, borders: bool = True, mode: str = "same"
) -> ndarray:
    if box_pts > 1:
        box = 1 / (box_pts**2) * np.ones(box_pts) * np.ones(box_pts)[:, np.newaxis]
        y_smooth = signal.convolve2d(y, box, mode=mode)
        if borders:
            y_smooth[0 : int(box_pts / 2 + 1), :] = y_smooth[int(box_pts / 2 + 2), :]
            y_smooth[-int(box_pts / 2 + 1) :, :] = y_smooth[-int(box_pts / 2 + 2), :]
            y_smooth[:, 0 : int(box_pts / 2 + 1)] = y_smooth[:, int(box_pts / 2 + 2)][
                :, np.newaxis
            ]
            y_smooth[:, -int(box_pts / 2 + 1) :] = y_smooth[:, -int(box_pts / 2 + 2)][
                :, np.newaxis
            ]
        else:
            y_smooth[0 : int(box_pts / 2 + 1), :] = y[0 : int(box_pts / 2 + 1), :]
            y_smooth[-int(box_pts / 2 + 1) :, :] = y[-int(box_pts / 2 + 1) :, :]
            y_smooth[:, 0 : int(box_pts / 2 + 1)] = y[:, 0 : int(box_pts / 2 + 1)]
            y_smooth[:, -int(box_pts / 2 + 1) :] = y[:, -int(box_pts / 2 + 1) :]
    else:
        y_smooth = y
    return y_smooth


def smooth(
    y: ndarray,
    box_pts: int,
    shape: Union[int, str] = "rectangular",
) -> ndarray:  # rectangular kernel for the smoothing
    box2_pts = int(2 * box_pts - 1)
    if isinstance(shape, int):
        y_smooth = np.ravel(
            pd.DataFrame(y)
            .rolling(box_pts, min_periods=1, center=True)
            .quantile(float(shape) / 100)
        )
    elif shape == "savgol":
        if box2_pts >= 5:
            y_smooth = savgol_filter(y, box2_pts, 3)
        else:
            y_smooth = y
    elif shape == "rectangular" or shape == "gaussian":
        if shape == "rectangular":
            box = np.ones(box2_pts) / box2_pts
        else:  # shape == "gaussian":
            vec = np.arange(-25, 26)
            box = norm.pdf(vec, scale=(box2_pts - 0.99) / 2.35) / np.sum(
                norm.pdf(vec, scale=(box2_pts - 0.99) / 2.35)
            )
        y_smooth = np.convolve(y, box, mode="same")
        y_smooth[0 : int((len(box) - 1) / 2)] = y[0 : int((len(box) - 1) / 2)]
        y_smooth[-int((len(box) - 1) / 2) :] = y[-int((len(box) - 1) / 2) :]
    return y_smooth
