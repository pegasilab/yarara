"""
This modules does XXX
"""

import platform
from typing import Literal, Optional, Tuple, Union, overload

import matplotlib
import matplotlib.pylab as plt
import numpy as np
from matplotlib.axes import Axes
from numpy import ndarray
from scipy import ndimage

from .stats import IQ, smooth2d


def init_matplotlib() -> None:
    """
    Intializes the Matplotlib backend that works best for a given system
    """

    # TODO: Michael have a look
    if platform.system() == "Linux":
        matplotlib.use("Agg", force=True)
    else:
        matplotlib.use("Qt5Agg", force=True)


def plot_color_box(
    color: str = "r",
    font: str = "bold",
    lw: int = 2,
    ax: Optional[Axes] = None,
    side_: Union[
        Literal["top"], Literal["bottom"], Literal["left"], Literal["right"], Literal["all"]
    ] = "all",
    ls: str = "-",
) -> None:
    if ls == "-":
        ls = "solid"

    if ax is None:
        ax = plt.gca()

    if side_ == "all":
        side = ["top", "bottom", "left", "right"]
    else:
        side = [side_]
    for axis in side:
        ax.spines[axis].set_linewidth(lw)
        ax.spines[axis].set_color(color)
        if ax.spines[axis].get_linestyle() != ls:  # to win a but of time
            ax.spines[axis].set_linestyle(ls)

    ax.tick_params(axis="x", which="both", colors=color)
    ax.tick_params(axis="y", which="both", colors=color)

    if font == "bold":
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontweight("bold")

        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight("bold")


@overload
def my_colormesh(
    x: ndarray,
    y: ndarray,
    z: ndarray,
    return_output: Literal[False] = False,
    cmap: str = "seismic",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    zoom: int = 1,
    shading: Optional[Literal["flat", "nearest", "gouraud", "auto"]] = "auto",
    order: int = 3,
    smooth_box: int = 1,
) -> None:
    pass


@overload
def my_colormesh(
    x: ndarray,
    y: ndarray,
    z: ndarray,
    return_output: Literal[True],
    cmap: str = "seismic",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    zoom: int = 1,
    shading: Optional[Literal["flat", "nearest", "gouraud", "auto"]] = "auto",
    order: int = 3,
    smooth_box: int = 1,
) -> Tuple[ndarray, ndarray, ndarray]:
    pass


def my_colormesh(
    x: ndarray,
    y: ndarray,
    z: ndarray,
    return_output: bool = False,
    cmap: str = "seismic",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    zoom: int = 1,
    shading: Optional[Literal["flat", "nearest", "gouraud", "auto"]] = "auto",
    order: int = 3,
    smooth_box: int = 1,
) -> Optional[Tuple[ndarray, ndarray, ndarray]]:

    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]
    x, y = np.meshgrid(x, y)

    x = np.hstack([x, x[:, -1][:, np.newaxis] + dx])
    x = np.vstack([x, x[-1, :]])

    y = np.hstack([y, y[:, -1][:, np.newaxis]])
    y = np.vstack([y, y[-1, :] + dy])

    z = np.hstack([z, z[:, -1][:, np.newaxis]])
    z = np.vstack([z, z[-1, :]])

    z = smooth2d(z, smooth_box, borders=False)

    Z: ndarray = ndimage.zoom(z, zoom, order=order)
    X: ndarray = ndimage.zoom(x, zoom, order=order)
    Y: ndarray = ndimage.zoom(y, zoom, order=order)

    if return_output:
        return X, Y, Z
    else:
        plt.pcolormesh(X, Y, Z, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)  # type: ignore


def auto_axis(vec: ndarray, axis: str = "y", m: int = 3) -> None:
    iq = IQ(vec)
    q1 = np.nanpercentile(vec, 25)
    q3 = np.nanpercentile(vec, 75)
    ax = plt.gca()
    if axis == "y":
        val1 = [ax.get_ylim()[0], q1 - m * iq][q1 - m * iq > ax.get_ylim()[0]]
        val2 = [ax.get_ylim()[1], q3 + m * iq][q3 + m * iq < ax.get_ylim()[1]]
        plt.ylim(val1, val2)
    else:
        val1 = [ax.get_xlim()[0], q1 - m * iq][q1 - m * iq > ax.get_xlim()[0]]
        val2 = [ax.get_xlim()[1], q3 + m * iq][q3 + m * iq < ax.get_xlim()[1]]
        plt.xlim(val1, val2)
