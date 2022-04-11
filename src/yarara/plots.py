"""
This modules does XXX
"""

import platform

import matplotlib


def init_matplotlib() -> None:
    """
    Intializes the Matplotlib backend that works best for a given system
    """

    # TODO: Michael have a look
    if platform.system() == "Linux":
        matplotlib.use("Agg", force=True)
    else:
        matplotlib.use("Qt5Agg", force=True)


def plot_color_box(color="r", font="bold", lw=2, ax=None, side="all", ls="-"):
    if ls == "-":
        ls = "solid"

    if ax is None:
        ax = plt.gca()
    if side == "all":
        side = ["top", "bottom", "left", "right"]
    else:
        side = [side]
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


def my_colormesh(
    x,
    y,
    z,
    cmap="seismic",
    vmin=None,
    vmax=None,
    zoom=1,
    shading="auto",
    return_output=False,
    order=3,
    smooth_box=1,
):

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

    Z = ndimage.zoom(z, zoom, order=order)
    X = ndimage.zoom(x, zoom, order=order)
    Y = ndimage.zoom(y, zoom, order=order)

    if return_output:
        return X, Y, Z
    else:
        plt.pcolormesh(X, Y, Z, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)


def auto_axis(vec, axis="y", m=3):
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
