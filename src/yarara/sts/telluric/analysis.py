from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Literal, Optional, Union

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ... import io
from ...paths import paths, root
from ...stats import clustering
from ...util import doppler_r

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_telluric(
    self: spec_time_series,
    sub_dico: str = "matching_anchors",
    suppress_broad: bool = True,
    delta_window: int = 5,
    telluric_tag: Union[Literal["h2o"], Literal["o2"], Literal["telluric"]] = "telluric",
    weighted: bool = False,
    reference: Union[bool, Literal["norm"], Literal["master_snr"]] = True,
    ratio: bool = False,
    normalisation: Union[Literal["left"], Literal["slope"]] = "slope",
    ccf_oversampling: int = 3,
) -> None:

    """
    Plot all the RASSINE spectra in the same plot

    Args:
        mask_name: The telluric tag used to find a mask to cross correlate with the spectrum

                   Mask should be located in MASK_CCF and have the filename mask_telluric_TELLURICTAG.txt

    Parameters
    ----------
    sub_dico : The sub_dictionnary used to  select the continuum
    continuum : The continuum to select (either linear or cubic)
    reference : True/False or 'norm', True use the matching anchors of reference, False use the continuum of each spectrum, norm use the continuum normalised spectrum (not )
    display_ccf : display all the ccf
    normalisation : 'left' or 'slope'. if left normalise the CCF by the most left value, otherwise fit a line between the two highest point
    planet : True/False to use the flux containing the injected planet or not
    """

    assert telluric_tag in ["h2o", "o2", "telluric"]
    kw = "_planet" * self.planet
    if kw != "":
        logging.info("PLANET ACTIVATED")

    logging.info("RECIPE : COMPUTE TELLURIC CCF MOMENT")

    self.import_table()

    if np.nanmax(self.table.berv) - np.nanmin(self.table.berv) < 5:
        reference = False
        ratio = False
        logging.warn(
            "BERV SPAN too low to consider ratio spectra as reliable. Diff spectra will be used."
        )

    berv_max = np.max(abs(self.table["berv" + kw]))
    directory = self.directory
    planet = self.planet

    rv_shift = np.array(self.table["rv_shift"]) * 1000

    files = paths.sorted_rassine_pickles(self)

    if sub_dico is None:
        sub_dico = self.dico_actif
    logging.info(f"DICO {sub_dico} used")

    test = self.spectrum(num=0, sub_dico=sub_dico, norm=True)
    grid = test.x
    flux = test.y
    dg = grid[1] - grid[0]
    file_ref = self.import_spectrum()
    ccf_sigma = int(file_ref["parameters"]["fwhm_ccf"] * 10 / 3e5 * 6000 / dg)

    # TODO: does mask differ from telluric_tag?
    mask: NDArray[np.float64] = np.genfromtxt(
        paths.generic_mask_ccf + "mask_telluric_" + telluric_tag + ".txt"
    )
    mask = mask[mask[:, 0].argsort()]
    # mask = mask[(mask[:,0]>6200)&(mask[:,0]<6400)]
    wave_tel = 0.5 * (mask[:, 0] + mask[:, 1])
    mask = mask[(wave_tel < np.max(grid)) & (wave_tel > np.min(grid))]
    wave_tel = wave_tel[(wave_tel < np.max(grid)) & (wave_tel > np.min(grid))]

    test.clip(min=[mask[0, 0], None], max=[mask[-1, 0], None])
    test.rolling(window=ccf_sigma, quantile=1)  # to suppress telluric in broad asorption line
    tt, matrix = clustering((test.roll < 0.97).astype("int"), tresh=0.5, num=0.5)
    t = np.array([k[0] for k in tt]) == 1
    matrix = matrix[t, :]

    keep_telluric = np.ones(len(wave_tel)).astype("bool")
    for j in range(len(matrix)):
        left = test.x[matrix[j, 0]]
        right = test.x[matrix[j, 1]]

        c1 = np.sign(doppler_r(wave_tel, 30000)[0] - left)
        c2 = np.sign(doppler_r(wave_tel, 30000)[1] - left)
        c3 = np.sign(doppler_r(wave_tel, 30000)[0] - right)
        c4 = np.sign(doppler_r(wave_tel, 30000)[1] - right)
        keep_telluric = keep_telluric & ((c1 == c2) * (c1 == c3) * (c1 == c4))

    if (sum(keep_telluric) > 25) & (
        suppress_broad
    ):  # to avoid rejecting all tellurics for cool stars
        mask = mask[keep_telluric]
    logging.info("%.0f lines available in the telluric mask" % (len(mask)))
    plt.figure()
    plt.plot(grid, flux)
    for j in 0.5 * (mask[:, 0] + mask[:, 1]):
        plt.axvline(x=j, color="k")

    self.yarara_ccf(
        sub_dico=sub_dico,
        mask=mask,
        mask_name=telluric_tag,
        weighted=weighted,
        delta_window=delta_window,
        reference=reference,
        plot=True,
        save=False,
        ccf_oversampling=ccf_oversampling,
        normalisation=normalisation,
        ratio=ratio,
        rv_borders=10.0,
        rv_range=float(berv_max + 7),
        rv_sys_=0.0,
        rv_shift_=rv_shift,
    )

    plt.figure(figsize=(6, 6))
    plt.axes((0.15, 0.3, 0.8, 0.6))
    self.ccf_rv.yerr *= 0
    self.ccf_rv.yerr += 50
    self.ccf_rv.plot(modulo=365.25, label="%s ccf rv" % (telluric_tag))
    plt.scatter(self.table.jdb % 365.25, self.table.berv * 1000, color="b", label="berv")
    plt.legend()
    plt.ylabel("RV [m/s]")
    plt.xlabel("Time %365.25 [days]")
    plt.axes((0.15, 0.08, 0.8, 0.2))
    plt.axhline(y=0, color="k", alpha=0.5)
    plt.errorbar(
        self.table.jdb % 365.25,
        self.table.berv * 1000 - self.ccf_rv.y,
        self.ccf_rv.yerr,
        fmt="ko",
    )
    self.berv_offset = np.nanmedian(self.table.berv * 1000 - self.ccf_rv.y) / 1000
    print("\n [INFO] Difference with the BERV : %.0f m/s" % (self.berv_offset * 1000))
    plt.ylabel(r"$\Delta$RV [m/s]")
    plt.savefig(self.dir_root + "IMAGES/telluric_control_check_%s.pdf" % (telluric_tag))

    output = self.ccf_timeseries

    mask_fail = (
        abs(self.table.berv * 1000 - self.ccf_rv.y) > 1000
    )  # rv derive to different fromn the berv -> CCF not ttrustworthly
    mask_fail = mask_fail | np.isnan(self.ccf_rv.y)
    if sum(mask_fail):
        logging.warn(
            "There are %.0f datapoints incompatible with the BERV values" % (sum(mask_fail))
        )
        for k in range(len(output)):
            output[k][mask_fail] = np.nanmedian(output[k][~mask_fail])

    for i, j in enumerate(files):
        file = pd.read_pickle(j)
        file["parameters"][telluric_tag + "_ew"] = output[0][i]
        file["parameters"][telluric_tag + "_contrast"] = output[1][i]
        file["parameters"][telluric_tag + "_rv"] = output[2][i]
        # warning: ccf_timeseries convention has changed during YARARA lifetime
        file["parameters"][telluric_tag + "_fwhm"] = output[7][i]
        file["parameters"][telluric_tag + "_center"] = output[9][i]
        file["parameters"][telluric_tag + "_depth"] = output[11][i]
        io.pickle_dump(file, open(j, "wb"))

    self.yarara_analyse_summary()
