from __future__ import annotations

import glob as glob
import time
from typing import TYPE_CHECKING, List

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import io
from ..analysis import table, tableXY
from ..plots import my_colormesh
from ..stats import find_nearest, smooth, smooth2d
from ..util import doppler_r, print_box

if TYPE_CHECKING:
    from . import spec_time_series


# =============================================================================
#  ACTIVITY CORRECTION
# =============================================================================

# activity
def yarara_correct_activity(
    self: spec_time_series,
    sub_dico: str = "matching_telluric",
    continuum: str = "linear",
    wave_min: int = 3900,
    wave_max: int = 4400,
    smooth_corr: int = 5,
    reference: str = "median",
    rv_shift: str = "none",
    proxy_corr: List[str] = ["CaII"],
) -> None:
    """
    Display the time-series spectra with proxies and its correlation

    Parameters
    ----------
    sub_dico : The sub_dictionnary used to  select the continuum
    continuum : The continuum to select (either linear or cubic)
    wave_min : Minimum x axis limit
    wave_max : Maximum x axis limit
    zoom : int-type, to improve the resolution of the 2D plot
    smooth_map = int-type, smooth the 2D plot by gaussian 2D convolution
    smooth_corr = smooth thecoefficient  ofcorrelation curve
    reference : 'median', 'snr' or 'master' to select the reference normalised spectrum usedin the difference
    berv_shift : True/False to move in terrestrial rest-frame
    proxy_corr : keyword  of the first proxies from RASSINE dictionnary to use in the correlation
    proxy_detrending : Degree of the polynomial fit to detrend the proxy

    cmap : cmap of the 2D plot
    dwin : window correction increase by dwin to slightly correct above around the peak of correlation


    """

    print_box("\n---- RECIPE : CORRECTION ACTIVITY (CCF MOMENTS) ----\n")

    directory = self.directory

    zoom = self.zoom
    smooth_map = self.smooth_map
    cmap = self.cmap
    planet: bool = self.planet
    low_cmap = self.low_cmap * 100
    high_cmap = self.high_cmap * 100
    self.import_material()
    load = self.material

    epsilon = 1e-12

    kw = "_planet" * planet
    if kw != "":
        print("\n---- PLANET ACTIVATED ----")

    if sub_dico is None:
        sub_dico = self.dico_actif
    print("\n---- DICO %s used ----\n" % (sub_dico))

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    flux = []
    snr = []
    conti = []
    prox = []
    jdb = []
    rv = []

    self.import_table()
    for prox_name in proxy_corr:
        prox.append(np.array(self.table[prox_name]))
    proxy = np.array(prox)

    for i, j in enumerate(files):
        file = pd.read_pickle(j)
        if not i:
            wave = file["wave"]
        snr.append(file["parameters"]["SNR_5500"])
        conti.append(file[sub_dico]["continuum_" + continuum])
        flux.append(file["flux" + kw] / file[sub_dico]["continuum_" + continuum])
        try:
            jdb.append(file["parameters"]["jdb"])
        except:
            jdb.append(i)
        try:
            rv.append(file["parameters"][rv_shift])
        except:
            rv.append(0)

    step = file[sub_dico]["parameters"]["step"]

    wave = np.array(wave)
    flux = np.array(flux)
    conti = np.array(conti)
    snr = np.array(snr)
    proxy = np.array(prox)
    jdb = np.array(jdb)
    rv = np.array(rv)
    mean_rv = np.mean(rv)
    rv = rv - mean_rv

    #        proxy = tableXY(jdb,proxy)
    #        proxy.substract_polyfit(proxy_detrending)
    #        proxy = proxy.detrend_poly.y

    if reference == "snr":
        ref = flux[snr.argmax()]
    elif reference == "median":
        print("[INFO] Reference spectrum : median")
        ref = np.median(flux, axis=0)
    elif reference == "master":
        print("[INFO] Reference spectrum : master")
        ref = np.array(load["reference_spectrum"])
    elif type(reference) == int:
        print("[INFO] Reference spectrum : spectrum %.0f" % (reference))
        ref = flux[reference]
    else:
        ref = 0 * np.median(flux, axis=0)

    if low_cmap is None:
        low_cmap = np.percentile(flux - ref, 2.5)
    if high_cmap is None:
        high_cmap = np.percentile(flux - ref, 97.5)

    diff = smooth2d(flux - ref, smooth_map)
    diff_backup = diff.copy()

    if np.sum(rv) != 0:
        for j in tqdm(np.arange(len(flux))):
            test = tableXY(wave, diff[j], 0 * wave)
            test.x = doppler_r(test.x, rv[j] * 1000)[1]
            test.interpolate(new_grid=wave, method="cubic", replace=True, interpolate_x=False)
            diff[j] = test.y

    collection = table(diff.T)
    base_vec = np.vstack([np.ones(len(flux)), proxy])
    collection.fit_base(base_vec, num_sim=1)

    collection.coeff_fitted[:, 1] = smooth(
        collection.coeff_fitted[:, 1], smooth_corr, shape="savgol"
    )

    correction = collection.coeff_fitted.dot(base_vec)
    correction = np.transpose(correction)

    correction_backup = correction.copy()
    if np.sum(rv) != 0:
        for j in tqdm(np.arange(len(flux))):
            test = tableXY(wave, correction[j], 0 * wave)
            test.x = doppler_r(test.x, rv[j] * 1000)[0]
            test.interpolate(new_grid=wave, method="cubic", replace=True, interpolate_x=False)
            correction_backup[j] = test.y

    index_min_backup = int(find_nearest(wave, doppler_r(wave[0], rv.max() * 1000)[0])[0])
    index_max_backup = int(find_nearest(wave, doppler_r(wave[-1], rv.min() * 1000)[0])[0])
    correction_backup[:, 0:index_min_backup] = 0
    correction_backup[:, index_max_backup:] = 0

    diff2_backup = diff_backup - correction_backup

    new_conti = conti * (diff_backup + ref) / (diff2_backup + ref + epsilon)
    new_continuum = new_conti.copy()
    new_continuum[flux == 0] = conti[flux == 0]
    new_continuum = self.uncorrect_hole(new_continuum, conti)

    # plot end

    idx_min = 0
    idx_max = len(wave)

    if wave_min is not None:
        idx_min = find_nearest(wave, wave_min)[0]
    if wave_max is not None:
        idx_max = find_nearest(wave, wave_max)[0] + 1

    if (idx_min == 0) & (idx_max == 1):
        idx_max = find_nearest(wave, np.min(wave) + 500)[0] + 1

    new_wave = wave[int(idx_min) : int(idx_max)]

    fig = plt.figure(figsize=(21, 9))

    plt.axes([0.05, 0.66, 0.90, 0.25])
    my_colormesh(
        new_wave,
        np.arange(len(diff)),
        100 * diff_backup[:, int(idx_min) : int(idx_max)],
        zoom=zoom,
        vmin=low_cmap,
        vmax=high_cmap,
        cmap=cmap,
    )
    plt.tick_params(direction="in", top=True, right=True, labelbottom=False)
    plt.ylabel("Spectra  indexes (time)", fontsize=14)
    plt.ylim(0, None)
    ax = plt.gca()
    cbaxes = fig.add_axes([0.95, 0.66, 0.01, 0.25])
    ax1 = plt.colorbar(cax=cbaxes)
    ax1.ax.set_ylabel(r"$\Delta$ flux normalised [%]", fontsize=14)

    plt.axes([0.05, 0.375, 0.90, 0.25], sharex=ax, sharey=ax)
    my_colormesh(
        new_wave,
        np.arange(len(diff)),
        100 * diff2_backup[:, int(idx_min) : int(idx_max)],
        zoom=zoom,
        vmin=low_cmap,
        vmax=high_cmap,
        cmap=cmap,
    )
    plt.tick_params(direction="in", top=True, right=True, labelbottom=False)
    plt.ylabel("Spectra  indexes (time)", fontsize=14)
    plt.ylim(0, None)
    ax = plt.gca()
    cbaxes2 = fig.add_axes([0.95, 0.375, 0.01, 0.25])
    ax2 = plt.colorbar(cax=cbaxes2)
    ax2.ax.set_ylabel(r"$\Delta$ flux normalised [%]", fontsize=14)

    plt.axes([0.05, 0.09, 0.90, 0.25], sharex=ax, sharey=ax)
    my_colormesh(
        new_wave,
        np.arange(len(diff)),
        100 * correction_backup[:, int(idx_min) : int(idx_max)],
        zoom=zoom,
        vmin=low_cmap,
        vmax=high_cmap,
        cmap=cmap,
    )
    plt.tick_params(direction="in", top=True, right=True, labelbottom=True)
    plt.ylabel("Spectra  indexes (time)", fontsize=14)
    plt.xlabel(r"Wavelength [$\AA$]", fontsize=14)
    plt.ylim(0, None)
    ax = plt.gca()
    cbaxes3 = fig.add_axes([0.95, 0.09, 0.01, 0.25])
    ax3 = plt.colorbar(cax=cbaxes3)
    ax3.ax.set_ylabel(r"$\Delta$ flux normalised [%]", fontsize=14)

    plt.savefig(self.dir_root + "IMAGES/Correction_activity.png")

    correction_activity = correction_backup
    to_be_saved = {"wave": wave, "correction_map": correction_activity}
    io.pickle_dump(
        to_be_saved,
        open(self.dir_root + "CORRECTION_MAP/map_matching_activity.p", "wb"),
    )

    print("\nComputation of the new continua, wait ... \n")
    time.sleep(0.5)

    i = -1
    for j in tqdm(files):
        i += 1
        file = pd.read_pickle(j)
        output = {"continuum_" + continuum: new_continuum[i]}
        file["matching_activity"] = output
        file["matching_activity"]["parameters"] = {
            "reference_spectrum": reference,
            "sub_dico_used": sub_dico,
            "proxy_used": proxy_corr,
            "step": step + 1,
        }
        io.save_pickle(j, file)

    self.dico_actif = "matching_activity"
