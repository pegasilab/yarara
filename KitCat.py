#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Tue Nov 20 10:34:00 2018

@author: Cretignier Michael 
@university University of Geneva

"""
import matplotlib
import platform

if platform.system() == "Linux":
    matplotlib.use("Agg", force=True)
else:
    matplotlib.use("Qt5Agg", force=True)

import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import pickle
from tqdm import tqdm
import my_classes as myc
import my_functions as myf
from scipy.signal import argrelextrema
from colorama import Fore
from scipy.interpolate import interp1d
from matplotlib.widgets import Slider, Button, RadioButtons
import getopt
import sys
import time
import os

# =============================================================================
# PARAMETERS
# =============================================================================

star = None
Teff = None
instrument = None
feedback = False
ext = ""
generic = False

flux_max_telluric = 0.999  # minimum depth to considrer the tellurics
ratio_tel = 0.001  # stellar lines contamined removed
ratio_blend = 0.10  # stellar lines contaminated by blends removed
oversampling = 5  # oversampling of the s1d spectrum
vicinity = 5  # vicinity window for the local maxima algorithm
line_depth_min = 0.10  # minimum line depth to be considered in the stellar mask
line_depth_max = 0.95  # minimum line depth to be considered in the stellar mask
fwhm_ccf = 5.0  # FWHM of the CCF for the weighting of Xavier

min_wave_mask = 3000  # minimum wavelength of the mask (bad RASSINE normalisation)
max_wave_mask = 10000  # maximum wavelength of the mask (telluric oxygen)

excluded_region = [[6865, 6930], [7590, 7705]]  # oxygen bands rejected
telluric_killed = (
    0.20  # maximum telluric contamination allowed in relative contamination
)

weights = "depth_rel"  # line_depth, depth_rel or min_depth
kernel_smooth = None

if len(sys.argv) > 1:
    optlist, args = getopt.getopt(sys.argv[1:], "s:t:i:f:b:e:g:F:k:v:")
    for j in optlist:
        if j[0] == "-s":
            star = j[1]
        elif j[0] == "-t":
            Teff = int(j[1])
        elif j[0] == "-i":
            instrument = j[1]
        elif j[0] == "-f":
            feedback = bool(j[1])
        elif j[0] == "-b":
            min_wave_mask = int(j[1])
        elif j[0] == "-e":
            max_wave_mask = int(j[1])
        elif j[0] == "-g":
            generic = int(j[1])
        elif j[0] == "-F":
            fwhm_ccf = float(j[1])
        elif j[0] == "-k":
            kernel_smooth = int(j[1])
        elif j[0] == "-v":
            vicinity = int(j[1])

cwd = os.getcwd()
root = "/".join(cwd.split("/")[:-1])

directory = root + "/Yarara/" + star + "/data/s1d/" + instrument + "/KITCAT/"

print("Star being reduced : %s" % (star))

# =============================================================================
# IMPORTATION OF THE DATA PREPROCESS BY YARARA
# =============================================================================

kernel_smoothing_length = [5, kernel_smooth][int(kernel_smooth is not None)]
kernel_length_first_deri = 10
kernel_length_second_deri = 15

if fwhm_ccf > 15:
    line_depth_min = 0.03  # minimum line depth to be considered in the stellar mask
    line_depth_max = 0.40  # minimum line depth to be considered in the stellar mask
    vicinity = 1000

if not feedback:

    crit1_depth = -1.40
    crit2_width = -1.30

    diff_continuum_tresh = 1  # update : change for treshold=1 the 29.01.21 and using diff_continuum_rel, old value 0.25 with diff_continuum
    asym_ddflux_tresh = 0.25

    rel_depth_tresh = 0.15
    max_width_tresh = 13.0  # update : change for treshold=13 the 29.01.21, old value 10


Spectre = pd.read_pickle(directory + "kitcat_spectrum" + ext + ".p")
berv = Spectre["berv_mean"]  # telluric position from the master spectrum
berv_min = Spectre["berv_min"] - 4  # security of 4kms to account telluric width
berv_max = Spectre["berv_max"] + 4  # security of 4kms to account telluric width
if generic:
    print(Fore.YELLOW + "\n[INFO] Production of a generic mask \n" + Fore.WHITE)
    berv_min = -130  # to account for close star rv_sys +/100kms
    berv_max = 130

rv_sys = Spectre["rv_sys"]
if Teff is None:
    Teff = Spectre["t_eff"]
    if Teff is None:
        Teff = 5500

print(
    Fore.YELLOW
    + "\n[INFO] Stellar effective temperature : %.0f \n" % (Teff)
    + Fore.WHITE
)

if (Teff < 4500) | (Teff > 6500):
    print(
        Fore.RED
        + "\n[WARNING] Stellar effective temperature out of the VALD Teff range [4500,6500]"
        + Fore.WHITE
    )

print(
    Fore.YELLOW
    + "\n[INFO] Importation of the high SNR spectrum : LOADING \n"
    + Fore.WHITE
)

spectreI, gridi, spectre_tI, spectre_rejectedI = (
    Spectre["flux"],
    Spectre["wave"],
    Spectre["flux_telluric"],
    Spectre["rejected"],
)
spectreI *= Spectre["correction_factor"]

if feedback:
    fig = plt.figure(figsize=(14, 7))
    plt.subplots_adjust(left=0.10, bottom=0.25, top=0.95, hspace=0.30)
    plt.title("Selection of the smoothing kernel length")
    plt.plot(gridi, spectreI, color="b", alpha=0.4)
    (l1,) = plt.plot(
        gridi,
        myf.smooth(
            spectreI,
            int(kernel_smoothing_length),
            shape=["savgol", "gaussian"][int(fwhm_ccf > 15)],
        ),
        color="k",
    )
    axcolor = "whitesmoke"
    axsmoothing = plt.axes([0.2, 0.1, 0.40, 0.03], facecolor=axcolor)
    ssmoothing = Slider(
        axsmoothing,
        "Kernel length",
        1,
        50,
        valinit=int(kernel_smoothing_length),
        valstep=1,
    )

    resetax = plt.axes([0.8, 0.05, 0.1, 0.1])
    button = Button(resetax, "Reset", color=axcolor, hovercolor="0.975")

    rax = plt.axes([0.65, 0.05, 0.10, 0.10], facecolor=axcolor)
    radio = RadioButtons(
        rax, ("rectangular", "gaussian", "savgol"), active=[2, 1][int(fwhm_ccf > 15)]
    )

    class Index:
        shape = "savgol"

        def update(self, val):
            smoothing = ssmoothing.val
            l1.set_ydata(myf.smooth(spectreI, int(smoothing), shape=self.shape))
            fig.canvas.draw_idle()

        def change_kernel(self, label):
            self.shape = label

    callback = Index()
    ssmoothing.on_changed(callback.update)
    radio.on_clicked(callback.change_kernel)
    radio.on_clicked(callback.update)

    def reset(event):
        ssmoothing.reset()

    button.on_clicked(reset)

    plt.show(block=False)
    loop = myf.sphinx("Press ENTER when you are satisfied")
    plt.close("all")

    smoothing_shape = callback.shape
    smoothing_length = ssmoothing.val

    kernel_smoothing_length = smoothing_length
    kernel_smoothing_shape = smoothing_shape
else:
    kernel_smoothing_shape = "savgol"


spectrei = myf.smooth(spectreI, kernel_smoothing_length, shape=kernel_smoothing_shape)

grid = np.linspace(gridi.min(), gridi.max(), len(gridi) * oversampling)

Interpol = interp1d(
    gridi, spectrei, kind="cubic", bounds_error=False, fill_value="extrapolate"
)
spectre = Interpol(grid)

Interpol_conti = interp1d(
    gridi,
    Spectre["continuum"],
    kind="cubic",
    bounds_error=False,
    fill_value="extrapolate",
)
conti = Interpol_conti(grid)

Interpol2 = interp1d(
    gridi, spectre_tI, kind="cubic", bounds_error=False, fill_value="extrapolate"
)
spectre_t = Interpol2(grid)

if np.sum(spectre_rejectedI):
    Interpol_rejection = interp1d(
        gridi,
        spectre_rejectedI,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    spectre_rejected = Interpol_rejection(grid)
    spectre_rejected = spectre_rejected.astype("int")
else:
    spectre_rejected = np.zeros(len(gridi))

print(
    Fore.YELLOW + "\n[INFO] Importation of the high SNR spectrum : DONE \n" + Fore.WHITE
)

# =============================================================================
# EXTREMA LOCALISATION
# =============================================================================

print(
    Fore.YELLOW
    + "\n[INFO] Computation of all the local extrema in the spectrum : LOADING \n"
    + Fore.WHITE
)

index, flux = myf.local_max(-spectre, vicinity)
flux = -flux
index = index.astype("int")
wave = grid[index]


index2, flux2 = myf.local_max(spectre, vicinity)
index2 = index2.astype("int")
wave2 = grid[index2]

if wave[0] < wave2[0]:
    wave2 = np.insert(wave2, 0, grid[0])
    flux2 = np.insert(flux2, 0, spectre[0])
    index2 = np.insert(index2, 0, 0)

if wave[-1] > wave2[-1]:
    wave2 = np.insert(wave2, -1, grid[-1])
    flux2 = np.insert(flux2, -1, spectre[-1])
    index2 = np.insert(index2, -1, len(grid) - 1)

memory = np.hstack([-1 * np.ones(len(wave)), np.ones(len(wave2))])
stack_wave = np.hstack([wave, wave2])
stack_flux = np.hstack([flux, flux2])
stack_index = np.hstack([index, index2])

memory = memory[stack_wave.argsort()]
stack_flux = stack_flux[stack_wave.argsort()]
stack_wave = stack_wave[stack_wave.argsort()]
stack_index = stack_index[stack_index.argsort()]

if len(np.unique(memory[::2])) > 1:  # if two consecutive maxima minima
    trash, matrix = myf.clustering(memory, 0.01, 0)

    delete_liste = []
    for j in range(len(matrix)):
        numero = np.arange(matrix[j, 0], matrix[j, 1] + 2)
        fluxes = stack_flux[numero].argsort()
        if trash[j][0] == 1:
            delete_liste.append(numero[fluxes[0:-1]])
        else:
            delete_liste.append(numero[fluxes[1:]])
    delete_liste = np.hstack(delete_liste)

    memory = np.delete(memory, delete_liste)
    stack_flux = np.delete(stack_flux, delete_liste)
    stack_wave = np.delete(stack_wave, delete_liste)
    stack_index = np.delete(stack_index, delete_liste)

minima = np.where(memory == -1)[0]
maxima = np.where(memory == 1)[0]

index = stack_index[minima]
index2 = stack_index[maxima]
flux = stack_flux[minima]
flux2 = stack_flux[maxima]
wave = stack_wave[minima]
wave2 = stack_wave[maxima]

matrix_wave = np.vstack([wave, wave2[0:-1], wave2[1:]]).T
matrix_flux = np.vstack([flux, flux2[0:-1], flux2[1:]]).T
matrix_index = np.vstack([index, index2[0:-1], index2[1:]]).T

mask_line = ((1 - matrix_flux[:, 0]) > line_depth_min) & (
    (1 - matrix_flux[:, 0]) < line_depth_max
)

matrix_wave = matrix_wave[mask_line]
matrix_flux = matrix_flux[mask_line]
matrix_index = matrix_index[mask_line]

Depth = abs(matrix_flux[:, 0] - np.mean([matrix_flux[:, 1], matrix_flux[:, 2]], axis=0))

mask_line = Depth > 0.01  # to remove spurious line

matrix_wave = matrix_wave[mask_line]
matrix_flux = matrix_flux[mask_line]
matrix_index = matrix_index[mask_line]

print(
    Fore.YELLOW
    + "\n[INFO] Computation of all the local extrema in the spectrum : DONE \n"
    + Fore.WHITE
)

plt.figure(figsize=(21, 7))
plt.title("Step 1")
plt.plot(grid, spectre, color="k")
plt.scatter(
    grid[matrix_index[:, 0]], spectre[matrix_index[:, 0]], s=10, color="r", zorder=100
)
plt.scatter(
    grid[matrix_index[:, 1]], spectre[matrix_index[:, 1]], s=10, color="g", zorder=101
)
plt.scatter(
    grid[matrix_index[:, 2]], spectre[matrix_index[:, 2]], s=10, color="g", zorder=102
)
if feedback:
    plt.show()
if not feedback:
    plt.show(block=False)
    plt.close("all")

print(Fore.YELLOW + "\n[INFO] Number of lines : " + str(len(matrix_index)) + Fore.WHITE)

# =============================================================================
# WINDOWS COMPUTATION
# =============================================================================

HW = np.min(
    np.vstack(
        [
            abs(matrix_wave[:, 0] - matrix_wave[:, 1]),
            abs(matrix_wave[:, 0] - matrix_wave[:, 2]),
        ]
    ),
    axis=0,
)
HW_sign = np.diag(
    np.vstack(
        [matrix_wave[:, 0] - matrix_wave[:, 1], matrix_wave[:, 0] - matrix_wave[:, 2]]
    )[
        np.argmin(
            np.vstack(
                [
                    abs(matrix_wave[:, 0] - matrix_wave[:, 1]),
                    abs(matrix_wave[:, 0] - matrix_wave[:, 2]),
                ]
            ),
            axis=0,
        ),
        :,
    ]
)
HD = np.min(
    np.vstack(
        [
            abs(matrix_flux[:, 0] - matrix_flux[:, 1]),
            abs(matrix_flux[:, 0] - matrix_flux[:, 2]),
        ]
    ),
    axis=0,
)
Width = matrix_wave[:, 2] - matrix_wave[:, 1]
Depth = abs(matrix_flux[:, 0] - np.mean([matrix_flux[:, 1], matrix_flux[:, 2]], axis=0))


def graphique_clean2(
    par, par2, par_name="", par_name2="", mask=False, outliers_clipping=False
):
    t = myc.tableXY(np.log10(par), np.log10(par2))
    t.yerr = np.zeros(len(t.y))
    t.xerr = np.zeros(len(t.y))
    if outliers_clipping:
        t.rm_outliers(who="both")
    return t


if feedback:
    t = graphique_clean2(HD, HW)
    t.joint_plot(columns=[r"$log_{10}(Depth)$", r"$log_{10}(Width)$"])
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    (l1,) = plt.plot(
        np.linspace(xlim[0], xlim[1], 100),
        np.ones(100) * 0.5 * (ylim[0] + ylim[1]),
        color="r",
        lw=2.5,
    )
    (l2,) = plt.plot(
        np.ones(100) * 0.5 * (xlim[0] + xlim[1]),
        np.linspace(ylim[0], ylim[1], 100),
        color="r",
        lw=2.5,
    )
    l = plt.text(
        xlim[0] + 0.2 * (xlim[1] - xlim[0]),
        ylim[0] + 0.9 * (ylim[1] - ylim[0]),
        "Number of lines kept %.0f over %.0f"
        % (
            np.sum(
                (t.x > 0.5 * (xlim[0] + xlim[1])) & (t.y > 0.5 * (ylim[0] + ylim[1]))
            ),
            len(matrix_index),
        ),
    )

    class Index:
        y = 0.5 * (ylim[0] + ylim[1])
        x = 0.5 * (xlim[0] + xlim[1])

        def update_data(self, newx, newy):
            self.y = newy
            self.x = newx
            l1.set_ydata(np.ones(100) * self.y)
            l2.set_xdata(np.ones(100) * self.x)
            l.set_text(
                "Number of lines kept %.0f over %.0f"
                % (np.sum((t.x > self.x) & (t.y > self.y)), len(matrix_index))
            )
            plt.gcf().canvas.draw_idle()

    callback = Index()

    def onclick(event):
        newx = event.xdata
        newy = event.ydata
        if event.dblclick:
            callback.update_data(newx, newy)

    plt.gcf().canvas.mpl_connect("button_press_event", onclick)

    plt.show(block=False)
    loop = myf.sphinx("Press ENTER when you are satisfied")
    plt.close("all")

    crit1_depth = callback.x
    crit2_width = callback.y

mask = (np.log10(HD) > crit1_depth) & (np.log10(HW) > crit2_width)

Dico = {
    "wave": matrix_wave[:, 0],
    "wave_left": matrix_wave[:, 1],
    "wave_right": matrix_wave[:, 2],
    "idx": matrix_index[:, 0],
    "idx_left": matrix_index[:, 1],
    "idx_right": matrix_index[:, 2],
    "flux": matrix_flux[:, 0],
    "line_depth": 1 - matrix_flux[:, 0],
    "flux_left": matrix_flux[:, 1],
    "flux_right": matrix_flux[:, 2],
    "dist_continuum": (1 - np.min([matrix_flux[:, 1], matrix_flux[:, 2]], axis=0))
    / (1 - matrix_flux[:, 0]),
    "width": Width,
    "depth_rel": Depth,
    "min_width": HW,
    "min_width_signed": HW_sign,
    "min_depth": HD,
    "valid": mask,
    "line_nb": np.arange(len(mask)),
}

Dico = pd.DataFrame(Dico)
Dico_all = Dico.copy()
Dico_all["qc"] = 2

Dico = Dico.loc[Dico["valid"] == True]
Dico = Dico.drop(columns="valid")
Dico = Dico.reset_index(drop=True)

plt.figure(figsize=(21, 7))
plt.title("Step 2")
plt.plot(grid, spectre, color="k")
plt.scatter(Dico["wave"], Dico["flux"], s=10, color="r", zorder=10)
plt.scatter(Dico["wave_left"], Dico["flux_left"], s=10, color="b", zorder=10)
plt.scatter(Dico["wave_right"], Dico["flux_right"], s=10, color="g", zorder=10)
if feedback:
    plt.show()
if not feedback:
    plt.show(block=False)
    plt.close("all")

Dico["min_width"] = np.min(
    np.vstack(
        [abs(Dico["wave"] - Dico["wave_left"]), abs(Dico["wave"] - Dico["wave_right"])]
    ),
    axis=0,
)
Dico["min_width_signed"] = np.diag(
    np.vstack([Dico["wave"] - Dico["wave_left"], Dico["wave"] - Dico["wave_right"]])[
        np.argmin(
            np.vstack(
                [
                    abs(Dico["wave"] - Dico["wave_left"]),
                    abs(Dico["wave"] - Dico["wave_right"]),
                ]
            ),
            axis=0,
        ),
        :,
    ]
)
Dico["min_depth"] = np.min(
    np.vstack(
        [abs(Dico["flux"] - Dico["flux_left"]), abs(Dico["flux"] - Dico["flux_right"])]
    ),
    axis=0,
)
Dico["width"] = Dico["wave_right"] - Dico["wave_left"]
Dico["depth_rel"] = abs(
    Dico["flux"] - np.mean([Dico["flux_left"], Dico["flux_right"]], axis=0)
)
Dico["line_depth"] = 1 - Dico["flux"]

Dico = Dico.loc[Dico["idx_left"] != Dico["idx_right"]]
Dico = Dico.loc[Dico["idx_right"] > Dico["idx"]]
Dico = Dico.loc[Dico["idx_left"] < Dico["idx"]]
Dico = Dico.reset_index(drop=True)

Dico_all.loc[np.array(Dico["line_nb"]), "qc"] = 3

print(Fore.YELLOW + "\n[INFO] Number of lines : " + str(len(Dico)) + Fore.WHITE)

# computation EW

dwave = np.median(np.gradient(grid))
i0 = np.array(Dico["idx"])
i1 = np.array(Dico["idx_left"])
i2 = np.array(Dico["idx_right"])

EW = []
rejection_crit = []
for k in range(len(i1)):
    EW.append(
        dwave
        * np.sum(
            abs(0.5 * (spectre[i1[k] : i2[k]] + spectre[i1[k] + 1 : i2[k] + 1]) - 1)
        )
    )
    rejection_crit.append(sum(spectre_rejected[i1[k] : i2[k]]) == 0)
EW = np.array(EW)
rejection_crit = np.array(rejection_crit).astype("bool")
Dico["equivalent_width"] = EW
Dico["valid"] = rejection_crit

print(
    Fore.YELLOW
    + "\n[INFO] Nb lines rejected by exclusion regions : %.0f \n"
    % (sum(Dico["valid"] == False))
    + Fore.WHITE
)

Dico = Dico.loc[Dico["valid"] == True]
Dico = Dico.drop(columns="valid")
Dico = Dico.reset_index(drop=True)

i0 = np.array(Dico["idx"])
i1 = np.array(Dico["idx_left"])
i2 = np.array(Dico["idx_right"])

# computation RV weight
flux_gradient = np.gradient(spectre) / dwave
weight_vec = grid**2 * flux_gradient**2 / (spectre + 1e-6)

factor_fwhm = np.array(Dico["wave"] / np.min(grid))

di_left = np.round((i0 - i1) * dwave / Dico["wave"] * 3e5, 0)
di_right = np.round((i2 - i0) * dwave / Dico["wave"] * 3e5, 0)

i1_bis = i0 - np.round(fwhm_ccf * Dico["wave"] / (dwave * 3e5), 0).astype(
    "int"
)  # maximum 5kms window for weight computation
i2_bis = i0 + np.round(fwhm_ccf * Dico["wave"] / (dwave * 3e5), 0).astype(
    "int"
)  # maximum 5kms window for weight computation

weight_rv = []
weight_rv_xav = []
for k in range(len(i1)):
    indice_left = [i1[k], i1_bis[k]][int(i1[k] < i1_bis[k])]
    indice_right = [i2[k], i2_bis[k]][int(i2[k] < i2_bis[k])]
    weight_rv.append(np.sum(weight_vec[indice_left : indice_right + 1]))
    weight_rv_xav.append(np.sum(weight_vec[i1_bis[k] : i2_bis[k] + 1]))
weight_rv = np.array(weight_rv)
weight_rv /= np.nanpercentile(weight_rv, 95)
# weight_rv/= np.max(weight_rv)
weight_rv[weight_rv <= 0] = np.min(abs(weight_rv[weight_rv != 0]))
Dico["weight_rv"] = (weight_rv) ** (0.5)

weight_rv_xav = np.array(weight_rv_xav)
weight_rv_xav /= np.nanpercentile(weight_rv_xav, 95)
weight_rv_xav[weight_rv_xav <= 0] = np.min(abs(weight_rv_xav[weight_rv_xav != 0]))

Dico["weight_rv_sym"] = (weight_rv_xav) ** (0.5)

Dico["mean_grad"] = 0.5 * (
    abs(Dico["flux"] - Dico["flux_left"]) / (Dico["wave"] - Dico["wave_left"])
    + abs(Dico["flux"] - Dico["flux_right"]) / (Dico["wave_right"] - Dico["wave"])
)
Dico["mean_grad"] *= factor_fwhm
Dico["mean_grad"] /= np.max(Dico["mean_grad"])


# =============================================================================
# FIT OF THE CENTRAL WAVELENGTH POSITION
# =============================================================================

c_lum = 299.792e6


def fit_spectral_line_core(idx_center, grid, flux, continuum, width_rv=5, Plot=False):
    coordinates = np.zeros((len(idx_center), 67))
    loop = 0
    for j in tqdm(idx_center):
        windows = int(width_rv / (np.diff(grid[j : j + 2]) / grid[j] * c_lum / 1000))
        spectrum_line = flux[j - windows : j + windows + 1]
        grid_line = grid[j - windows : j + windows + 1]
        continuum_line = continuum[j - windows : j + windows + 1]

        mini1 = np.argmin(spectrum_line)
        std_depth = np.sqrt(
            abs(
                spectrum_line[mini1]
                * (1 + (spectrum_line[mini1] / continuum_line[mini1]) ** 2)
            )
        ) / abs(continuum_line[mini1])
        try:
            line = myc.tableXY(
                grid_line,
                spectrum_line / continuum_line,
                np.sqrt(abs(spectrum_line)) / continuum_line,
            )
            line.recenter()
            line.fit_poly(d=2)
            errors = np.sqrt(np.diag(line.cov))
            line.interpolate(replace=False)
            center = -0.5 * line.poly_coefficient[1] / line.poly_coefficient[0]
            depth = np.polyval(line.poly_coefficient, center) + line.ymean
            center = center + line.xmean
            coordinates[loop, 0] = center
            coordinates[loop, 1] = 1 - depth
            coordinates[loop, 2] = 1 - (line.y.min() + line.ymean)
            coordinates[loop, 3] = 1 - (line.y_interp.min() + line.ymean)
            coordinates[loop, 4] = std_depth
            coordinates[loop, 5] = 0.5 * np.sqrt(
                (errors[1] / line.poly_coefficient[0]) ** 2
                + (errors[0] * line.poly_coefficient[1] / line.poly_coefficient[0] ** 2)
                ** 2
            )
            coordinates[loop, 6] = line.chi2
        except:
            pass
        loop += 1

        if Plot:
            newx = np.linspace(np.min(line.x), np.max(line.x), 100)
            plt.plot(
                newx + line.xmean,
                np.polyval(line.poly_coefficient, newx) + line.ymean,
                color="r",
            )
            line.decenter()
            line.plot()
    return coordinates


coordinates = fit_spectral_line_core(
    np.array(Dico["idx"]), grid, spectre, np.ones(len(spectre)), width_rv=1, Plot=False
)  # get a visual feedback on the fit

Dico["wave_fitted"] = coordinates[:, 0]
Dico["wave_chi2"] = coordinates[:, 6]
Dico = Dico.loc[abs(Dico["wave"] - Dico["wave_fitted"]) < 0.03]

Dico["freq_mask0"] = myf.doppler_r(np.array(Dico["wave_fitted"]), rv_sys * 1000)[1]

Dico = Dico.loc[Dico["freq_mask0"] < max_wave_mask]
Dico = Dico.loc[Dico["freq_mask0"] > min_wave_mask]

for limites in excluded_region:
    Dico = Dico.loc[
        (Dico["freq_mask0"] < limites[0]) | (Dico["freq_mask0"] > limites[1])
    ]

Dico = Dico.reset_index(drop=True)

plt.figure(figsize=(21, 7))
plt.title("Step 3")
plt.plot(grid, spectre, color="k")
plt.scatter(Dico["wave"], Dico["flux"], s=10, color="r", zorder=10)
plt.scatter(Dico["wave_left"], Dico["flux_left"], s=10, color="b", zorder=10)
plt.scatter(Dico["wave_right"], Dico["flux_right"], s=10, color="g", zorder=10)
if feedback:
    plt.show()
if not feedback:
    plt.show(block=False)
    plt.close("all")

Dico_all.loc[np.array(Dico["line_nb"]), "qc"] = 4

print(Fore.YELLOW + "\n[INFO] Number of lines : " + str(len(Dico)) + Fore.WHITE)

print(
    Fore.YELLOW
    + "\n[INFO] Computation of all the local extrema in the spectrum : DONE \n"
    + Fore.WHITE
)

# =============================================================================
# MORPHOLOGICAL CRITERION
# =============================================================================

print(
    Fore.YELLOW
    + "\n[INFO] Computation of morphological parameters on each line : LOADING \n"
    + Fore.WHITE
)

Dico["tri_flux"] = (Dico["flux_left"] - Dico["flux_right"]) * (
    Dico["wave"] - Dico["wave_left"]
) / (Dico["wave_left"] - Dico["wave_right"]) + Dico[
    "flux_left"
]  # flux level intersection at the line center between the line linking left and right maxima
Dico["max_depth"] = np.max(
    np.vstack(
        [abs(Dico["flux"] - Dico["flux_left"]), abs(Dico["flux"] - Dico["flux_right"])]
    ),
    axis=0,
)  # maximum difference of flux between the line center and one of the maxima
Dico["max_width"] = np.max(
    np.vstack(
        [abs(Dico["wave"] - Dico["wave_left"]), abs(Dico["wave"] - Dico["wave_right"])]
    ),
    axis=0,
)  # maximum difference of wave between the line center and one of the maxima
Dico["diff_continuum"] = (
    abs(Dico["flux_left"] - Dico["flux_right"]) / Dico["max_depth"]
)  # absolute difference of flux between the two maxima normalised by the previous parameter
Dico["diff_continuum_signed"] = (Dico["flux_left"] - Dico["flux_right"]) / Dico[
    "max_depth"
]  # difference of flux between the two maxima normalised by the previous parameter
Dico["diff_continuum_rel"] = Dico["diff_continuum"] / Dico["depth_rel"]
Dico["tri_param"] = (
    abs(Dico["tri_flux"] - Dico["flux"]) / Dico["max_depth"]
)  # triangular parameter (for a line such that HW_max =HW : 1 = symetric, 0.5 = asymmetric)


def tri_area(x1, y1, x2, y2, x3, y3):
    c1 = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    c2 = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
    c3 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    p = (c1 + c2 + c3) / 2
    return np.sqrt(p * (p - c1) * (p - c2) * (p - c3))


tri_left = tri_area(
    Dico["wave_left"],
    Dico["flux_left"],
    Dico["wave"],
    Dico["flux"],
    Dico["wave"],
    Dico["tri_flux"],
)  # triangular area formed by left maxima, tri flux, line center
tri_right = tri_area(
    Dico["wave_right"],
    Dico["flux_right"],
    Dico["wave"],
    Dico["flux"],
    Dico["wave"],
    Dico["tri_flux"],
)  # triangular area formed by right maxima, tri flux, line center
Dico["tri_sum"] = tri_left + tri_right  # total area triangular
Dico["asym_area_tri"] = (tri_left - tri_right) / Dico[
    "tri_sum"
]  # area difference normalised by the total area

# computation of the EW

Dico["win_xav"] = (
    0.82 * 8 / 3e5 * Dico["wave"]
)  # initial windows of 8 pixel used by Xavier
area_left = []
area_right = []
for j in tqdm(Dico.index):
    area_left.append(np.mean(spectre[Dico.loc[j, "idx_left"] : Dico.loc[j, "idx"] + 1]))
    area_right.append(
        np.mean(spectre[Dico.loc[j, "idx"] : Dico.loc[j, "idx_right"] + 1])
    )

area_left, area_right = np.array(area_left), np.array(
    area_right
)  # mean flux using the windows of Xavier (but the the good line center)
Dico["mean_flux"] = (area_left + area_right) / 2
Dico["asym_mean_flux"] = (area_left - area_right) / Dico[
    "mean_flux"
]  # difference of flux level

area_left2 = []
area_right2 = []
for j in tqdm(Dico.index):
    windows = np.min(
        [
            Dico.loc[j, "idx_right"] - Dico.loc[j, "idx"],
            Dico.loc[j, "idx"] - Dico.loc[j, "idx_left"],
        ]
    )
    area_left2.append(
        np.mean(spectre[Dico.loc[j, "idx"] - windows : Dico.loc[j, "idx"] + 1])
    )
    area_right2.append(
        np.mean(spectre[Dico.loc[j, "idx"] : Dico.loc[j, "idx"] + windows + 1])
    )

area_left2, area_right2 = np.array(area_left2), np.array(
    area_right2
)  # mean flux using the windows of Xavier (but the the good line center)
Dico["mean_flux2"] = (area_left2 + area_right2) / 2
Dico["asym_mean_flux2"] = (area_left2 - area_right2) / Dico[
    "mean_flux2"
]  # difference of flux level

# derivative criterion

diff_spectrum = np.gradient(spectre) / np.gradient(grid)

# begin of the smoothing kernel

if feedback:
    fig = plt.figure(figsize=(14, 7))
    plt.subplots_adjust(left=0.10, bottom=0.25, top=0.95, hspace=0.30)
    plt.title("Selection of the smoothing kernel length")
    plt.plot(grid, diff_spectrum, color="b", alpha=0.4)
    (l1,) = plt.plot(
        grid,
        myf.smooth(diff_spectrum, int(kernel_length_first_deri), shape="rectangular"),
        color="k",
    )
    axcolor = "whitesmoke"
    axsmoothing = plt.axes([0.2, 0.1, 0.40, 0.03], facecolor=axcolor)
    ssmoothing = Slider(
        axsmoothing,
        "Kernel length",
        1,
        40,
        valinit=int(kernel_length_first_deri),
        valstep=1,
    )

    resetax = plt.axes([0.8, 0.05, 0.1, 0.1])
    button = Button(resetax, "Reset", color=axcolor, hovercolor="0.975")

    rax = plt.axes([0.65, 0.05, 0.10, 0.10], facecolor=axcolor)
    radio = RadioButtons(rax, ("rectangular", "gaussian", "savgol"), active=0)

    class Index:
        shape = "rectangular"

        def update(self, val):
            smoothing = ssmoothing.val
            l1.set_ydata(myf.smooth(diff_spectrum, int(smoothing), shape=self.shape))
            fig.canvas.draw_idle()

        def change_kernel(self, label):
            self.shape = label

    callback = Index()
    ssmoothing.on_changed(callback.update)
    radio.on_clicked(callback.change_kernel)
    radio.on_clicked(callback.update)

    def reset(event):
        ssmoothing.reset()

    button.on_clicked(reset)

    plt.show(block=False)
    loop = myf.sphinx("Press ENTER when you are satisfied")
    plt.close("all")

    smoothing_shape = callback.shape
    smoothing_length = ssmoothing.val

    kernel_length_first_deri = smoothing_length
    kernel_shape_first_deri = smoothing_shape
else:
    kernel_shape_first_deri = "gaussian"

diff_spectrum = myf.smooth(
    diff_spectrum, kernel_length_first_deri, shape=kernel_shape_first_deri
)

# end of the smoothing

diff_min = []
diff_max = []
contrast_deri = []
contrast_deri_norm = []
cm = []
cm_norm = []
cm_left = []
cm_right = []
for j in tqdm(Dico.index):
    mini = diff_spectrum[Dico.loc[j, "idx_left"] : Dico.loc[j, "idx_right"] + 1].min()
    maxi = diff_spectrum[Dico.loc[j, "idx_left"] : Dico.loc[j, "idx_right"] + 1].max()
    diff_min.append(mini)  # highest derivtive left wing (HDL)
    diff_max.append(maxi)  # highest derivtive right wing (HDR)
    contrast_deri.append(
        (mini + maxi) / 2
    )  # mean of previous parameter (symmetric = 0)
    contrast_deri_norm.append(
        (mini + maxi) / (2 * abs(mini - maxi))
    )  # previous parameter normalised by the absolute derivative
    mini_wave = diff_spectrum[
        int(Dico.loc[j, "idx_left"]) : int(Dico.loc[j, "idx_right"] + 1)
    ].argmin()
    maxi_wave = diff_spectrum[
        int(Dico.loc[j, "idx_left"]) : int(Dico.loc[j, "idx_right"] + 1)
    ].argmax()
    cm_left.append(
        grid[int(Dico.loc[j, "idx_left"]) : int(Dico.loc[j, "idx_right"] + 1)][
            mini_wave
        ]
        - grid[int(Dico.loc[j, "idx"])]
    )  # distance from line center of HDL
    cm_right.append(
        grid[int(Dico.loc[j, "idx_left"]) : int(Dico.loc[j, "idx_right"] + 1)][
            maxi_wave
        ]
        - grid[int(Dico.loc[j, "idx"])]
    )  # distance from line center of HDR
    center_of_mass = (
        np.mean(
            [
                grid[int(Dico.loc[j, "idx_left"]) : int(Dico.loc[j, "idx_right"] + 1)][
                    mini_wave
                ],
                grid[int(Dico.loc[j, "idx_left"]) : int(Dico.loc[j, "idx_right"] + 1)][
                    maxi_wave
                ],
            ]
        )
        - grid[int(Dico.loc[j, "idx"])]
    )  # mean of previous parameters
    center_of_mass_norm = center_of_mass / abs(
        grid[int(Dico.loc[j, "idx_left"]) : int(Dico.loc[j, "idx_right"] + 1)][
            mini_wave
        ]
        - grid[int(Dico.loc[j, "idx_left"]) : int(Dico.loc[j, "idx_right"] + 1)][
            maxi_wave
        ]
    )  # previous parameter normalised
    cm.append(center_of_mass)
    cm_norm.append(center_of_mass_norm)


diff_min, diff_max, contrast_deri, contrast_deri_norm = (
    np.array(diff_min),
    np.array(diff_max),
    np.array(contrast_deri),
    np.array(contrast_deri_norm),
)
cm = np.array(cm)
cm_norm = np.array(cm_norm)
cm_left = np.array(cm_left)
cm_right = np.array(cm_right)

# second derivative

diff_diff_spectrum = np.gradient(diff_spectrum) / np.gradient(grid)

# begin of the smoothing

if feedback:
    fig = plt.figure(figsize=(14, 7))
    plt.subplots_adjust(left=0.10, bottom=0.25, top=0.95, hspace=0.30)
    plt.title("Selection of the smoothing kernel length")
    plt.plot(grid, diff_diff_spectrum, color="b", alpha=0.4)
    (l1,) = plt.plot(
        grid,
        myf.smooth(
            diff_diff_spectrum, int(kernel_length_second_deri), shape="rectangular"
        ),
        color="k",
    )
    axcolor = "whitesmoke"
    axsmoothing = plt.axes([0.2, 0.1, 0.40, 0.03], facecolor=axcolor)
    ssmoothing = Slider(
        axsmoothing,
        "Kernel length",
        1,
        40,
        valinit=int(kernel_length_second_deri),
        valstep=1,
    )

    resetax = plt.axes([0.8, 0.05, 0.1, 0.1])
    button = Button(resetax, "Reset", color=axcolor, hovercolor="0.975")

    rax = plt.axes([0.65, 0.05, 0.10, 0.10], facecolor=axcolor)
    radio = RadioButtons(rax, ("rectangular", "gaussian", "savgol"), active=0)

    class Index:
        shape = "rectangular"

        def update(self, val):
            smoothing = ssmoothing.val
            l1.set_ydata(
                myf.smooth(diff_diff_spectrum, int(smoothing), shape=self.shape)
            )
            fig.canvas.draw_idle()

        def change_kernel(self, label):
            self.shape = label

    callback = Index()
    ssmoothing.on_changed(callback.update)
    radio.on_clicked(callback.change_kernel)
    radio.on_clicked(callback.update)

    def reset(event):
        ssmoothing.reset()

    button.on_clicked(reset)

    plt.show(block=False)
    loop = myf.sphinx("Press ENTER when you are satisfied")
    plt.close("all")

    smoothing_shape = callback.shape
    smoothing_length = ssmoothing.val

    kernel_length_second_deri = smoothing_length
    kernel_shape_second_deri = smoothing_shape
else:
    kernel_shape_second_deri = "gaussian"

diff_diff_spectrum = myf.smooth(
    diff_diff_spectrum, kernel_length_second_deri, shape=kernel_shape_second_deri
)

# end of the smoothing

dd_min = np.where(
    (
        np.in1d(np.arange(len(grid)), argrelextrema(diff_diff_spectrum, np.less)[0])
        & (diff_diff_spectrum < 0)
    )
    == True
)[
    0
]  # find all dd minima with dd<0 (clue to define the windows later)
dd_max = np.where(
    (
        np.in1d(np.arange(len(grid)), argrelextrema(diff_diff_spectrum, np.greater)[0])
        & (diff_diff_spectrum > 0)
    )
    == True
)[
    0
]  # find all dd maxima with dd>0 (clue to identify blends later)

dd_min = np.hstack([0, dd_min, len(grid)])

distance = dd_min - np.array(Dico["idx"])[:, np.newaxis]

pos = np.array(
    [np.argmax(distance[j, distance[j, :] < 0]) for j in range(len(distance))]
)
dd_left = dd_min[pos]
dd_right = dd_min[pos + 1]

# fixe the border to min width if no minima
for j in range(len(dd_left)):
    if dd_left[j] < Dico.loc[j, "idx_left"]:
        dd_left[j] = Dico.loc[j, "idx_left"]
    if dd_right[j] > Dico.loc[j, "idx_right"]:
        dd_right[j] = Dico.loc[j, "idx_right"]

save_diff = dd_max - dd_right[:, np.newaxis]
save_diff2 = dd_max - dd_left[:, np.newaxis]
num_dd_max = np.sum(
    (save_diff * save_diff2) < 0, axis=1
)  # find the number of maxima with dd>0 between the two dd minima (blend clue)
num_dd_max2 = np.sum(
    (
        (dd_max - np.array(Dico["idx_right"])[:, np.newaxis])
        * (dd_max - np.array(Dico["idx_left"])[:, np.newaxis])
    )
    < 0,
    axis=1,
)  # same as before with the two flux maxima (blend clue)
ddwave_left = grid[dd_left]
ddwave_right = grid[dd_right]
ddflux_left = spectre[dd_left]
ddflux_right = spectre[dd_right]
asym_ddflux = ddflux_left - ddflux_right
asym_ddflux_norm = asym_ddflux / Dico["max_depth"]

HW_dd = np.min(
    np.vstack([abs(Dico["wave"] - ddwave_left), abs(Dico["wave"] - ddwave_right)]),
    axis=0,
)
HW_dd_max = np.max(
    np.vstack([abs(Dico["wave"] - ddwave_left), abs(Dico["wave"] - ddwave_right)]),
    axis=0,
)
HW_dd_sign = np.diag(
    np.vstack([Dico["wave"] - ddwave_left, Dico["wave"] - ddwave_right])[
        np.argmin(
            np.vstack(
                [abs(Dico["wave"] - ddwave_left), abs(Dico["wave"] - ddwave_right)]
            ),
            axis=0,
        ),
        :,
    ]
)


nb_turnover = np.nan * np.ones(len(Dico.index))
mean_bis = np.nan * np.ones(len(Dico.index))
bis = np.nan * np.ones(len(Dico.index))
bis_contrast = np.nan * np.ones(len(Dico.index))
mean_abs_bis = np.nan * np.ones(len(Dico.index))
slope_bis_w = np.nan * np.ones(len(Dico.index))
intercept_bis_w = np.nan * np.ones(len(Dico.index))
med_bis = np.nan * np.ones(len(Dico.index))
med_abs_bis = np.nan * np.ones(len(Dico.index))
for j in tqdm(Dico.index):
    try:
        line = myc.tableXY(
            grid[int(Dico.loc[j, "idx_left"]) : int(Dico.loc[j, "idx_right"]) + 1],
            spectre[int(Dico.loc[j, "idx_left"]) : int(Dico.loc[j, "idx_right"]) + 1],
        )
        line.my_bisector(oversampling=5)
        bisec = myc.tableXY(
            line.bisector[1:, 1],
            line.bisector[1:, 0] - line.bisector[0, 0],
            line.bisector[1:, 2],
        )
        bisec.fit_line(perm=100)
        slope_bis_w[j] = bisec.lin_slope_w  # linear weighted slope of the bisector
        intercept_bis_w[
            j
        ] = bisec.lin_intercept_w  # linear weighted intersection of the bisector
        smoothed = myf.smooth(
            line.bisector[:, 0], 10
        )  # presmoothing to compute the local maxima
        mask2 = abs(np.diff(smoothed)) < 30
        mask2 = np.insert(mask2, -1, False)
        id1 = argrelextrema(smoothed[mask2], np.greater)[0]
        id2 = argrelextrema(-smoothed[mask2], np.greater)[0]
        nb_turnover[j] = sum(
            abs(np.diff(smoothed[mask2][np.hstack([id1, id2])])) > 0.002
        )  # only keeping significant turnover
        mean_bis[j] = (
            np.sum(line.bisector[1:, 0] * line.bisector[1:, 2] ** -2)
            / np.sum(line.bisector[1:, 2] ** -2)
            - line.bisector[0, 0]
        )  # weighted average of the bisector
        mean_abs_bis[j] = np.sum(
            abs(line.bisector[1:, 0] - line.bisector[0, 0]) * line.bisector[1:, 2] ** -2
        ) / np.sum(
            line.bisector[1:, 2] ** -2
        )  # weighted average of the absolute bisector
        bis[j] = (
            line.bisector[np.argmax(abs(line.bisector[:, 0] - line.bisector[0, 0])), 0]
            - line.bisector[0, 0]
        )  # bisspan
        bis_contrast[j] = np.max(-line.bisector[:, 0] + line.bisector[0, 0]) - np.min(
            -line.bisector[:, 0] + line.bisector[0, 0]
        )  # abs bisspan
    except (ValueError, IndexError):
        pass

print(
    Fore.YELLOW
    + "\n[INFO] Computation of morphological parameters on each line : DONE \n"
    + Fore.WHITE
)


Dico_new = {
    "min_deri": diff_min,
    "max_deri": diff_max,
    "contrast_deri": contrast_deri,
    "contrast_deri_norm": contrast_deri_norm,
    "cm_deri": cm,
    "cm_deri_norm": cm_norm,
    "cm_left": cm_left,
    "cm_right": cm_right,
    "nb_turn_bis": nb_turnover,
    "mean_bis": mean_bis,
    "bis": bis,
    "bis_contrast": bis_contrast,
    "mean_abs_bis": mean_abs_bis,
    "slope_bis_w": slope_bis_w,
    "intercept_bis_w": intercept_bis_w,
    "num_dd_max": num_dd_max,
    "num_dd_max2": num_dd_max2,
    "ddwave_left": ddwave_left,
    "ddwave_right": ddwave_right,
    "asym_ddflux": asym_ddflux,
    "asym_ddflux_norm": asym_ddflux_norm,
    "min_width_dd": HW_dd,
    "min_width_dd_signed": HW_dd_sign,
    "max_width_dd": HW_dd_max,
}


Dico_new = pd.DataFrame(Dico_new)
Dico = pd.concat([Dico, Dico_new], axis=1)

Dico["win_mic"] = Dico[["min_width_dd", "min_width"]].mean(axis=1)
Dico["win_asym_left"] = np.nan
Dico["win_asym_right"] = np.nan

w = np.array(Dico["wave"])
win = np.array(Dico["win_mic"])
left_i = np.array(Dico["idx_left"])
right_i = np.array(Dico["idx_right"])
for k in tqdm(range(len(w))):
    i, j = w[k], win[k]
    sub_grid = grid[int(left_i[k]) : int(right_i[k]) + 1]
    sub_spectre = spectre[int(left_i[k]) : int(right_i[k]) + 1]
    left = int(myf.find_nearest(sub_grid, i - j)[0])
    right = int(myf.find_nearest(sub_grid, i + j)[0])
    center = int(myf.find_nearest(sub_grid, i)[0])
    f1 = myf.find_nearest(sub_spectre[left + 1 : center + 1], sub_spectre[right])
    f2 = myf.find_nearest(sub_spectre[center:right], sub_spectre[left])
    if abs(f2[-1]) < abs(f1[-1]):
        Dico.loc[k, "win_asym_right"] = abs(i - sub_grid[center:right][f2[0]])
        Dico.loc[k, "win_asym_left"] = j
    if abs(f2[-1]) > abs(f1[-1]):
        Dico.loc[k, "win_asym_right"] = j
        Dico.loc[k, "win_asym_left"] = abs(i - sub_grid[left + 1 : center + 1][f1[0]])


plt.figure(figsize=(21, 7))
plt.title("Step 4")
plt.plot(grid, spectre, color="k")
plt.scatter(Dico["wave"], Dico["flux"], s=10, color="r", zorder=10)
plt.scatter(Dico["wave_left"], Dico["flux_left"], s=10, color="b", zorder=10)
plt.scatter(Dico["wave_right"], Dico["flux_right"], s=10, color="g", zorder=10)
if feedback:
    plt.show()
if not feedback:
    plt.show(block=False)
    plt.close("all")


print(Fore.YELLOW + "\n[INFO] Number of lines : " + str(len(Dico)) + Fore.WHITE)

print(
    Fore.YELLOW
    + "\n[INFO] Computation of the telluric contamination : LOADING \n"
    + Fore.WHITE
)


indext, fluxt = myf.local_max(-spectre_t, vicinity)
fluxt = -fluxt
indext = indext.astype("int")
wavet = grid[indext]

mask = fluxt < flux_max_telluric
indext = indext[mask]
wavet = wavet[mask]
fluxt = fluxt[mask]

wave_tel = []
depth_tel = []
contam = []
for i in tqdm(Dico.index):
    mask2 = ((1 - fluxt) / (Dico.loc[i, "min_depth"])) > ratio_tel
    max_t = myf.doppler_r(wavet[mask2], (berv_max - berv) * 1000)[0]
    min_t = myf.doppler_r(wavet[mask2], (berv_min - berv) * 1000)[0]

    max_t = wavet[mask2] * ((1 + 1.55e-8) * (1 + (berv_max - berv) / 299792.458))
    min_t = wavet[mask2] * ((1 + 1.55e-8) * (1 + (berv_min - berv) / 299792.458))

    c1 = np.sign((Dico.loc[i, "wave"] + Dico.loc[i, "win_mic"]) - min_t)
    c2 = np.sign((Dico.loc[i, "wave"] + Dico.loc[i, "win_mic"]) - max_t)
    c3 = np.sign((Dico.loc[i, "wave"] - Dico.loc[i, "win_mic"]) - min_t)
    c4 = np.sign((Dico.loc[i, "wave"] - Dico.loc[i, "win_mic"]) - max_t)

    if np.product((c1 == c2) * (c1 == c3) * (c1 == c4)) == 1:
        wave_tel.append(np.nan)
        depth_tel.append(0)
        contam.append(0)
    else:
        loc_tellu = np.where((c1 == c2) * (c1 == c3) * (c1 == c4) == False)[0]
        max_cont = (1 - fluxt)[mask2][loc_tellu].argmax()
        wave_tel.append(wavet[mask2][loc_tellu][max_cont])
        depth_tel.append((1 - fluxt)[mask2][loc_tellu].max())
        contam.append(1)

wave_tel = np.array(wave_tel)
depth_tel = np.array(depth_tel)
contam = np.array(contam)

Dico["telluric"] = contam
Dico["telluric_depth"] = depth_tel
Dico["telluric_wave"] = wave_tel
Dico["rel_contam"] = Dico["telluric_depth"] / Dico["line_depth"]
Dico = Dico.loc[Dico["rel_contam"] <= telluric_killed]

Dico = Dico.reset_index(drop=True)

plt.figure(figsize=(21, 7))
plt.title("Step 5")
plt.plot(grid, spectre, color="k")
plt.scatter(Dico["wave"], Dico["flux"], s=10, color="r", zorder=10)
plt.scatter(Dico["wave_left"], Dico["flux_left"], s=10, color="b", zorder=10)
plt.scatter(Dico["wave_right"], Dico["flux_right"], s=10, color="g", zorder=10)
if feedback:
    plt.show()
if not feedback:
    plt.show(block=False)
    plt.close("all")

Dico_all.loc[np.array(Dico["line_nb"]), "qc"] = 6

print(Fore.YELLOW + "\n[INFO] Number of lines : " + str(len(Dico)) + Fore.WHITE)

print(
    Fore.YELLOW
    + "\n[INFO] Computation of the telluric contamination : DONE \n"
    + Fore.WHITE
)

Dico2 = {"spectre": Spectre, "catalogue": Dico}
Dico2["parameters"] = {
    "rv_sys": rv_sys,
    "berv_mean": berv,
    "berv_min": berv_min,
    "berv_max": berv_max,
    "flux_max_telluric": flux_max_telluric,
    "ratio_telluric": ratio_tel,
    "oversampling": oversampling,
    "kernel_length_f0": kernel_smoothing_length,
    "kernel_length_f1": kernel_length_first_deri,
    "kernel_length_f2": kernel_length_second_deri,
    "kernel_f0": kernel_smoothing_shape,
    "kernel_f1": kernel_shape_first_deri,
    "kernel_f2": kernel_shape_second_deri,
    "vicinity": vicinity,
    "weights": weights,
}

Dico_backup = Dico.copy()

myf.pickle_dump(Dico2, open(directory + "kitcat_mask_" + star + ext + ".p", "wb"))
print(
    Fore.YELLOW
    + "\n[INFO] The table has been saved under : %s \n"
    % (directory + "kitcat_mask_" + star + ext + ".p")
    + Fore.WHITE
)

# =============================================================================
# VALD CROSS MATCHING
# =============================================================================

all_vald_table = pd.read_pickle(root + "/Python/VALD/vald_database.p")
loc_temp = myf.find_nearest(
    np.array([T[0:4] for T in all_vald_table.keys()]).astype("int"), Teff
)[0][0]
vald_table = all_vald_table[list(all_vald_table.keys())[loc_temp]]
vald_table = vald_table.loc[
    (vald_table["wave_vald"] > np.min(Dico["freq_mask0"]))
    & (vald_table["wave_vald"] < np.max(Dico["freq_mask0"]))
]
twins = vald_table["wave_vald"].value_counts()
twins = twins.loc[twins > 1]
for twin in twins.keys():
    vald_table = vald_table.loc[vald_table["wave_vald"] != twin]

vald_table = vald_table.reset_index()
wave_model = myf.doppler_r(np.array(vald_table["wave_vald"]), rv_sys * 1000)[0]
depth_model = np.array(vald_table["depth_vald"])

left = np.array(Dico["wave_left"])
right = np.array(Dico["wave_right"])

wave1 = wave_model > left[:, np.newaxis]
wave2 = wave_model < right[:, np.newaxis]
crit = wave1 * wave2
nb_blends = np.sum(crit, axis=1) - 1
Dico["nb_blends"] = nb_blends

vald_table["depth_vald_corrected"] = vald_table["depth_vald"]
vald_table = vald_table.reset_index(drop=True)

rassine_accuracy = 0.01 + 0.07 * np.array(Dico["wave"] < 4500)
Dico["continuum"] = conti[np.array(Dico["idx"])]
Dico["line_depth_std"] = (
    np.sqrt(1 + Dico["continuum"] ** 2)
    * np.sqrt(abs(Dico["line_depth"] * Dico["continuum"]))
    / Dico["continuum"] ** 2
)
Dico["line_depth_std"] = np.sqrt(Dico["line_depth_std"] ** 2 + rassine_accuracy**2)

Dico_unblended = Dico.loc[Dico["nb_blends"] == 0]
Dico_unblended = Dico_unblended.reset_index(drop=True)

my_wave = np.array(Dico_unblended["wave_fitted"])
my_depth = np.array(Dico_unblended["line_depth"])

match = np.sum(crit[nb_blends == 0] * np.arange(len(wave_model)), axis=1)
match_vald = my_wave - wave_model[match]
loop = "n"

typ_shift = [-0.03, 0.03]

if len(match) != len(Dico_unblended):
    print(
        Fore.RED
        + "\n[WARNING] Problem during ther matching of VALD and KitCat, unable to correct the abundances"
        + Fore.WHITE
    )
else:
    Dico_unblended["element"] = np.array(vald_table.loc[match, "element"])
    Dico_unblended["log_gf"] = np.array(vald_table.loc[match, "log_gf"])
    Dico_unblended["vald_depth"] = np.array(vald_table.loc[match, "depth_vald"])
    Dico_unblended["diff_vald_wave"] = abs(match_vald)
    Dico_unblended["diff_vald_depth"] = (
        Dico_unblended["line_depth"] - Dico_unblended["vald_depth"]
    )
    typ_shift = [
        np.median(Dico_unblended["diff_vald_wave"])
        - 1.5 * myf.IQ(Dico_unblended["diff_vald_wave"]),
        np.median(Dico_unblended["diff_vald_wave"])
        + 1.5 * myf.IQ(Dico_unblended["diff_vald_wave"]),
    ]

    Dico_unblended = Dico_unblended.loc[
        (Dico_unblended["diff_vald_wave"] < 0.04)
        & (Dico_unblended["vald_depth"] > 0.02)
    ]
    Dico_unblended = Dico_unblended.reset_index(drop=True)
    count_element = Dico_unblended["element"].value_counts()
    element_kept = count_element.loc[count_element >= 20].keys()
    size = np.where((np.array([x**2 for x in range(8)]) > len(element_kept)) == 1)[0][
        0
    ]
    if len(element_kept):
        plt.figure(figsize=(12, 15))
        for idx_plot, elem in enumerate(element_kept):
            plt.subplot(size, size, idx_plot + 1)
            plt.title(elem)
            vald_depth = Dico_unblended.loc[
                Dico_unblended["element"] == elem, "vald_depth"
            ]
            my_depth = Dico_unblended.loc[
                Dico_unblended["element"] == elem, "diff_vald_depth"
            ]
            my_depth_std = Dico_unblended.loc[
                Dico_unblended["element"] == elem, "line_depth_std"
            ]
            line = myc.tableXY(vald_depth, my_depth, my_depth_std)
            line.clip(min=[0.03, None], replace=True)
            line.rm_outliers()
            mask = myf.rm_outliers(line.yerr, m=2, kind="inter")[0]
            line.masked(mask)
            line.x = np.insert(line.x, len(line.x), 1)
            line.y = np.insert(line.y, len(line.y), 0)
            line.yerr = np.insert(line.yerr, len(line.yerr), 0.001)
            line.xerr = np.insert(line.xerr, len(line.xerr), 0)
            line.plot()

            plt.scatter(
                vald_depth,
                my_depth,
                c=Dico_unblended.loc[Dico_unblended["element"] == elem, "wave"],
                cmap="jet",
                zorder=2,
            )
            line.fit_poly(Draw=True, d=3, color="k")

            vald_table.loc[
                vald_table["element"] == elem, "depth_vald_corrected"
            ] = vald_table.loc[
                vald_table["element"] == elem, "depth_vald"
            ] + np.polyval(
                line.poly_coefficient,
                vald_table.loc[vald_table["element"] == elem, "depth_vald"],
            )
            plt.scatter(
                vald_table.loc[vald_table["element"] == elem, "depth_vald"],
                np.polyval(
                    line.poly_coefficient,
                    vald_table.loc[vald_table["element"] == elem, "depth_vald"],
                ),
                color="k",
                alpha=0.2,
            )
            plt.xlabel("VALD depth")
            plt.ylabel(r"$\Delta$ Line depth")
            plt.xlim(0, 1)
            plt.ylim(-1, 1)
            plt.plot([0, 1], [0, 0], color="gray", alpha=0.5)
        plt.subplots_adjust(
            top=0.95, left=0.08, right=0.96, bottom=0.10, wspace=0.45, hspace=0.55
        )
        plt.savefig(directory + "Correction_depth.png")
        plt.show(block=False)
        if feedback:
            loop = myf.sphinx("Do you confirm this correction ? (y/n)", rep=["y", "n"])
        else:
            loop = "y"
        plt.close("all")

vald_table.loc[vald_table["depth_vald_corrected"] < 0, "depth_vald_corrected"] = 0
vald_table.loc[vald_table["depth_vald_corrected"] > 1, "depth_vald_corrected"] = 1

if loop == "n":
    vald_table["depth_vald_corrected"] = vald_table["depth_vald"]
else:
    plt.figure()
    plt.plot(grid, spectre, color="k")
    wave_model = myf.doppler_r(np.array(vald_table["wave_vald"]), rv_sys * 1000)[0]
    depth_model = np.array(vald_table["depth_vald"])
    depth_model_corrected = np.array(vald_table["depth_vald_corrected"])

    uncorrected = myc.tableXY(wave_model, 1 - depth_model)
    corrected = myc.tableXY(wave_model, 1 - depth_model_corrected)
#    plt.xlim(6000,6001)
#    uncorrected.myscatter(liste=np.array(vald_table['element']),num=False,factor=100)
#    corrected.myscatter(liste=np.array(vald_table['element']),num=False,factor=100)
#    plt.xlim(5000,6001)
#    plt.show(block=False)

wave_model = myf.doppler_r(np.array(vald_table["wave_vald"]), rv_sys * 1000)[0]
depth_model = np.array(vald_table["depth_vald_corrected"])

left = np.array(Dico["wave"] - Dico["win_mic"])
right = np.array(Dico["wave"] + Dico["win_mic"])

wave1 = wave_model > left[:, np.newaxis]
wave2 = wave_model < right[:, np.newaxis]
crit = wave1 * wave2
nb_blends = np.sum(crit, axis=1) - 1
Dico["nb_blends"] = nb_blends

main_line_index = []
contam = []
delta_wave = []
for j in range(len(crit)):
    loc_blend = np.where(crit[j])[0]
    diff_wave = Dico.loc[j, "wave_fitted"] - wave_model[loc_blend]
    diff_depth = Dico.loc[j, "line_depth"] - depth_model[loc_blend]
    main_line = (diff_wave > typ_shift[0]) & (diff_wave < typ_shift[1])
    if sum(main_line):
        diff_wave2 = diff_wave[main_line]
        diff_depth2 = diff_depth[main_line]
        final_main_line = np.argmin(np.sqrt(diff_wave2**2 + diff_depth2**2))
        main_line_index.append(loc_blend[main_line][final_main_line])
        delta_wave.append(diff_wave[main_line][final_main_line])
        if len(loc_blend) > 1:
            blend_line = np.ones(len(loc_blend)).astype("bool")
            blend_line[np.arange(len(loc_blend))[main_line][final_main_line]] = False
            highest_blend = np.max(depth_model[loc_blend][blend_line])
            contam.append(highest_blend / Dico.loc[j, "line_depth"])
        else:
            contam.append(0)
    else:
        main_line_index.append(-99.9)
        contam.append(-99.9)
        delta_wave.append(np.nan)


delta_wave = np.array(delta_wave)
outliers = myf.rm_outliers(delta_wave, m=2, kind="inter")[0]
delta_wave[~outliers] = np.nan
main_line_index = np.array(main_line_index)
main_line_index[np.arange(len(outliers))[~outliers]] = -99.9


Dico["contam_blend"] = contam
kw = [
    "wave_vald",
    "depth_vald",
    "depth_vald_corrected",
    "element",
    "atomic_number",
    "log_gf",
    "E_low",
    "E_up",
    "J_low",
    "J_up",
    "lande_low",
    "lande_up",
    "lande_mean",
    "damp_rad",
    "damp_stark",
    "damp_waals",
    "zeeman",
    "ionisation_energy",
    "mass_ion",
]
for keyword in kw:
    Dico[keyword] = np.nan

i = -1
for j in tqdm(main_line_index):
    i += 1
    if j != -99.9:
        for keyword in kw:
            Dico.loc[i, keyword] = vald_table.loc[j, keyword]

Dico["diff_wave_vald"] = Dico["freq_mask0"] - Dico["wave_vald"]
Dico["blend_crit"] = (
    (Dico["contam_blend"] < ratio_blend) | (Dico["contam_blend"] == -99.9)
).astype("int")

print(Fore.YELLOW + "\n[INFO] Number of lines : " + str(len(Dico)) + Fore.WHITE)

print(
    Fore.YELLOW
    + "\n[INFO] Computation of the telluric contamination : DONE \n"
    + Fore.WHITE
)

Dico2 = {"spectre": Spectre, "catalogue": Dico}
Dico2["parameters"] = {
    "rv_sys": rv_sys,
    "berv_mean": berv,
    "berv_min": berv_min,
    "berv_max": berv_max,
    "flux_max_telluric": flux_max_telluric,
    "ratio_telluric": ratio_tel,
    "oversampling": oversampling,
    "kernel_length_f0": kernel_smoothing_length,
    "kernel_length_f1": kernel_length_first_deri,
    "kernel_length_f2": kernel_length_second_deri,
    "kernel_f0": kernel_smoothing_shape,
    "kernel_f1": kernel_shape_first_deri,
    "kernel_f2": kernel_shape_second_deri,
    "vicinity": vicinity,
    "weights": weights,
}

Dico_backup = Dico.copy()

myf.pickle_dump(Dico2, open(directory + "kitcat_mask_" + star + ext + ".p", "wb"))
myf.pickle_dump(
    Dico2, open(directory + "intermediate_kitcat_mask_" + star + ext + ".p", "wb")
)
print(
    Fore.YELLOW
    + "\n[INFO] The table has been saved under : %s and copied in %s \n"
    % (
        directory + "kitcat_mask_" + star + ext + ".p",
        directory + "intermediate_kitcat_mask_" + star + ext + ".p",
    )
    + Fore.WHITE
)

# =============================================================================
# MORPHOLOGICAL CLIPPING 1
# =============================================================================

par1 = 10 ** Dico["diff_continuum_rel"]
par2 = 10 ** abs(Dico["asym_ddflux_norm"])

if feedback:
    t = graphique_clean2(par1, par2)
    t.joint_plot(columns=[r"$diff continuum$", r"$asym dd flux$"])
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    (l1,) = plt.plot(
        np.linspace(xlim[0], xlim[1], 100),
        np.ones(100) * 0.5 * (ylim[0] + ylim[1]),
        color="r",
        lw=2.5,
    )
    (l2,) = plt.plot(
        np.ones(100) * 0.5 * (xlim[0] + xlim[1]),
        np.linspace(ylim[0], ylim[1], 100),
        color="r",
        lw=2.5,
    )
    l = plt.text(
        xlim[0] + 0.2 * (xlim[1] - xlim[0]),
        ylim[0] + 0.9 * (ylim[1] - ylim[0]),
        "Number of lines kept %.0f over %.0f"
        % (
            np.sum(
                (t.x < 0.5 * (xlim[0] + xlim[1])) & (t.y < 0.5 * (ylim[0] + ylim[1]))
            ),
            len(Dico),
        ),
    )

    class Index:
        y = 0.5 * (ylim[0] + ylim[1])
        x = 0.5 * (xlim[0] + xlim[1])

        def update_data(self, newx, newy):
            self.y = newy
            self.x = newx
            l1.set_ydata(np.ones(100) * self.y)
            l2.set_xdata(np.ones(100) * self.x)
            l.set_text(
                "Number of lines kept %.0f over %.0f"
                % (np.sum((t.x < self.x) & (t.y < self.y)), len(Dico))
            )
            plt.gcf().canvas.draw_idle()

    callback = Index()

    def onclick(event):
        newx = event.xdata
        newy = event.ydata
        if event.dblclick:
            callback.update_data(newx, newy)

    plt.gcf().canvas.mpl_connect("button_press_event", onclick)

    plt.show(block=False)
    loop = myf.sphinx("Press ENTER when you are satisfied")
    plt.close("all")

    diff_continuum_tresh = callback.x
    asym_ddflux_tresh = callback.y

mask = (np.log10(par1) < diff_continuum_tresh) & (np.log10(par2) < asym_ddflux_tresh)
mask = mask & Dico["num_dd_max"] == 1
# mask = mask&Dico['num_dd_max2']==1

Dico_backup["morpho_crit"] = np.array(mask).astype("int")
index_backup = np.array(Dico.index[mask])

Dico["valid"] = mask
Dico = Dico.loc[(Dico["valid"] == True)]
Dico = Dico.drop(columns="valid")
Dico = Dico.reset_index(drop=True)

plt.figure(figsize=(21, 7))
plt.title("Step 6")
plt.plot(grid, spectre, color="k")
plt.scatter(Dico["wave"], Dico["flux"], s=10, color="r", zorder=10)
plt.scatter(Dico["wave_left"], Dico["flux_left"], s=10, color="b", zorder=10)
plt.scatter(Dico["wave_right"], Dico["flux_right"], s=10, color="g", zorder=10)
if feedback:
    plt.show()
if not feedback:
    plt.show(block=False)
    plt.close("all")

Dico_all.loc[np.array(Dico["line_nb"]), "qc"] = 7

# =============================================================================
# MORPHOLOGICAL CLIPPING 2
# =============================================================================

Dico["width_kms"] = Dico["win_mic"] * 3e5 / Dico["freq_mask0"]

par1 = 10 ** Dico["depth_rel"]
par2 = 10 ** Dico["width_kms"]

if feedback:
    t = graphique_clean2(par1, par2)
    t.joint_plot(columns=[r"$relative depth$", r"$width [km/s]$"])
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    (l1,) = plt.plot(
        np.linspace(xlim[0], xlim[1], 100),
        np.ones(100) * 0.5 * (ylim[0] + ylim[1]),
        color="r",
        lw=2.5,
    )
    (l2,) = plt.plot(
        np.ones(100) * 0.2 * (xlim[0] + xlim[1]),
        np.linspace(ylim[0], ylim[1], 100),
        color="r",
        lw=2.5,
    )
    l = plt.text(
        xlim[0] + 0.2 * (xlim[1] - xlim[0]),
        ylim[0] + 0.9 * (ylim[1] - ylim[0]),
        "Number of lines kept %.0f over %.0f"
        % (
            np.sum(
                (t.x > 0.2 * (xlim[0] + xlim[1])) & (t.y < 0.5 * (ylim[0] + ylim[1]))
            ),
            len(Dico),
        ),
    )

    class Index:
        y = 0.5 * (ylim[0] + ylim[1])
        x = 0.5 * (xlim[0] + xlim[1])

        def update_data(self, newx, newy):
            self.y = newy
            self.x = newx
            l1.set_ydata(np.ones(100) * self.y)
            l2.set_xdata(np.ones(100) * self.x)
            l.set_text(
                "Number of lines kept %.0f over %.0f"
                % (np.sum((t.x > self.x) & (t.y < self.y)), len(Dico))
            )
            plt.gcf().canvas.draw_idle()

    callback = Index()

    def onclick(event):
        newx = event.xdata
        newy = event.ydata
        if event.dblclick:
            callback.update_data(newx, newy)

    plt.gcf().canvas.mpl_connect("button_press_event", onclick)

    plt.show(block=False)
    loop = myf.sphinx("Press ENTER when you are satisfied")
    plt.close("all")

    rel_depth_tresh = callback.x
    max_width_tresh = callback.y

mask = (np.log10(par1) > rel_depth_tresh) & (np.log10(par2) < max_width_tresh)

Dico_backup.loc[index_backup.astype("int"), "morpho_crit"] = np.array(mask).astype(
    "int"
)

Dico["valid"] = mask
Dico = Dico.loc[(Dico["valid"] == True)]
Dico = Dico.drop(columns="valid")
Dico = Dico.reset_index(drop=True)

plt.figure(figsize=(21, 7))
plt.title("Step 7")
plt.plot(grid, spectre, color="k")
plt.scatter(Dico["wave"], Dico["flux"], s=10, color="r", zorder=10)
plt.scatter(Dico["wave_left"], Dico["flux_left"], s=10, color="b", zorder=10)
plt.scatter(Dico["wave_right"], Dico["flux_right"], s=10, color="g", zorder=10)
if feedback:
    plt.show()
if not feedback:
    plt.show(block=False)
    plt.close("all")

Dico_all.loc[np.array(Dico["line_nb"]), "qc"] = 0

Dico2 = {"spectre": Spectre, "catalogue": Dico_backup}
Dico2["parameters"] = {
    "rv_sys": rv_sys,
    "berv_mean": berv,
    "berv_min": berv_min,
    "berv_max": berv_max,
    "flux_max_telluric": flux_max_telluric,
    "ratio_telluric": ratio_tel,
    "oversampling": oversampling,
    "kernel_length_f0": kernel_smoothing_length,
    "kernel_length_f1": kernel_length_first_deri,
    "kernel_length_f2": kernel_length_second_deri,
    "kernel_f0": kernel_smoothing_shape,
    "kernel_f1": kernel_shape_first_deri,
    "kernel_f2": kernel_shape_second_deri,
    "vicinity": vicinity,
    "tresh_rel_depth": rel_depth_tresh,
    "tresh_max_width": max_width_tresh,
    "tresh_asym_ddflux": asym_ddflux_tresh,
    "tresh_diff_continuum": diff_continuum_tresh,
    "weights": weights,
}

myf.pickle_dump(Dico2, open(directory + "kitcat_mask_" + star + ext + ".p", "wb"))
print(
    Fore.YELLOW
    + "\n[INFO] The table has been saved under : %s \n"
    % (directory + "kitcat_mask_" + star + ext + ".p")
    + Fore.WHITE
)


# =============================================================================
# MAKE DIFFERENT MASK WITH DEPTH CRITERION
# =============================================================================

l1 = 0
l2 = 1
w1 = 0
w2 = 10000

# Dico['mask_deep'] = Dico[weights]
# Dico['mask_medium'] = Dico[weights]
# Dico['mask_shallow'] = Dico[weights]
#
# Dico['mask_blue'] = Dico[weights]
# Dico['mask_green'] = Dico[weights]
# Dico['mask_red'] = Dico[weights]
#
# all_line_depths = np.cumsum(np.sort(Dico[weights]))/np.sum(Dico[weights])
#
# l1 = np.sort(Dico[weights])[myf.find_nearest(all_line_depths,0.33)[0]]
# l2 = np.sort(Dico[weights])[myf.find_nearest(all_line_depths,0.66)[0]]
#
# l1 = np.array(l1)[0]
# l2 = np.array(l2)[0]
#
# Dico.loc[Dico['mask_deep']<l2,'mask_deep'] = 0
# Dico.loc[(Dico['mask_medium']<l1)|(Dico['mask_medium']>l2),'mask_medium'] = 0
# Dico.loc[Dico['mask_shallow']>l1,'mask_shallow'] = 0
#
# w1 = Dico['freq_mask0'][myf.find_nearest(np.cumsum(Dico[weights])/np.sum(Dico[weights]),0.33)[0]]
# w2 = Dico['freq_mask0'][myf.find_nearest(np.cumsum(Dico[weights])/np.sum(Dico[weights]),0.66)[0]]
#
# w1 = np.array(w1)[0]
# w2 = np.array(w2)[0]
#
# Dico.loc[Dico['freq_mask0']>w1,'mask_blue'] = 0
# Dico.loc[(Dico['freq_mask0']<w1)|(Dico['freq_mask0']>w2),'mask_green'] = 0
# Dico.loc[Dico['freq_mask0']<w2,'mask_red'] = 0

# =============================================================================
# SAVE THE MASK
# =============================================================================
loop = "y"
if feedback:
    loop = myf.sphinx("Save the stellar mask (y/n)?", rep=["y", "n"])
Dico = Dico.reset_index(drop=True)
Dico["morpho_crit"] = 1

if loop == "y":
    Dico2 = {"spectre": Spectre, "catalogue": Dico}
    Dico2["parameters"] = {
        "rv_sys": rv_sys,
        "berv_mean": berv,
        "berv_min": berv_min,
        "berv_max": berv_max,
        "flux_max_telluric": flux_max_telluric,
        "ratio_telluric": ratio_tel,
        "oversampling": oversampling,
        "kernel_length_f0": kernel_smoothing_length,
        "kernel_length_f1": kernel_length_first_deri,
        "kernel_length_f2": kernel_length_second_deri,
        "kernel_f0": kernel_smoothing_shape,
        "kernel_f1": kernel_shape_first_deri,
        "kernel_f2": kernel_shape_second_deri,
        "vicinity": vicinity,
        "depth1": l1,
        "depth2": l2,
        "wave1": w1,
        "wave2": w2,
        "tresh_depth": crit1_depth,
        "tresh_width": crit2_width,
        "tresh_rel_depth": rel_depth_tresh,
        "tresh_max_width": max_width_tresh,
        "tresh_asym_ddflux": asym_ddflux_tresh,
        "tresh_diff_continuum": diff_continuum_tresh,
        "weights": weights,
    }

    if len(Dico2["catalogue"]) < 100:
        print(
            Fore.YELLOW
            + "\n[INFO] Current value of stellar lines in the table (%.0f) smaller than the default value (100). Supression of morphological criterion.\n"
            % (len(Dico2["catalogue"]))
            + Fore.WHITE
        )
        Dico = Dico_backup.loc[Dico_backup["blend_crit"] == 1]
        Dico = Dico.reset_index(drop=True)
        Dico2["catalogue"] = Dico.copy()

    if len(Dico2["catalogue"]) < 100:
        print(
            Fore.YELLOW
            + "\n[INFO] Current value of stellar lines in the table (%.0f) smaller than the default value (100). Supression any criterion.\n"
            % (len(Dico2["catalogue"]))
            + Fore.WHITE
        )
        Dico2["catalogue"] = Dico_backup.copy()

    myf.pickle_dump(
        Dico2, open(directory + "kitcat_cleaned_mask_" + star + ext + ".p", "wb")
    )

    Dico_harps_like = Dico[Dico["rel_contam"] < ratio_tel]
    wave_left = np.array(Dico_harps_like["wave_fitted"] - Dico_harps_like["win_mic"])
    wave_right = np.array(Dico_harps_like["wave_fitted"] + Dico_harps_like["win_mic"])
    depths = np.array(Dico_harps_like[weights])
    wave_left = myf.doppler_r(wave_left, rv_sys * 1000)[1]
    wave_right = myf.doppler_r(wave_right, rv_sys * 1000)[1]
    mask_star = np.array([wave_left, wave_right, depths]).T

    np.savetxt(
        root + "/Python/MASK_CCF/kitcat_cleaned_" + star + ext + ".txt", mask_star
    )

    print(Fore.YELLOW + "\n[INFO] Your table is ready !\n" + Fore.WHITE)
    print(
        Fore.YELLOW
        + "\n[INFO] The table has been saved under : %s \n"
        % (directory + "kitcat_cleaned_mask_" + star + ext + ".p")
        + Fore.WHITE
    )
    print(
        Fore.YELLOW
        + "\n[INFO] The mask has been saved under : %s \n"
        % (root + "/Python/MASK_CCF/kitcat_cleaned_" + star + ext + ".txt")
        + Fore.WHITE
    )


time.sleep(1)
plt.close("all")
time.sleep(1)

# =============================================================================
# FINAL PLOT
# =============================================================================


if False:
    Dico = pd.read_pickle(directory + "catalogue_" + star + ext + ".p")["catalogue"]
    Spectre = pd.read_pickle(directory + "catalogue_" + star + ext + ".p")["spectre"]

    grid = Spectre["wave"]
    spectre = Spectre["flux"]

    plt.plot(grid, spectre, color="k")
    plt.scatter(Dico["wave_fitted"], 1 - Dico["line_depth"], color="r", zorder=10)
    plt.scatter(Dico["wave_left"], Dico["flux_left"], color="g", zorder=10)
    plt.scatter(Dico["wave_right"], Dico["flux_right"], color="g", zorder=10)
    for j in np.arange(len(Dico)):
        plt.axvline(x=Dico["wave_fitted"][j], color="k", alpha=0.6)
        plt.axvspan(
            xmin=Dico["wave_fitted"][j] - Dico["win_mic"][j],
            xmax=Dico["wave_fitted"][j] + Dico["win_mic"][j],
            color="k",
            alpha=0.3,
        )
