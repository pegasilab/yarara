#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 18:49:52 2020

@author: Cretignier Michael 
@university University of Geneva
"""


import getopt
import logging
import os
import sys
import time
from typing import Dict, Literal, Union

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

import yarara.stages
from yarara.sts import spec_time_series
from yarara.util import print_iter

# =============================================================================
# PARAMETERS
# =============================================================================

star = "HD110315"
ins = "HARPN"
input_product = "s1d"

stage_start = 0  # always start at 0
stage_break = 29  # break included
close_figure = True
planet_activated: bool = False
reference = None
import warnings

# Disable the new Pandas performance warnings due to fragmentation
try:
    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
except Exception:
    pass

if len(sys.argv) > 1:
    optlist, args = getopt.getopt(sys.argv[1:], "s:i:b:e:c:p:a:l:r:f:d:v:k:D:S:m:")
    for j in optlist:
        if j[0] == "-s":
            star = j[1]
        elif j[0] == "-i":
            ins = j[1]
        elif j[0] == "-b":
            stage_start = int(j[1])
        elif j[0] == "-e":
            stage_break = int(j[1])
            # TODO: options below are never used
        elif j[0] == "-p":
            planet_activated = int(j[1]) != 0
        elif j[0] == "-r":
            reference = j[1]
        elif j[0] == "-d":
            close_figure = bool(j[1])


cwd = os.getcwd()
root = "/".join(cwd.split("/")[:-1])
directory_yarara = root + "/spectra/"
directory_to_dace = directory_yarara + star + "/data/s1d/spectroDownload"
directory_rassine = "/".join(directory_to_dace.split("/")[0:-1]) + "/" + ins
directory_reduced_rassine = directory_rassine + "/STACKED/"
directory_workspace = directory_rassine + "/WORKSPACE/"

# =============================================================================
# TABLE
# =============================================================================
#
# stage = 0  (optionnal functions to be run only once manually)
# stage = 1 (basic operation and statistics)
# stage = 2 (cosmics correction)
# stage = 3 (telluric water correction)
# stage = 4 (telluric oxygen correction)
# stage = 5 (contam pca correction)
# stage = 6 (telluric pca correction)
# stage = 7 (ccf moment/activity correction)
# stage = 8 (frog correction ghost_a)
# stage = 9 (frog correction ghost_b)
# stage = 10 (fourier correction)
# stage = 11 (correct continuum)
# stage = 12 (correct mad)
# stage = 13 (atmospheric parameters)
# stage = 14 (correct brute)

#
# =============================================================================

# =============================================================================
# BEGINNING OF THE TRIGGER
# =============================================================================


begin = time.time()
all_time = [begin]
time_step: Dict[Union[str, int], Union[int, float]] = {"begin": 0}
instrument = ins[0:5]


def get_time_step(step):
    now = time.time()
    time_step[step] = now - all_time[-1]
    all_time.append(now)


# =============================================================================
# RUN YARARA
# =============================================================================

sts = spec_time_series(directory_workspace)
# file_test = sts.import_spectrum()
sts.planet = planet_activated
sts.instrument = ins

print(f"------------------------\n STAR LOADED : {sts.starname} \n------------------------")

print(
    f"\n [INFO] Complete analysis {ins} {['with', 'without'][int(reference is None)]} reference spectrum launched...\n"
)


# =============================================================================
# LOAD DATA IN YARARA
# =============================================================================

for stage in range(stage_start, stage_break):
    if stage == 0:
        # sts.yarara_inject_planet()
        sts.yarara_simbad_query()
        sts.yarara_analyse_summary()
        sts.import_table()

        sts.flux_error(ron=11)
        sts.continuum_error()
        sts.yarara_exploding_pickle()

        sts.yarara_check_rv_sys()
        sts.yarara_check_fwhm()
        plt.close("all")

        sts.yarara_ccf(
            mask=sts.read_ccf_mask(sts.mask_harps),
            mask_name=sts.mask_harps,
            ccf_oversampling=1,
            plot=True,
            save=True,
            rv_range=None,
        )

        sts.yarara_correct_secular_acc(update_rv=True)

        sts.scale_cmap()

        # sous option
        sts.suppress_low_snr_spectra(suppress=False)

        # sous option
        sts.yarara_suppress_doubtful_spectra(suppress=False)

        # sts.supress_time_spectra(num_min=None, num_max=None)
        # sts.split_instrument(instrument=ins)

        if close_figure:
            plt.close("all")

        get_time_step("preprocessing")

    if stage == 1:
        # STATISTICS
        sts.yarara_simbad_query()
        # sts.yarara_star_info(sp_type='K1V', Mstar=1.0, Rstar=1.0)

        sts.yarara_add_step_dico("matching_diff", 0, sub_dico_used="matching_anchors")

        # sts.yarara_add_step_dico('matching_brute',0,chain=True)

        # COLOR TEMPLATE
        logging.info("Compute bolometric constant")
        sts.yarara_flux_constant()
        sts.yarara_color_template()

        sts.yarara_map_1d_to_2d(instrument=ins)

        # ERROR BARS ON FLUX AND CONTINUUM
        print("Add flux errors")
        sts.yarara_non_zero_flux()
        sts.flux_error(ron=11)
        sts.continuum_error()

        # BERV SUMMARY
        sts.yarara_berv_summary(sub_dico="matching_diff", dbin_berv=3, nb_plot=2)

        # ACTIVITY
        logging.info("Compute activity proxy")
        sts.yarara_activity_index(sub_dico="matching_diff")

        # table
        sts.yarara_obs_info(pd.DataFrame(data=[ins] * len(sts.table), columns=["instrument"]))
        sts.import_table()

        logging.info("Make SNR statistics figure")
        sts.snr_statistic()
        logging.info("Make DRS RV summary figure")
        sts.dace_statistic()

        sts.yarara_transit_def(period=100000, T0=0, duration=0.0001, auto=True)

        logging.info("Crop spectra")
        w0 = sts.yarara_get_first_wave()
        sts.yarara_cut_spectrum(wave_min=w0, wave_max=6865.00)

        if close_figure:
            plt.close("all")

        sts.yarara_exploding_pickle()

        get_time_step("statistics")

        if sts.starname == "Sun":
            # TODO: not tested
            sts.yarara_correct_smooth(sub_dico="matching_diff", reference="median", window_ang=1)

    # sts.yarara_inject_planet(amp=[0.4,0.4,0.4,0.4],period=[7.142,27.123,101.543,213.594],phase=[0,0,0,0])

    if stage == 2:
        yarara.stages.matching_cosmics(sts=sts, reference=reference, close_figure=close_figure)
        get_time_step("matching_cosmics")

    if stage == 3:
        yarara.stages.matching_telluric(sts=sts, reference=reference, close_figure=close_figure)
        get_time_step("matching_telluric")

    if stage == 4:
        yarara.stages.matching_oxygen(sts=sts, reference=reference, close_figure=close_figure)
        get_time_step("matching_oxygen")

    if stage == 5:
        # TODO: stage
        sts.yarara_activity_index()
        yarara.stages.matching_contam(
            sts,
            reference=reference,
            frog_file=root + "/Python/Material/Contam_HARPN.p",
            close_figure=close_figure,
        )
        get_time_step("matching_contam")

    if stage == 6:
        yarara.stages.matching_pca(sts=sts, reference=reference, close_figure=close_figure)
        get_time_step("matching_pca")

    ref: Union[Literal["master"], Literal["median"]] = ["master", "median"][reference is None]  # type: ignore

    if stage == 7:
        yarara.stages.matching_activity(
            sts=sts,
            reference=reference,
            ref=ref,
            close_figure=close_figure,
            input_dico="matching_pca",
        )
        get_time_step("matching_activity")

    if stage == 8:
        yarara.stages.matching_ghost_a(
            sts=sts,
            ref=ref,
            close_figure=close_figure,
            frog_file=root + "/Python/Material/Ghost_" + ins + ".p",
        )
        get_time_step("matching_ghost_a")

    if stage == 9:
        yarara.stages.matching_ghost_b(sts=sts, ref=ref, close_figure=close_figure)
        get_time_step("matching_ghost_b")

    if stage == 10:
        yarara.stages.matching_fourier(sts=sts, reference=reference, close_figure=close_figure)
        get_time_step("matching_fourier")

    if stage == 11:
        yarara.stages.matching_smooth(
            sts=sts, ref=ref, reference=reference, close_figure=close_figure
        )
        get_time_step("matching_smooth")

    if stage == 12:
        yarara.stages.matching_mad(sts=sts, reference=reference, close_figure=close_figure)
        get_time_step("matching_mad")

    if stage == 13:
        yarara.stages.stellar_atmos1(sts=sts, reference=reference, close_figure=close_figure)
        get_time_step("stellar_atmos1")

    if stage == 14:
        yarara.stages.matching_brute(sts=sts, reference=reference, close_figure=close_figure)
        get_time_step("matching_brute")


# =============================================================================
# SAVE INFO TIME
# =============================================================================

print_iter(time.time() - begin)

if True:
    table_time = pd.DataFrame(time_step.values(), index=time_step.keys(), columns=["time_step"])
    table_time["frac_time"] = 100 * table_time["time_step"] / np.sum(table_time["time_step"])
    table_time["time_step"] /= 60  # convert in minutes

    filename_time = sts.dir_root + f"REDUCTION_INFO/Time_informations_reduction_{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())}.csv"
    table_time.to_csv(filename_time)
