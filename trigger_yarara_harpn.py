#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 18:49:52 2020

@author: Cretignier Michael 
@university University of Geneva
"""


import getopt
import os
import sys
import time

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from yarara.io import pickle_dump
from yarara.sts import spec_time_series
from yarara.util import print_iter

# =============================================================================
# PARAMETERS
# =============================================================================

star = "HD110315"
ins = "HARPN"
input_product = "s1d"

stage = 1000  # begin at 1 when data are already processed
stage_break = 29  # break included
cascade = True
close_figure = True
planet_activated: bool = False
rassine_full_auto = 0
bin_length = 1
fast = False
reference = None
verbose = 2
prefit_planet = False
drs_version = "old"
sub_dico_to_analyse = "matching_diff"
m_clipping = 3
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
            stage = int(j[1])
        elif j[0] == "-e":
            stage_break = int(j[1])
        elif j[0] == "-c":
            cascade = int(j[1])
        elif j[0] == "-p":
            planet_activated = int(j[1]) != 0
        elif j[0] == "-a":
            rassine_full_auto = int(j[1])
        elif j[0] == "-l":
            bin_length = int(j[1])
        elif j[0] == "-r":
            reference = j[1]
        elif j[0] == "-f":
            fast = int(j[1])
        elif j[0] == "-d":
            close_figure = int(j[1])
        elif j[0] == "-v":
            verbose = int(j[1])
        elif j[0] == "-k":
            prefit_planet = int(j[1])
        elif j[0] == "-D":
            drs_version = j[1]
        elif j[0] == "-S":
            sub_dico_to_analyse = j[1]
        elif j[0] == "-m":
            m_clipping = int(j[1])


cwd = os.getcwd()
root = "/".join(cwd.split("/")[:-1])
directory_yarara = root + "/Yarara/"
directory_to_dace = directory_yarara + star + "/data/s1d/spectroDownload"
directory_rassine = "/".join(directory_to_dace.split("/")[0:-1]) + "/" + ins
directory_reduced_rassine = directory_rassine + "/STACKED/"
directory_workspace = directory_rassine + "/WORKSPACE/"

# =============================================================================
# TABLE
# =============================================================================
#
# stage = -2 (preprocessing importation from dace)
# stage = -1 (launch trigger RASSINE)
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
# stage = 15 (run kitcat)
# stage = 16 (atmospheric parameters 2)
# stage = 17 (activity proxy)
# stage = 18 (correction profile)
# stage = 19 (compute all ccf)
# stage = 20 (all map)
# stage = 21 (compute lbl)
# stage = 22 (compute dbd)
# stage = 23 (compute aba)
# stage = 24 (compute bt)
# stage = 25 (compute wbw)
# stage = 26 (correct morpho + 1 year line rejection)
# stage = 27 (final graphic summary) ---------------- BEGIN OF THE TIME DOMAIN
# stage = 28 (SHELL correction)
# stage = 29 (COLOR LBL PCA correction)
# stage = 30 (LBL PCA correction)
# stage = 31 (BIS correction)
# stage = 32 (General Analysis)
# stage = 33 (l1 periodogram)
# stage = 34 (hierarchical periodogram)
# stage = 35 (light the directory)
# stage = 36 (yarara qc file creation)

#
# =============================================================================

# =============================================================================
# BEGINNING OF THE TRIGGER
# =============================================================================


begin = time.time()
all_time = [begin]
time_step = {"begin": 0}
button = 0
instrument = ins[0:5]


def get_time_step(step):
    now = time.time()
    time_step[step] = now - all_time[-1]
    all_time.append(now)


def break_func(stage):
    button = 1
    if stage >= stage_break:
        return 99
    else:
        if cascade:
            return stage + 1
        else:
            return stage


# =============================================================================
# RUN YARARA
# =============================================================================

sts = spec_time_series(directory_workspace)
# file_test = sts.import_spectrum()
sts.planet = planet_activated
sts.instrument = ins

print("------------------------\n STAR LOADED : %s \n------------------------" % (sts.starname))

if fast:
    print(
        "\n [INFO] Fast analysis %s %s reference spectrum launched...\n"
        % (ins, ["with", "without"][int(reference is None)])
    )
else:
    print(
        "\n [INFO] Complete analysis %s %s reference spectrum launched...\n"
        % (ins, ["with", "without"][int(reference is None)])
    )


# if stage == 999:
#     sts.yarara_recreate_dico("matching_fourier")
#     sts.yarara_recreate_dico("matching_telluric")
#     sts.yarara_recreate_dico("matching_oxygen")
#     sts.yarara_recreate_dico("matching_contam")
#     sts.yarara_recreate_dico("matching_ghost_a")
#     sts.yarara_recreate_dico("matching_ghost_b")
#     sts.yarara_recreate_dico("matching_smooth")

if (
    stage == 777
):  # to rerun the pipeline automatically from the last step in case of a crash (however pickle may be corrupted...)
    sts.import_dico_tree()
    sts.import_material()
    last_dico = np.array(sts.dico_tree["dico"])[-1]
    if "reference_spectrum_backup" in sts.material.keys():
        reference = "master"
    else:
        reference = None
        stage_break = 13
    vec = {
        "matching_diff": 1,
        "matching_cosmics": 2,
        "matching_telluric": 3,
        "matching_oxygen": 4,
        "matching_pca": 5,
        "matching_activity": 6,
        "matching_ghost_a": 7,
        "matching_ghost_b": 8,
        "matching_thar": 9,
        "matching_fourier": 10,
        "matching_smooth": 11,
        "matching_mad": 12,
        "matching_brute": 13,
        "matching_morpho": 24,
        "matching_shell": 26,
        "matching_color": 26,
        "matching_empca": 27,
    }
    stage = vec[last_dico] + 1

# =============================================================================
# LOAD DATA IN YARARA
# =============================================================================


if stage == 1:
    # ADAPT INPUT DATA
    sts.yarara_analyse_summary(rm_old=True)
    sts.yarara_add_step_dico("matching_diff", 0, sub_dico_used="matching_anchors")
    sts.yarara_exploding_pickle()
    stage = break_func(stage)

# =============================================================================
# COSMICS
# =============================================================================

# sts.yarara_inject_planet(amp=[0.4,0.4,0.4,0.4],period=[7.142,27.123,101.543,213.594],phase=[0,0,0,0])

if stage == 2:
    # needs mask_telluric_telluric.txt, type MaskCCF

    # CORRECT COSMICS
    sub_dico = "matching_diff"

    sts.yarara_correct_cosmics(sub_dico=sub_dico, continuum="linear", k_sigma=5)

    if reference is None:
        # MEDIAN MASTER
        sts.yarara_median_master(
            sub_dico="matching_cosmics",
            method="mean",  # if smt else than max, the classical weighted average of v1.0
        )

        sts.yarara_telluric(
            sub_dico="matching_cosmics",
            reference="norm",
            ratio=True,
        )

        ## Works until there

        sts.yarara_master_ccf(sub_dico="matching_cosmics", name_ext="_telluric")

        sts.yarara_median_master(
            sub_dico="matching_cosmics",
            method="mean",  # if smt else than max, the classical weighted average of v1.0
        )

    if close_figure:
        plt.close("all")

    get_time_step("matching_cosmics")
    stage = break_func(stage)


# =============================================================================
# TELLURIC WATER
# =============================================================================

if stage == 3:
    # needs mask_telluric_telluric.txt, type MaskCCF
    # needs mask_telluric_h2o.txt

    # CORRECT TELLURIC WATER

    # TELLURIC FOR MEDIAN MASTER
    if reference is None:
        # MEDIAN MASTER FOR TELLURIC CORRECTION
        sts.yarara_median_master(
            sub_dico="matching_cosmics",
            method="mean",  # if smt else than max, the classical weighted average of v1.0
        )

    sts.yarara_telluric(
        sub_dico="matching_cosmics",
        telluric_tag="telluric",
        reference="norm",
        ratio=True,
        normalisation="left",
    )

    if reference is None:
        # MEDIAN MASTER FOR TELLURIC CORRECTION
        sts.yarara_median_master(
            sub_dico="matching_cosmics",
            method="mean",  # if smt else than max, the classical weighted average of v1.0
        )

    sts.yarara_telluric(
        sub_dico="matching_cosmics",
        telluric_tag="h2o",
        reference="norm",
        ratio=True,
        normalisation="left",
    )

    # CORRECT WATER TELLURIC WITH CCF
    sts.yarara_correct_telluric_proxy(
        sub_dico="matching_cosmics",
        sub_dico_output="telluric",
        reference="master",
        proxies_corr=["h2o_depth", "h2o_fwhm"],
        wave_min_correction_=4400.0,
        min_r_corr_=0.4,
        sigma_ext=2,
    )

    if close_figure:
        plt.close("all")

    get_time_step("matching_telluric")

    stage = break_func(stage)

# =============================================================================
# TELLURIC OXYGEN
# =============================================================================

if stage == 4:
    # needs mask_telluric_telluric.txt, type MaskCCF
    # needs mask_telluric_o2.txt, type MaskCCF

    # CORRECT OXYGEN WATER
    if reference is None:
        # MEDIAN MASTER FOR TELLURIC CORRECTION
        sts.yarara_median_master(
            sub_dico="matching_cosmics",
            method="mean",  # if smt else than max, the classical weighted average of v1.0
        )

    # TELLURIC FOR MEDIAN MASTER
    sts.yarara_telluric(
        sub_dico="matching_cosmics",
        telluric_tag="o2",
        reference="norm",
        ratio=True,
        normalisation="left",
    )

    # CORRECT WATER TELLURIC WITH CCF
    sts.yarara_correct_telluric_proxy(
        sub_dico="matching_telluric",
        sub_dico_output="oxygen",
        reference="master",
        proxies_corr=["h2o_depth", "h2o_fwhm", "o2_depth", "o2_fwhm"],
        wave_min_correction_=4400,
        min_r_corr_=0.4,
        sigma_ext=2,
    )

    sts.yarara_correct_oxygen(
        sub_dico="matching_oxygen",
        oxygene_bands=[[5787, 5830]],
        reference="master",
    )

    if close_figure:
        plt.close("all")

    get_time_step("matching_oxygen")

    stage = break_func(stage)

# =============================================================================
# THAR CONTAM
# =============================================================================

if stage == 5:

    # input Contam_HARPN

    # CORRECT TH CONTAM
    sts.import_table()
    nb_spec = sum(sts.table.jdb < 57505)
    if nb_spec:
        if nb_spec > 15:
            jdb_range = [53000, 57505, 0]
        else:
            jdb_range = [0, 100000, 1]

        if reference is None:
            # MEDIAN MASTER
            sts.yarara_median_master(
                sub_dico="matching_cosmics",
                method="mean",  # if smt else than max, the classical weighted average of v1.0
                suppress_telluric=False,
                jdb_range=jdb_range,
            )

        sts.yarara_produce_mask_contam(frog_file=root + "/Python/Material/Contam_HARPN.p")
        # sts.yarara_produce_mask_telluric(telluric_tresh=0.001)

        sts.yarara_correct_frog(
            correction="contam",
            sub_dico="matching_oxygen",
            reference="master",
            wave_min=5710,
            wave_max=5840,
            berv_shift="berv",
            wave_max_train=7000,
            pca_comp_kept=3,
            algo_pca="pca",
            threshold_contam=0.5,
            equal_weight=True,
            complete_analysis=False,
            rcorr_min=0,
        )

        if close_figure:
            plt.close("all")

    get_time_step("matching_contam")
    stage = break_func(stage)

# =============================================================================
# TELLURIC PCA
# =============================================================================

if stage == 6:

    # Input K5 type MaskCCF
    # CORRECT TELLURIC WITH PCA
    file_test = sts.import_spectrum()
    if "matching_contam" in file_test.keys():
        sts.dico_actif = "matching_contam"
    else:
        sts.dico_actif = "matching_oxygen"

    if reference is None:
        sts.yarara_median_master(
            sub_dico=None,
            method="mean",  # if smt else than max, the classical weighted average of v1.0
        )

    sts.yarara_correct_telluric_gradient(
        sub_dico_detection="matching_cosmics",
        sub_dico_correction=None,
        inst_resolution=110000,
        continuum="linear",
        reference="master",
        equal_weight=True,
        wave_min_correction=4400,
        calib_std=1e-3,
        nb_pca_comp_kept=None,
        nb_pca_max_kept=3,
    )

    sts.yarara_telluric(sub_dico="matching_pca", reference="norm", ratio=True)

    sts.yarara_master_ccf(
        sub_dico="matching_pca",
        name_ext="_telluric",
        rvs=np.array(sts.table["berv"]) * 1000,
    )

    sts.yarara_telluric(sub_dico="matching_cosmics", reference="norm", ratio=True)

    sts.yarara_master_ccf(
        sub_dico="matching_cosmics",
        name_ext="_telluric",
        rvs=np.array(sts.table["berv"]) * 1000,
    )

    if close_figure:
        plt.close("all")

    get_time_step("matching_pca")
    stage = break_func(stage)

# =============================================================================
# CCF MOMENTS
# =============================================================================
ref = ["master", "median"][reference is None]

if stage == 7:
    # CORRECT ACTIVITY WITH PROXIES + CCF MOMENT

    # ACTIVITY
    print("\n Compute activity proxy")
    sts.yarara_activity_index(sub_dico="matching_pca", continuum="linear")

    sts.yarara_ccf(
        mask=sts.mask_harps,
        plot=True,
        sub_dico="matching_pca",
        ccf_oversampling=1,
        rv_range=None,
    )

    proxy = ["Kernel_CaII", "CaII"][reference is None]  # sts.yarara_determine_optimal_Sindex()
    print("\n Optimal proxy of activity : %s" % (proxy))

    sts.yarara_correct_activity(
        sub_dico="matching_pca",
        proxy_corr=[proxy, "ccf_fwhm", "ccf_contrast"],
        smooth_corr=1,
        reference=ref,
    )

    if close_figure:
        plt.close("all")

    get_time_step("matching_activity")
    stage = break_func(stage)

# =============================================================================
# GHOST
# =============================================================================

if stage == 8:
    # Material/Ghost_{instrument}.p type InstrumentGhost

    # CORRECT FROG
    sts.dico_actif = "matching_activity"

    sts.yarara_correct_borders_pxl(
        pixels_to_reject=np.hstack([np.arange(1, 6), np.arange(4092, 4097)])
    )
    sts.yarara_produce_mask_frog(frog_file=root + "/Python/Material/Ghost_" + ins + ".p")

    sts.yarara_correct_frog(
        correction="ghost_a",
        sub_dico="matching_activity",
        reference=ref,
        berv_shift="berv",
        threshold_contam=1,
        equal_weight=False,
        pca_comp_kept=3,
        complete_analysis=False,
    )

    if close_figure:
        plt.close("all")

    get_time_step("matching_ghost_a")
    stage = break_func(stage)


# =============================================================================
# GHOSTB
# =============================================================================

if stage == 9:
    # CORRECT GHOST B CONTAM
    sts.yarara_correct_frog(
        correction="ghost_b",
        sub_dico="matching_ghost_a",
        reference=ref,
        berv_shift="berv",
        wave_max_train=4100,
        pca_comp_kept=2,
        threshold_contam=1,
        equal_weight=True,
        complete_analysis=False,
    )

    if close_figure:
        plt.close("all")

    get_time_step("matching_ghost_b")

    stage = break_func(stage)


# =============================================================================
# FOURIER
# =============================================================================

if stage == 10:
    # CORRECT INTERFERENCE PATTERN

    # MEDIAN MASTER
    if reference is None:
        sts.yarara_median_master(
            sub_dico="matching_ghost_b",
            method="mean",  # if smt else than max, the classical weighted average of v1.0
            jdb_range=[0, 100000, 1],
        )

    # CORRECT PATTERN IN FOURIER SPACE
    sts.yarara_correct_pattern(
        sub_dico="matching_ghost_b",
        continuum="linear",
        reference="master",
        correct_blue=True,
        correct_red=True,
        width_range=[2.5, 3.5],
        jdb_range=[0, 100000],
    )  # all the time a pattern on HARPN

    if close_figure:
        plt.close("all")

    get_time_step("matching_fourier")
    stage = break_func(stage)

# =============================================================================
# CONTINUUM
# =============================================================================

if stage == 11:
    # CORRECT CONTINUUM

    sts.yarara_correct_smooth(
        sub_dico="matching_fourier", continuum="linear", reference=ref, window_ang=5
    )

    if reference is None:
        sts.yarara_retropropagation_correction(
            correction_map="matching_smooth",
            sub_dico="matching_cosmics",
            continuum="linear",
        )

    if close_figure:
        plt.close("all")

    get_time_step("matching_smooth")

    stage = break_func(stage)

# =============================================================================
# MAD
# =============================================================================

if stage == 12:
    # CORRECT MAD
    counter_mad_removed = sts.yarara_correct_mad(
        sub_dico="matching_smooth",
        continuum="linear",
        k_sigma=2,
        k_mad=2,
        n_iter=1,
        ext=["0", "1"][int(reference == "master")],
    )

    spectrum_removed = counter_mad_removed > [0.15, 0.15][int(reference == "master")]
    sts.supress_time_spectra(liste=spectrum_removed)

    sts.yarara_ccf(
        mask=sts.mask_harps,
        plot=True,
        sub_dico="matching_mad",
        ccf_oversampling=1,
        rv_range=None,
    )

    if close_figure:
        plt.close("all")

    get_time_step("matching_mad")
    stage = break_func(stage)

# =============================================================================
# STELLAR ATMOS
# =============================================================================

if stage == 13:
    # TEMPERATURE + MODEL FIT

    sts.yarara_median_master(
        sub_dico="matching_mad",
        method="median",  # if smt else than max, the classical weighted average of v1.0
        suppress_telluric=False,
        shift_spectrum=False,
    )

    if reference != "master":
        sts.import_material()
        load = sts.material

        load["reference_spectrum_backup"] = load["reference_spectrum"].copy()
        pickle_dump(load, open(sts.directory + "Analyse_material.p", "wb"))

    sts.yarara_cut_spectrum(wave_min=None, wave_max=6834)

    #    sts.yarara_stellar_atmos(sub_dico="matching_diff", reference="master", continuum="linear")

    #    sts.yarara_correct_continuum_absorption(model=None, T=None, g=None)

    sts.yarara_ccf(
        mask=sts.mask_harps,
        plot=True,
        sub_dico="matching_pca",
        ccf_oversampling=1,
        rv_range=None,
    )
    sts.yarara_ccf(
        mask=sts.mask_harps,
        plot=True,
        sub_dico="matching_mad",
        ccf_oversampling=1,
        rv_range=None,
    )

    #    sts.yarara_snr_curve()
    #    sts.snr_statistic(version=2)
    #    sts.yarara_kernel_caii(
    #     contam=True,
    #     mask="CaII_HD128621",
    #     power_snr=None,
    #     noise_kernel="unique",
    #     wave_max=None,
    #     doppler_free=False,
    # )

    if close_figure:
        plt.close("all")

    get_time_step("stellar_atmos1")
    stage = break_func(stage)

# =============================================================================
# BRUTE
# =============================================================================

if stage == 14:
    # CORRECT BRUTE
    sts.yarara_correct_brute(
        sub_dico="matching_mad",
        continuum="linear",
        min_length=3,
        k_sigma=2,
        percent_removed=10,  # None if sphinx
        ghost2=False,
        borders_pxl=True,
    )

    if close_figure:
        plt.close("all")

    get_time_step("matching_brute")
    stage = break_func(stage)


# =============================================================================
# SAVE INFO TIME
# =============================================================================

print_iter(time.time() - begin)

if button:
    table_time = pd.DataFrame(time_step.values(), index=time_step.keys(), columns=["time_step"])
    table_time["frac_time"] = 100 * table_time["time_step"] / np.sum(table_time["time_step"])
    table_time["time_step"] /= 60  # convert in minutes

    filename_time = sts.dir_root + "REDUCTION_INFO/Time_informations_reduction_%s.csv" % (
        time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    )
    table_time.to_csv(filename_time)
