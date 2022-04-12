from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:07:14 2019

@author: Cretignier Michael 
@university University of Geneva
"""

# =============================================================================
# Yet Another RAssine Related Arborescence (YARARA)
# =============================================================================

import datetime
import glob as glob
import logging
import os
import time
import warnings

import matplotlib.cm as cmx
import matplotlib.colors as mplcolors
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from colorama import Fore
from scipy.interpolate import interp1d
from tqdm import tqdm

from . import Rassine_functions as ras
from . import io
from . import my_classes as myc
from . import my_functions as myf
from . import sts
from .paths import cwd, root
from .util import print_iter, yarara_artefact_suppressed

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# =============================================================================
# PRODUCE THE DACE TABLE SUMMARIZING RV TIMESERIES
# =============================================================================


class spec_time_series(object):
    def __init__(self, directory):
        if len(directory.split("/")) == 1:
            directory = root + "/Yarara/" + directory + "/data/s1d/HARPS03/WORKSPACE/"
        if directory[-1] != "/":
            directory = directory + "/"

        self.directory = directory
        self.starname = directory.split("/Yarara/")[-1].split("/")[0].split("=")[0]
        self.dir_root = directory.split("WORKSPACE/")[0]
        self.dir_yarara = directory.split("Yarara/")[0] + "Yarara/"
        self.cmap = "plasma"
        self.low_cmap = -0.005
        self.high_cmap = 0.005
        self.zoom = 1
        self.smooth_map = 1
        self.planet = False
        self.sp_type = None
        self.rv_sys = None
        self.teff = None
        self.log_g = None
        self.bv = None
        self.fwhm = None
        self.wave = None
        self.infos = {}
        self.ram = []

        self.dico_actif = "matching_diff"

        self.all_dicos = [
            "matching_diff",
            "matching_cosmics",
            "matching_fourier",
            "matching_telluric",
            "matching_oxy_bands",
            "matching_oxygen",
            "matching_pca",
            "matching_activity",
            "matching_ghost_a",
            "matching_ghost_b",
            "matching_database",
            "matching_stitching",
            "matching_berv",
            "matching_thar",
            "matching_contam",
            "matching_smooth",
            "matching_profile",
            "matching_mad",
        ]

        self.light_dicos = [
            "matching_diff",
            "matching_pca",
            "matching_activity",
            "matching_mad",
        ]

        self.planet_fitted = {}

        if not os.path.exists(self.directory + "Analyse_ccf.p"):
            ccf_summary = {"star_info": {"name": self.starname}}
            io.pickle_dump(ccf_summary, open(self.directory + "Analyse_ccf.p", "wb"))

        if not os.path.exists(self.directory + "Analyse_material.p"):
            file = pd.read_pickle(glob.glob(self.directory + "RASSI*.p")[0])
            wave = file["wave"]
            dico = {
                "wave": wave,
                "correction_factor": np.ones(len(wave)),
                "reference_spectrum": np.ones(len(wave)),
                "color_template": np.ones(len(wave)),
                "blaze_correction": np.ones(len(wave)),
                "rejected": np.zeros(len(wave)),
            }
            dico = pd.DataFrame(dico)
            io.pickle_dump(dico, open(self.directory + "Analyse_material.p", "wb"))

        if os.path.exists(self.directory + "/RASSINE_Master_spectrum.p"):
            master = pd.read_pickle(self.directory + "/RASSINE_Master_spectrum.p")
            master = master["flux"] / master["output"]["continuum_linear"]
            os.system(
                "mv "
                + self.directory
                + "RASSINE_Master_spectrum.p "
                + self.directory
                + "ras_Master_spectrum.p "
            )

        if not os.path.exists(self.dir_root + "IMAGES/"):
            os.system("mkdir " + self.dir_root + "IMAGES/")

        if not os.path.exists(self.dir_root + "CCF_MASK/"):
            os.system("mkdir " + self.dir_root + "CCF_MASK/")

        if not os.path.exists(self.dir_root + "PCA/"):
            os.system("mkdir " + self.dir_root + "PCA/")

        if not os.path.exists(self.dir_root + "REDUCTION_INFO/"):
            os.system("mkdir " + self.dir_root + "REDUCTION_INFO/")

        if not os.path.exists(self.dir_root + "STAR_INFO/"):
            os.system("mkdir " + self.dir_root + "STAR_INFO/")

        if not os.path.exists(self.dir_root + "STAR_INFO/Stellar_info_" + self.starname + ".p"):
            dico = {
                "Name": self.starname,
                "Simbad_name": {"fixed": "-"},
                "Sp_type": {"fixed": "G2V"},
                "Ra": {"fixed": "00 00 00.0000"},
                "Dec": {"fixed": "00 00 00.0000"},
                "Pma": {"fixed": 0.0},
                "Pmd": {"fixed": 0.0},
                "Rv_sys": {"fixed": 0.0},
                "Mstar": {"fixed": 1.0},
                "Rstar": {"fixed": 1.0},
                "magU": {"fixed": -26.0},
                "magB": {"fixed": -26.2},
                "magV": {"fixed": -26.8},
                "magR": {"fixed": -26.8},
                "UB": {"fixed": 0.0},
                "BV": {"fixed": 0.6},
                "VR": {"fixed": 0.0},
                "Dist_pc": {"fixed": 0.0},
                "Teff": {"fixed": 5775},
                "Log_g": {"fixed": 4.5},
                "FeH": {"fixed": 0.0},
                "Vsini": {"fixed": 2.0},
                "Vmicro": {"fixed": 1.0},
                "Prot": {"fixed": 25},
                "Pmag": {"fixed": 11},
                "FWHM": {"fixed": 6.0},
                "Contrast": {"fixed": 0.5},
                "CCF_delta": {"fixed": 5},
                "stellar_template": {"fixed": "MARCS_T5750_g4.5"},
            }

            io.pickle_dump(
                dico,
                open(
                    self.dir_root + "STAR_INFO/Stellar_info_" + self.starname + ".p",
                    "wb",
                ),
            )

        self.import_star_info()
        sp = self.star_info["Sp_type"]["fixed"][0]
        self.mask_harps = ["G2", "K5", "M2"][int((sp == "K") | (sp == "M")) + int(sp == "M")]
        try:
            self.fwhm = self.star_info["FWHM"]["YARARA"]
        except:
            self.fwhm = self.star_info["FWHM"]["fixed"]

        try:
            self.contrast = self.star_info["Contrast"]["YARARA"]
        except:
            self.contrast = self.star_info["Contrast"]["fixed"]

        if not os.path.exists(self.dir_root + "KEPLERIAN/"):
            os.system("mkdir " + self.dir_root + "KEPLERIAN/")

        if not os.path.exists(self.dir_root + "KEPLERIAN/MCMC/"):
            os.system("mkdir " + self.dir_root + "KEPLERIAN/MCMC/")

        if not os.path.exists(self.dir_root + "PERIODOGRAM/"):
            os.system("mkdir " + self.dir_root + "PERIODOGRAM/")

        if not os.path.exists(self.dir_root + "FILM"):
            os.system("mkdir " + self.dir_root + "FILM")

        if not os.path.exists(self.dir_root + "DETECTION_LIMIT/"):
            os.system("mkdir " + self.dir_root + "DETECTION_LIMIT/")

        if not os.path.exists(self.dir_root + "CORRECTION_MAP/"):
            os.system("mkdir " + self.dir_root + "CORRECTION_MAP/")

    def import_rassine_output(self: spec_time_series, return_name=False, kw1=None, kw2=None):
        return sts.io.import_rassine_output(return_name, kw1, kw2)

    def import_star_info(self: spec_time_series):
        return sts.io.import_star_info(self)

    def import_table(self: spec_time_series):
        return sts.io.import_table(self)

    def import_material(self: spec_time_series):
        return sts.io.import_material(self)

    def import_dico_tree(self: spec_time_series):
        return sts.io.import_dico_tree(self)

    def import_spectrum(self: spec_time_series, num=None):
        return sts.io.import_spectrum(self, num)

    def yarara_star_info(
        self: spec_time_series,
        Rv_sys=None,
        simbad_name=None,
        magB=None,
        magV=None,
        magR=None,
        BV=None,
        VR=None,
        sp_type=None,
        Mstar=None,
        Rstar=None,
        Vsini=None,
        Vmicro=None,
        Teff=None,
        log_g=None,
        FeH=None,
        Prot=None,
        Fwhm=None,
        Contrast=None,
        CCF_delta=None,
        Pmag=None,
        stellar_template=None,
    ):
        return sts.io.yarara_star_info(
            self,
            Rv_sys,
            simbad_name,
            magB,
            magV,
            magR,
            BV,
            VR,
            sp_type,
            Mstar,
            Rstar,
            Vsini,
            Vmicro,
            Teff,
            log_g,
            FeH,
            Prot,
            Fwhm,
            Contrast,
            CCF_delta,
            Pmag,
            stellar_template,
        )

    def yarara_master_ccf(self: spec_time_series, sub_dico="matching_diff", name_ext="", rvs=None):
        return sts.limbo.yarara_master_ccf(self, sub_dico, name_ext, rvs)

    def yarara_poissonian_noise(
        self: spec_time_series, noise_wanted=1 / 100, wave_ref=None, flat_snr=True, seed=9
    ):
        return sts.util.yarara_poissonian_noise(self, noise_wanted, wave_ref, flat_snr, seed)

    def yarara_obs_info(
        self: spec_time_series,
        kw=[None, None],
        jdb=None,
        berv=None,
        rv=None,
        airmass=None,
        texp=None,
        seeing=None,
        humidity=None,
    ):
        return sts.io.yarara_obs_info(self, kw, jdb, berv, rv, airmass, texp, seeing, humidity)

    def yarara_get_orders(self: spec_time_series):
        return sts.extract.yarara_get_orders(self)

    def yarara_get_pixels(self: spec_time_series):
        return sts.extract.yarara_get_pixels(self)

    def supress_time_spectra(
        self: spec_time_series,
        liste=None,
        jdb_min=None,
        jdb_max=None,
        num_min=None,
        num_max=None,
        supress=False,
        name_ext="temp",
    ):
        return sts.io.supress_time_spectra(
            self, liste, jdb_min, jdb_max, num_min, num_max, supress, name_ext
        )

    def yarara_analyse_summary(self: spec_time_series, rm_old=False):
        return sts.io.yarara_analyse_summary(self, rm_old)

    def yarara_get_berv_value(
        self: spec_time_series,
        time_value,
        Draw=False,
        new=True,
        light_graphic=False,
        save_fig=True,
    ):
        return sts.extract.yarara_get_berv_value(self, time, Draw, new, light_graphic, save_fig)

    def yarara_non_zero_flux(self: spec_time_series, spectrum=None, min_value=None):
        return sts.util.yarara_non_zero_flux(self, spectrum, min_value)

    def yarara_median_master_backup(
        self,
        sub_dico="matching_diff",
        method="mean",
        continuum="linear",
        supress_telluric=True,
        shift_spectrum=False,
        telluric_tresh=0.001,
        wave_min=5750,
        wave_max=5900,
        jdb_range=[-100000, 100000, 1],
        mask_percentile=[None, 50],
        save=True,
    ):
        return sts.util.yarara_median_master_backup(
            self,
            sub_dico,
            method,
            continuum,
            supress_telluric,
            shift_spectrum,
            telluric_tresh,
            wave_min,
            wave_max,
            jdb_range,
            mask_percentile,
            save,
        )

    def yarara_median_master(
        self: spec_time_series,
        sub_dico="matching_diff",
        continuum="linear",
        method="max",
        smooth_box=7,
        supress_telluric=True,
        shift_spectrum=False,
        wave_min=5750,
        wave_max=5900,
        bin_berv=10,
        bin_snr=None,
        telluric_tresh=0.001,
        jdb_range=[-100000, 100000, 1],
        mask_percentile=[None, 50],
        save=True,
    ):
        return sts.util.yarara_median_master(
            self,
            sub_dico,
            continuum,
            method,
            smooth_box,
            supress_telluric,
            shift_spectrum,
            wave_min,
            wave_max,
            bin_berv,
            bin_snr,
            telluric_tresh,
            jdb_range,
            mask_percentile,
            save,
        )

    def yarara_cut_spectrum(self: spec_time_series, wave_min=None, wave_max=None):
        return sts.util.yarara_cut_spectrum(self, wave_min, wave_max)

    def yarara_activity_index(
        self: spec_time_series,
        sub_dico="matching_diff",
        continuum="linear",
        plot=True,
        debug=False,
        calib_std=0,
        optimize=False,
        substract_map=[],
        add_map=[],
        p_noise=1 / np.inf,
        save=True,
    ):
        return sts.processing.yarara_activity_index(
            self,
            sub_dico,
            continuum,
            plot,
            debug,
            calib_std,
            optimize,
            substract_map,
            add_map,
            p_noise,
            save,
        )

    def yarara_telluric(
        self: spec_time_series,
        sub_dico="matching_anchors",
        continuum="linear",
        suppress_broad=True,
        delta_window=5,
        mask=None,
        weighted=False,
        reference=True,
        display_ccf=False,
        ratio=False,
        normalisation="slope",
        ccf_oversampling=3,
        wave_max=None,
        wave_min=None,
    ):
        return sts.telluric.yarara_telluric(
            self,
            sub_dico,
            continuum,
            suppress_broad,
            delta_window,
            mask,
            weighted,
            reference,
            display_ccf,
            ratio,
            normalisation,
            ccf_oversampling,
            wave_max,
            wave_min,
        )

    def yarara_ccf(
        self: spec_time_series,
        sub_dico="matching_diff",
        continuum="linear",
        mask=None,
        mask_name=None,
        ccf_name=None,
        mask_col="weight_rv",
        treshold_telluric=1,
        ratio=False,
        element=None,
        reference=True,
        weighted=True,
        plot=False,
        display_ccf=False,
        save=True,
        save_ccf_profile=False,
        normalisation="left",
        del_outside_max=False,
        bis_analysis=False,
        ccf_oversampling=1,
        rv_range=None,
        rv_borders=None,
        delta_window=5,
        debug=False,
        rv_sys=None,
        rv_shift=None,
        speed_up=True,
        force_brute=False,
        wave_min=None,
        wave_max=None,
        squared=True,
        p_noise=1 / np.inf,
        substract_map=[],
        add_map=[],
    ):
        return sts.processing.yarara_ccf(
            self,
            sub_dico,
            continuum,
            mask,
            mask_name,
            ccf_name,
            mask_col,
            treshold_telluric,
            ratio,
            element,
            reference,
            weighted,
            plot,
            display_ccf,
            save,
            save_ccf_profile,
            normalisation,
            del_outside_max,
            bis_analysis,
            ccf_oversampling,
            rv_range,
            rv_borders,
            delta_window,
            debug,
            rv_sys,
            rv_shift,
            speed_up,
            force_brute,
            wave_min,
            wave_max,
            squared,
            p_noise,
            substract_map,
            add_map,
        )

    def yarara_map(
        self: spec_time_series,
        sub_dico="matching_diff",
        continuum="linear",
        planet=False,
        modulo=None,
        unit=1.0,
        wave_min=4000,
        wave_max=4300,
        time_min=None,
        time_max=None,
        index="index",
        ratio=False,
        reference="median",
        berv_shift=False,
        rv_shift=False,
        new=True,
        Plot=True,
        substract_map=[],
        add_map=[],
        correction_factor=True,
        p_noise=1 / np.inf,
    ):
        return sts.processing.yarara_map(
            self,
            sub_dico,
            continuum,
            planet,
            modulo,
            unit,
            wave_min,
            wave_max,
            time_min,
            time_max,
            index,
            ratio,
            reference,
            berv_shift,
            rv_shift,
            new,
            Plot,
            substract_map,
            add_map,
            correction_factor,
            p_noise,
        )

    def yarara_correct_pattern(
        self: spec_time_series,
        sub_dico="matching_diff",
        continuum="linear",
        wave_min=6000,
        wave_max=6100,
        reference="median",
        width_range=[0.1, 20],
        correct_blue=True,
        correct_red=True,
        jdb_range=None,
    ):
        return sts.instrument.yarara_correct_pattern(
            self,
            sub_dico,
            continuum,
            wave_min,
            wave_max,
            reference,
            width_range,
            correct_blue,
            correct_red,
            jdb_range,
        )

    def yarara_correct_smooth(
        self: spec_time_series,
        sub_dico="matching_diff",
        continuum="linear",
        reference="median",
        wave_min=4200,
        wave_max=4300,
        window_ang=5,
    ):
        return sts.outliers.yarara_correct_smooth(
            self, sub_dico, continuum, reference, wave_min, wave_max, window_ang
        )

    def yarara_retropropagation_correction(
        self: spec_time_series,
        correction_map="matching_smooth",
        sub_dico="matching_cosmics",
        continuum="linear",
    ):
        return sts.processing.yarara_retropropagation_correction(
            self, correction_map, sub_dico, continuum
        )

    def yarara_correct_telluric_proxy(
        self: spec_time_series,
        sub_dico="matching_fourier",
        sub_dico_output="telluric",
        continuum="linear",
        wave_min=5700,
        wave_max=5900,
        reference="master",
        berv_shift="berv",
        smooth_corr=1,
        proxies_corr=["h2o_depth", "h2o_fwhm"],
        proxies_detrending=None,
        wave_min_correction=4400,
        wave_max_correction=None,
        min_r_corr=0.40,
        sigma_ext=2,
    ):
        return sts.telluric.yarara_correct_telluric_proxy(
            self,
            sub_dico,
            sub_dico_output,
            continuum,
            wave_min,
            wave_max,
            reference,
            berv_shift,
            smooth_corr,
            proxies_corr,
            proxies_detrending,
            wave_min_correction,
            wave_max_correction,
            min_r_corr,
            sigma_ext,
        )

    def yarara_correct_oxygen(
        self: spec_time_series,
        sub_dico="matching_telluric",
        continuum="linear",
        berv_shift="berv",
        reference="master",
        wave_min=5760,
        wave_max=5850,
        oxygene_bands=[[5787, 5835], [6275, 6340], [6800, 6950]],
    ):
        return sts.telluric.yarara_correct_oxygen(
            self, sub_dico, continuum, berv_shift, reference, wave_min, wave_max, oxygene_bands
        )

    def yarara_correct_telluric_gradient(
        self: spec_time_series,
        sub_dico_detection="matching_fourier",
        sub_dico_correction="matching_oxygen",
        continuum="linear",
        wave_min_train=4200,
        wave_max_train=5000,
        wave_min_correction=4400,
        wave_max_correction=6600,
        smooth_map=1,
        berv_shift="berv",
        reference="master",
        inst_resolution=110000,
        debug=False,
        equal_weight=True,
        nb_pca_comp=20,
        nb_pca_comp_kept=None,
        nb_pca_max_kept=5,
        calib_std=1e-3,
    ):
        return sts.telluric.yarara_correct_telluric_gradient(
            self,
            sub_dico_detection,
            sub_dico_correction,
            continuum,
            wave_min_train,
            wave_max_train,
            wave_min_correction,
            wave_max_correction,
            smooth_map,
            berv_shift,
            reference,
            inst_resolution,
            debug,
            equal_weight,
            nb_pca_comp,
            nb_pca_comp_kept,
            nb_pca_max_kept,
            calib_std,
        )

    def yarara_correct_activity(
        self: spec_time_series,
        sub_dico="matching_telluric",
        continuum="linear",
        wave_min=3900,
        wave_max=4400,
        smooth_corr=5,
        reference="median",
        rv_shift="none",
        proxy_corr=["CaII"],
    ):
        return sts.activity.yarara_correct_activity(
            self,
            sub_dico,
            continuum,
            wave_min,
            wave_max,
            smooth_corr,
            reference,
            rv_shift,
            proxy_corr,
        )

    def yarara_correct_cosmics(
        self: spec_time_series,
        sub_dico="matching_diff",
        continuum="linear",
        k_sigma=3,
        bypass_warning=True,
    ):
        return sts.outliers.yarara_correct_cosmics(
            self, sub_dico, continuum, k_sigma, bypass_warning
        )

    def yarara_correct_mad(
        self: spec_time_series,
        sub_dico="matching_diff",
        continuum="linear",
        k_sigma=2,
        k_mad=2,
        n_iter=1,
        ext="0",
    ):
        return sts.outliers.yarara_correct_mad(
            self, sub_dico, continuum, k_sigma, k_mad, n_iter, ext
        )

    def yarara_produce_mask_contam(
        self: spec_time_series, frog_file=root + "/Python/Material/Contam_HARPN.p"
    ):
        return sts.instrument.yarara_produce_mask_contam(self, frog_file)

    def yarara_produce_mask_frog(
        self: spec_time_series, frog_file=root + "/Python/Material/Ghost_HARPS03.p"
    ):
        return sts.instrument.yarara_produce_mask_frog(self, frog_file)

    def yarara_correct_frog(
        self: spec_time_series,
        sub_dico="matching_diff",
        continuum="linear",
        correction="stitching",
        berv_shift=False,
        wave_min=3800,
        wave_max=3975,
        wave_min_train=3700,
        wave_max_train=6000,
        complete_analysis=False,
        reference="median",
        equal_weight=True,
        nb_pca_comp=10,
        pca_comp_kept=None,
        rcorr_min=0,
        treshold_contam=0.5,
        algo_pca="empca",
    ) -> None:
        return sts.instrument.yarara_correct_frog(
            self,
            sub_dico,
            continuum,
            correction,
            berv_shift,
            wave_min,
            wave_max,
            wave_min_train,
            wave_max_train,
        )

    def yarara_correct_borders_pxl(
        self: spec_time_series, pixels_to_reject=[2, 4095], min_shift=-30, max_shift=30
    ):
        return sts.instrument.yarara_correct_borders_pxl(
            self, pixels_to_reject, min_shift, max_shift
        )

    def yarara_correct_brute(
        self: spec_time_series,
        sub_dico="matching_mad",
        continuum="linear",
        reference="median",
        win_roll=1000,
        min_length=5,
        percent_removed=10,
        k_sigma=2,
        extended=10,
        ghost2="HARPS03",
        borders_pxl=False,
    ) -> None:
        return sts.outliers.yarara_correct_brute(
            self,
            sub_dico,
            continuum,
            reference,
            win_roll,
            min_length,
            percent_removed,
            k_sigma,
            extended,
            ghost2,
            borders_pxl,
        )

    def uncorrect_hole(self: spec_time_series, conti, conti_ref, values_forbidden=[0, np.inf]):
        return sts.processing.uncorrect_hole(self, conti, conti_ref, values_forbidden)
