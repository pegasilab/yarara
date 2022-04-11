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
from .util import print_iter, yarara_artefact_suppressed

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# =============================================================================
# PRODUCE THE DACE TABLE SUMMARIZING RV TIMESERIES
# =============================================================================

cwd = os.getcwd()
root = "/".join(cwd.split("/")[:-1])




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

    # =============================================================================
    # IMPORT ALL RASSINE DICTIONNARY
    # =============================================================================

    # io
    def import_rassine_output(self, return_name=False, kw1=None, kw2=None):
        """
        Import all the RASSINE dictionnaries in a list

        Parameters
        ----------
        return_name : True/False to also return the filenames

        Returns
        -------
        Return the list containing all thedictionnary

        """

        directory = self.directory

        files = glob.glob(directory + "RASSI*.p")
        if len(files) <= 1:  # 1 when merged directory
            print("No RASSINE file found in the directory : %s" % (directory))
            if return_name:
                return [], []
            else:
                return []
        else:
            files = np.sort(files)
            file = []
            for i, j in enumerate(files):
                self.debug = j
                file.append(pd.read_pickle(j))

                if kw1 is not None:
                    file[-1] = file[-1][kw1]

                if kw2 is not None:
                    file[-1] = file[-1][kw2]

            if return_name:
                return file, files
            else:
                return file

    # =============================================================================
    # IMPORT SUMMARY TABLE
    # =============================================================================

    # io
    def import_star_info(self):
        self.star_info = pd.read_pickle(
            self.dir_root + "STAR_INFO/Stellar_info_" + self.starname + ".p"
        )

    # io
    def import_table(self):
        self.table = pd.read_pickle(self.directory + "Analyse_summary.p")

    # io
    def import_material(self):
        self.material = pd.read_pickle(self.directory + "Analyse_material.p")

    # =============================================================================
    # IMPORT THE FULL DICO CHAIN
    # =============================================================================

    # io
    def import_dico_tree(self):
        file_test = self.import_spectrum()
        kw = list(file_test.keys())
        kw_kept = []
        kw_chain = []
        for k in kw:
            if len(k.split("matching_")) == 2:
                kw_kept.append(k)
        kw_kept = np.array(kw_kept)

        info = []
        for n in kw_kept:

            try:
                s = file_test[n]["parameters"]["step"]
                dico = file_test[n]["parameters"]["sub_dico_used"]
                info.append([n, s, dico])
            except:
                pass
        info = pd.DataFrame(info, columns=["dico", "step", "dico_used"])
        self.dico_tree = info.sort_values(by="step")

    # =============================================================================
    # IMPORT a RANDOM SPECTRUM
    # =============================================================================

    # def copy_spectrum(self):
    #     directory = self.directory
    #     files = glob.glob(directory + "RASSI*.p")
    #     files = np.sort(files)[0]

    #     if not os.path.exists(self.dir_root + "TEMP/"):
    #         os.system("mkdir " + self.dir_root + "TEMP/")

    #     os.system("cp " + files + " " + self.dir_root + "TEMP/")

    # io
    def import_spectrum(self, num=None):
        """
        Import a pickle file of a spectrum to get fast common information shared by all spectra

        Parameters
        ----------
        num : index of the spectrum to extract (if None random selection)

        Returns
        -------
        Return the open pickle file

        """

        directory = self.directory
        files = glob.glob(directory + "RASSI*.p")
        files = np.sort(files)
        if not len(files):
            files = glob.glob(directory.replace("WORKSPACE", "TEMP") + "RASSI*.p")

        if num is None:
            try:
                num = np.random.choice(np.arange(len(files)), 1)
                file = files[num][0]
            except:
                file = files[0]

        else:
            try:
                file = files[num]
            except:
                file = files[0]
        return pd.read_pickle(file)

    # io
    def yarara_star_info(
        self,
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

        kw = [
            "Rv_sys",
            "Simbad_name",
            "Sp_type",
            "magB",
            "magV",
            "magR",
            "BV",
            "VR",
            "Mstar",
            "Rstar",
            "Vsini",
            "Vmicro",
            "Teff",
            "Log_g",
            "FeH",
            "Prot",
            "FWHM",
            "Contrast",
            "Pmag",
            "stellar_template",
            "CCF_delta",
        ]
        val = [
            Rv_sys,
            simbad_name,
            sp_type,
            magB,
            magV,
            magR,
            BV,
            VR,
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
            Pmag,
            stellar_template,
            CCF_delta,
        ]

        self.import_star_info()
        self.import_table()
        self.import_material()

        table = self.table
        snr = np.array(table["snr"]).argmax()
        file_test = self.import_spectrum(num=snr)

        for i, j in zip(kw, val):
            if j is not None:
                if type(j) != list:
                    j = ["fixed", j]
                if i in self.star_info.keys():
                    self.star_info[i][j[0]] = j[1]
                else:
                    self.star_info[i] = {j[0]: j[1]}

        # TODO:
        # Here, we removed the Gray temperature and the MARCS atlas atmospheric model
        # initialization

        io.pickle_dump(
            self.star_info,
            open(self.dir_root + "STAR_INFO/Stellar_info_" + self.starname + ".p", "wb"),
        )

    # limbo
    def yarara_master_ccf(self, sub_dico="matching_diff", name_ext="", rvs=None):
        self.import_table()

        vrad, ccfs = (self.all_ccf_saved[sub_dico][0], self.all_ccf_saved[sub_dico][1])

        if rvs is None:
            rvs = self.ccf_rv.y.copy()

        med_rv = np.nanmedian(rvs)
        rvs -= med_rv

        new_ccf = []
        for j in range(len(ccfs.T)):
            ccf = myc.tableXY(vrad - rvs[j], ccfs[:, j])
            ccf.interpolate(new_grid=vrad, method="linear", fill_value=np.nan)
            new_ccf.append(ccf.y)
        new_ccf = np.array(new_ccf)
        new_vrad = vrad - med_rv
        stack = np.sum(new_ccf, axis=0)
        stack /= np.nanpercentile(stack, 95)
        half = 0.5 * (1 + np.nanmin(stack))

        master_ccf = myc.tableXY(new_vrad, stack)
        master_ccf.supress_nan()
        master_ccf.interpolate(replace=True, method="cubic")

        new_vrad = master_ccf.x
        stack = master_ccf.y

        v1 = new_vrad[new_vrad < 0][myf.find_nearest(stack[new_vrad < 0], half)[0][0]]
        v2 = new_vrad[new_vrad > 0][myf.find_nearest(stack[new_vrad > 0], half)[0][0]]

        vmin = np.nanmin(new_vrad[~np.isnan(stack)])
        vmax = np.nanmax(new_vrad[~np.isnan(stack)])

        vlim = np.min([abs(vmin), abs(vmax)])
        vmin = -vlim
        vmax = vlim

        contrast = 1 - np.nanmin(stack)

        plt.figure()
        plt.plot(new_vrad, stack, color="k", label="Contrast = %.1f %%" % (100 * contrast))

        extension = ["YARARA", "HARPS", ""][int(name_ext != "") + int(name_ext == "_telluric")]

        if extension == "YARARA":
            self.fwhm = np.round((v2 - v1) / 1000, 2)
            io.pickle_dump(
                {"vrad": new_vrad, "ccf_power": stack},
                open(self.dir_root + "MASTER/MASTER_CCF_KITCAT.p", "wb"),
            )
            try:
                old = pd.read_pickle(self.dir_root + "MASTER/MASTER_CCF_HARPS.p")
                plt.plot(old["vrad"], old["ccf_power"], alpha=0.5, color="k", ls="--")
            except:
                pass
        elif extension == "HARPS":
            io.pickle_dump(
                {"vrad": new_vrad, "ccf_power": stack},
                open(self.dir_root + "MASTER/MASTER_CCF_HARPS.p", "wb"),
            )
        else:
            try:
                old = pd.read_pickle(self.dir_root + "MASTER/MASTER_CCF" + name_ext + ".p")
                plt.plot(old["vrad"], old["ccf_power"], alpha=0.5, color="k", ls="--")
            except:
                pass
            io.pickle_dump(
                {"vrad": new_vrad, "ccf_power": stack},
                open(self.dir_root + "MASTER/MASTER_CCF" + name_ext + ".p", "wb"),
            )

        plt.xlim(vmin, vmax)

        plt.plot(
            [v1, v2],
            [half, half],
            color="r",
            label="FHWM = %.2f kms" % ((v2 - v1) / 1000),
        )
        plt.scatter([v1, v2], [half, half], color="r", edgecolor="k", zorder=10)
        plt.scatter([0], [np.nanmin(stack)], color="k", edgecolor="k", zorder=10)
        plt.axvline(x=0, ls=":", color="k", alpha=0.5)
        plt.legend()
        plt.grid()
        plt.xlabel("RV [m/s]", fontsize=13)
        plt.ylabel("Flux normalised", fontsize=13)
        plt.title("%s" % (self.starname), fontsize=14)

        self.yarara_star_info(Contrast=[extension, np.round(contrast, 3)])
        self.yarara_star_info(Fwhm=[extension, np.round((v2 - v1) / 1000, 2)])

        plt.savefig(self.dir_root + "IMAGES/MASTER_CCF" + name_ext + ".pdf")

    # util
    def yarara_poissonian_noise(self, noise_wanted=1 / 100, wave_ref=None, flat_snr=True, seed=9):
        self.import_table()
        self.import_material()

        if noise_wanted:
            master = np.sqrt(
                np.array(self.material.reference_spectrum * self.material.correction_factor)
            )  # used to scale the snr continuum into errors bars
            snrs = pd.read_pickle(self.dir_root + "WORKSPACE/Analyse_snr.p")

            if not flat_snr:
                if wave_ref is None:
                    current_snr = np.array(self.table.snr_computed)
                else:
                    i = myf.find_nearest(snrs["wave"], wave_ref)[0]
                    current_snr = snrs["snr_curve"][:, i]

                curves = current_snr * np.ones(len(master))[:, np.newaxis]
            else:
                curves = snrs["snr_curve"]  # snr curves representing the continuum snr

            snr_wanted = 1 / noise_wanted
            diff = 1 / snr_wanted**2 - 1 / curves**2
            diff[diff < 0] = 0
            # snr_to_degrate = 1/np.sqrt(diff)

            noise = np.sqrt(diff)
            noise[np.isnan(noise)] = 0

            noise_values = noise * master[:, np.newaxis].T
            np.random.seed(seed=seed)
            matrix_noise = np.random.randn(len(self.table.jdb), len(self.material.wave))
            matrix_noise *= noise_values
        else:
            matrix_noise = np.zeros((len(self.table.jdb), len(self.material.wave)))
            noise_values = np.zeros((len(self.table.jdb), len(self.material.wave)))

        return matrix_noise, noise_values

    # io
    def yarara_obs_info(
        self,
        kw=[None, None],
        jdb=None,
        berv=None,
        rv=None,
        airmass=None,
        texp=None,
        seeing=None,
        humidity=None,
    ):
        """
        Add some observationnal information in the RASSINE files and produce a summary table

        Parameters
        ----------
        kw: list-like with format [keyword,array]
        jdb : array-like with same size than the number of files in the directory
        berv : array-like with same size than the number of files in the directory
        rv : array-like with same size than the number of files in the directory
        airmass : array-like with same size than the number of files in the directory
        texp : array-like with same size than the number of files in the directory
        seeing : array-like with same size than the number of files in the directory
        humidity : array-like with same size than the number of files in the directory

        """

        directory = self.directory

        files = glob.glob(directory + "RASSI*.p")
        files = np.sort(files)

        nb = len(files)

        if type(kw) == pd.core.frame.DataFrame:  # in case of a pandas dataframe
            kw = [list(kw.keys()), [i for i in np.array(kw).T]]
        else:
            try:
                if len(kw[1]) == 1:
                    kw[1] = [kw[1][0]] * nb
            except TypeError:
                kw[1] = [kw[1]] * nb

            kw[0] = [kw[0]]
            kw[1] = [kw[1]]

        for i, j in enumerate(files):
            file = pd.read_pickle(j)
            for kw1, kw2 in zip(kw[0], kw[1]):
                if kw1 is not None:
                    if len(kw1.split("ccf_")) - 1:
                        file["ccf_gaussian"][kw1.split("ccf_")[1]] = kw2[i]
                    else:
                        file["parameters"][kw1] = kw2[i]
            if jdb is not None:
                file["parameters"]["jdb"] = jdb[i]
            if berv is not None:
                file["parameters"]["berv"] = berv[i]
            if rv is not None:
                file["parameters"]["rv"] = rv[i]
            if airmass is not None:
                file["parameters"]["airmass"] = airmass[i]
            if texp is not None:
                file["parameters"]["texp"] = texp[i]
            if seeing is not None:
                file["parameters"]["seeing"] = seeing[i]
            if humidity is not None:
                file["parameters"]["humidity"] = humidity[i]
            io.save_pickle(j, file)

        self.yarara_analyse_summary()

    # extract
    def yarara_get_orders(self):
        self.import_material()
        mat = self.material
        orders = np.array(mat["orders_rnr"])
        orders = myf.map_rnr(orders)
        orders = np.round(orders, 0)
        self.orders = orders
        return orders

    # extract
    def yarara_get_pixels(self):
        self.import_material()
        mat = self.material
        pixels = np.array(mat["pixels_rnr"])
        pixels = myf.map_rnr(pixels)
        pixels = np.round(pixels, 0)
        self.pixels = pixels
        return pixels

    # io
    def supress_time_spectra(
        self,
        liste=None,
        jdb_min=None,
        jdb_max=None,
        num_min=None,
        num_max=None,
        supress=False,
        name_ext="temp",
    ):
        """
        Supress spectra according to time

        Parameters
        ----------

        jdb or num are both inclusive in lower and upper limit
        snr_cutoff : treshold value of the cutoff below which spectra are removed
        supress : True/False to delete of simply hide the files

        """

        self.import_table()
        jdb = self.table["jdb"]
        name = self.table["filename"]
        directory = "/".join(np.array(name)[0].split("/")[:-1])

        if num_min is not None:
            jdb_min = jdb[num_min]

        if num_max is not None:
            jdb_max = jdb[num_max]

        if jdb_min is None:
            jdb_min = jdb_max

        if jdb_max is None:
            jdb_max = jdb_min

        if (num_min is None) & (num_max is None) & (jdb_min is None) & (jdb_max is None):
            jdb_min = 1e9
            jdb_max = -1e9

        if liste is None:
            mask = np.array((jdb >= jdb_min) & (jdb <= jdb_max))
        else:
            if type(liste[0]) == np.bool_:
                mask = liste
            else:
                mask = np.in1d(np.arange(len(jdb)), liste)

        if sum(mask):
            idx = np.arange(len(jdb))[mask]
            print(" [INFO] Following spectrum indices will be supressed : ", idx)
            print(" [INFO] Number of spectrum supressed : %.0f \n" % (sum(mask)))
            maps = glob.glob(self.dir_root + "CORRECTION_MAP/*.p")
            if len(maps):
                for names in maps:
                    correction_map = pd.read_pickle(names)
                    correction_map["correction_map"] = np.delete(
                        correction_map["correction_map"], idx, axis=0
                    )
                    io.pickle_dump(correction_map, open(names, "wb"))
                    print("%s modified" % (names.split("/")[-1]))

            name = name[mask]

            files_to_process = np.sort(np.array(name))
            for j in files_to_process:
                if supress:
                    print("File deleted : %s " % (j))
                    os.system("rm " + j)
                else:
                    new_name = name_ext + "_" + j.split("/")[-1]
                    print("File renamed : %s " % (directory + "/" + new_name))
                    os.system("mv " + j + " " + directory + "/" + new_name)

            os.system("rm " + directory + "/Analyse_summary.p")
            os.system("rm " + directory + "/Analyse_summary.csv")
            self.yarara_analyse_summary()

            self.supress_time_RV(mask)

    # =============================================================================
    # MAKE SUMMARY
    # =============================================================================

    # io
    def yarara_analyse_summary(self, rm_old=False):
        """
        Produce a summary table with the RASSINE files of the specified directory

        """

        directory = self.directory
        if rm_old:
            os.system("rm " + directory + "Analyse_summary.p")
            os.system("rm " + directory + "Analyse_summary.csv")

        test, names = self.import_rassine_output(return_name=True)

        # info obs

        ins = np.array([test[j]["parameters"].get("instrument") for j in range(len(test))])
        snr = np.array([test[j]["parameters"].get("SNR_5500") for j in range(len(test))])
        snr2 = np.array([test[j]["parameters"].get("SNR_computed") for j in range(len(test))])
        jdb = np.array([test[j]["parameters"].get("jdb") for j in range(len(test))])
        mjd = np.array([test[j]["parameters"].get("mjd") for j in range(len(test))])
        berv = np.array([test[j]["parameters"].get("berv") for j in range(len(test))])
        berv_planet = np.array(
            [test[j]["parameters"].get("berv_planet") for j in range(len(test))]
        )
        lamp_offset = np.array(
            [test[j]["parameters"].get("lamp_offset") for j in range(len(test))]
        )
        rv = np.array([test[j]["parameters"].get("rv") for j in range(len(test))])
        rv_shift = np.array([test[j]["parameters"].get("RV_shift") for j in range(len(test))])
        rv_sec = np.array([test[j]["parameters"].get("RV_sec") for j in range(len(test))])
        rv_moon = np.array([test[j]["parameters"].get("RV_moon") for j in range(len(test))])
        airmass = np.array([test[j]["parameters"].get("airmass") for j in range(len(test))])
        texp = np.array([test[j]["parameters"].get("texp") for j in range(len(test))])
        seeing = np.array([test[j]["parameters"].get("seeing") for j in range(len(test))])
        humidity = np.array([test[j]["parameters"].get("humidity") for j in range(len(test))])
        rv_planet = np.array([test[j]["parameters"].get("rv_planet") for j in range(len(test))])
        rv_dace = np.array([test[j]["parameters"].get("rv_dace") for j in range(len(test))])
        rv_dace_std = np.array(
            [test[j]["parameters"].get("rv_dace_std") for j in range(len(test))]
        )
        flux_balance = np.array(
            [test[j]["parameters"].get("flux_balance") for j in range(len(test))]
        )
        transit = np.array([test[j]["parameters"].get("transit_in") for j in range(len(test))])
        night_drift = np.array([test[j]["parameters"].get("drift_used") for j in range(len(test))])
        night_drift_std = np.array(
            [test[j]["parameters"].get("drift_used_std") for j in range(len(test))]
        )

        # info activity

        rhk = np.array([test[j]["parameters"].get("RHK") for j in range(len(test))])
        rhk_std = np.array([test[j]["parameters"].get("RHK_std") for j in range(len(test))])
        kernel_rhk = np.array([test[j]["parameters"].get("Kernel_CaII") for j in range(len(test))])
        kernel_rhk_std = np.array(
            [test[j]["parameters"].get("Kernel_CaII_std") for j in range(len(test))]
        )
        ca2k = np.array([test[j]["parameters"].get("CaIIK") for j in range(len(test))])
        ca2k_std = np.array([test[j]["parameters"].get("CaIIK_std") for j in range(len(test))])
        ca2h = np.array([test[j]["parameters"].get("CaIIH") for j in range(len(test))])
        ca2h_std = np.array([test[j]["parameters"].get("CaIIH_std") for j in range(len(test))])
        ca2 = np.array([test[j]["parameters"].get("CaII") for j in range(len(test))])
        ca2_std = np.array([test[j]["parameters"].get("CaII_std") for j in range(len(test))])
        ca1 = np.array([test[j]["parameters"].get("CaI") for j in range(len(test))])
        ca1_std = np.array([test[j]["parameters"].get("CaI_std") for j in range(len(test))])
        mg1 = np.array([test[j]["parameters"].get("MgI") for j in range(len(test))])
        mg1_std = np.array([test[j]["parameters"].get("MgI_std") for j in range(len(test))])
        mg1a = np.array([test[j]["parameters"].get("MgIa") for j in range(len(test))])
        mg1a_std = np.array([test[j]["parameters"].get("MgIa_std") for j in range(len(test))])
        mg1b = np.array([test[j]["parameters"].get("MgIb") for j in range(len(test))])
        mg1b_std = np.array([test[j]["parameters"].get("MgIb_std") for j in range(len(test))])
        mg1c = np.array([test[j]["parameters"].get("MgIc") for j in range(len(test))])
        mg1c_std = np.array([test[j]["parameters"].get("MgIc_std") for j in range(len(test))])
        d3 = np.array([test[j]["parameters"].get("HeID3") for j in range(len(test))])
        d3_std = np.array([test[j]["parameters"].get("HeID3_std") for j in range(len(test))])
        nad = np.array([test[j]["parameters"].get("NaD") for j in range(len(test))])
        nad_std = np.array([test[j]["parameters"].get("NaD_std") for j in range(len(test))])
        nad1 = np.array([test[j]["parameters"].get("NaD1") for j in range(len(test))])
        nad1_std = np.array([test[j]["parameters"].get("NaD1_std") for j in range(len(test))])
        nad2 = np.array([test[j]["parameters"].get("NaD2") for j in range(len(test))])
        nad2_std = np.array([test[j]["parameters"].get("NaD2_std") for j in range(len(test))])
        ha = np.array([test[j]["parameters"].get("Ha") for j in range(len(test))])
        ha_std = np.array([test[j]["parameters"].get("Ha_std") for j in range(len(test))])
        hb = np.array([test[j]["parameters"].get("Hb") for j in range(len(test))])
        hb_std = np.array([test[j]["parameters"].get("Hb_std") for j in range(len(test))])
        hc = np.array([test[j]["parameters"].get("Hc") for j in range(len(test))])
        hc_std = np.array([test[j]["parameters"].get("Hc_std") for j in range(len(test))])
        hd = np.array([test[j]["parameters"].get("Hd") for j in range(len(test))])
        hd_std = np.array([test[j]["parameters"].get("Hd_std") for j in range(len(test))])
        heps = np.array([test[j]["parameters"].get("Heps") for j in range(len(test))])
        heps_std = np.array([test[j]["parameters"].get("Heps_std") for j in range(len(test))])

        wbk = np.array([test[j]["parameters"].get("WB_K") for j in range(len(test))])
        wbk_std = np.array([test[j]["parameters"].get("WB_K_std") for j in range(len(test))])
        wbh = np.array([test[j]["parameters"].get("WB_H") for j in range(len(test))])
        wbh_std = np.array([test[j]["parameters"].get("WB_H_std") for j in range(len(test))])
        wb = np.array([test[j]["parameters"].get("WB") for j in range(len(test))])
        wb_std = np.array([test[j]["parameters"].get("WB_std") for j in range(len(test))])
        wb_h1 = np.array([test[j]["parameters"].get("WB_H1") for j in range(len(test))])
        wb_h1_std = np.array([test[j]["parameters"].get("WB_H1_std") for j in range(len(test))])
        kernel_wb = np.array([test[j]["parameters"].get("Kernel_WB") for j in range(len(test))])
        kernel_wb_std = np.array(
            [test[j]["parameters"].get("Kernel_WB_std") for j in range(len(test))]
        )

        cb = np.array([test[j]["parameters"].get("CB") for j in range(len(test))])
        cb_std = np.array([test[j]["parameters"].get("CB_std") for j in range(len(test))])
        cb2 = np.array([test[j]["parameters"].get("CB2") for j in range(len(test))])
        cb2_std = np.array([test[j]["parameters"].get("CB2_std") for j in range(len(test))])

        # pca shells coefficient

        shell_compo = np.array(
            [test[j]["parameters"].get("shell_fitted") for j in range(len(test))]
        )
        shell1 = np.array([test[j]["parameters"].get("shell_v1") for j in range(len(test))])
        shell2 = np.array([test[j]["parameters"].get("shell_v2") for j in range(len(test))])
        shell3 = np.array([test[j]["parameters"].get("shell_v3") for j in range(len(test))])
        shell4 = np.array([test[j]["parameters"].get("shell_v4") for j in range(len(test))])
        shell5 = np.array([test[j]["parameters"].get("shell_v5") for j in range(len(test))])

        # pca rv vectors
        pca_compo = np.array([test[j]["parameters"].get("pca_fitted") for j in range(len(test))])
        pca1 = np.array([test[j]["parameters"].get("pca_v1") for j in range(len(test))])
        pca2 = np.array([test[j]["parameters"].get("pca_v2") for j in range(len(test))])
        pca3 = np.array([test[j]["parameters"].get("pca_v3") for j in range(len(test))])
        pca4 = np.array([test[j]["parameters"].get("pca_v4") for j in range(len(test))])
        pca5 = np.array([test[j]["parameters"].get("pca_v5") for j in range(len(test))])

        # telluric ccf

        vec_background = np.array(
            [test[j]["parameters"].get("proxy_background") for j in range(len(test))]
        )

        # telluric ccf

        tell_ew = np.array([test[j]["parameters"].get("telluric_ew") for j in range(len(test))])
        tell_contrast = np.array(
            [test[j]["parameters"].get("telluric_contrast") for j in range(len(test))]
        )
        tell_rv = np.array([test[j]["parameters"].get("telluric_rv") for j in range(len(test))])
        tell_fwhm = np.array(
            [test[j]["parameters"].get("telluric_fwhm") for j in range(len(test))]
        )
        tell_center = np.array(
            [test[j]["parameters"].get("telluric_center") for j in range(len(test))]
        )
        tell_depth = np.array(
            [test[j]["parameters"].get("telluric_depth") for j in range(len(test))]
        )

        # telluric ccf h2o

        h2o_ew = np.array([test[j]["parameters"].get("h2o_ew") for j in range(len(test))])
        h2o_contrast = np.array(
            [test[j]["parameters"].get("h2o_contrast") for j in range(len(test))]
        )
        h2o_rv = np.array([test[j]["parameters"].get("h2o_rv") for j in range(len(test))])
        h2o_fwhm = np.array([test[j]["parameters"].get("h2o_fwhm") for j in range(len(test))])
        h2o_center = np.array([test[j]["parameters"].get("h2o_center") for j in range(len(test))])
        h2o_depth = np.array([test[j]["parameters"].get("h2o_depth") for j in range(len(test))])

        # telluric ccf o2

        o2_ew = np.array([test[j]["parameters"].get("o2_ew") for j in range(len(test))])
        o2_contrast = np.array(
            [test[j]["parameters"].get("o2_contrast") for j in range(len(test))]
        )
        o2_rv = np.array([test[j]["parameters"].get("o2_rv") for j in range(len(test))])
        o2_fwhm = np.array([test[j]["parameters"].get("o2_fwhm") for j in range(len(test))])
        o2_center = np.array([test[j]["parameters"].get("o2_center") for j in range(len(test))])
        o2_depth = np.array([test[j]["parameters"].get("o2_depth") for j in range(len(test))])

        # teff

        t_eff = np.array([test[j]["parameters"].get("Teff") for j in range(len(test))])
        t_eff_std = np.array([test[j]["parameters"].get("Teff_std") for j in range(len(test))])

        # ghost

        ghost = np.array([test[j]["parameters"].get("ghost") for j in range(len(test))])

        # sas
        sas = np.array([test[j]["parameters"].get("SAS") for j in range(len(test))])
        sas_std = np.array([test[j]["parameters"].get("SAS_std") for j in range(len(test))])
        sas1y = np.array([test[j]["parameters"].get("SAS1Y") for j in range(len(test))])
        sas1y_std = np.array([test[j]["parameters"].get("SAS1Y_std") for j in range(len(test))])
        bis = np.array([test[j]["parameters"].get("BIS") for j in range(len(test))])
        bis_std = np.array([test[j]["parameters"].get("BIS_std") for j in range(len(test))])
        bis2 = np.array([test[j]["parameters"].get("BIS2") for j in range(len(test))])
        bis2_std = np.array([test[j]["parameters"].get("BIS2_std") for j in range(len(test))])

        # parabola ccf

        try:
            para_rv = np.array([test[j]["ccf_parabola"].get("para_rv") for j in range(len(test))])
            para_depth = np.array(
                [test[j]["ccf_parabola"].get("para_depth") for j in range(len(test))]
            )
        except KeyError:
            para_rv = np.zeros(len(snr))
            para_depth = np.zeros(len(snr))

        # gaussian ccf
        try:
            ccf_ew = np.array([test[j]["ccf"].get("ew") for j in range(len(test))])
            ccf_contrast = np.array(
                [test[j]["ccf_gaussian"].get("contrast") for j in range(len(test))]
            )
            ccf_contrast_std = np.array(
                [test[j]["ccf_gaussian"].get("contrast_std") for j in range(len(test))]
            )
            ccf_rv = np.array([test[j]["ccf_gaussian"].get("rv") for j in range(len(test))])
            ccf_rv_std = np.array(
                [test[j]["ccf_gaussian"].get("rv_std") for j in range(len(test))]
            )
            ccf_fwhm = (
                np.array([test[j]["ccf_gaussian"].get("fwhm") for j in range(len(test))]) * 2.355
            )
            ccf_fwhm_std = (
                np.array([test[j]["ccf_gaussian"].get("fwhm_std") for j in range(len(test))])
                * 2.355
            )
            ccf_offset = np.array(
                [test[j]["ccf_gaussian"].get("offset") for j in range(len(test))]
            )
            ccf_offset_std = np.array(
                [test[j]["ccf_gaussian"].get("offset_std") for j in range(len(test))]
            )
            ccf_vspan = np.array([test[j]["ccf_gaussian"].get("vspan") for j in range(len(test))])
            ccf_vspan_std = np.array(
                [test[j]["ccf_gaussian"].get("rv_std") for j in range(len(test))]
            )
            ccf_svrad_phot = np.array(
                [test[j]["ccf_gaussian"].get("rv_std_phot") for j in range(len(test))]
            )

        except KeyError:
            ccf_ew = np.zeros(len(snr))
            ccf_contrast = np.zeros(len(snr))
            ccf_contrast_std = np.zeros(len(snr))
            ccf_rv = np.zeros(len(snr))
            ccf_rv_std = np.zeros(len(snr))
            ccf_fwhm = np.zeros(len(snr))
            ccf_fwhm_std = np.zeros(len(snr))
            ccf_offset = np.zeros(len(snr))
            ccf_offset_std = np.zeros(len(snr))
            ccf_vspan = np.zeros(len(snr))
            ccf_vspan_std = np.zeros(len(snr))
            ccf_svrad_phot = np.zeros(len(snr))

        dico = pd.DataFrame(
            {
                "filename": names,
                "ins": ins,
                "snr": snr,
                "snr_computed": snr2,
                "flux_balance": flux_balance,
                "jdb": jdb,
                "mjd": mjd,
                "berv": berv,
                "berv_planet": berv_planet,
                "lamp_offset": lamp_offset,
                "rv": rv,
                "rv_shift": rv_shift,
                "rv_sec": rv_sec,
                "rv_moon": rv_moon,
                "rv_planet": rv_planet,
                "rv_dace": rv_dace,
                "rv_dace_std": rv_dace_std,
                "rv_drift": night_drift,
                "rv_drift_std": night_drift_std,
                "airmass": airmass,
                "texp": texp,
                "seeing": seeing,
                "humidity": humidity,
                "transit_in": transit,
                "RHK": rhk,
                "RHK_std": rhk_std,
                "Kernel_CaII": kernel_rhk,
                "Kernel_CaII_std": kernel_rhk_std,
                "CaIIK": ca2k,
                "CaIIK_std": ca2k_std,
                "CaIIH": ca2h,
                "CaIIH_std": ca2h_std,
                "CaII": ca2,
                "CaII_std": ca2_std,
                "CaI": ca1,
                "CaI_std": ca1_std,
                "MgI": mg1,
                "MgI_std": mg1_std,
                "MgIa": mg1a,
                "MgIa_std": mg1a_std,
                "MgIb": mg1b,
                "MgIb_std": mg1b_std,
                "MgIc": mg1c,
                "MgIc_std": mg1c_std,
                "HeID3": d3,
                "HeID3_std": d3_std,
                "NaD": nad,
                "NaD_std": nad_std,
                "NaD1": nad1,
                "NaD1_std": nad1_std,
                "NaD2": nad2,
                "NaD2_std": nad2_std,
                "Ha": ha,
                "Ha_std": ha_std,
                "Hb": hb,
                "Hb_std": hb_std,
                "Hc": hc,
                "Hc_std": hc_std,
                "Hd": hd,
                "Hd_std": hd_std,
                "Heps": heps,
                "Heps_std": heps_std,
                "WBK": wbk,
                "WBK_std": wbk_std,
                "WBH": wbh,
                "WBH_std": wbh_std,
                "WB": wb,
                "WB_std": wb_std,
                "WB_H1": wb_h1,
                "WB_H1_std": wb_h1_std,
                "Kernel_WB": kernel_wb,
                "Kernel_WB_std": kernel_wb_std,
                "CB": cb,
                "CB_std": cb_std,
                "CB2": cb2,
                "CB2_std": cb2_std,
                "shell_fitted": shell_compo,
                "shell1": shell1,
                "shell2": shell2,
                "shell3": shell3,
                "shell4": shell4,
                "shell5": shell5,
                "pca_fitted": pca_compo,
                "pca1": pca1,
                "pca2": pca2,
                "pca3": pca3,
                "pca4": pca4,
                "pca5": pca5,
                "telluric_ew": tell_ew,
                "telluric_contrast": tell_contrast,
                "telluric_rv": tell_rv,
                "telluric_fwhm": tell_fwhm,
                "telluric_center": tell_center,
                "telluric_depth": tell_depth,
                "h2o_ew": h2o_ew,
                "h2o_contrast": h2o_contrast,
                "h2o_rv": h2o_rv,
                "h2o_fwhm": h2o_fwhm,
                "h2o_center": h2o_center,
                "h2o_depth": h2o_depth,
                "o2_ew": o2_ew,
                "o2_contrast": o2_contrast,
                "o2_rv": o2_rv,
                "o2_fwhm": o2_fwhm,
                "o2_center": o2_center,
                "o2_depth": o2_depth,
                "proxy_background": vec_background,
                "ccf_rv_para": para_rv,
                "ccf_depth_para": para_depth,
                "ccf_ew": ccf_ew,
                "ccf_contrast": ccf_contrast,
                "ccf_contrast_std": ccf_contrast_std,
                "ccf_vspan": ccf_vspan,
                "ccf_vspan_std": ccf_vspan_std,
                "ccf_rv": ccf_rv,
                "ccf_rv_std": ccf_rv_std,
                "ccf_rv_std_phot": ccf_svrad_phot,
                "ccf_fwhm": ccf_fwhm,
                "ccf_fwhm_std": ccf_fwhm_std,
                "ccf_offset": ccf_offset,
                "ccf_offset_std": ccf_offset_std,
                "Teff": t_eff,
                "Teff_std": t_eff_std,
                "ghost": ghost,
                "SAS": sas,
                "SAS_std": sas_std,
                "SAS1Y": sas1y,
                "SAS1Y_std": sas1y_std,
                "BIS": bis,
                "BIS_std": bis_std,
                "BIS2": bis2,
                "BIS2_std": bis2_std,
            }
        )

        if os.path.exists(directory + "Analyse_summary.p"):
            summary = pd.read_pickle(directory + "Analyse_summary.p")
            merge = summary.merge(dico, on="filename", how="outer", indicator=True)
            update = pd.DataFrame({"filename": np.array(merge["filename"])})
            merge = merge.drop(columns="filename")
            summary = summary.drop(columns="filename")
            dico = dico.drop(columns="filename")

            all_keys = np.array(summary.keys())
            for j in all_keys:
                try:
                    merge.loc[merge["_merge"] != "left_only", j + "_x"] = merge.loc[
                        merge["_merge"] != "left_only", j + "_y"
                    ]
                    update[j] = merge[j + "_x"]
                except KeyError:
                    update[j] = merge[j]

            all_keys = np.setdiff1d(np.array(dico.keys()), np.array(summary.keys()))
            if len(all_keys) != 0:
                for j in all_keys:
                    update[j] = merge[j]
            dico = update

        dico = dico.sort_values(by="jdb").reset_index(drop=True)

        io.save_pickle(directory + "/Analyse_summary.p", dico)
        dico.to_csv(directory + "/Analyse_summary.csv")

        print("\n [INFO] Summary table updated")

    # =============================================================================
    # EXTRACT BERV AT A SPECIFIC TIME
    # =============================================================================

    # extract
    def yarara_get_berv_value(
        self, time_value, Draw=False, new=True, light_graphic=False, save_fig=True
    ):
        """Return the berv value for a given jdb date"""

        self.import_table()
        tab = self.table
        berv = myc.tableXY(tab["jdb"], tab["berv"], tab["berv"] * 0 + 0.3)
        berv.fit_sinus(guess=[30, 365.25, 0, 0, 0, 0], Draw=False)
        amp = berv.lmfit.params["amp"].value
        period = berv.lmfit.params["period"].value
        phase = berv.lmfit.params["phase"].value
        offset = berv.lmfit.params["c"].value

        berv_value = amp * np.sin(2 * np.pi * time_value / period + phase) + offset
        if Draw == True:
            t = np.linspace(0, period, 365)
            b = amp * np.sin(2 * np.pi * t / period + phase) + offset

            if new:
                plt.figure(figsize=(8.5, 7))
            plt.title(
                "BERV min : %.1f | BERV max : %.1f | BERV mean : %.1f"
                % (np.min(berv.y), np.max(berv.y), np.mean(berv.y)),
                fontsize=13,
            )
            berv.plot(modulo=period)
            plt.plot(t, b, color="k")
            if not light_graphic:
                plt.axvline(x=time_value % period, color="gray")
                plt.axhline(
                    y=berv_value,
                    color="gray",
                    label="BERV = %.1f [km/s]" % (berv_value),
                )
                plt.axhline(y=0, ls=":", color="k")
                plt.legend()
            plt.xlabel("Time %% %.2f" % (period), fontsize=16)
            plt.ylabel("BERV [km/s]", fontsize=16)
            if save_fig:
                plt.savefig(self.dir_root + "IMAGES/berv_values_summary.pdf")
        return berv_value

    # =============================================================================
    # YARARA NONE ZERO FLUX
    # =============================================================================

    # util
    def yarara_non_zero_flux(self, spectrum=None, min_value=None):
        file_test = self.import_spectrum()
        hole_left = file_test["parameters"]["hole_left"]
        hole_right = file_test["parameters"]["hole_right"]
        grid = file_test["wave"]
        mask = (grid < hole_left) | (grid > hole_right)

        directory = self.directory

        files = glob.glob(directory + "RASSI*.p")
        files = np.sort(files)

        if spectrum is None:
            for i, j in enumerate(files):
                file = pd.read_pickle(j)
                flux = file["flux"]
                zero = flux == 0
                if min_value is None:
                    min_value = np.min(flux[flux != 0])
                flux[mask & zero] = min_value
                io.pickle_dump(file, open(j, "wb"))
        else:
            print("[INFO] Removing null values of the spectrum")
            zero = spectrum <= 0
            if min_value is None:
                min_value = np.min(spectrum[spectrum > 0])
            spectrum[mask & zero] = min_value
            return spectrum

    # =============================================================================
    #     CREATE MEDIAN SPECTRUM TELLURIC SUPRESSED
    # =============================================================================

    # util
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
        """
        Produce a median master by masking region of the spectrum

        Parameters
        ----------

        sub_dico : The sub_dictionnary used to  select the continuum
        continuum : The continuum to select (either linear or cubic)
        telluric_tresh : Treshold used to cover the position of the contaminated wavelength
        wave_min : The minimum xlim axis
        wave_max : The maximum xlim axis

        """

        mask_percentile = [None, 50]

        myf.print_box("\n---- RECIPE : PRODUCE MASTER MEDIAN SPECTRUM ----\n")

        self.import_table()
        self.import_material()
        load = self.material
        epsilon = 1e-6

        planet = self.planet

        kw = "_planet" * planet
        if kw != "":
            print("\n---- PLANET ACTIVATED ----")

        if sub_dico is None:
            sub_dico = self.dico_actif
        print("\n---- DICO %s used ----" % (sub_dico))

        if self.table["telluric_fwhm"][0] is None:
            fwhm = np.array([3.0] * len(self.table.jdb))
        else:
            try:
                fwhm = np.array(self.table["telluric_fwhm"])
            except:
                fwhm = np.array([3.0] * len(self.table.jdb))

        fwhm_max = [5, 8][self.instrument[:-2] == "CORALIE"]
        fwhm_min = [2, 3][self.instrument[:-2] == "CORALIE"]
        fwhm_default = [3, 5][self.instrument[:-2] == "CORALIE"]
        if np.percentile(fwhm, 95) > fwhm_max:
            logging.warn(
                "[WARNING] FWHM of tellurics larger than %.0f km/s (%.1f), reduced to default value of %.0f km/s"
                % (fwhm_max, np.percentile(fwhm, 95), fwhm_default)
            )
            fwhm = np.array([fwhm_default] * len(self.table.jdb))
        if np.percentile(fwhm, 95) < fwhm_min:
            logging.warn(
                "FWHM of tellurics smaller than %.0f km/s (%.1f), increased to default value of %.0f km/s"
                % (fwhm_min, np.percentile(fwhm, 95), fwhm_default)
            )
            fwhm = np.array([fwhm_default] * len(self.table.jdb))

        print("\n [INFO] FWHM of tellurics : %.1f km/s" % (np.percentile(fwhm, 95)))

        all_flux = []
        all_conti = []
        for i, name in enumerate(np.array(self.table["filename"])):
            file = pd.read_pickle(name)
            self.debug = name
            if not i:
                wavelength = file["wave"]
                self.wave = wavelength

            f = file["flux" + kw]
            f_std = file["flux_err"]
            c = file[sub_dico]["continuum_" + continuum]
            c_std = file["continuum_err"]

            f_norm, f_norm_std = myf.flux_norm_std(f, f_std, c, c_std)

            all_flux.append(f)
            all_conti.append(c)

        all_flux = np.array(all_flux)
        all_conti = np.array(all_conti)
        all_flux_norm = all_flux / all_conti

        mask = np.ones(len(self.table.jdb)).astype("bool")
        if jdb_range[2]:
            mask = (np.array(self.table.jdb) > jdb_range[0]) & (
                np.array(self.table.jdb) < jdb_range[1]
            )
        else:
            mask = (np.array(self.table.jdb) < jdb_range[0]) | (
                np.array(self.table.jdb) > jdb_range[1]
            )

        if sum(mask) < 40:
            print(
                Fore.YELLOW
                + "\n [WARNING] Not enough spectra %s the specified temporal range"
                % (["inside", "outside"][jdb_range[2] == 0])
                + Fore.RESET
            )
            mask = np.ones(len(self.table.jdb)).astype("bool")
        else:
            print(
                "\n [INFO] %.0f spectra %s the specified temporal range can be used for the median\n"
                % (sum(mask), ["inside", "outside"][jdb_range[2] == 0])
            )

        all_flux = all_flux[mask]
        all_conti = all_conti[mask]
        all_flux_norm = all_flux_norm[mask]

        berv = np.array(self.table["berv" + kw])[mask]
        rv_shift = np.array(self.table["rv_shift"])[mask]
        berv = berv - rv_shift

        model = pd.read_pickle(root + "/Python/Material/model_telluric.p")
        grid = model["wave"]
        spectre = model["flux_norm"]
        telluric = myc.tableXY(grid, spectre)
        telluric.find_min()

        all_min = np.array([telluric.x_min, telluric.y_min, telluric.index_min]).T
        all_min = all_min[1 - all_min[:, 1] > telluric_tresh]
        all_width = np.round(
            all_min[:, 0] * fwhm[:, np.newaxis] / 3e5 / np.median(np.diff(wavelength)),
            0,
        )
        all_width = np.nanpercentile(all_width, 95, axis=0)

        if supress_telluric:
            borders = np.array([all_min[:, 2] - all_width, all_min[:, 2] + all_width]).T
            telluric_mask = myf.flat_clustering(len(grid), borders) != 0
            all_mask2 = []
            for j in tqdm(berv):
                mask = myc.tableXY(myf.doppler_r(grid, j * 1000)[0], telluric_mask, 0 * grid)
                mask.interpolate(new_grid=wavelength, method="linear", interpolate_x=False)
                all_mask2.append(mask.y != 0)
            all_mask2 = np.array(all_mask2).astype("float")
        else:
            all_mask2 = np.zeros(np.shape(all_flux_norm))

        #        i=-1
        #        for j in tqdm(berv):
        #            i+=1
        #            borders = np.array([all_min[:,2]-all_width[i],all_min[:,2]+all_width[i]]).T
        #            telluric_mask = myf.flat_clustering(len(grid),borders)!=0
        #            mask = myc.tableXY(myf.doppler_r(grid,j*1000)[0],telluric_mask)
        #            mask.interpolate(new_grid=wavelength,method='linear')
        #            all_mask2.append(mask.y!=0)
        #         all_mask2 = np.array(all_mask2).astype('float')

        if supress_telluric:
            telluric_mask = telluric.y < (1 - telluric_tresh)
            all_mask = []
            for j in tqdm(berv):
                mask = myc.tableXY(myf.doppler_r(grid, j * 1000)[0], telluric_mask, 0 * grid)
                mask.interpolate(new_grid=wavelength, method="linear", interpolate_x=False)
                all_mask.append(mask.y != 0)
            all_mask = np.array(all_mask).astype("float")
        else:
            all_mask = np.zeros(np.shape(all_flux_norm))

        if shift_spectrum:
            rv_star = np.array(self.table["ccf_rv"])
            rv_star[np.isnan(rv_star)] = np.nanmedian(rv_star)
            rv_star -= np.median(rv_star)
            i = -1
            if method == "median":
                for j in tqdm(rv_star):
                    i += 1
                    mask = myc.tableXY(
                        myf.doppler_r(wavelength, j * 1000)[1],
                        all_flux_norm[i],
                        0 * wavelength,
                    )
                    mask.interpolate(new_grid=wavelength, method="linear", interpolate_x=False)
                    all_flux_norm[i] = mask.y.copy()
            else:
                # print(len(rv))
                # print(np.shape(all_flux))
                for j in tqdm(rv_star):
                    i += 1
                    mask = myc.tableXY(
                        myf.doppler_r(wavelength, j * 1000)[1],
                        all_flux[i],
                        0 * wavelength,
                    )
                    mask.interpolate(new_grid=wavelength, method="linear", interpolate_x=False)
                    all_flux[i] = mask.y.copy()
                    mask = myc.tableXY(
                        myf.doppler_r(wavelength, j * 1000)[1],
                        all_conti[i],
                        0 * wavelength,
                    )
                    mask.interpolate(new_grid=wavelength, method="linear", interpolate_x=False)
                    all_conti[i] = mask.y.copy()

        # plt.plot(wavelength,np.product(all_mask,axis=0))
        # plt.plot(wavelength,np.product(all_mask2,axis=0))
        print(
            "Percent always contaminated metric1 : %.3f %%"
            % (np.sum(np.product(all_mask, axis=0)) / len(wavelength) * 100)
        )
        print(
            "Percent always contaminated metric2 : %.3f %%"
            % (np.sum(np.product(all_mask2, axis=0)) / len(wavelength) * 100)
        )

        all_mask_nan1 = 1 - all_mask
        all_mask_nan1[all_mask_nan1 == 0] = np.nan

        all_mask_nan2 = 1 - all_mask2
        all_mask_nan2[all_mask_nan2 == 0] = np.nan

        print(mask_percentile)
        if mask_percentile[0] is None:
            mask_percentile[0] = np.ones(len(wavelength)).astype("bool")

        print(
            np.shape(wavelength),
            np.shape(all_flux_norm),
            np.shape(mask_percentile[0]),
            np.shape(mask_percentile[1]),
        )

        med = np.zeros(len(wavelength))
        med[mask_percentile[0]] = np.nanpercentile(
            all_flux_norm[:, mask_percentile[0]], mask_percentile[1], axis=0
        )
        med[~mask_percentile[0]] = np.nanpercentile(
            all_flux_norm[:, ~mask_percentile[0]], 50, axis=0
        )

        del mask_percentile

        # med1 = np.nanmedian(all_flux_norm*all_mask_nan1,axis=0)
        # med2 = np.nanmedian(all_flux_norm*all_mask_nan2,axis=0)

        mean = np.nansum(all_flux, axis=0) / np.nansum(all_conti, axis=0)
        mean1 = np.nansum(all_flux * all_mask_nan1, axis=0) / (
            np.nansum(all_conti * all_mask_nan1, axis=0) + epsilon
        )
        mean2 = np.nansum(all_flux * all_mask_nan2, axis=0) / (
            np.nansum(all_conti * all_mask_nan2, axis=0) + epsilon
        )
        mean2[mean2 == 0] = np.nan
        mean1[mean1 == 0] = np.nan
        mean1[mean1 != mean1] = mean2[mean1 != mean1]
        mean1[mean1 != mean1] = med[mean1 != mean1]
        mean2[mean2 != mean2] = mean1[mean2 != mean2]
        # med1[med1!=med1] = mean1[med1!=med1]
        # med2[med2!=med2] = mean1[med2!=med2]
        all_flux_diff_med = all_flux_norm - med
        tresh = 1.5 * myf.IQ(np.ravel(all_flux_diff_med)) + np.nanpercentile(all_flux_diff_med, 75)

        mean1[mean1 > (1 + tresh)] = 1

        if method != "median":
            self.reference = (wavelength, mean1)
        else:
            self.reference = (wavelength, med)

        plt.figure(figsize=(16, 8))
        plt.plot(wavelength, med, color="b", ls="-", label="median")
        plt.plot(wavelength, mean1, color="g", ls="-", label="mean1")
        plt.plot(wavelength, mean2, color="r", ls="-", label="mean2")
        plt.legend(loc=2)
        # plt.plot(wavelength,med,color='b',ls='-.')
        # plt.plot(wavelength,med1,color='g',ls='-.')
        # plt.plot(wavelength,med2,color='r',ls='-.')

        all_flux_diff_mean = all_flux_norm - mean
        all_flux_diff_med = all_flux_norm - med
        all_flux_diff1_mean = all_flux_norm - mean1
        # all_flux_diff1_med = all_flux_norm - med1
        # all_flux_diff2_mean = all_flux_norm - mean2
        # all_flux_diff2_med = all_flux_norm - med2

        idx_min = int(myf.find_nearest(wavelength, wave_min)[0])
        idx_max = int(myf.find_nearest(wavelength, wave_max)[0])

        plt.figure(figsize=(16, 8))
        plt.subplot(2, 1, 2)
        plt.imshow(
            all_flux_diff_mean[::-1, idx_min:idx_max],
            aspect="auto",
            vmin=-0.005,
            vmax=0.005,
            cmap="plasma",
        )
        plt.imshow(all_mask2[::-1, idx_min:idx_max], aspect="auto", alpha=0.2, cmap="Reds")
        ax = plt.gca()
        plt.subplot(2, 1, 1, sharex=ax, sharey=ax)
        plt.imshow(
            all_flux_diff_mean[::-1, idx_min:idx_max],
            aspect="auto",
            vmin=-0.005,
            vmax=0.005,
            cmap="plasma",
        )
        plt.imshow(all_mask[::-1, idx_min:idx_max], aspect="auto", alpha=0.2, cmap="Reds")

        if len(berv) > 15:
            plt.figure(figsize=(16, 8))
            plt.subplot(2, 1, 1)

            plt.title("Median")
            myf.my_colormesh(
                wavelength[idx_min : idx_max + 1],
                np.arange(len(all_mask)),
                all_flux_diff_med[:, idx_min : idx_max + 1],
                vmin=-0.005,
                vmax=0.005,
                cmap="plasma",
            )
            ax = plt.gca()

            plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
            plt.title("Masked mean")
            myf.my_colormesh(
                wavelength[idx_min : idx_max + 1],
                np.arange(len(all_mask)),
                all_flux_diff1_mean[:, idx_min : idx_max + 1],
                vmin=-0.005,
                vmax=0.005,
                cmap="plasma",
            )

        load["wave"] = wavelength
        if method != "median":
            load["reference_spectrum"] = mean1
        else:
            load["reference_spectrum"] = med - np.median(all_flux_diff_med)

        ref = np.array(load["reference_spectrum"])
        ref = self.yarara_non_zero_flux(spectrum=ref, min_value=None)
        load["reference_spectrum"] = ref

        load.loc[load["reference_spectrum"] < 0, "reference_spectrum"] = 0

        if save:
            io.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))
        else:
            return np.array(load["reference_spectrum"])

    # util
    def yarara_median_master(
        self,
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
        """
        Produce a median master by masking region of the spectrum

        Parameters
        ----------

        sub_dico : The sub_dictionnary used to  select the continuum
        continuum : The continuum to select (either linear or cubic)
        telluric_tresh : Treshold used to cover the position of the contaminated wavelength
        wave_min : The minimum xlim axis
        wave_max : The maximum xlim axis

        """
        if method:
            self.yarara_median_master_backup(
                sub_dico=sub_dico,
                method=method,
                continuum=continuum,
                telluric_tresh=telluric_tresh,
                wave_min=wave_min,
                wave_max=wave_max,
                jdb_range=jdb_range,
                supress_telluric=supress_telluric,
                shift_spectrum=shift_spectrum,
                mask_percentile=mask_percentile,
                save=save,
            )

        if method == "max":
            myf.print_box("\n---- RECIPE : PRODUCE MASTER MAX SPECTRUM ----\n")

            self.import_table()
            self.import_material()
            load = self.material
            tab = self.table
            planet = self.planet

            epsilon = 1e-12

            kw = "_planet" * planet
            if kw != "":
                print("\n---- PLANET ACTIVATED ----")

            if sub_dico is None:
                sub_dico = self.dico_actif
            print("---- DICO %s used ----" % (sub_dico))

            all_flux = []
            all_conti = []
            snr = np.array(tab["snr"])
            for i, name in enumerate(np.array(tab["filename"])):
                file = pd.read_pickle(name)
                if not i:
                    wavelength = file["wave"]
                    self.wave = wavelength

                f = file["flux" + kw]
                f_std = file["flux_err"]
                c = file[sub_dico]["continuum_" + continuum]
                c_std = file["continuum_err"]

                f_norm, f_norm_std = myf.flux_norm_std(f, f_std, c, c_std)

                all_flux.append(f_norm)
                all_conti.append(file["matching_diff"]["continuum_" + continuum])

            all_conti = np.array(all_conti)
            all_flux = np.array(all_flux) * all_conti.copy()
            all_flux_backup = all_flux / all_conti
            med = np.nanmedian(all_flux_backup, axis=0)

            berv = np.array(tab["berv"])
            rv_shift = np.array(tab["rv_shift"])
            berv = berv - rv_shift

            sort = np.argsort(berv)
            snr = np.array(tab["snr"])
            snr_sorted = snr[sort]

            if bin_snr is not None:
                val = int(np.sum(snr_sorted**2) // (bin_snr**2))
                if val > 4:
                    bin_berv = val
                else:
                    print(
                        "The expected SNR cannot be reached since the total SNR is about : %.0f"
                        % (np.sqrt(np.sum(snr_sorted**2)))
                    )
                    print(
                        "The maximum value allowed : %.0f" % (np.sqrt(np.sum(snr_sorted**2) / 5))
                    )
                    bin_snr = np.sqrt(np.sum(snr_sorted**2) / 5) - 50
                    bin_berv = int(np.sum(snr_sorted**2) // (bin_snr**2))

            # plt.plot(np.sqrt(np.cumsum(snr_sorted**2)))
            snr_lim = np.linspace(0, np.sum(snr_sorted**2), bin_berv + 1)
            berv_bin = berv[sort][myf.find_nearest(np.cumsum(snr_sorted**2), snr_lim)[0]]
            berv_bin[0] -= 1  # to ensure the first point to be selected

            mask_bin = (berv > berv_bin[0:-1][:, np.newaxis]) & (
                berv <= berv_bin[1:][:, np.newaxis]
            )
            berv_bin = (
                berv_bin[:-1][np.sum(mask_bin, axis=1) != 0]
                + np.diff(berv_bin)[np.sum(mask_bin, axis=1) != 0] / 2
            )
            mask_bin = mask_bin[np.sum(mask_bin, axis=1) != 0]

            snr_stacked = []
            all_flux_norm = []
            all_snr = []
            for j in range(len(mask_bin)):
                all_flux_norm.append(
                    np.sum(all_flux[mask_bin[j]], axis=0)
                    / (np.sum(all_conti[mask_bin[j]], axis=0) + epsilon)
                )
                snr_stacked.append(np.sqrt(np.sum((snr[mask_bin[j]]) ** 2)))
                all_snr.append(snr[mask_bin[j]])
            all_flux_norm = np.array(all_flux_norm)

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            self.yarara_get_berv_value(0, Draw=True, new=False, save_fig=False, light_graphic=True)
            ax = plt.gca()
            for j in berv_bin:
                plt.axhline(y=j, color="k", alpha=0.2)
            plt.axhline(y=0, color="k", ls=":")

            plt.subplot(1, 2, 2, sharey=ax)
            plt.axhline(y=0, color="k", ls=":")

            plt.plot(snr_stacked, berv_bin, "bo-", alpha=0.3)
            curve = myc.tableXY(snr_stacked, berv_bin)
            curve.myscatter(
                num=False,
                liste=[len(all_snr[j]) for j in range(len(all_snr))],
                color="k",
                factor=50,
            )
            plt.xlabel("SNR stacked", fontsize=13)
            plt.ylabel("BERV [km/s]", fontsize=13)

            print("SNR of binned spetcra around %.0f" % (np.mean(snr_stacked)))

            for j in range(len(all_flux_norm)):
                all_flux_norm[j] = myf.smooth(all_flux_norm[j], shape="savgol", box_pts=smooth_box)

            mean1 = np.max(all_flux_norm, axis=0)

            self.reference_max = (wavelength, mean1)

            mean1 -= np.median(mean1 - self.reference[1])

            all_flux_diff_mean = all_flux_backup - mean1
            all_flux_diff_med = all_flux_backup - med
            all_flux_diff1_mean = all_flux_backup - self.reference[1]

            idx_min = int(myf.find_nearest(wavelength, wave_min)[0])
            idx_max = int(myf.find_nearest(wavelength, wave_max)[0])

            plt.figure(figsize=(16, 8))
            plt.subplot(3, 1, 1)

            plt.title("Median")
            myf.my_colormesh(
                wavelength[idx_min : idx_max + 1],
                np.arange(len(berv)),
                all_flux_diff_med[sort][:, idx_min : idx_max + 1],
                vmin=-0.005,
                vmax=0.005,
                cmap="plasma",
            )
            ax = plt.gca()

            plt.subplot(3, 1, 2, sharex=ax, sharey=ax)
            plt.title("Max")
            myf.my_colormesh(
                wavelength[idx_min : idx_max + 1],
                np.arange(len(berv)),
                all_flux_diff_mean[sort][:, idx_min : idx_max + 1],
                vmin=-0.005,
                vmax=0.005,
                cmap="plasma",
            )

            plt.subplot(3, 1, 3, sharex=ax, sharey=ax)
            plt.title("Masked weighted mean")
            myf.my_colormesh(
                wavelength[idx_min : idx_max + 1],
                np.arange(len(berv)),
                all_flux_diff1_mean[sort][:, idx_min : idx_max + 1],
                vmin=-0.005,
                vmax=0.005,
                cmap="plasma",
            )

            load["wave"] = wavelength
            load["reference_spectrum"] = mean1
            io.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))

    # =============================================================================
    # CUT SPECTRUM
    # =============================================================================

    # util
    def yarara_cut_spectrum(self, wave_min=None, wave_max=None):
        """Cut the spectrum time-series borders to reach the specified wavelength limits (included)
        There is no way to cancel this step ! Use it wisely."""

        myf.print_box("\n---- RECIPE : SPECTRA CROPING ----\n")

        directory = self.directory
        self.import_material()
        load = self.material

        if wave_min == "auto":
            w0 = np.min(load["wave"])
            a, b, c = self.yarara_map(
                "matching_diff",
                reference="norm",
                wave_min=w0,
                wave_max=w0 + 1000,
                Plot=False,
            )
            a = a[
                np.sum(a < 1e-5, axis=1) > 100
            ]  # only kept spectra with more than 10 values below 0

            if len(a):
                t = np.nanmean(np.cumsum(a < 1e-5, axis=1).T / np.sum(a < 1e-5, axis=1), axis=1)
                i0 = myf.find_nearest(t, 0.99)[0][
                    0
                ]  # supress wavelength range until 99% of nan values
                if i0:
                    wave_min = load["wave"][i0]
                    print(
                        " [INFO] Automatic detection of the blue edge spectrum found at %.2f AA\n"
                        % (wave_min)
                    )
                else:
                    wave_min = None
            else:
                wave_min = None

        maps = glob.glob(self.dir_root + "CORRECTION_MAP/*.p")
        if len(maps):
            for name in maps:
                correction_map = pd.read_pickle(name)
                old_wave = correction_map["wave"]
                length = len(old_wave)
                idx_min = 0
                idx_max = len(old_wave)
                if wave_min is not None:
                    idx_min = int(myf.find_nearest(old_wave, wave_min)[0])
                if wave_max is not None:
                    idx_max = int(myf.find_nearest(old_wave, wave_max)[0] + 1)
                correction_map["wave"] = old_wave[idx_min:idx_max]
                correction_map["correction_map"] = correction_map["correction_map"][
                    :, idx_min:idx_max
                ]
                io.pickle_dump(correction_map, open(name, "wb"))
                print("%s modified" % (name.split("/")[-1]))

        time.sleep(1)

        old_wave = np.array(load["wave"])
        length = len(old_wave)
        idx_min = 0
        idx_max = len(old_wave)
        if wave_min is not None:
            idx_min = int(myf.find_nearest(old_wave, wave_min)[0])
        if wave_max is not None:
            idx_max = int(myf.find_nearest(old_wave, wave_max)[0] + 1)

        new_wave = old_wave[idx_min:idx_max]
        wave_min = np.min(new_wave)
        wave_max = np.max(new_wave)

        load = load[idx_min:idx_max]
        load = load.reset_index(drop=True)
        io.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))

        try:
            file_kitcat = pd.read_pickle(self.dir_root + "KITCAT/kitcat_spectrum.p")
            wave_kit = file_kitcat["wave"]
            idx_min = 0
            idx_max = len(wave_kit)
            if wave_min is not None:
                idx_min = int(myf.find_nearest(wave_kit, wave_min)[0])
            if wave_max is not None:
                idx_max = int(myf.find_nearest(wave_kit, wave_max)[0] + 1)
            for kw in [
                "wave",
                "flux",
                "correction_factor",
                "flux_telluric",
                "flux_uncorrected",
                "continuum",
            ]:
                file_kitcat[kw] = np.array(file_kitcat[kw])[idx_min:idx_max]
            io.pickle_dump(file_kitcat, open(self.dir_root + "KITCAT/kitcat_spectrum.p", "wb"))
        except:
            pass

        files = glob.glob(directory + "RASSI*.p")
        files = np.sort(files)

        file_ref = self.import_spectrum()
        old_wave = np.array(file_ref["wave"])
        length = len(old_wave)
        idx_min = 0
        idx_max = len(old_wave)
        if wave_min is not None:
            idx_min = int(myf.find_nearest(old_wave, wave_min)[0])
        if wave_max is not None:
            idx_max = int(myf.find_nearest(old_wave, wave_max)[0] + 1)

        new_wave = old_wave[idx_min:idx_max]
        wave_min = np.min(new_wave)
        wave_max = np.max(new_wave)

        for j in tqdm(files):
            file = pd.read_pickle(j)
            file["parameters"]["wave_min"] = wave_min
            file["parameters"]["wave_max"] = wave_max

            anchors_wave = file["matching_anchors"]["anchor_wave"]
            mask = (anchors_wave >= wave_min) & (anchors_wave <= wave_max)
            file["matching_anchors"]["anchor_index"] = (
                file["matching_anchors"]["anchor_index"][mask] - idx_min
            )
            file["matching_anchors"]["anchor_flux"] = file["matching_anchors"]["anchor_flux"][mask]
            file["matching_anchors"]["anchor_wave"] = file["matching_anchors"]["anchor_wave"][mask]

            anchors_wave = file["output"]["anchor_wave"]
            mask = (anchors_wave >= wave_min) & (anchors_wave <= wave_max)
            file["output"]["anchor_index"] = file["output"]["anchor_index"][mask] - idx_min
            file["output"]["anchor_flux"] = file["output"]["anchor_flux"][mask]
            file["output"]["anchor_wave"] = file["output"]["anchor_wave"][mask]

            fields = file.keys()
            for field in fields:
                if type(file[field]) == dict:
                    sub_fields = file[field].keys()
                    for sfield in sub_fields:
                        if type(file[field][sfield]) == np.ndarray:
                            if len(file[field][sfield]) == length:
                                file[field][sfield] = file[field][sfield][idx_min:idx_max]
                elif type(file[field]) == np.ndarray:
                    if len(file[field]) == length:
                        file[field] = file[field][idx_min:idx_max]
            io.save_pickle(j, file)

    # =============================================================================
    # COMPUTE ALLTHE TEMPERATURE SENSITIVE RATIO
    # =============================================================================

    # processing
    def yarara_activity_index(
        self,
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
        """
        Produce the activity proxy time-series. Need to cancel the RV systemic of the star

        Parameters
        ----------
        sub_dico : The sub_dictionnary used to  select the continuum
        continuum : The continuum to select (either linear or cubic)
        rv : The RV of the star system
        plot : True/False, Plot the proxies time-series
        debug : True/False, Plot the intermediate graphicwith the spectrum and area extraction for the proxies
        ron : read-out-noise error injected by reading pixels
        calib_std : std error due to flat-field photon noise (5.34e-4 or 10.00e-4 Cretignier+20)
        """

        myf.print_box("\n---- RECIPE : ACTIVITY PROXIES EXTRACTION ----\n")

        directory = self.directory
        rv_sys = self.rv_sys
        self.import_table()

        self.import_material()
        load = self.material
        kw = "_planet" * self.planet
        if kw != "":
            print("\n---- PLANET ACTIVATED ----")
        time.sleep(1)

        epsilon = 1e-12
        save_kw = save

        try:
            self.import_star_info()
            bv = self.star_info["BV"]["fixed"]
            if np.isnan(bv):
                bv = 0
                print(Fore.YELLOW + " [WARNING] No BV value given for the star" + Fore.RESET)
            # conv_offset = -0.60 - 5.99*bv + 2.51*bv**2 #old calib
            # conv_slope = 4.55 - 7.30*bv + 3.61*bv**2 #old calib

            conv_offset = 1.13 - 10.13 * bv + 4.97 * bv**2  # old calib
            conv_slope = 5.76 - 10.0 * bv + 5.10 * bv**2  # old calib

        except:
            conv_offset = 0
            conv_slope = 1

        files = glob.glob(directory + "RASSI*.p")
        files = np.sort(files)

        # [center, half-window, hole_size, half-window-continuum,database_kw, subplot]
        Ca2H = [3968.47, 0.45, 0, 1.5, "CaIIH", None]
        Ca2K = [3933.66, 0.45, 0, 1.5, "CaIIK", None]
        Ca1 = [4226.72, 1.50, 0, 1.5, "CaI", 2]
        Mg1a = [5167.32, 0.50, 0, 1.5, "MgIa", 6]
        Mg1b = [5172.68, 0.50, 0, 1.5, "MgIb", 7]
        Mg1c = [5183.60, 0.50, 0, 1.5, "MgIc", 8]
        NaDl = [5889.95, 0.50, 0, 0, "NaD1", 3]
        NaDr = [5895.92, 0.50, 0, 0, "NaD2", 4]
        Ha = [6562.79, 0.35, 0, 0, "Ha", 9]
        Hb = [4861.35, 0.35, 0, 1.5, "Hb", 10]
        Hc = [4340.47, 0.35, 0, 1.5, "Hc", 11]
        Hd = [4101.73, 0.15, 0, 1.5, "Hd", 12]
        Heps = [3889.04, 0.10, 0, 1.5, "Heps", None]
        He1D3 = [5875.62, 0.15, 0, 0, "HeID3", 5]

        all_proxies = [
            Ca2H,
            Ca2K,
            Ca1,
            Mg1a,
            Mg1b,
            Mg1c,
            NaDl,
            NaDr,
            Ha,
            Hb,
            Hc,
            Hd,
            Heps,
            He1D3,
        ]

        # FeXIV = 5302.86 ;

        # calib_std = 5.34e-4 #from Cretignier+20 precision linear + clustering
        # calib_std = 10.00e-4 #from Cretignier+20 precision linear + clustering (new sun) related to flat field SNR

        fluxes = []
        err_fluxes = []
        jdb = []
        snrs = []
        count = 0

        file_random = self.import_spectrum()
        waves = file_random["wave"]
        all_prox_names = np.array(all_proxies)[:, 4]
        proxy_found = ((np.array(all_proxies)[:, 0] - np.nanmin(waves[0])) > 0) & (
            (np.nanmax(waves[0])) > 0
        )
        all_proxies = list(np.array(all_proxies)[proxy_found])

        dgrid = np.mean(np.diff(waves))
        for j in tqdm(files):
            count += 1
            file = pd.read_pickle(j)
            try:
                jdb.append(file["parameters"]["jdb"])
            except:
                jdb.append(count)

            snrs.append(file["parameters"]["SNR_5500"])

            f = file["flux" + kw]
            f_std = file["flux_err"]
            c = file[sub_dico]["continuum_" + continuum]
            c_std = file["continuum_err"]

            f_norm, f_norm_std = myf.flux_norm_std(f, f_std, c, c_std)
            dustbin, f_norm_std = myf.flux_norm_std(
                f, f_std, file["matching_diff"]["continuum_" + continuum], c_std
            )

            fluxes.append(f_norm)
            err_fluxes.append(f_norm_std)
            self.debug = j

        waves = np.array(waves)
        flux = np.array(fluxes)
        err_flux = np.array(err_fluxes)
        flux *= np.array(load["correction_factor"])
        err_flux *= np.array(load["correction_factor"])
        snrs = np.array(snrs)

        noise_matrix, noise_values = self.yarara_poissonian_noise(
            noise_wanted=p_noise, wave_ref=None, flat_snr=True
        )
        flux += noise_matrix
        err_flux = np.sqrt(err_flux**2 + noise_values**2)

        for maps in substract_map:
            flux = self.yarara_substract_map(flux, maps, correction_factor=True)

        for maps in add_map:
            flux = self.yarara_add_map(flux, maps, correction_factor=True)

        jdb = np.array(jdb)
        ref = snrs.argmax()
        wave_ref = waves
        wave = waves

        flux_ref = np.median(flux, axis=0)
        ratio = flux / (flux_ref + epsilon)
        try:
            mask_ghost = np.array(load["ghost_a"]).astype("bool")
        except:
            mask_ghost = np.zeros(len(flux_ref)).astype("bool")
        ratio[:, mask_ghost] = np.nan

        if rv_sys is None:
            if file_random["parameters"]["RV_sys"] is not None:
                rv_sys = 1000 * file_random["parameters"]["RV_sys"]
            else:
                rv_sys = 0
        else:
            rv_sys *= 1000

        def find_proxy(vec):
            center = myf.doppler_r(vec[0], rv_sys)[0]
            left = myf.doppler_r(vec[0] - vec[1], rv_sys)[0]
            right = myf.doppler_r(vec[0] + vec[1], rv_sys)[0]

            center_idx_proxy = myf.find_nearest(wave, center)[0]
            left_idx_proxy = myf.find_nearest(wave, left)[0]
            right_idx_proxy = myf.find_nearest(wave, right)[0]

            left = myf.doppler_r(vec[0] - vec[2], rv_sys)[0]
            right = myf.doppler_r(vec[0] + vec[2], rv_sys)[0]

            left_idx_hole = myf.find_nearest(wave, left)[0]
            right_idx_hole = myf.find_nearest(wave, right)[0]

            left = myf.doppler_r(vec[0] - vec[3], rv_sys)[0]
            right = myf.doppler_r(vec[0] + vec[3], rv_sys)[0]

            left_idx_cont = myf.find_nearest(wave, left)[0]
            right_idx_cont = myf.find_nearest(wave, right)[0]

            return (
                int(center_idx_proxy),
                int(left_idx_proxy),
                int(right_idx_proxy),
                int(left_idx_hole),
                int(right_idx_hole),
                int(left_idx_cont),
                int(right_idx_cont),
            )

        def extract_proxy(vec, kernel=None):
            c, l, r, l_hole, r_hole, l_cont, r_cont = find_proxy(vec)
            if kernel is None:
                continuum = 1
                if r != l:
                    r += 1
                if l_hole != r_hole:
                    r_hole += 1
                if l_cont != l:
                    r_cont += 1
                    continuum = np.hstack([ratio[:, l_cont:l], ratio[:, r:r_cont]])
                    continuum = np.nanmedian(continuum, axis=1)
                    continuum[np.isnan(continuum)] = 1
                proxy = np.sum(flux[:, l:r], axis=1) - np.sum(flux[:, l_hole:r_hole], axis=1)
                proxy_std = np.sum((err_flux[:, l:r]) ** 2, axis=1) - np.sum(
                    (err_flux[:, l_hole:r_hole]) ** 2, axis=1
                )
                proxy_std = np.sqrt(proxy_std)
                norm_proxy = (r - l) - (r_hole - l_hole)

                proxy /= continuum
                proxy_std /= continuum
            else:
                kernel /= np.sum(abs(kernel))
                proxy = np.sum((flux - np.median(flux, axis=0)) * kernel, axis=1)
                proxy_std = np.sum((kernel * err_flux) ** 2, axis=1)
                proxy_std = np.sqrt(proxy_std)
                norm_proxy = 1

            prox = myc.tableXY(jdb, proxy, proxy_std)
            prox.rms_w()
            proxy_rms = prox.rms
            windex = int(1 / dgrid)
            mask_proxy = abs(np.arange(len(flux.T)) - c) < windex
            slope = np.median(
                (flux[:, mask_proxy] - np.mean(flux[:, mask_proxy], axis=0))
                / ((proxy - np.mean(proxy))[:, np.newaxis]),
                axis=0,
            )

            s = myc.tableXY(wave[mask_proxy], slope)
            s.smooth(box_pts=7, shape="savgol")
            s.center_symmetrise(myf.doppler_r(vec[0], rv_sys)[0], replace=True)
            slope = s.y

            t = myc.table(flux[:, mask_proxy] - np.mean(flux[:, mask_proxy], axis=0))
            t.rms_w(1 / err_flux[:, mask_proxy] ** 2, axis=0)
            rslope = np.zeros(len(flux.T))
            rslope[mask_proxy] = slope
            rms = np.ones(len(flux.T))
            rms[mask_proxy] = t.rms

            rcorr = (
                rslope * proxy_rms / (rms + epsilon)
            )  # need good weighting of the proxy and the flux

            if norm_proxy:
                proxy /= norm_proxy
                proxy_std /= norm_proxy
                return proxy, proxy_std, rcorr, rslope
            else:
                return 0 * proxy, 0 * proxy_std, 0 * wave, 0 * wave

        save = {"null": 0}
        mask_activity = np.zeros(len(waves))
        all_rcorr = []
        all_rslope = []
        for p in all_proxies:
            c, l, r, lh, rh, lc, rc = find_proxy(p)
            mask_activity[l:r] = 1
            proxy, proxy_std, rcorr, rslope = extract_proxy(p)
            all_rcorr.append(rcorr)
            all_rslope.append(rslope)
            save[p[4]] = proxy
            save[p[4] + "_std"] = proxy_std
        del save["null"]

        all_rcorr = np.array(all_rcorr)
        all_rslope = np.array(all_rslope)

        for n in all_prox_names:
            if n not in save.keys():
                save[n] = np.zeros(len(jdb))
                save[n + "_std"] = np.zeros(len(jdb))

        save_backup = save.copy()

        if optimize:
            save = {"null": 0}
            mask_activity = np.zeros(len(waves))
            for i, p in enumerate(all_proxies):
                mask_activity[all_rslope[i] != 0] = 1
                proxy, proxy_std, dust, dust = extract_proxy(p, kernel=all_rslope[i])
                save[p[4]] = proxy
                save[p[4] + "_std"] = proxy_std
            del save["null"]

            self.all_kernels = all_rslope

        del ratio

        if debug:
            plt.figure(figsize=(18, 9))
            plt.subplot(3, 1, 1)
            plt.plot(wave_ref, all_rslope.T, color="k")
            plt.axhline(y=0, color="r")
            ax = plt.gca()
            plt.subplot(3, 1, 2, sharex=ax)
            plt.plot(wave_ref, all_rcorr.T, color="k")
            plt.axhline(y=0, color="r")
            plt.ylim(-1, 1)
            plt.subplot(3, 1, 3, sharex=ax)
            plt.plot(wave_ref, flux_ref, color="k")
            for p in all_proxies:
                center = myf.doppler_r(p[0], rv_sys)[0]
                hw = p[1]
                plt.axvspan(xmin=center - hw, xmax=center + hw, alpha=0.5, color="r")
                plt.axvline(x=center, color="r")

        load["activity_proxies"] = mask_activity.astype("int")
        io.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))

        if not np.sum(abs(save["CaIIK"])):
            save["CaIIK"] = save["Ha"] + 0.01
            save["CaIIH"] = save["Ha"] + 0.01
            save["CaIIK_std"] = save["Ha_std"]
            save["CaIIH_std"] = save["Ha_std"]
            save_backup["CaIIK"] = save_backup["Ha"]
            save_backup["CaIIH"] = save_backup["Ha"]
            save_backup["CaIIK_std"] = save_backup["Ha_std"]
            save_backup["CaIIH_std"] = save_backup["Ha_std"]

        save["CaII"] = 0.5 * (save["CaIIK"] + save["CaIIH"])
        save["CaII_std"] = 0.5 * np.sqrt((save["CaIIK_std"]) ** 2 + (save["CaIIH_std"]) ** 2)

        save["NaD"] = 0.5 * (save["NaD1"] + save["NaD2"])
        save["NaD_std"] = 0.5 * np.sqrt((save["NaD1_std"]) ** 2 + (save["NaD2_std"]) ** 2)

        save["MgI"] = 0.5 * (save["MgIa"] + save["MgIb"] + save["MgIc"])
        save["MgI_std"] = 0.5 * np.sqrt(
            (save["MgIa_std"]) ** 2 + (save["MgIb_std"]) ** 2 + (save["MgIc_std"]) ** 2
        )

        save_backup["CaII"] = 0.5 * (save_backup["CaIIK"] + save_backup["CaIIH"])
        save_backup["CaII_std"] = 0.5 * np.sqrt(
            (save_backup["CaIIK_std"]) ** 2 + (save_backup["CaIIH_std"]) ** 2
        )

        shift = np.mean(save["CaII"]) - np.mean(save_backup["CaII"])

        save["RHK"] = conv_slope * np.log10(save["CaII"].copy() - shift) + conv_offset
        save["RHK_std"] = (
            conv_slope
            * (save["CaII_std"].copy() + calib_std)
            / abs(save["CaII"].copy() - shift)
            / np.log(10)
        )

        save_backup["RHK"] = conv_slope * np.log10(save_backup["CaII"].copy()) + conv_offset
        save_backup["RHK_std"] = (
            conv_slope
            * (save_backup["CaII_std"].copy() + calib_std)
            / abs(save_backup["CaII"].copy())
            / np.log(10)
        )

        self.all_proxies = save
        self.all_proxies_name = list(save.keys())[::2]

        all_proxies.append([0, 0, 0, 0, "CaII", None])
        all_proxies.append([0, 0, 0, 0, "NaD", None])
        all_proxies.append([0, 0, 0, 0, "MgI", None])
        all_proxies.append([0, 0, 0, 0, "RHK", 1])

        for j in list(save.keys()):
            if len(j.split("_std")) == 2:
                save[j] += calib_std

        if save_kw:
            print("\n Saving activity proxies...")
            if False:  # no more used 02.07.21
                for i, j in enumerate(files):
                    file = pd.read_pickle(j)
                    # print('File (%.0f/%.0f) %s SNR %.0f reduced'%(i+1,len(files),j,snrs[i]))
                    for p in all_proxies:
                        file["parameters"][p[4]] = save[p[4]][i]
                        file["parameters"][p[4] + "_std"] = save[p[4] + "_std"][i]
                    io.pickle_dump(file, open(j, "wb"))

                self.yarara_analyse_summary()
            else:
                self.yarara_obs_info(kw=pd.DataFrame(save))

        self.ca2k = myc.tableXY(jdb, save["CaIIK"], save["CaIIK_std"] + calib_std)
        self.ca2h = myc.tableXY(jdb, save["CaIIH"], save["CaIIH_std"] + calib_std)
        self.ca2 = myc.tableXY(jdb, save["CaII"], save["CaII_std"] + calib_std)
        self.rhk = myc.tableXY(jdb, save["RHK"], save["RHK_std"])
        self.mg1 = myc.tableXY(jdb, save["MgI"], save["MgI_std"] + calib_std)
        self.mga = myc.tableXY(jdb, save["MgIa"], save["MgIa_std"] + calib_std)
        self.mgb = myc.tableXY(jdb, save["MgIb"], save["MgIb_std"] + calib_std)
        self.mgc = myc.tableXY(jdb, save["MgIc"], save["MgIc_std"] + calib_std)
        self.nad = myc.tableXY(jdb, save["NaD"], save["NaD_std"] + calib_std)
        self.nad1 = myc.tableXY(jdb, save["NaD1"], save["NaD1_std"] + calib_std)
        self.nad2 = myc.tableXY(jdb, save["NaD2"], save["NaD2_std"] + calib_std)
        self.ha = myc.tableXY(jdb, save["Ha"], save["Ha_std"] + calib_std)
        self.hb = myc.tableXY(jdb, save["Hb"], save["Hb_std"] + calib_std)
        self.hc = myc.tableXY(jdb, save["Hc"], save["Hc_std"] + calib_std)
        self.hd = myc.tableXY(jdb, save["Hd"], save["Hd_std"] + calib_std)
        self.heps = myc.tableXY(jdb, save["Heps"], save["Heps_std"] + calib_std)
        self.hed3 = myc.tableXY(jdb, save["HeID3"], save["HeID3_std"] + calib_std)
        self.ca1 = myc.tableXY(jdb, save["CaI"], save["CaI_std"] + calib_std)

        self.infos["latest_dico_activity"] = sub_dico

        if plot:
            phase = myf.get_phase(np.array(self.table.jdb), 365.25)
            for name, modulo, phase_mod in zip(["", "_1year"], [None, 365.25], [None, phase]):
                titles = [
                    "CaII H&K",
                    "CaI",
                    "NaD_left",
                    "NaD_right",
                    "HeID3",
                    "MgIa",
                    "MgIb",
                    "MgIc",
                    r"$H_\alpha$",
                    r"$H_\beta$",
                    r"$H_\gamma$",
                    r"$H_\delta$",
                ]
                plt.figure(figsize=(20, 10))
                plt.subplot(3, 4, 1)
                ax = plt.gca()
                for p in all_proxies:
                    if p[5] is not None:
                        num = p[5]
                        plt.subplot(3, 4, num, sharex=ax)
                        plt.title(titles[num - 1])

                        vec = myc.tableXY(
                            jdb,
                            save_backup[p[4]],
                            save_backup[p[4] + "_std"] + calib_std,
                        )
                        vec.plot(
                            capsize=0,
                            zorder=1,
                            color=["k", "r"][int(optimize)],
                            modulo=modulo,
                            phase_mod=phase,
                        )

                        if optimize:
                            vec2 = myc.tableXY(jdb, save[p[4]], save[p[4] + "_std"] + calib_std)
                            vec2.y -= np.mean(vec2.y)
                            vec2.y += np.mean(vec.y)
                            vec2.plot(
                                capsize=0,
                                color="k",
                                zorder=10,
                                modulo=modulo,
                                phase_mod=phase,
                            )

                        myf.auto_axis(vec.y, m=5)
                        plt.xlabel("Time")
                        plt.ylabel("Proxy [unit arb.]")
                plt.subplots_adjust(
                    left=0.07,
                    right=0.93,
                    top=0.95,
                    bottom=0.08,
                    wspace=0.3,
                    hspace=0.35,
                )
                plt.savefig(self.dir_root + "IMAGES/all_proxies" + name + ".pdf")

    # =============================================================================
    # COMPUTE THE TELLURIC CCF
    # =============================================================================

    # telluric
    def yarara_telluric(
        self,
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

        """
        Plot all the RASSINE spectra in the same plot

        Parameters
        ----------
        sub_dico : The sub_dictionnary used to  select the continuum
        continuum : The continuum to select (either linear or cubic)
        mask : The telluric mask used to cross correlate with the spectrum (mask should be located in MASK_CCF)
        reference : True/False or 'norm', True use the matching anchors of reference, False use the continuum of each spectrum, norm use the continuum normalised spectrum (not )
        display_ccf : display all the ccf
        normalisation : 'left' or 'slope'. if left normalise the CCF by the most left value, otherwise fit a line between the two highest point
        planet : True/False to use the flux containing the injected planet or not

        """

        kw = "_planet" * self.planet
        if kw != "":
            print("\n---- PLANET ACTIVATED ----")

        myf.print_box("\n---- RECIPE : COMPUTE TELLURIC CCF MOMENT ----\n")

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

        files = glob.glob(directory + "RASSI*.p")
        files = np.sort(files)

        if sub_dico is None:
            sub_dico = self.dico_actif
        print("---- DICO %s used ----" % (sub_dico))

        one_file = pd.read_pickle(files[0])
        grid = one_file["wave"]
        flux = one_file["flux" + kw] / one_file[sub_dico]["continuum_linear"]
        dg = grid[1] - grid[0]
        ccf_sigma = int(one_file["parameters"]["fwhm_ccf"] * 10 / 3e5 * 6000 / dg)
        test = myc.tableXY(grid, flux)

        telluric_tag = "telluric"
        if mask is None:
            mask = "telluric"

        if type(mask) == str:
            if mask == "h2o":
                telluric_tag = "h2o"
            elif mask == "o2":
                telluric_tag = "o2"
            mask = np.genfromtxt(root + "/Python/MASK_CCF/mask_telluric_" + mask + ".txt")
            mask = mask[mask[:, 0].argsort()]
            # mask = mask[(mask[:,0]>6200)&(mask[:,0]<6400)]

        wave_tel = 0.5 * (mask[:, 0] + mask[:, 1])
        mask = mask[(wave_tel < np.max(grid)) & (wave_tel > np.min(grid))]
        wave_tel = wave_tel[(wave_tel < np.max(grid)) & (wave_tel > np.min(grid))]

        test.clip(min=[mask[0, 0], None], max=[mask[-1, 0], None])
        test.rolling(window=ccf_sigma, quantile=1)  # to supress telluric in broad asorption line
        tt, matrix = myf.clustering((test.roll < 0.97).astype("int"), tresh=0.5, num=0.5)
        t = np.array([k[0] for k in tt]) == 1
        matrix = matrix[t, :]

        keep_telluric = np.ones(len(wave_tel)).astype("bool")
        for j in range(len(matrix)):
            left = test.x[matrix[j, 0]]
            right = test.x[matrix[j, 1]]

            c1 = np.sign(myf.doppler_r(wave_tel, 30000)[0] - left)
            c2 = np.sign(myf.doppler_r(wave_tel, 30000)[1] - left)
            c3 = np.sign(myf.doppler_r(wave_tel, 30000)[0] - right)
            c4 = np.sign(myf.doppler_r(wave_tel, 30000)[1] - right)
            keep_telluric = keep_telluric & ((c1 == c2) * (c1 == c3) * (c1 == c4))

        if (sum(keep_telluric) > 25) & (
            suppress_broad
        ):  # to avoid rejecting all tellurics for cool stars
            mask = mask[keep_telluric]
        print("\n [INFO] %.0f lines available in the telluric mask" % (len(mask)))
        plt.figure()
        plt.plot(grid, flux)
        for j in 0.5 * (mask[:, 0] + mask[:, 1]):
            plt.axvline(x=j, color="k")

        self.yarara_ccf(
            sub_dico=sub_dico,
            continuum=continuum,
            mask=mask,
            weighted=weighted,
            delta_window=delta_window,
            reference=reference,
            plot=True,
            save=False,
            ccf_oversampling=ccf_oversampling,
            display_ccf=display_ccf,
            normalisation=normalisation,
            ratio=ratio,
            rv_borders=10,
            rv_range=int(berv_max + 7),
            rv_sys=0,
            rv_shift=rv_shift,
            wave_max=wave_max,
            wave_min=wave_min,
        )

        plt.figure(figsize=(6, 6))
        plt.axes([0.15, 0.3, 0.8, 0.6])
        self.ccf_rv.yerr *= 0
        self.ccf_rv.yerr += 50
        self.ccf_rv.plot(modulo=365.25, label="%s ccf rv" % (telluric_tag))
        plt.scatter(self.table.jdb % 365.25, self.table.berv * 1000, color="b", label="berv")
        plt.legend()
        plt.ylabel("RV [m/s]")
        plt.xlabel("Time %365.25 [days]")
        plt.axes([0.15, 0.08, 0.8, 0.2])
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
            file["parameters"][telluric_tag + "_fwhm"] = output[5][i]
            file["parameters"][telluric_tag + "_center"] = output[6][i]
            file["parameters"][telluric_tag + "_depth"] = output[7][i]
            io.pickle_dump(file, open(j, "wb"))

        self.yarara_analyse_summary()

    # =============================================================================
    # COMPUTE THE CCF OF THE RASSINE SPECTRUM
    # =============================================================================

    # processing
    def yarara_ccf(
        self,
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
        """
        Compute the CCF of a spectrum, reference to use always the same continuum (matching_anchors highest SNR).
        Display_ccf to plot all the individual CCF. Plot to plot the FWHM, contrast and RV.

        Parameters
        ----------
        sub_dico : The sub_dictionnary used to  select the continuum
        continuum : The continuum to select (either linear or cubic)
        mask : The line mask used to cross correlate with the spectrum (mask should be located in MASK_CCF otherwise KITCAT dico)
        mask_col : Column of the KitCat column to use for the weight
        threshold_telluric : Maximum telluric contamination to keep a stellar line in the mask
        reference : True/False or 'norm', True use the matching anchors of reference, False use the continuum of each spectrum, norm use the continuum normalised spectrum (not )
        plot : True/False to plot the RV time-series
        display_ccf : display all the ccf subproduct
        save : True/False to save the informations iun summary table
        normalisation : 'left' or 'slope'. if left normalise the CCF by the most left value, otherwise fit a line between the two highest point
        del_outside maximam : True/False to delete the CCF outside the two bump in personal mask
        speed_up : remove region from the CCF not crossed by a line in the mask to speed up the code
        force_brute : force to remove the region excluded by the brute mask

        """

        directory = self.directory
        planet = self.planet

        def replace_none(y, yerr):
            if yerr is None:
                return np.nan, 1e6
            else:
                return y, yerr

        if rv_range is None:
            rv_range = int(3 * self.fwhm)
            print("\n [INFO] RV range updated to : %.1f kms" % (rv_range))

        if rv_borders is None:
            rv_borders = int(2 * self.fwhm)
            print("\n [INFO] RV borders updated to : %.1f kms" % (rv_borders))

        kw = "_planet" * planet
        if kw != "":
            print("\n---- PLANET ACTIVATED ----")

        self.import_table()

        files = glob.glob(directory + "RASSI*.p")
        files = np.array(self.table["filename"])  # updated 29.10.21 to allow ins_merged
        files = np.sort(files)

        flux = []
        snr = []
        conti = []
        flux_err = []
        jdb = []
        berv = []

        epsilon = 1e-12

        file_random = self.import_spectrum()
        self.import_table()
        self.import_material()
        load = self.material

        mask_loc = mask_name
        if mask_name is None:
            mask_name = "No name"
            mask_loc = "No name"

        if mask is None:
            mask = self.mask_harps
            mask_name = self.mask_harps

        if type(mask) == str:
            if mask.split(".")[-1] == "p":
                loc_mask = self.dir_root + "KITCAT/"
                mask_name = mask
                mask_loc = loc_mask + mask
                dico = pd.read_pickle(mask_loc)["catalogue"]
                dico = dico.loc[dico["rel_contam"] < treshold_telluric]
                if "valid" in dico.keys():
                    dico = dico.loc[dico["valid"]]
                if element is not None:
                    dico = dico.loc[dico["element"] == element]
                mask = np.array([np.array(dico["freq_mask0"]), np.array(dico[mask_col])]).T
                mask = mask[mask[:, 1] != 0]
                print("\n [INFO] Nb lines in the CCF mask : %.0f" % (len(dico)))

            else:
                mask_name = mask
                mask_loc = root + "/Python/MASK_CCF/" + mask + ".txt"
                mask = np.genfromtxt(mask_loc)
                mask = np.array([0.5 * (mask[:, 0] + mask[:, 1]), mask[:, 2]]).T

        if type(mask) == pd.core.frame.DataFrame:
            dico = mask
            mask = np.array(
                [
                    np.array(mask["freq_mask0"]).astype("float"),
                    np.array(mask[mask_col]).astype("float"),
                ]
            ).T

        print("\n [INFO] CCF mask selected : %s \n" % (mask_loc))

        if rv_sys is None:
            if file_random["parameters"]["RV_sys"] is not None:
                rv_sys = 1000 * file_random["parameters"]["RV_sys"]
            else:
                rv_sys = 0
        else:
            rv_sys *= 1000

        if rv_shift is None:
            rv_shift = np.zeros(len(files))

        print("\n [INFO] RV sys : %.2f [km/s] \n" % (rv_sys / 1000))

        mask[:, 0] = myf.doppler_r(mask[:, 0], rv_sys)[0]

        for i, j in enumerate(files):
            file = pd.read_pickle(j)
            if not i:
                wave = file["wave"]
            snr.append(file["parameters"]["SNR_5500"])
            try:
                jdb.append(file["parameters"]["jdb"])
            except:
                jdb.append(i)
            try:
                berv.append(file["parameters"]["berv"])
            except:
                berv.append(0)

            f = file["flux" + kw]
            f_std = file["flux_err"]
            c = file[sub_dico]["continuum_" + continuum] + epsilon
            c_std = file["continuum_err"]

            f_norm, f_norm_std = myf.flux_norm_std(f, f_std, c, c_std)

            flux.append(f_norm)
            flux_err.append(f_norm_std)
            conti.append(c - epsilon)

        wave = np.array(wave)
        flux = np.array(flux)
        flux_err = np.array(flux_err)
        snr = np.array(snr)
        conti = np.array(conti)
        jdb = np.array(jdb)
        berv = np.array(berv)
        grid = wave

        flux, flux_err, wave = self.yarara_map(
            sub_dico=sub_dico,
            planet=self.planet,
            wave_min=None,
            wave_max=None,
            Plot=False,
            reference=False,
            substract_map=substract_map,
            add_map=add_map,
            correction_factor=False,
        )  # 04.08.21 in order to include add and substract map

        flux *= np.array(load["correction_factor"])
        flux_err *= np.array(load["correction_factor"])

        noise_matrix, noise_values = self.yarara_poissonian_noise(
            noise_wanted=p_noise, wave_ref=None, flat_snr=True
        )
        flux += noise_matrix
        flux_err = np.sqrt(flux_err**2 + noise_values**2)

        if reference == True:
            norm_factor = np.array(load["color_template"])
            flux *= norm_factor
            flux_err *= norm_factor
        elif reference == "norm":
            if ratio:
                norm_factor = np.array(load["reference_spectrum"]) * np.array(
                    load["correction_factor"]
                )
                norm_factor[norm_factor == 0] = 1
                flux /= norm_factor
                flux_err /= norm_factor
            else:
                pass
        elif reference == "master_snr":
            norm_factor = np.array(load["master_snr_curve"]) ** 2
            norm_factor[np.isnan(norm_factor)] = 1
            norm_factor *= np.nanmean(np.array(load["color_template"])) / np.nanmean(norm_factor)
            flux *= norm_factor
            flux_err *= norm_factor
        else:
            flux *= conti
            flux_err *= conti

        if sub_dico == "matching_brute":
            force_brute = True

        mask_shifted = myf.doppler_r(mask[:, 0], (rv_range + 5) * 1000)

        if force_brute:
            brute_mask = np.array(load["mask_brute"])
            used_region = ((grid) >= mask_shifted[1][:, np.newaxis]) & (
                (grid) <= mask_shifted[0][:, np.newaxis]
            )
            line_killed = np.sum(brute_mask * used_region, axis=1) == 0
            mask = mask[line_killed]
            mask_shifted = myf.doppler_r(mask[:, 0], (rv_range + 5) * 1000)

        mask = mask[
            (myf.doppler_r(mask[:, 0], 30000)[0] < grid.max())
            & (myf.doppler_r(mask[:, 0], 30000)[1] > grid.min()),
            :,
        ]  # supres line farther than 30kms
        if wave_min is not None:
            mask = mask[mask[:, 0] > wave_min, :]
        if wave_max is not None:
            mask = mask[mask[:, 0] < wave_max, :]

        print("\n [INFO] Nb lines in the mask : %.0f \n" % (len(mask)))

        mask_min = np.min(mask[:, 0])
        mask_max = np.max(mask[:, 0])

        # supress useless part of the spectra to speed up the CCF
        grid_min = int(myf.find_nearest(grid, myf.doppler_r(mask_min, -100000)[0])[0])
        grid_max = int(myf.find_nearest(grid, myf.doppler_r(mask_max, 100000)[0])[0])
        grid = grid[grid_min:grid_max]

        log_grid = np.linspace(np.log10(grid).min(), np.log10(grid).max(), len(grid))
        dgrid = log_grid[1] - log_grid[0]
        # dv = (10**(dgrid)-1)*299.792e6

        # computation of region free of spectral line to increase code speed

        if speed_up:
            used_region = ((10**log_grid) >= mask_shifted[1][:, np.newaxis]) & (
                (10**log_grid) <= mask_shifted[0][:, np.newaxis]
            )
            used_region = (np.sum(used_region, axis=0) != 0).astype("bool")
            print(
                "\n [INFO] Percentage of the spectrum used : %.1f [%%] (%.0f) \n"
                % (100 * sum(used_region) / len(grid), len(grid))
            )
            time.sleep(1)
        else:
            used_region = np.ones(len(grid)).astype("bool")

        if (
            not os.path.exists(self.dir_root + "CCF_MASK/CCF_" + mask_name.split(".")[0] + ".fits")
        ) | (force_brute):
            print(
                "\n [INFO] CCF mask reduced for the first time, wait for the static mask producing...\n"
            )
            time.sleep(1)
            mask_wave = np.log10(mask[:, 0])
            mask_contrast = mask[:, 1] * weighted + (1 - weighted)

            mask_hole = (
                mask[:, 0] > myf.doppler_r(file_random["parameters"]["hole_left"], -30000)[0]
            ) & (mask[:, 0] < myf.doppler_r(file_random["parameters"]["hole_right"], 30000)[0])
            mask_contrast[mask_hole] = 0

            log_grid_mask = np.arange(
                log_grid.min() - 10 * dgrid,
                log_grid.max() + 10 * dgrid + dgrid / 10,
                dgrid / 11,
            )
            log_mask = np.zeros(len(log_grid_mask))

            # mask_contrast /= np.sqrt(np.nansum(mask_contrast**2)) #UPDATE 04.05.21 (DOES NOT WORK)

            match = myf.identify_nearest(mask_wave, log_grid_mask)
            for j in np.arange(-delta_window, delta_window + 1, 1):
                log_mask[match + j] = (mask_contrast) ** (1 + int(squared))

            plt.figure()
            plt.plot(log_grid_mask, log_mask)

            if (not force_brute) & (mask_name != "No name"):
                hdu = fits.PrimaryHDU(np.array([log_grid_mask, log_mask]).T)
                hdul = fits.HDUList([hdu])
                hdul.writeto(self.dir_root + "CCF_MASK/CCF_" + mask_name.split(".")[0] + ".fits")
                print(
                    "\n [INFO] CCF mask saved under : %s"
                    % (self.dir_root + "CCF_MASK/CCF_" + mask_name.split(".")[0] + ".fits")
                )
        else:
            print(
                "\n [INFO] CCF mask found : %s"
                % (self.dir_root + "CCF_MASK/CCF_" + mask_name.split(".")[0] + ".fits")
            )
            log_grid_mask, log_mask = fits.open(
                self.dir_root + "CCF_MASK/CCF_" + mask_name.split(".")[0] + ".fits"
            )[0].data.T

        log_template = interp1d(
            log_grid_mask,
            log_mask,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )(log_grid)

        flux = flux[:, grid_min:grid_max]
        flux_err = flux_err[:, grid_min:grid_max]

        amplitudes = []
        amplitudes_std = []
        rvs = []
        rvs_std = []
        fwhms = []
        fwhms_std = []
        ew = []
        ew_std = []
        centers = []
        centers_std = []
        depths = []
        depths_std = []
        bisspan = []
        bisspan_std = []
        b0s = []
        b1s = []
        b2s = []
        b3s = []
        b4s = []

        if display_ccf:
            plt.figure()

        now = datetime.datetime.now()
        print(
            "\n Computing CCF (Current time %.0fh%.0fm%.0fs) \n"
            % (now.hour, now.minute, now.second)
        )

        all_flux = []
        for j, i in enumerate(files):
            all_flux.append(
                interp1d(
                    np.log10(myf.doppler_r(grid, rv_shift[j])[0]),
                    flux[j],
                    kind="cubic",
                    bounds_error=False,
                    fill_value="extrapolate",
                )(log_grid)
            )
        all_flux = np.array(all_flux)

        all_flux_err = []
        for j, i in enumerate(files):
            all_flux_err.append(
                interp1d(
                    np.log10(myf.doppler_r(grid, rv_shift[j])[0]),
                    flux_err[j],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )(log_grid)
            )
        all_flux_err = np.array(all_flux_err)

        vrad, ccf_power, ccf_power_std = myf.ccf(
            log_grid[used_region],
            all_flux[:, used_region],
            log_template[used_region],
            rv_range=rv_range,
            oversampling=ccf_oversampling,
            spec1_std=all_flux_err[:, used_region],
        )  # to compute on all the ccf simultaneously

        now = datetime.datetime.now()
        print("")
        print(
            "\n CCF computed (Current time %.0fh%.0fm%.0fs) \n"
            % (now.hour, now.minute, now.second)
        )

        try:
            self.all_ccf_saved[sub_dico] = (vrad, ccf_power, ccf_power_std)
        except AttributeError:
            self.all_ccf_saved = {sub_dico: (vrad, ccf_power, ccf_power_std)}

        ccf_ref = np.median(ccf_power, axis=1)
        continuum_ccf = np.argmax(ccf_ref)
        top_ccf = np.argsort(ccf_ref)[
            -int(len(ccf_ref) / 2) :
        ]  # roughly half of a CCF is made of the continuum

        ccf_snr = 1 / (
            np.std((ccf_power - ccf_ref[:, np.newaxis])[top_ccf], axis=0)
            / np.mean(ccf_power[continuum_ccf])
        )
        print(" [INFO] SNR CCF continuum median : %.0f\n" % (np.median(ccf_snr)))

        # noise_ccf = ccf_power_std
        # w = noise_ccf/(np.gradient(ccf_ref)/np.gradient(vrad)+epsilon)[:,np.newaxis]
        # w[w==0] = np.min(w[w!=0])/10
        # svrad_phot = 1/np.sqrt(np.sum(1/w**2,axis=0))
        # scaling = 820/np.mean(np.gradient(vrad)) #to penalize oversampling in vrad
        # svrad_phot*=scaling
        # self.svrad_phot = svrad_phot

        noise_ccf = [
            (np.sqrt(ccf_ref / np.max(ccf_ref)) * ccf_ref[continuum_ccf])[:, np.newaxis] / ccf_snr,
            ccf_power_std,
        ][
            int(ratio)
        ]  # assume that the noise in the continuum is white (okay for matching_mad but wrong when tellurics are still there)
        sigma_rv = noise_ccf / (abs(np.gradient(ccf_ref)) / np.gradient(vrad))[:, np.newaxis]
        w_rv = (1 / sigma_rv) ** 2
        svrad_phot = 1 / np.sqrt(np.sum(w_rv, axis=0))
        scaling = np.sqrt(820 / np.mean(np.gradient(vrad)))  # to penalize oversampling in vrad
        svrad_phot *= scaling

        svrad_phot[svrad_phot == 0] = 2 * np.max(svrad_phot)  # in case of null values

        print(" [INFO] Photon noise RV median : %.2f m/s\n " % (np.median(svrad_phot)))

        svrad_phot2 = {}
        svrad_phot2["rv"] = 10 ** (
            0.98 * np.log10(svrad_phot) - 3.08
        )  # from photon noise simulations Photon_noise_CCF.py
        svrad_phot2["contrast"] = 10 ** (
            0.98 * np.log10(svrad_phot) - 3.58
        )  # from photon noise simulations Photon_noise_CCF.py
        svrad_phot2["fwhm"] = 10 ** (
            0.98 * np.log10(svrad_phot) - 2.94
        )  # from photon noise simulations Photon_noise_CCF.py
        svrad_phot2["center"] = 10 ** (
            0.98 * np.log10(svrad_phot) - 2.83
        )  # from photon noise simulations Photon_noise_CCF.py
        svrad_phot2["depth"] = 10 ** (
            0.97 * np.log10(svrad_phot) - 3.62
        )  # from photon noise simulations Photon_noise_CCF.py
        svrad_phot2["ew"] = 10 ** (
            0.97 * np.log10(svrad_phot) - 3.47
        )  # from photon noise simulations Photon_noise_CCF.py
        svrad_phot2["vspan"] = 10 ** (
            0.98 * np.log10(svrad_phot) - 2.95
        )  # from photon noise simulations Photon_noise_CCF.py

        self.svrad_phot = svrad_phot2["rv"]

        print(
            " [INFO] Photon noise RV from calibration : %.2f m/s\n "
            % (np.median(svrad_phot2["rv"]) * 1000)
        )

        for j, i in enumerate(files):
            if bis_analysis:
                print("File (%.0f/%.0f) %s SNR %.0f reduced" % (j + 1, len(files), i, snr[j]))
            file = pd.read_pickle(i)
            # log_spectrum = interp1d(np.log10(grid), flux[j], kind='cubic', bounds_error=False, fill_value='extrapolate')(log_grid)
            # vrad, ccf_power_old = myf.ccf2(log_grid, log_spectrum,  log_grid_mask, log_mask)
            # vrad, ccf_power_old = myf.ccf(log_grid, log_spectrum, log_template, rv_range=45, oversampling=ccf_oversampling)
            ccf_power_old = ccf_power[:, j]
            ccf_power_old_std = ccf_power_std[:, j]
            ccf = myc.tableXY(vrad / 1000, ccf_power_old, ccf_power_old_std)
            ccf.yerr = np.sqrt(abs(ccf.y))

            ccf.y *= -1
            ccf.find_max(vicinity=5)

            ccf.diff(replace=False)
            ccf.deri.y = np.abs(ccf.deri.y)
            for jj in range(3):
                ccf.deri.find_max(vicinity=4 - jj)
                if len(ccf.deri.x_max) > 1:
                    break

            first_max = ccf.deri.x_max[np.argsort(ccf.deri.y_max)[-1]]
            second_max = ccf.deri.x_max[np.argsort(ccf.deri.y_max)[-2]]

            ccf.y *= -1
            if (np.min(abs(ccf.x_max - 0.5 * (first_max + second_max))) < 5) & (self.fwhm < 15):
                center = ccf.x_max[np.argmin(abs(ccf.x_max - 0.5 * (first_max + second_max)))]
            else:
                center = ccf.x[ccf.y.argmin()]
            ccf.x -= center

            if not del_outside_max:
                mask = (ccf.x > -rv_borders) & (ccf.x < rv_borders)
                ccf.supress_mask(mask)
            else:
                ccf.find_max(vicinity=10)
                ccf.index_max = np.sort(ccf.index_max)
                mask = np.zeros(len(ccf.x)).astype("bool")
                mask[ccf.index_max[0] : ccf.index_max[1] + 1] = True
                ccf.supress_mask(mask)

            if normalisation == "left":
                norm = ccf.y[0]
            else:
                max1 = np.argmax(ccf.y[0 : int(len(ccf.y) / 2)])
                max2 = np.argmax(ccf.y[int(len(ccf.y) / 2) :]) + int(len(ccf.y) / 2)
                fmax1 = ccf.y[max1]
                fmax2 = ccf.y[max2]
                norm = (fmax2 - fmax1) / (max2 - max1) * (np.arange(len(ccf.y)) - max2) + fmax2
            ccf.yerr /= norm
            ccf.y /= norm

            if ratio:
                ccf.yerr *= 0
                ccf.yerr += 0.01

            if display_ccf:
                ccf.plot(color=None)

            # bis #interpolated on 10 m/s step
            moments = np.zeros(5)
            b0 = []
            b1 = []
            b2 = []
            b3 = []
            b4 = []

            if bis_analysis:
                ccf.x *= 1000
                drv = 10
                border = np.min([abs(ccf.x.min()), abs(ccf.x.max())])
                border = (border // drv) * drv
                grid_vrad = np.arange(-border, border + drv, drv)
                ccf.interpolate(new_grid=grid_vrad, replace=False)
                ccf.interpolated.yerr *= 0
                ccf.interpolated.y = (ccf.interpolated.y - ccf.interpolated.y.min()) / (
                    ccf.interpolated.y.max() - ccf.interpolated.y.min()
                )
                ccf.interpolated.my_bisector(oversampling=10, between_max=True)
                ccf.interpolated.bis.clip(min=[0.01, None], max=[0.7, None])
                bis = ccf.interpolated.bis
                bis.interpolate(new_grid=np.linspace(0.01, 0.7, 20))
                bis.y -= bis.y[0]
                if save:
                    save_bis = {
                        "bis_flux": bis.x,
                        "bis_rv": bis.y,
                        "bis_rv_std": bis.yerr,
                    }
                    file["ccf_bis"] = save_bis
                bis.y = np.gradient(bis.y)
                for p in range(5):
                    moments[p] = np.sum(bis.x**p * bis.y)
                b0.append(moments[0])
                b1.append(moments[1])
                b2.append(moments[2])
                b3.append(moments[3])
                b4.append(moments[4])
                ccf.x /= 1000
            else:
                b0.append(0)
                b1.append(0)
                b2.append(0)
                b3.append(0)
                b4.append(0)

            ccf.clip(min=[-0.5, None], max=[0.5, None], replace=False)
            if len(ccf.clipped.x) < 7:
                ccf.clip(min=[-2, None], max=[2, None], replace=False)
            ccf.clipped.fit_poly()
            a, b, c = ccf.clipped.poly_coefficient
            para_center = -b / (2 * a) + center
            para_depth = a * (-b / (2 * a)) ** 2 + b * (-b / (2 * a)) + c
            centers.append(para_center)
            depths.append(1 - para_depth)

            EW = np.sum(ccf.y - 1) / len(ccf.y)
            ew.append(EW)
            save_ccf = {
                "ccf_flux": ccf.y,
                "ccf_flux_std": ccf.yerr,
                "ccf_rv": ccf.x + center,
                "reference": reference,
                "ew": EW,
            }

            para_ccf = {"para_rv": para_center, "para_depth": para_depth}

            ccf.fit_gaussian(Plot=False)  # ,guess=[-self.contrast,0,self.fwhm/2.355,1])

            rv_ccf = ccf.params["cen"].value + center
            rv_ccf_std = ccf.params["cen"].stderr
            rv_ccf, rv_ccf_std = replace_none(rv_ccf, rv_ccf_std)
            rv_ccf_std = svrad_phot2["rv"][j]
            factor = rv_ccf_std / abs(rv_ccf)
            scaling_noise = {
                "amp": 0.32,
                "wid": 1.33,
                "depth": 0.29,
                "center": 1.79,
                "bisspan": 1.37,
                "ew": 0.42,
            }

            contrast_ccf = -ccf.params["amp"].value
            contrast_ccf_std = ccf.params["amp"].stderr
            contrast_ccf, contrast_ccf_std = replace_none(contrast_ccf, contrast_ccf_std)
            contrast_ccf_std = svrad_phot2["contrast"][
                j
            ]  # abs(contrast_ccf)*factor*scaling_noise['amp']

            wid_ccf = ccf.params["wid"].value
            wid_ccf_std = ccf.params["wid"].stderr
            wid_ccf, wid_ccf_std = replace_none(wid_ccf, wid_ccf_std)
            wid_ccf_std = svrad_phot2["fwhm"][j]  # abs(wid_ccf)*factor*scaling_noise['wid']

            offset_ccf = ccf.params["offset"].value
            offset_ccf_std = ccf.params["offset"].stderr
            offset_ccf, offset_ccf_std = replace_none(offset_ccf, offset_ccf_std)

            amplitudes.append(contrast_ccf)
            amplitudes_std.append(contrast_ccf_std)
            rvs.append(rv_ccf)
            rvs_std.append(rv_ccf_std)
            fwhms.append(wid_ccf)
            fwhms_std.append(wid_ccf_std)
            bisspan.append(rv_ccf - para_center)
            bisspan_ccf_std = svrad_phot2["vspan"][
                j
            ]  # abs(rv_ccf - para_center)*factor*scaling_noise['bisspan']
            bisspan_std.append(bisspan_ccf_std)

            ew_std.append(svrad_phot2["ew"][j])  # abs(EW)*factor*scaling_noise['ew'])
            centers_std.append(
                svrad_phot2["center"][j]
            )  # abs(para_center)*factor*scaling_noise['center'])
            depths_std.append(
                svrad_phot2["depth"][j]
            )  # abs(1-para_depth)*factor*scaling_noise['depth'])

            save_ccf["ew_std"] = ew_std
            para_ccf["para_rv_std"] = centers_std
            para_ccf["para_depth_std"] = depths_std

            file["ccf"] = save_ccf
            file["ccf_parabola"] = para_ccf

            b0s.append(moments[0])
            b1s.append(moments[1])
            b2s.append(moments[2])
            b3s.append(moments[3])
            b4s.append(moments[4])
            if save:
                save_gauss = {
                    "contrast": contrast_ccf,
                    "contrast_std": contrast_ccf_std,
                    "rv": rv_ccf,
                    "rv_std": rv_ccf_std,
                    "rv_std_phot": svrad_phot2["rv"][j],
                    "fwhm": wid_ccf,
                    "fwhm_std": wid_ccf_std,
                    "offset": offset_ccf,
                    "offset_std": offset_ccf_std,
                    "vspan": rv_ccf - para_center,
                    "vspan_std": bisspan_std,
                    "b0": moments[0],
                    "b1": moments[1],
                    "b2": moments[2],
                    "b3": moments[3],
                    "b4": moments[4],
                }

                file["ccf_gaussian"] = save_gauss
                #
                #                    ccf.my_bisector(between_max=True,oversampling=50)
                #                    bis = myc.tableXY(ccf.bisector[5::50,1],ccf.bisector[5::50,0]+center,ccf.bisector[5::50,2])
                #
                #                    save_bis = {'bis_flux':bis.x,'bis_rv':bis.y,'bis_rv_std':bis.yerr}
                #                    file['ccf_bis'] = save_bis

                io.pickle_dump(file, open(i, "wb"))

        # try:
        #     rvs_std = np.array(self.table['rv_dace_std'])/1000
        # except:
        #     pass

        rvs_std_backup = np.array(self.table["rv_dace_std"]) / 1000
        rvs_std = svrad_phot2["rv"]
        rvs_std[rvs_std == 0] = rvs_std_backup[rvs_std == 0]

        fwhms = np.array(fwhms).astype("float") * 2.355
        fwhms_std = np.array(fwhms_std).astype("float") * 2.355

        self.warning_rv_borders = False
        if np.median(fwhms) > (rv_borders / 1.5):
            print("[WARNING] The CCF is larger than the RV borders for the fit")
            self.warning_rv_borders = True

        self.ccf_rv = myc.tableXY(jdb, np.array(rvs) * 1000, np.array(rvs_std) * 1000)
        self.ccf_centers = myc.tableXY(jdb, np.array(centers) * 1000, np.array(centers_std) * 1000)
        self.ccf_contrast = myc.tableXY(jdb, amplitudes, amplitudes_std)
        self.ccf_depth = myc.tableXY(jdb, depths, depths_std)
        self.ccf_fwhm = myc.tableXY(jdb, fwhms, fwhms_std)
        self.ccf_vspan = myc.tableXY(jdb, np.array(bisspan) * 1000, np.array(bisspan_std) * 1000)
        self.ccf_ew = myc.tableXY(jdb, np.array(ew), np.array(ew_std))
        self.ccf_bis0 = myc.tableXY(jdb, b0s, np.sqrt(2) * np.array(rvs_std) * 1000)
        self.ccf_timeseries = np.array(
            [
                ew,
                ew_std,
                amplitudes,
                amplitudes_std,
                rvs,
                rvs_std,
                svrad_phot2["rv"],
                fwhms,
                fwhms_std,
                centers,
                centers_std,
                depths,
                depths_std,
                b0s,
                bisspan,
                bisspan_std,
            ]
        )
        self.ccf_rv.rms_w()
        self.ccf_centers.rms_w()
        self.ccf_rv_shift = center

        ccf_infos = pd.DataFrame(
            self.ccf_timeseries.T,
            columns=[
                "ew",
                "ew_std",
                "contrast",
                "contrast_std",
                "rv",
                "rv_std",
                "rv_std_phot",
                "fwhm",
                "fwhm_std",
                "center",
                "center_std",
                "depth",
                "depth_std",
                "b0",
                "bisspan",
                "bisspan_std",
            ],
        )
        ccf_infos["jdb"] = jdb
        ccf_infos = {
            "table": ccf_infos,
            "creation_date": datetime.datetime.now().isoformat(),
        }

        if not os.path.exists(self.directory + "Analyse_ccf.p"):
            ccf_summary = {"star_info": {"name": self.starname}}
            io.pickle_dump(ccf_summary, open(self.directory + "/Analyse_ccf.p", "wb"))

        if ccf_name is None:
            ccf_name = sub_dico

        if save:
            if mask_name != "No name":
                file_summary_ccf = pd.read_pickle(self.directory + "Analyse_ccf.p")
                try:
                    file_summary_ccf["CCF_" + mask_name.split(".")[0]][ccf_name] = ccf_infos
                except KeyError:
                    file_summary_ccf["CCF_" + mask_name.split(".")[0]] = {ccf_name: ccf_infos}

                io.pickle_dump(file_summary_ccf, open(self.directory + "/Analyse_ccf.p", "wb"))

        self.infos["latest_dico_ccf"] = ccf_name

        self.yarara_analyse_summary()

        if save_ccf_profile:
            self.yarara_ccf_save(mask_name.split(".")[0], sub_dico)

        if plot:
            plt.figure(figsize=(12, 10))
            plt.subplot(4, 2, 1)
            self.ccf_rv.plot(
                label=r"rms : %.2f | $\sigma_{\gamma}$ : %.2f"
                % (self.ccf_rv.rms, np.median(self.svrad_phot) * 1000)
            )
            plt.legend()
            plt.title("RV", fontsize=14)
            ax = plt.gca()

            plt.subplot(4, 2, 3, sharex=ax)  # .scatter(jdb,ew,color='k')
            self.ccf_ew.plot()
            plt.title("EW", fontsize=14)
            plt.ylim(
                np.nanpercentile(ew, 25) - 1.5 * myf.IQ(ew),
                np.nanpercentile(ew, 75) + 1.5 * myf.IQ(ew),
            )

            plt.subplot(4, 2, 5, sharex=ax)  # .scatter(jdb,amplitudes,color='k')
            self.ccf_contrast.plot()
            plt.title("Contrast", fontsize=14)
            plt.ylim(
                np.nanpercentile(amplitudes, 25) - 1.5 * myf.IQ(amplitudes),
                np.nanpercentile(amplitudes, 75) + 1.5 * myf.IQ(amplitudes),
            )

            plt.subplot(4, 2, 4, sharex=ax)  # .scatter(jdb,fwhms,color='k')
            self.ccf_fwhm.plot()
            plt.title("FWHM", fontsize=14)
            plt.ylim(
                np.nanpercentile(fwhms, 25) - 1.5 * myf.IQ(fwhms),
                np.nanpercentile(fwhms, 75) + 1.5 * myf.IQ(fwhms),
            )

            plt.subplot(4, 2, 6, sharex=ax)  # .scatter(jdb,depths,color='k')
            self.ccf_depth.plot()
            plt.title("Depth", fontsize=14)
            plt.ylim(
                np.nanpercentile(depths, 25) - 1.5 * myf.IQ(depths),
                np.nanpercentile(depths, 75) + 1.5 * myf.IQ(depths),
            )

            plt.subplot(4, 2, 2, sharex=ax, sharey=ax)
            self.ccf_centers.plot(label="rms : %.2f" % (self.ccf_centers.rms))
            plt.legend()
            plt.title("Center", fontsize=14)

            plt.subplot(4, 2, 7, sharex=ax).scatter(jdb, b0s, color="k")
            plt.title("BIS", fontsize=14)

            plt.subplot(4, 2, 8, sharex=ax)  # .scatter(jdb,bisspan,color='k')
            self.ccf_vspan.plot()
            plt.title(r"RV $-$ Center (VSPAN)", fontsize=14)
            plt.ylim(
                np.nanpercentile(self.ccf_vspan.y, 25) - 1.5 * myf.IQ(self.ccf_vspan.y),
                np.nanpercentile(self.ccf_vspan.y, 75) + 1.5 * myf.IQ(self.ccf_vspan.y),
            )
            plt.subplots_adjust(
                left=0.07, right=0.93, top=0.95, bottom=0.08, wspace=0.3, hspace=0.3
            )

            if bis_analysis:
                plt.figure(figsize=(12, 10))
                plt.subplot(3, 2, 1).scatter(jdb, b0s, color="k")
                ax = plt.gca()
                plt.title("B0", fontsize=14)
                plt.subplot(3, 2, 2, sharex=ax).scatter(
                    jdb, self.ccf_rv.y - self.ccf_centers.y, color="k"
                )
                plt.title("RV-Center", fontsize=14)
                plt.subplot(3, 2, 3, sharex=ax).scatter(jdb, b1s, color="k")
                plt.title("B1", fontsize=14)
                plt.subplot(3, 2, 4, sharex=ax).scatter(jdb, b2s, color="k")
                plt.title("B2", fontsize=14)
                plt.subplot(3, 2, 5, sharex=ax).scatter(jdb, b3s, color="k")
                plt.title("B3", fontsize=14)
                plt.subplot(3, 2, 6, sharex=ax).scatter(jdb, b4s, color="k")
                plt.title("B4", fontsize=14)

        return {
            "rv": self.ccf_rv,
            "cen": self.ccf_centers,
            "contrast": self.ccf_contrast,
            "fwhm": self.ccf_fwhm,
            "vspan": self.ccf_vspan,
        }

    # =============================================================================
    # VISUALISE THE RASSINE TIMESERIES AND ITS CORRELATION WITH A PROXY
    # =============================================================================

    # processing
    def yarara_map(
        self,
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

        """
        Display the time-series spectra with proxies and its correlation

        Parameters
        ----------
        sub_dico : The sub_dictionnary used to  select the continuum
        continuum : The continuum to select (either linear or cubic)
        wave_min : Minimum x axis limit
        wave_max : Maximum x axis limit
        time_min : Minimum y axis limit
        time_max : Maximum y axis limit
        index : 'index' or 'time' if 'time', display the time-series with blank color if no spectra at specific time  (need roughly equidistant time-series spectra)
        zoom : int-type, to improve the resolution of the 2D plot
        smooth_map = int-type, smooth the 2D plot by gaussian 2D convolution
        reference : 'median', 'snr' or 'master' to select the reference normalised spectrum usedin the difference
        berv_shift : keyword column to use to move in terrestrial rest-frame, km/s speed
        rv_shift : keyword column to use to shift spectra, m/s speed
        cmap : cmap of the 2D plot
        low_cmap : vmin cmap colorbar
        high_cmap : vmax cmap colorbar

        """

        directory = self.directory
        self.import_material()
        load = self.material

        self.import_table()

        kw = "_planet" * planet
        if kw != "":
            print("\n---- PLANET ACTIVATED ----")

        zoom = self.zoom
        smooth_map = self.smooth_map
        cmap = self.cmap
        low_cmap = self.low_cmap
        high_cmap = self.high_cmap

        files = glob.glob(directory + "RASSI*.p")
        files = np.array(self.table["filename"])  # updated 29.10.21 to allow ins_merged

        files = np.sort(files)

        flux = []
        err_flux = []
        snr = []
        jdb = []
        berv = []
        rv = []

        epsilon = 1e-12

        for i, j in enumerate(files):
            self.current_file = j
            file = pd.read_pickle(j)
            if not i:
                wave = file["wave"]
            f = file["flux" + kw]
            f_std = file["flux_err"]
            c = file[sub_dico]["continuum_" + continuum] + epsilon
            c_std = file["continuum_err"]
            f_norm, f_norm_std = myf.flux_norm_std(f, f_std, c, c_std)

            snr.append(file["parameters"]["SNR_5500"])
            flux.append(f_norm)
            err_flux.append(f_norm_std)

            try:
                jdb.append(file["parameters"]["jdb"])
            except:
                jdb.append(i)
            if type(berv_shift) != np.ndarray:
                try:
                    berv.append(file["parameters"][berv_shift])
                except:
                    berv.append(0)
            else:
                berv = berv_shift
            if type(rv_shift) != np.ndarray:
                try:
                    rv.append(file["parameters"][rv_shift])
                except:
                    rv.append(0)
            else:
                rv = rv_shift

        del self.current_file

        wave = np.array(wave)
        flux = np.array(flux)
        err_flux = np.array(err_flux)
        snr = np.array(snr)
        jdb = np.array(jdb)
        berv = np.array(berv)
        rv = np.array(rv)
        rv = rv - np.median(rv)

        self.debug = flux

        if correction_factor:
            flux *= np.array(load["correction_factor"])
            err_flux *= np.array(load["correction_factor"])

        for maps in substract_map:
            flux = self.yarara_substract_map(flux, maps, correction_factor=correction_factor)

        for maps in add_map:
            flux = self.yarara_add_map(flux, maps, correction_factor=correction_factor)

        idx_min = 0
        idx_max = len(wave)
        idx2_min = 0
        idx2_max = len(flux)

        if wave_min is not None:
            idx_min, val, dist = myf.find_nearest(wave, wave_min)
            if val < wave_min:
                idx_min += 1
        if wave_max is not None:
            idx_max, val, dist = myf.find_nearest(wave, wave_max)
            idx_max += 1
            if val > wave_max:
                idx_max -= 1

        if time_min is not None:
            idx2_min = time_min
        if time_max is not None:
            idx2_max = time_max + 1

        if (idx_min == 0) & (idx_max == 0):
            idx_max = myf.find_nearest(wave, np.min(wave) + (wave_max - wave_min))[0]

        noise_matrix, noise_values = self.yarara_poissonian_noise(
            noise_wanted=p_noise, wave_ref=None, flat_snr=True
        )
        flux += noise_matrix
        err_flux = np.sqrt(err_flux**2 + noise_values**2)

        old_length = len(wave)
        wave = wave[int(idx_min) : int(idx_max)]
        flux = flux[int(idx2_min) : int(idx2_max), int(idx_min) : int(idx_max)]
        err_flux = err_flux[int(idx2_min) : int(idx2_max), int(idx_min) : int(idx_max)]

        snr = snr[int(idx2_min) : int(idx2_max)]
        jdb = jdb[int(idx2_min) : int(idx2_max)]
        berv = berv[int(idx2_min) : int(idx2_max)]

        if np.sum(abs(rv)) != 0:
            for j in tqdm(np.arange(len(flux))):
                test = myc.tableXY(wave, flux[j], 0 * wave)
                test.x = myf.doppler_r(test.x, rv[j])[1]
                test.interpolate(new_grid=wave, method="cubic", replace=True)
                flux[j] = test.y

        if reference == "snr":
            ref = flux[snr.argmax()]
        elif reference == "median":
            ref = np.median(flux, axis=0)
        elif reference == "master":
            ref = (np.array(load["reference_spectrum"]) * np.array(load["correction_factor"]))[
                int(idx_min) : int(idx_max)
            ]
        elif type(reference) == int:
            ref = flux[reference]
        elif type(reference) == np.ndarray:
            ref = reference.copy()
            if len(ref) == old_length:
                ref = ref[int(idx_min) : int(idx_max)]
        else:
            ref = 0 * np.median(flux, axis=0)
            low_cmap = 0
            high_cmap = 1

        if low_cmap is None:
            low_cmap = np.percentile(flux - ref, 2.5)
        if high_cmap is None:
            high_cmap = np.percentile(flux - ref, 97.5)

        if ratio:
            diff = myf.smooth2d(flux / (ref + epsilon), smooth_map)
            low_cmap = 1 - 0.005
            high_cmap = 1 + 0.005
        else:
            diff = myf.smooth2d(flux - ref, smooth_map)

        if modulo is not None:
            diff = self.yarara_map_folded(diff, modulo=modulo, jdb=jdb)[0]

        if np.sum(abs(berv)) != 0:
            for j in tqdm(np.arange(len(flux))):
                test = myc.tableXY(wave, diff[j], 0 * wave)
                test.x = myf.doppler_r(test.x, berv[j] * 1000)[1]
                test.interpolate(new_grid=wave, method="cubic", replace=True)
                diff[j] = test.y

        self.map = (wave, diff)

        if index != "index":
            dtime = np.median(np.diff(jdb))
            liste_time = np.arange(jdb.min(), jdb.max() + dtime, dtime)
            match_time = myf.match_nearest(liste_time, jdb)

            snr2 = np.nan * np.ones(len(liste_time))
            jdb2 = np.nan * np.ones(len(liste_time))
            diff2 = np.median(diff) * np.ones((len(liste_time), len(wave)))

            snr2[match_time[:, 0].astype("int")] = snr[match_time[:, 1].astype("int")]
            jdb2[match_time[:, 0].astype("int")] = jdb[match_time[:, 1].astype("int")]
            diff2[match_time[:, 0].astype("int"), :] = diff[match_time[:, 1].astype("int"), :]

            snr = snr2
            jdb = jdb2
            diff = diff2
        if Plot:
            if new:
                fig = plt.figure(figsize=(24, 6))
                plt.axes([0.1, 0.1, 0.8, 0.8])
            myf.my_colormesh(
                wave,
                np.arange(len(diff)),
                diff * unit,
                zoom=zoom,
                vmin=low_cmap * unit,
                vmax=high_cmap * unit,
                cmap=cmap,
            )
            plt.tick_params(direction="in", top=True, right=True)
            plt.ylabel("Spectra  indexes (time)", fontsize=14)
            plt.xlabel(r"Wavelength [$\AA$]", fontsize=14)
            plt.ylim(0, None)
            if new:
                cbaxes = fig.add_axes([0.86 + 0.04, 0.1, 0.01, 0.8])
                ax = plt.colorbar(cax=cbaxes)
                ax.ax.set_ylabel(r"$\Delta$ flux normalised")
        return diff, err_flux, wave

    # =============================================================================
    # INTERFERENCE CORRECTION
    # =============================================================================

    # instrument
    def yarara_correct_pattern(
        self,
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

        """
        Suppress interferency pattern produced by a material of a certain width by making a Fourier filtering.
        Width_min is the minimum width possible for the material in mm.

        Parameters
        ----------
        sub_dico : The sub_dictionnary used to  select the continuum
        continuum : The continuum to select (either linear or cubic)
        wave_min : Minimum x axis limit for the plot
        wave_max : Maximum x axis limit for the plot
        zoom : int-type, to improve the resolution of the 2D plot
        smooth_map = int-type, smooth the 2D plot by gaussian 2D convolution
        reference : 'median', 'snr' or 'master' to select the reference normalised spectrum usedin the difference
        cmap : cmap of the 2D plot
        width_min : minimum width in mm above which to search for a peak in Fourier space
        correct_blue : enable correction of the blue detector for HARPS
        correct_blue : enable correction of the red detector for HARPS

        """

        directory = self.directory

        zoom = self.zoom
        smooth_map = self.smooth_map
        cmap = self.cmap
        planet = self.planet
        epsilon = 1e-6
        self.import_material()
        self.import_table()
        load = self.material

        myf.print_box("\n---- RECIPE : CORRECTION FRINGING ----\n")

        kw = "_planet" * planet
        if kw != "":
            print("\n---- PLANET ACTIVATED ----")

        if sub_dico is None:
            sub_dico = self.dico_actif
        print("\n---- DICO %s used ----" % (sub_dico))

        files = glob.glob(directory + "RASSI*.p")
        files = np.sort(files)

        flux = []
        conti = []
        snr = []
        jdb = []
        hl = []
        hr = []

        for i, j in enumerate(files):
            file = pd.read_pickle(j)
            if not i:
                wave = file["wave"]
            snr.append(file["parameters"]["SNR_5500"])
            try:
                jdb.append(file["parameters"]["jdb"])
            except:
                jdb.append(i)
            try:
                hl.append(file["parameters"]["hole_left"])
            except:
                hl.append(None)
            try:
                hr.append(file["parameters"]["hole_right"])
            except:
                hr.append(None)

            flux.append(file["flux" + kw] / file[sub_dico]["continuum_" + continuum])
            conti.append(file[sub_dico]["continuum_" + continuum])

        step = file[sub_dico]["parameters"]["step"]

        wave = np.array(wave)
        flux = np.array(flux)
        conti = np.array(conti)
        snr = np.array(snr)
        jdb = np.array(jdb)

        idx_min = int(myf.find_nearest(wave, wave_min)[0])
        idx_max = int(myf.find_nearest(wave, wave_max)[0])

        mask = np.zeros(len(snr)).astype("bool")
        if jdb_range is not None:
            mask = (np.array(self.table.jdb) > jdb_range[0]) & (
                np.array(self.table.jdb) < jdb_range[1]
            )

        if reference == "median":
            if sum(~mask) < 50:
                print("Not enough spectra out of the temporal specified range")
                mask = np.zeros(len(snr)).astype("bool")
            else:
                print(
                    "%.0f spectra out of the specified temporal range can be used for the median"
                    % (sum(~mask))
                )

        if reference == "snr":
            ref = flux[snr.argmax()]
        elif reference == "median":
            print("[INFO] Reference spectrum : median")
            ref = np.median(flux[~mask], axis=0)
        elif reference == "master":
            print("[INFO] Reference spectrum : master")
            ref = np.array(load["reference_spectrum"])
        elif type(reference) == int:
            print("[INFO] Reference spectrum : spectrum %.0f" % (reference))
            ref = flux[reference]
        else:
            ref = 0 * np.median(flux, axis=0)

        print("[INFO] Pattern analysis for range mm : ", width_range)
        # low = np.percentile(flux-ref,2.5)
        # high = np.percentile(flux-ref,97.5)
        old_diff = myf.smooth2d(flux - ref, smooth_map)
        low = np.percentile(flux / (ref + epsilon), 2.5)
        high = np.percentile(flux / (ref + epsilon), 97.5)

        diff = myf.smooth2d(flux / (ref + epsilon), smooth_map) - 1  # changed for a ratio 21-01-20
        diff[diff == -1] = 0
        diff_backup = diff.copy()

        if jdb_range is None:
            fig = plt.figure(figsize=(18, 6))

            plt.axes([0.06, 0.28, 0.7, 0.65])

            myf.my_colormesh(
                wave[idx_min:idx_max],
                np.arange(len(diff)),
                diff[:, idx_min:idx_max],
                zoom=zoom,
                vmin=low,
                vmax=high,
                cmap=cmap,
            )
            plt.tick_params(direction="in", top=True, right=True, labelbottom=False)
            plt.ylabel("Spectra  indexes (time)", fontsize=14)
            ax = plt.gca()
            cbaxes = fig.add_axes([0.7 + 0.06, 0.28, 0.01, 0.65])
            plt.colorbar(cax=cbaxes)

            plt.axes([0.82, 0.28, 0.07, 0.65], sharey=ax)
            plt.plot(snr, np.arange(len(snr)), "k-")
            plt.tick_params(direction="in", top=True, right=True, labelleft=False)
            plt.xlabel("SNR", fontsize=14)

            plt.axes([0.90, 0.28, 0.07, 0.65], sharey=ax)
            plt.plot(jdb, np.arange(len(snr)), "k-")
            plt.tick_params(direction="in", top=True, right=True, labelleft=False)
            plt.xlabel("jdb", fontsize=14)

            plt.axes([0.06, 0.08, 0.7, 0.2], sharex=ax)
            plt.plot(wave[idx_min:idx_max], flux[snr.argmax()][idx_min:idx_max], color="k")
            plt.xlabel(r"Wavelength [$\AA$]", fontsize=14)
            plt.ylabel("Flux normalised", fontsize=14)
            plt.tick_params(direction="in", top=True, right=True)

            plt.show(block=False)
            index_sphinx = int(myf.sphinx("Which index present a clear pattern ?"))
            index_sphinx2 = int(myf.sphinx("Which index present no pattern ?"))
            plt.close()
        else:
            snr = np.array(self.table.snr)

            if sum(mask):
                i1 = np.argmax(snr[mask])
                index_sphinx = np.arange(len(snr))[mask][i1]

                if sum(mask) == len(mask):
                    index_sphinx2 = -1
                else:
                    i2 = np.argmax(snr[~mask])
                    index_sphinx2 = np.arange(len(snr))[~mask][i2]
            else:
                index_sphinx = -1
                index_sphinx2 = -1
                print("[INFO] No spectrum contaminated by interference pattern")

        print("Index spectrum containing a pattern : %.0f" % (index_sphinx))
        print("Index spectrum not containing a pattern : %.0f" % (index_sphinx2))

        time.sleep(1)

        if index_sphinx >= 0:

            for j in tqdm(range(len(diff))):
                diff_pattern = myc.tableXY(wave, diff[j].copy(), 0 * wave)
                diff_pattern.rolling(
                    window=10000, median=False
                )  # HARDCODE PARAMETER rolling filter to remove telluric power in the fourrier space
                diff_pattern.y[abs(diff_pattern.y) > 3 * diff_pattern.roll_IQ] = (
                    0
                    * np.random.randn(np.sum(abs(diff_pattern.y) > 3 * diff_pattern.roll_IQ))
                    * np.median(diff_pattern.roll_IQ)
                )
                diff[j] = diff_pattern.y

            diff_pattern = myc.tableXY(2 / wave[::-1], diff[index_sphinx][::-1])

            if np.float(index_sphinx2) >= 0:
                diff_flat = myc.tableXY(2 / wave[::-1], diff[index_sphinx2][::-1], 0 * wave)
            else:
                diff_flat = myc.tableXY(2 / wave[::-1], np.median(diff, axis=0)[::-1], 0 * wave)

            new_grid = np.linspace(diff_pattern.x.min(), diff_pattern.x.max(), len(diff_pattern.x))
            diff_pattern.interpolate(new_grid=new_grid, interpolate_x=False)
            diff_flat.interpolate(new_grid=new_grid, interpolate_x=False)

            # diff_pattern.rolling(window=1000) #HARDCODE PARAMETER rolling filter to remove telluric power in the fourrier space
            # diff_pattern.y[abs(diff_pattern.y)>3*diff_pattern.roll_IQ] = np.median(diff_pattern.y)
            # diff_flat.rolling(window=1000) #rolling filter to remove telluric power in the fourrier space
            # diff_flat.y[abs(diff_flat.y)>3*diff_flat.roll_IQ] = np.median(diff_flat.y)

            dl = np.diff(new_grid)[0]

            fft_pattern = np.fft.fft(diff_pattern.y)
            fft_flat = np.fft.fft(diff_flat.y)

            plt.figure()
            plt.plot(np.abs(fft_pattern))
            plt.plot(np.abs(fft_flat))

            diff_fourrier = np.abs(fft_pattern) - np.abs(fft_flat)
            diff_fourrier_pos = diff_fourrier[0 : int(len(diff_fourrier) / 2) + 1]

            width = (
                1e-7
                * abs(np.fft.fftfreq(len(diff_fourrier)))[0 : int(len(diff_fourrier) / 2) + 1]
                / dl
            )  # transformation of the frequency in mm of the material

            maximum = np.argmax(
                diff_fourrier_pos[(width > width_range[0]) & (width < width_range[1])]
            )
            freq_maximum_ref = width[(width > width_range[0]) & (width < width_range[1])][maximum]

            plt.figure()
            plt.plot(width, diff_fourrier_pos, color="k")
            print(
                "\n[INFO] The interference pattern is produced by material with a width of %.3f mm"
                % (freq_maximum_ref)
            )

            new_diff = diff.copy()
            hard_window = 50  # window extraction of the fourier power excess

            index_corrected_pattern_red = []
            index_corrected_pattern_blue = []
            timer_red = -1
            timer_blue = -1

            for j in range(len(diff)):

                diff_pattern = myc.tableXY(2 / wave[::-1], diff[j][::-1], 0 * wave)
                diff_pattern.interpolate(new_grid=new_grid, interpolate_x=False)
                diff_pattern_backup = diff_pattern.y.copy()
                # diff_pattern.rolling(window=1000)  #HARDCODE PARAMETER rolling filter to remove telluric power in the fourrier space
                # diff_pattern.y[abs(diff_pattern.y)>3*diff_pattern.roll_IQ] = np.median(diff_pattern.y)
                emergency = 1

                if (hl[j] == -99.9) | (hr[j] == -99.9):
                    emergency = 0
                    highest = [0, 0, 0]
                else:
                    dstep_clust = abs(np.mean(np.diff(diff[j])))
                    if dstep_clust == 0:
                        dstep_clust = np.mean(abs(np.mean(np.diff(diff, axis=1), axis=1)))
                    mask = myf.clustering(np.cumsum(diff[j]), dstep_clust, 0)[-1]
                    highest = mask[mask[:, 2].argmax()]

                    left = (
                        2 / wave[int(highest[0] - 1.0 / np.diff(wave)[0])]
                    )  # add 1.0 angstrom of security around the gap
                    right = (
                        2 / wave[int(highest[1] + 1.0 / np.diff(wave)[0])]
                    )  # add 1.0 angstrom of security around the gap

                    if hr[j] is not None:
                        if (hr[j] - wave[int(highest[1])]) > 10:
                            print(
                                "The border right of the gap is not the same than the one of the header"
                            )
                            emergency = 0
                    if hl[j] is not None:
                        if (hl[j] - wave[int(highest[0])]) > 10:
                            print(
                                "The border left of the gap is not the same than the one of the header"
                            )
                            emergency = 0

                if (highest[2] > 1000) & (emergency):  # because gap between ccd is large
                    left = myf.find_nearest(diff_pattern.x, left)[0]
                    right = myf.find_nearest(diff_pattern.x, right)[0]

                    left, right = (
                        right[0],
                        left[0],
                    )  # because xaxis is reversed in 1/lambda space

                    fft_pattern_left = np.fft.fft(diff_pattern.y[0:left])
                    fft_flat_left = np.fft.fft(diff_flat.y[0:left])
                    diff_fourrier_left = np.abs(fft_pattern_left) - np.abs(fft_flat_left)

                    fft_pattern_right = np.fft.fft(diff_pattern.y[right:])
                    fft_flat_right = np.fft.fft(diff_flat.y[right:])
                    diff_fourrier_right = np.abs(fft_pattern_right) - np.abs(fft_flat_right)

                    width_right = (
                        1e-7
                        * abs(np.fft.fftfreq(len(diff_fourrier_right)))[
                            0 : int(len(diff_fourrier_right) / 2) + 1
                        ]
                        / dl
                    )  # transformation of the frequency in mm of the material
                    width_left = (
                        1e-7
                        * abs(np.fft.fftfreq(len(diff_fourrier_left)))[
                            0 : int(len(diff_fourrier_left) / 2) + 1
                        ]
                        / dl
                    )  # transformation of the frequency in mm of the material

                    diff_fourrier_pos_left = diff_fourrier_left[
                        0 : int(len(diff_fourrier_left) / 2) + 1
                    ]
                    diff_fourrier_pos_right = diff_fourrier_right[
                        0 : int(len(diff_fourrier_right) / 2) + 1
                    ]

                    maxima_left = myc.tableXY(
                        width_left[(width_left > width_range[0]) & (width_left < width_range[1])],
                        myf.smooth(
                            diff_fourrier_pos_left[
                                (width_left > width_range[0]) & (width_left < width_range[1])
                            ],
                            3,
                        ),
                    )
                    maxima_left.find_max(vicinity=int(hard_window / 2))

                    maxima_right = myc.tableXY(
                        width_right[
                            (width_right > width_range[0]) & (width_right < width_range[1])
                        ],
                        myf.smooth(
                            diff_fourrier_pos_right[
                                (width_right > width_range[0]) & (width_right < width_range[1])
                            ],
                            3,
                        ),
                    )
                    maxima_right.find_max(vicinity=int(hard_window / 2))

                    five_maxima_left = maxima_left.x_max[np.argsort(maxima_left.y_max)[::-1]][0:10]
                    five_maxima_right = maxima_right.x_max[np.argsort(maxima_right.y_max)[::-1]][
                        0:10
                    ]

                    five_maxima_left_y = maxima_left.y_max[np.argsort(maxima_left.y_max)[::-1]][
                        0:10
                    ]
                    five_maxima_right_y = maxima_right.y_max[np.argsort(maxima_right.y_max)[::-1]][
                        0:10
                    ]

                    thresh_left = 10 * np.std(diff_fourrier_pos_left)
                    thresh_right = 10 * np.std(diff_fourrier_pos_right)

                    five_maxima_left = five_maxima_left[five_maxima_left_y > thresh_left]
                    five_maxima_right = five_maxima_right[five_maxima_right_y > thresh_right]

                    if len(five_maxima_left) > 0:
                        where, freq_maximum_left, dust = myf.find_nearest(
                            five_maxima_left, freq_maximum_ref
                        )
                        maximum_left = maxima_left.index_max[np.argsort(maxima_left.y_max)[::-1]][
                            0:10
                        ]
                        maximum_left = maximum_left[five_maxima_left_y > thresh_left][where]
                    else:
                        freq_maximum_left = 0
                    if len(five_maxima_right) > 0:
                        where, freq_maximum_right, dust = myf.find_nearest(
                            five_maxima_right, freq_maximum_ref
                        )
                        maximum_right = maxima_right.index_max[
                            np.argsort(maxima_right.y_max)[::-1]
                        ][0:10]
                        maximum_right = maximum_right[five_maxima_right_y > thresh_right][where]
                    else:
                        freq_maximum_right = 0

                    offset_left = np.where(
                        ((width_left > width_range[0]) & (width_left < width_range[1])) == True
                    )[0][0]
                    offset_right = np.where(
                        ((width_right > width_range[0]) & (width_right < width_range[1])) == True
                    )[0][0]

                    if ((abs(freq_maximum_left - freq_maximum_ref) / freq_maximum_ref) < 0.05) & (
                        correct_red
                    ):
                        print(
                            "[INFO] Correcting night %.0f (r). Interference produced a width of %.3f mm"
                            % (j, freq_maximum_left)
                        )
                        timer_red += 1
                        index_corrected_pattern_red.append(timer_red)

                        # left
                        smooth = myc.tableXY(
                            np.arange(len(diff_fourrier_pos_left)),
                            np.ravel(
                                pd.DataFrame(diff_fourrier_pos_left)
                                .rolling(hard_window, min_periods=1, center=True)
                                .std()
                            ),
                        )
                        smooth.find_max(vicinity=int(hard_window / 2))
                        maxi = myf.find_nearest(smooth.x_max, maximum_left + offset_left)[1]

                        smooth.diff(replace=False)
                        smooth.deri.rm_outliers(m=5, kind="sigma")

                        loc_slope = np.arange(len(smooth.deri.mask))[~smooth.deri.mask]
                        cluster = myf.clustering(loc_slope, hard_window / 2, 0)[
                            0
                        ]  # half size of the rolling window
                        dist = np.ravel([np.mean(k) - maxi for k in cluster])
                        closest = np.sort(np.abs(dist).argsort()[0:2])

                        mini1 = cluster[closest[0]].min()
                        mini2 = cluster[closest[1]].max()

                        fft_pattern_left[mini1:mini2] = fft_flat_left[mini1:mini2]
                        fft_pattern_left[-mini2:-mini1] = fft_flat_left[-mini2:-mini1]
                        diff_pattern.y[0:left] = np.real(np.fft.ifft(fft_pattern_left))
                    if ((abs(freq_maximum_right - freq_maximum_ref) / freq_maximum_ref) < 0.05) & (
                        correct_blue
                    ):
                        print(
                            "[INFO] Correcting night %.0f (b). Interference produced a width of %.3f mm"
                            % (j, freq_maximum_right)
                        )
                        # right
                        timer_blue += 1
                        index_corrected_pattern_blue.append(timer_blue)

                        smooth = myc.tableXY(
                            np.arange(len(diff_fourrier_pos_right)),
                            np.ravel(
                                pd.DataFrame(diff_fourrier_pos_right)
                                .rolling(hard_window, min_periods=1, center=True)
                                .std()
                            ),
                        )
                        smooth.find_max(vicinity=int(hard_window / 2))
                        maxi = myf.find_nearest(smooth.x_max, maximum_right + offset_right)[1]
                        smooth.diff(replace=False)
                        smooth.deri.rm_outliers(
                            m=5, kind="sigma"
                        )  # find peak in fourier space and width from derivative

                        loc_slope = np.arange(len(smooth.deri.mask))[~smooth.deri.mask]
                        cluster = myf.clustering(loc_slope, hard_window / 2, 0)[
                            0
                        ]  # half size of the rolling window
                        dist = np.ravel([np.mean(k) - maxi for k in cluster])
                        closest = np.sort(np.abs(dist).argsort()[0:2])

                        mini1 = cluster[closest[0]].min()
                        mini2 = cluster[closest[1]].max()

                        fft_pattern_right[mini1:mini2] = fft_flat_right[mini1:mini2]
                        fft_pattern_right[-mini2:-mini1] = fft_flat_right[-mini2:-mini1]
                        diff_pattern.y[right:] = np.real(np.fft.ifft(fft_pattern_right))

                        # final

                    correction = myc.tableXY(
                        diff_pattern.x,
                        diff_pattern_backup - diff_pattern.y,
                        0 * diff_pattern.x,
                    )
                    correction.interpolate(new_grid=2 / wave[::-1], interpolate_x=False)
                    correction.x = 2 / correction.x[::-1]
                    correction.y = correction.y[::-1]
                    # diff_pattern.x = 2/diff_pattern.x[::-1]
                    # diff_pattern.y = diff_pattern.y[::-1]
                    # diff_pattern.interpolate(new_grid=wave)
                    new_diff[j] = diff_backup[j] - correction.y

                else:
                    fft_pattern = np.fft.fft(diff_pattern.y)
                    diff_fourrier = np.abs(fft_pattern) - np.abs(fft_flat)
                    diff_fourrier_pos = diff_fourrier[0 : int(len(diff_fourrier) / 2) + 1]
                    width = (
                        1e-7
                        * abs(np.fft.fftfreq(len(diff_fourrier)))[
                            0 : int(len(diff_fourrier) / 2) + 1
                        ]
                        / dl
                    )  # transformation of the frequency in mm of the material

                    maxima = myc.tableXY(
                        width[(width > width_range[0]) & (width < width_range[1])],
                        diff_fourrier_pos[(width > width_range[0]) & (width < width_range[1])],
                    )
                    maxima.find_max(vicinity=50)
                    five_maxima = maxima.x_max[np.argsort(maxima.y_max)[::-1]][0:5]
                    match = int(np.argmin(abs(five_maxima - freq_maximum_ref)))
                    freq_maximum = five_maxima[match]
                    maximum = maxima.index_max[np.argsort(maxima.y_max)[::-1]][0:5][match]
                    offset = np.where(
                        ((width > width_range[0]) & (width < width_range[1])) == True
                    )[0][0]

                    if (abs(freq_maximum - freq_maximum_ref) / freq_maximum) < 0.10:
                        print(
                            "[INFO] Correcting night %.0f. The interference pattern is produced by material with a width of %.3f mm"
                            % (j, freq_maximum)
                        )
                        timer_blue += 1
                        index_corrected_pattern_red.append(timer_red)

                        smooth = myc.tableXY(
                            np.arange(len(diff_fourrier_pos)),
                            np.ravel(
                                pd.DataFrame(diff_fourrier_pos)
                                .rolling(100, min_periods=1, center=True)
                                .std()
                            ),
                        )
                        smooth.find_max(vicinity=30)
                        maxi = myf.find_nearest(smooth.x_max, maximum + offset)[1]

                        smooth.diff(replace=False)
                        smooth.deri.rm_outliers(m=5, kind="sigma")

                        loc_slope = np.arange(len(smooth.deri.mask))[~smooth.deri.mask]
                        cluster = myf.clustering(loc_slope, 50, 0)[
                            0
                        ]  # half size of the rolling window
                        dist = np.ravel([np.mean(k) - maxi for k in cluster])

                        closest = np.sort(np.abs(dist).argsort()[0:2])

                        mini1 = cluster[closest[0]].min()
                        if len(closest) > 1:
                            mini2 = cluster[closest[1]].max()
                        else:
                            mini2 = mini1 + 1

                        fft_pattern[mini1:mini2] = fft_flat[mini1:mini2]
                        fft_pattern[-mini2:-mini1] = fft_flat[-mini2:-mini1]
                        diff_pattern.y = np.real(np.fft.ifft(fft_pattern))
                        diff_pattern.x = 2 / diff_pattern.x[::-1]
                        diff_pattern.y = diff_pattern.y[::-1]
                        diff_pattern.interpolate(new_grid=wave, interpolate_x=False)
                        new_diff[j] = diff_pattern.y

            self.index_corrected_pattern_red = index_corrected_pattern_red
            self.index_corrected_pattern_blue = index_corrected_pattern_blue

            correction = diff_backup - new_diff

            ratio2_backup = new_diff + 1

            new_conti = conti * flux / (ref * ratio2_backup + epsilon)
            new_continuum = new_conti.copy()
            new_continuum[flux == 0] = conti[flux == 0]

            diff2_backup = flux * conti / new_continuum - ref

            new_conti = flux * conti / (diff2_backup + ref + epsilon)

            new_continuum = new_conti.copy()
            new_continuum[flux == 0] = conti[flux == 0]

            low_cmap = self.low_cmap * 100
            high_cmap = self.high_cmap * 100

            fig = plt.figure(figsize=(21, 9))
            plt.axes([0.05, 0.66, 0.90, 0.25])
            myf.my_colormesh(
                wave[idx_min:idx_max],
                np.arange(len(diff)),
                100 * old_diff[:, idx_min:idx_max],
                zoom=zoom,
                vmin=low_cmap,
                vmax=high_cmap,
                cmap=cmap,
            )
            plt.tick_params(direction="in", top=True, right=True, labelbottom=False)
            plt.ylabel("Spectra  indexes (time)", fontsize=14)
            ax = plt.gca()
            cbaxes = fig.add_axes([0.95, 0.66, 0.01, 0.25])
            ax1 = plt.colorbar(cax=cbaxes)
            ax1.ax.set_ylabel(r"$\Delta$ flux normalised [%]", fontsize=14)

            plt.axes([0.05, 0.375, 0.90, 0.25], sharex=ax, sharey=ax)
            myf.my_colormesh(
                wave[idx_min:idx_max],
                np.arange(len(new_diff)),
                100 * diff2_backup[:, idx_min:idx_max],
                zoom=zoom,
                vmin=low_cmap,
                vmax=high_cmap,
                cmap=cmap,
            )
            plt.tick_params(direction="in", top=True, right=True, labelbottom=False)
            plt.ylabel("Spectra  indexes (time)", fontsize=14)
            ax = plt.gca()
            cbaxes2 = fig.add_axes([0.95, 0.375, 0.01, 0.25])
            ax2 = plt.colorbar(cax=cbaxes2)
            ax2.ax.set_ylabel(r"$\Delta$ flux normalised [%]", fontsize=14)

            plt.axes([0.05, 0.09, 0.90, 0.25], sharex=ax, sharey=ax)
            myf.my_colormesh(
                wave[idx_min:idx_max],
                np.arange(len(new_diff)),
                100 * old_diff[:, idx_min:idx_max] - 100 * diff2_backup[:, idx_min:idx_max],
                vmin=low_cmap,
                vmax=high_cmap,
                zoom=zoom,
                cmap=cmap,
            )
            plt.xlabel(r"Wavelength [$\AA$]", fontsize=14)
            plt.ylabel("Spectra  indexes (time)", fontsize=14)
            plt.ylim(0, None)
            ax = plt.gca()
            cbaxes3 = fig.add_axes([0.95, 0.09, 0.01, 0.25])
            ax3 = plt.colorbar(cax=cbaxes3)
            ax3.ax.set_ylabel(r"$\Delta$ flux normalised [%]", fontsize=14)

            plt.savefig(self.dir_root + "IMAGES/Correction_pattern.png")

            pre_map = np.zeros(np.shape(diff2_backup))
            if sub_dico == "matching_fourier":
                spec = self.import_spectrum()
                sub_dico = spec[sub_dico]["parameters"]["sub_dico_used"]
                step -= 1
                pre_map = pd.read_pickle(self.dir_root + "CORRECTION_MAP/map_matching_fourier.p")[
                    "correction_map"
                ]

            correction_pattern = old_diff - diff2_backup
            to_be_saved = {"wave": wave, "correction_map": correction_pattern + pre_map}
            io.pickle_dump(
                to_be_saved,
                open(self.dir_root + "CORRECTION_MAP/map_matching_fourier.p", "wb"),
            )

            print("\nComputation of the new continua, wait ... \n")
            time.sleep(0.5)

            i = -1
            for j in tqdm(files):
                i += 1
                file = pd.read_pickle(j)
                output = {"continuum_" + continuum: new_continuum[i]}
                file["matching_fourier"] = output
                file["matching_fourier"]["parameters"] = {
                    "pattern_width": freq_maximum_ref,
                    "width_cutoff": width_range,
                    "index_corrected_red": np.array(index_corrected_pattern_red),
                    "index_corrected_blue": np.array(index_corrected_pattern_blue),
                    "reference_pattern": index_sphinx,
                    "reference_flat": index_sphinx2,
                    "reference_spectrum": reference,
                    "sub_dico_used": sub_dico,
                    "step": step + 1,
                }
                io.save_pickle(j, file)

            self.dico_actif = "matching_fourier"

            plt.show(block=False)

            self.fft_output = np.array([diff, new_diff, conti, new_continuum])

    # outliers
    def yarara_correct_smooth(
        self,
        sub_dico="matching_diff",
        continuum="linear",
        reference="median",
        wave_min=4200,
        wave_max=4300,
        window_ang=5,
    ):

        myf.print_box("\n---- RECIPE : CORRECTION SMOOTH ----\n")

        directory = self.directory

        zoom = self.zoom
        smooth_map = self.smooth_map
        low_cmap = self.low_cmap * 100
        high_cmap = self.high_cmap * 100
        cmap = self.cmap
        planet = self.planet

        self.import_material()
        self.import_table()
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
        # flux_err = []
        conti = []
        snr = []
        for i, j in enumerate(files):
            file = pd.read_pickle(j)
            if not i:
                wave = file["wave"]
                dgrid = file["parameters"]["dwave"]
                try:
                    hl = file["parameters"]["hole_left"]
                except:
                    hl = None
                try:
                    hr = file["parameters"]["hole_right"]
                except:
                    hr = None

            snr.append(file["parameters"]["SNR_5500"])

            f = file["flux" + kw]
            f_std = file["flux_err"]
            c = file[sub_dico]["continuum_" + continuum]
            c_std = file["continuum_err"]
            f_norm, f_norm_std = myf.flux_norm_std(f, f_std, c, c_std)
            flux.append(f_norm)
            # flux_err.append(f_norm_std)
            conti.append(c)

        step = file[sub_dico]["parameters"]["step"]

        snr = np.array(snr)
        wave = np.array(wave)
        all_flux = np.array(flux)
        # all_flux_std = np.array(flux_err)
        conti = np.array(conti)

        if reference == "snr":
            ref = all_flux[snr.argmax()]
        elif reference == "median":
            print("[INFO] Reference spectrum : median")
            ref = np.median(all_flux, axis=0)
        elif reference == "master":
            print("[INFO] Reference spectrum : master")
            ref = np.array(load["reference_spectrum"])
        elif type(reference) == int:
            print("[INFO] Reference spectrum : spectrum %.0f" % (reference))
            ref = all_flux[reference]
        else:
            ref = 0 * np.median(all_flux, axis=0)

        diff_ref = all_flux.copy() - ref

        flux_ratio = (all_flux.copy() / (ref + epsilon)) - 1

        box_pts = int(window_ang / dgrid)

        for k in tqdm(range(len(all_flux))):
            spec_smooth = myf.smooth(
                myf.smooth(flux_ratio[k], box_pts=box_pts, shape=50),
                box_pts=box_pts,
                shape="savgol",
            )
            if hl is not None:
                i1 = int(myf.find_nearest(wave, hl)[0])
                i2 = int(myf.find_nearest(wave, hr)[0])
                spec_smooth[i1 - box_pts : i2 + box_pts] = 0

            flux_ratio[k] -= spec_smooth

        flux_ratio += 1
        flux_ratio *= ref

        diff_ref2 = flux_ratio - ref

        del flux_ratio

        correction_smooth = diff_ref - diff_ref2

        new_conti = conti * (diff_ref + ref) / (diff_ref2 + ref + epsilon)
        new_continuum = new_conti.copy()
        new_continuum[flux == 0] = conti[flux == 0].copy()
        new_continuum[new_continuum != new_continuum] = conti[
            new_continuum != new_continuum
        ].copy()  # to supress mystic nan appearing
        new_continuum[np.isnan(new_continuum)] = conti[np.isnan(new_continuum)].copy()
        new_continuum[new_continuum == 0] = conti[new_continuum == 0].copy()
        new_continuum = self.uncorrect_hole(new_continuum, conti)

        idx_min = 0
        idx_max = len(wave)

        if wave_min is not None:
            idx_min = myf.find_nearest(wave, wave_min)[0]
        if wave_max is not None:
            idx_max = myf.find_nearest(wave, wave_max)[0] + 1

        if (idx_min == 0) & (idx_max == 1):
            idx_max = myf.find_nearest(wave, np.min(wave) + 200)[0] + 1

        new_wave = wave[int(idx_min) : int(idx_max)]

        fig = plt.figure(figsize=(21, 9))

        plt.axes([0.05, 0.66, 0.90, 0.25])
        myf.my_colormesh(
            new_wave,
            np.arange(len(diff_ref)),
            100 * diff_ref[:, int(idx_min) : int(idx_max)],
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
        myf.my_colormesh(
            new_wave,
            np.arange(len(diff_ref)),
            100 * diff_ref2[:, int(idx_min) : int(idx_max)],
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
        myf.my_colormesh(
            new_wave,
            np.arange(len(diff_ref)),
            100
            * (
                diff_ref[:, int(idx_min) : int(idx_max)]
                - diff_ref2[:, int(idx_min) : int(idx_max)]
            ),
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

        if sub_dico == "matching_diff":
            plt.savefig(self.dir_root + "IMAGES/Correction_diff.png")
            spec = self.import_spectrum()
            name = "diff"
            recenter = spec[sub_dico]["parameters"]["recenter"]
            ref_name = spec[sub_dico]["parameters"]["reference_continuum"]
            savgol_window = spec[sub_dico]["parameters"]["savgol_window"]
            sub_dico = spec[sub_dico]["parameters"]["sub_dico_used"]
        else:
            plt.savefig(self.dir_root + "IMAGES/Correction_smooth.png")
            to_be_saved = {"wave": wave, "correction_map": correction_smooth}
            io.pickle_dump(
                to_be_saved,
                open(self.dir_root + "CORRECTION_MAP/map_matching_smooth.p", "wb"),
            )
            name = "smooth"
            recenter = False
            ref_name = str(reference)
            savgol_window = 0

        # diff_ref2 = flux_ratio - ref
        # correction_smooth = diff_ref - diff_ref2
        # new_conti = conti*(diff_ref+ref)/(diff_ref2+ref+epsilon)
        # new_continuum = new_conti.copy()
        # new_continuum = self.uncorrect_hole(new_continuum,conti)

        print("Computation of the new continua, wait ... \n")
        time.sleep(0.5)
        count_file = -1
        for j in tqdm(files):
            count_file += 1
            file = pd.read_pickle(j)
            conti = new_continuum[count_file]
            mask = yarara_artefact_suppressed(
                file[sub_dico]["continuum_" + continuum],
                conti,
                larger_than=50,
                lower_than=-50,
            )
            conti[mask] = file[sub_dico]["continuum_" + continuum][mask]
            output = {"continuum_" + continuum: conti}
            file["matching_" + name] = output
            file["matching_" + name]["parameters"] = {
                "reference_continuum": ref_name,
                "sub_dico_used": sub_dico,
                "savgol_window": savgol_window,
                "window_ang": window_ang,
                "step": step + 1,
                "recenter": recenter,
            }
            io.save_pickle(j, file)

    # processing
    def yarara_retropropagation_correction(
        self,
        correction_map="matching_smooth",
        sub_dico="matching_cosmics",
        continuum="linear",
    ):

        # we introduce the continuum correction (post-processing of rassine normalisation) inside the cosmics correction
        # it allow to not lose the output product of rassine (no need of backup)
        # allow to rerun the code iteratively from the beginning
        # do not use matching_diff + substract_map['cosmics','smooth'] simultaneously otherwise 2 times corrections
        # rurunning the cosmics recipes will kill this correction, therefore a sphinx warning is included in the recipes
        # when a loop is rerun (beginning at fourier correction or water correction), make sure to finish completely the loop

        myf.print_box("\n---- RECIPE : RETROPROPAGATION CORRECTION MAP ----\n")

        directory = self.directory

        planet = self.planet

        self.import_material()
        self.import_table()
        file_test = self.import_spectrum()

        try:
            hl = file_test["parameters"]["hole_left"]
        except:
            hl = None
        try:
            hr = file_test["parameters"]["hole_right"]
        except:
            hr = None

        wave = np.array(self.material["wave"])
        if hl is not None:
            i1 = int(myf.find_nearest(wave, hl)[0])
            i2 = int(myf.find_nearest(wave, hr)[0])

        epsilon = 1e-12

        kw = "_planet" * planet
        if kw != "":
            print("\n---- PLANET ACTIVATED ----")

        files = glob.glob(directory + "RASSI*.p")
        files = np.sort(files)

        correction_retro = pd.read_pickle(
            self.dir_root + "CORRECTION_MAP/map_" + correction_map + ".p"
        )["correction_map"]

        m = pd.read_pickle(
            self.dir_root + "CORRECTION_MAP/map_" + sub_dico + ".p"
        )  # allow the iterative process to be run
        m["correction_map"] += correction_retro
        io.pickle_dump(m, open(self.dir_root + "CORRECTION_MAP/map_" + sub_dico + ".p", "wb"))

        print("\nComputation of the new continua, wait ... \n")
        time.sleep(0.5)
        count_file = -1
        for j in tqdm(files):
            count_file += 1
            file = pd.read_pickle(j)
            conti = file["matching_cosmics"]["continuum_" + continuum]
            flux = file["flux" + kw]

            flux_norm_corrected = flux / conti - correction_retro[count_file]
            new_conti = flux / (flux_norm_corrected + epsilon)  # flux/flux_norm_corrected

            new_continuum = new_conti.copy()
            new_continuum[flux == 0] = conti[flux == 0]
            new_continuum[new_continuum != new_continuum] = conti[
                new_continuum != new_continuum
            ]  # to supress mystic nan appearing
            new_continuum[np.isnan(new_continuum)] = conti[np.isnan(new_continuum)]
            new_continuum[new_continuum == 0] = conti[new_continuum == 0]
            if hl is not None:
                new_continuum[i1:i2] = conti[i1:i2]

            file[sub_dico]["continuum_" + continuum] = new_continuum
            io.save_pickle(j, file)

    # =============================================================================
    #  TELLURIC CORRECTION
    # =============================================================================

    # telluric
    def yarara_correct_telluric_proxy(
        self,
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
        reference : 'median', 'snr' or 'master' to select the reference normalised spectrum usedin the difference
        berv_shift : True/False to move in terrestrial rest-frame
        proxy1_corr : keyword  of the first proxies from RASSINE dictionnary to use in the correlation
        proxy1_detrending : Degree of the polynomial fit to detrend the proxy
        proxy2_corr : keyword  of the second proxies from RASSINE dictionnary to use in the correlation
        proxy2_detrending : Degree of the polynomial fit to detrend the proxy
        cmap : cmap of the 2D plot
        min_wave_correction : wavelength limit above which to correct
        min_r_corr : minimum correlation coefficient of one of the two proxies to consider a line as telluric
        dwin : window correction increase by dwin to slightly correct above around the peak of correlation
        positive_coeff : The correction can only be absorption line profile moving and no positive


        """

        if sub_dico_output == "telluric":
            myf.print_box("\n---- RECIPE : CORRECTION TELLURIC WATER ----\n")
            name = "water"
        elif sub_dico_output == "oxygen":
            myf.print_box("\n---- RECIPE : CORRECTION TELLURIC OXYGEN ----\n")
            name = "oxygen"
        else:
            myf.print_box("\n---- RECIPE : CORRECTION TELLURIC PROXY ----\n")
            name = "telluric"

        directory = self.directory

        zoom = self.zoom
        smooth_map = self.smooth_map
        low_cmap = self.low_cmap
        high_cmap = self.high_cmap
        cmap = self.cmap
        planet = self.planet

        self.import_material()
        self.import_table()
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
        err_flux = []
        snr = []
        conti = []
        prox = []
        jdb = []
        berv = []
        rv_shift = []

        for i, j in enumerate(files):
            file = pd.read_pickle(j)
            if not i:
                wave = file["wave"]
                hole_left = file["parameters"]["hole_left"]
                hole_right = file["parameters"]["hole_right"]
                dgrid = file["parameters"]["dwave"]
            snr.append(file["parameters"]["SNR_5500"])
            for proxy_name in proxies_corr:
                prox.append(file["parameters"][proxy_name])

            f = file["flux" + kw]
            f_std = file["flux_err"]
            c = file[sub_dico]["continuum_" + continuum]
            c_std = file["continuum_err"]
            f_norm, f_norm_std = myf.flux_norm_std(f, f_std, c, c_std)
            flux.append(f_norm)
            conti.append(c)
            err_flux.append(f_norm_std)

            try:
                jdb.append(file["parameters"]["jdb"])
            except:
                jdb.append(i)
            try:
                berv.append(file["parameters"][berv_shift])
            except:
                berv.append(0)
            try:
                rv_shift.append(file["parameters"]["RV_shift"])
            except:
                rv_shift.append(0)

        step = file[sub_dico]["parameters"]["step"]

        wave = np.array(wave)
        flux = np.array(flux)
        err_flux = np.array(err_flux)
        conti = np.array(conti)
        snr = np.array(snr)
        proxy = np.array(prox)
        proxy = np.reshape(proxy, (len(proxy) // len(proxies_corr), len(proxies_corr)))

        jdb = np.array(jdb)
        berv = np.array(berv)
        rv_shift = np.array(rv_shift)
        mean_berv = np.mean(berv)
        berv = berv - mean_berv - rv_shift

        if proxies_detrending is None:
            proxies_detrending = [0] * len(proxies_corr)

        for k in range(len(proxies_corr)):
            proxy1 = myc.tableXY(jdb, proxy[:, k])
            proxy1.substract_polyfit(proxies_detrending[k])
            proxy[:, k] = proxy1.detrend_poly.y

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

        low = np.percentile(flux - ref, 2.5)
        high = np.percentile(flux - ref, 97.5)

        ratio = myf.smooth2d(flux / (ref + 1e-6), smooth_map)
        ratio_backup = ratio.copy()

        diff_backup = myf.smooth2d(flux - ref, smooth_map)

        if np.sum(abs(berv)) != 0:
            for j in tqdm(np.arange(len(flux))):
                test = myc.tableXY(wave, ratio[j], err_flux[j] / (ref + 1e-6))
                test.x = myf.doppler_r(test.x, berv[j] * 1000)[1]
                test.interpolate(new_grid=wave, method="cubic", replace=True, interpolate_x=False)
                ratio[j] = test.y
                err_flux[j] = test.yerr

        t = myc.table(ratio)
        t.rms_w(1 / (err_flux) ** 2, axis=0)

        rslope = []
        rcorr = []
        for k in range(len(proxies_corr)):
            rslope.append(
                np.median(
                    (ratio - np.mean(ratio, axis=0))
                    / ((proxy[:, k] - np.mean(proxy[:, k]))[:, np.newaxis]),
                    axis=0,
                )
            )
            rcorr.append(abs(rslope[-1] * np.std(proxy[:, k]) / (t.rms + epsilon)))

        # rcorr1 = abs(rslope1*np.std(proxy1)/np.std(ratio,axis=0))
        # rcorr2 = abs(rslope2*np.std(proxy2)/np.std(ratio,axis=0))

        rslope = np.array(rslope)
        rcorr = np.array(rcorr)

        rcorr = np.max(rcorr, axis=0)
        r_corr = myc.tableXY(wave, rcorr)
        r_corr.smooth(box_pts=smooth_corr, shape="savgol", replace=True)
        rcorr = r_corr.y

        if wave_min_correction is None:
            wave_min_correction = np.min(wave)

        if wave_max_correction is None:
            wave_max_correction = np.max(wave)

        if min_r_corr is None:
            min_r_corr = np.percentile(rcorr[wave < 5400], 75) + 1.5 * myf.IQ(rcorr[wave < 5400])
            print(
                "\n [INFO] Significative R Pearson detected as %.2f based on wavelength smaller than 5400 \AA"
                % (min_r_corr)
            )

        first_guess_position = (
            (rcorr > min_r_corr) & (wave > wave_min_correction) & (wave < wave_max_correction)
        )  # only keep >0.4 and redder than 4950 AA
        second_guess_position = first_guess_position

        # fwhm_telluric = np.median(self.table['telluric_fwhm'])
        fwhm_telluric = self.star_info["FWHM"][""]  # 09.08.21
        val, borders = myf.clustering(first_guess_position, 0.5, 1)
        val = np.array([np.product(v) for v in val]).astype("bool")
        borders = borders[val]
        wave_tel = wave[(0.5 * (borders[:, 0] + borders[:, 1])).astype("int")]
        extension = np.round(sigma_ext * fwhm_telluric / 3e5 * wave_tel / dgrid, 0).astype("int")
        borders[:, 0] -= extension
        borders[:, 1] += extension
        borders[:, 2] = borders[:, 1] - borders[:, 0] + 1
        borders = myf.merge_borders(borders)
        second_guess_position = myf.flat_clustering(len(wave), borders).astype("bool")

        guess_position = np.arange(len(second_guess_position))[second_guess_position]

        correction = np.zeros((len(wave), len(jdb)))

        len_segment = 10000
        print("\n")
        for k in range(len(guess_position) // len_segment + 1):
            print(
                " [INFO] Segment %.0f/%.0f being reduced\n"
                % (k + 1, len(guess_position) // len_segment + 1)
            )
            second_guess_position = guess_position[k * len_segment : (k + 1) * len_segment]
            # print(second_guess_position)

            collection = myc.table(ratio.T[second_guess_position])

            base_vec = np.vstack(
                [np.ones(len(flux))] + [proxy[:, k] for k in range(len(proxies_corr))]
            )
            # rm outliers and define weight for the fit
            weights = (1 / (err_flux / (ref + 1e-6)) ** 2).T[second_guess_position]
            IQ = myf.IQ(collection.table, axis=1)
            Q1 = np.nanpercentile(collection.table, 25, axis=1)
            Q3 = np.nanpercentile(collection.table, 75, axis=1)
            sup = Q3 + 1.5 * IQ
            inf = Q1 - 1.5 * IQ
            out = (collection.table > sup[:, np.newaxis]) | (collection.table < inf[:, np.newaxis])
            weights[out] = np.min(weights) / 100

            collection.fit_base(base_vec, weight=weights, num_sim=1)

            correction[second_guess_position] = collection.coeff_fitted.dot(base_vec)

        correction = np.transpose(correction)
        correction[correction == 0] = 1

        correction_backup = correction.copy()
        if np.sum(abs(berv)) != 0:
            for j in tqdm(np.arange(len(flux))):
                test = myc.tableXY(wave, correction[j], 0 * wave)
                test.x = myf.doppler_r(test.x, berv[j] * 1000)[0]
                test.interpolate(new_grid=wave, method="cubic", replace=True, interpolate_x=False)
                correction_backup[j] = test.y

        index_min_backup = int(
            myf.find_nearest(wave, myf.doppler_r(wave[0], berv.max() * 1000)[0])[0]
        )
        index_max_backup = int(
            myf.find_nearest(wave, myf.doppler_r(wave[-1], berv.min() * 1000)[0])[0]
        )
        correction_backup[:, 0:index_min_backup] = 1
        correction_backup[:, index_max_backup:] = 1
        index_hole_right = int(
            myf.find_nearest(wave, hole_right + 1)[0]
        )  # correct 1 angstrom band due to stange artefact at the border of the gap
        index_hole_left = int(
            myf.find_nearest(wave, hole_left - 1)[0]
        )  # correct 1 angstrom band due to stange artefact at the border of the gap
        correction_backup[:, index_hole_left : index_hole_right + 1] = 1

        #        if positive_coeff:
        #            correction_backup[correction_backup>0] = 0

        ratio2_backup = ratio_backup - correction_backup + 1

        # print(psutil.virtual_memory().percent)

        del correction_backup
        del correction
        del err_flux

        new_conti = conti * flux / (ref * ratio2_backup + epsilon)
        new_continuum = new_conti.copy()
        new_continuum[flux == 0] = conti[flux == 0]

        del ratio2_backup
        del ratio_backup

        diff2_backup = flux * conti / new_continuum - ref

        # plot end

        idx_min = 0
        idx_max = len(wave)

        if wave_min is not None:
            idx_min = myf.find_nearest(wave, wave_min)[0]
        if wave_max is not None:
            idx_max = myf.find_nearest(wave, wave_max)[0] + 1

        new_wave = wave[int(idx_min) : int(idx_max)]

        fig = plt.figure(figsize=(21, 9))

        plt.axes([0.05, 0.66, 0.90, 0.25])
        myf.my_colormesh(
            new_wave,
            np.arange(len(diff_backup)),
            diff_backup[:, int(idx_min) : int(idx_max)],
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
        plt.colorbar(cax=cbaxes)

        plt.axes([0.05, 0.375, 0.90, 0.25], sharex=ax, sharey=ax)
        myf.my_colormesh(
            new_wave,
            np.arange(len(diff_backup)),
            diff2_backup[:, int(idx_min) : int(idx_max)],
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
        plt.colorbar(cax=cbaxes2)

        plt.axes([0.05, 0.09, 0.90, 0.25], sharex=ax, sharey=ax)
        myf.my_colormesh(
            new_wave,
            np.arange(len(diff_backup)),
            (diff_backup - diff2_backup)[:, int(idx_min) : int(idx_max)],
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
        plt.colorbar(cax=cbaxes3)

        plt.savefig(self.dir_root + "IMAGES/Correction_" + name + ".png")

        correction_water = diff_backup - diff2_backup
        to_be_saved = {"wave": wave, "correction_map": correction_water}
        io.pickle_dump(
            to_be_saved,
            open(
                self.dir_root + "CORRECTION_MAP/map_matching_" + sub_dico_output + ".p",
                "wb",
            ),
        )

        print("\nComputation of the new continua, wait ... \n")
        time.sleep(0.5)

        if sub_dico == "matching_" + sub_dico_output:
            spec = self.import_spectrum()
            sub_dico = spec[sub_dico]["parameters"]["sub_dico_used"]

        i = -1
        for j in tqdm(files):
            i += 1
            file = pd.read_pickle(j)
            output = {"continuum_" + continuum: new_continuum[i]}
            file["matching_" + sub_dico_output] = output
            file["matching_" + sub_dico_output]["parameters"] = {
                "reference_spectrum": reference,
                "sub_dico_used": sub_dico,
                "proxies": proxies_corr,
                "min_wave_correction ": wave_min_correction,
                "minimum_r_corr": min_r_corr,
                "step": step + 1,
            }
            io.save_pickle(j, file)

        self.dico_actif = "matching_" + sub_dico_output

    # =============================================================================
    # OXYGENE CORRECTION
    # =============================================================================

    # telluric
    def yarara_correct_oxygen(
        self,
        sub_dico="matching_telluric",
        continuum="linear",
        berv_shift="berv",
        reference="master",
        wave_min=5760,
        wave_max=5850,
        oxygene_bands=[[5787, 5835], [6275, 6340], [6800, 6950]],
    ):

        """
        Display the time-series spectra with proxies and its correlation

        Parameters
        ----------
        sub_dico : The sub_dictionnary used to  select the continuum
        continuum : The continuum to select (either linear or cubic)
        wave_min : Minimum x axis limit
        wave_max : Maximum x axis limit
        smooth_map = int-type, smooth the 2D plot by gaussian 2D convolution
        berv_shift : True/False to move in terrestrial rest-frame
        cmap : cmap of the 2D plot
        low_cmap : vmin cmap colorbar
        high_cmap : vmax cmap colorbar

        """

        myf.print_box("\n---- RECIPE : CORRECTION TELLURIC OXYGEN ----\n")

        directory = self.directory

        zoom = self.zoom
        smooth_map = self.smooth_map
        cmap = self.cmap
        planet = self.planet
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
        flux_err = []
        conti = []
        snr = []
        jdb = []
        berv = []
        rv_shift = []

        for i, j in enumerate(files):
            file = pd.read_pickle(j)
            if not i:
                wave = file["wave"]
                hole_left = file["parameters"]["hole_left"]
                hole_right = file["parameters"]["hole_right"]
            snr.append(file["parameters"]["SNR_5500"])

            f = file["flux" + kw]
            f_std = file["flux_err"]
            c = file[sub_dico]["continuum_" + continuum]
            c_std = file["continuum_err"]
            f_norm, f_norm_std = myf.flux_norm_std(f, f_std, c, c_std)
            flux.append(f_norm)
            flux_err.append(f_norm_std)
            conti.append(c)
            try:
                jdb.append(file["parameters"]["jdb"])
            except:
                jdb.append(i)
            if type(berv_shift) != np.ndarray:
                try:
                    berv.append(file["parameters"][berv_shift])
                except:
                    berv.append(0)
            else:
                berv = berv_shift
            try:
                rv_shift.append(file["parameters"]["RV_shift"])
            except:
                rv_shift.append(0)

        step = file[sub_dico]["parameters"]["step"]

        wave = np.array(wave)
        flux = np.array(flux)
        flux_err = np.array(flux_err)
        conti = np.array(conti)
        snr = np.array(snr)
        jdb = np.array(jdb)
        rv_shift = np.array(rv_shift)
        berv = np.array(berv)
        mean_berv = np.mean(berv)
        berv = berv - mean_berv - rv_shift

        def idx_wave(wavelength):
            return int(myf.find_nearest(wave, wavelength)[0])

        if reference == "snr":
            ref = flux[snr.argmax()]
        elif reference == "median":
            ref = np.median(flux, axis=0)
        elif reference == "master":
            ref = np.array(load["reference_spectrum"])
        elif type(reference) == int:
            ref = flux[reference]
        else:
            ref = 0 * np.median(flux, axis=0)

        diff_ref = myf.smooth2d(flux - ref, smooth_map)
        ratio_ref = myf.smooth2d(flux / (ref + epsilon), smooth_map)

        diff_backup = diff_ref.copy()
        ratio_backup = ratio_ref.copy()

        if np.sum(abs(berv)) != 0:
            for j in tqdm(np.arange(len(flux))):
                test = myc.tableXY(wave, ratio_ref[j], flux_err[j] / (ref + epsilon))
                test.x = myf.doppler_r(test.x, berv[j] * 1000)[1]
                test.interpolate(new_grid=wave, method="cubic", replace=True, interpolate_x=False)
                ratio_ref[j] = test.y
                flux_err[j] = test.yerr

        inside_oxygene_mask = np.zeros(len(ratio_ref.T))
        for k in range(len(oxygene_bands)):
            first = myf.find_nearest(wave, oxygene_bands[k][0])[0]
            last = myf.find_nearest(wave, oxygene_bands[k][1])[0]
            inside_oxygene_mask[int(first) : int(last)] = 1
        # inside_oxygene[wave>6600] = 0  #reject band [HYPERPARAMETER HARDCODED]
        inside_oxygene = inside_oxygene_mask.astype("bool")

        vec = ratio_ref.T[inside_oxygene]
        collection = myc.table(vec)

        print(np.shape(flux_err))

        weights = 1 / (flux_err) ** 2
        weights = weights.T[inside_oxygene]
        IQ = myf.IQ(collection.table, axis=1)
        Q1 = np.nanpercentile(collection.table, 25, axis=1)
        Q3 = np.nanpercentile(collection.table, 75, axis=1)
        sup = Q3 + 1.5 * IQ
        inf = Q1 - 1.5 * IQ
        out = (collection.table > sup[:, np.newaxis]) | (collection.table < inf[:, np.newaxis])
        weights[out] = np.min(weights) / 100

        base_vec = np.vstack(
            [np.ones(len(flux)), jdb - np.median(jdb), jdb**2 - np.median(jdb**2)]
        )  # fit offset + para trend par oxygene line (if binary drift substract)
        collection.fit_base(base_vec, weight=weights, num_sim=1)
        correction = np.zeros((len(wave), len(jdb)))
        correction[inside_oxygene] = collection.coeff_fitted.dot(base_vec)
        correction = np.transpose(correction)
        correction[correction == 0] = 1
        correction_backup = correction.copy()

        if np.sum(abs(berv)) != 0:
            for j in tqdm(np.arange(len(flux))):
                test = myc.tableXY(wave, correction[j], 0 * wave)
                test.x = myf.doppler_r(test.x, berv[j] * 1000)[0]
                test.interpolate(new_grid=wave, method="cubic", replace=True, interpolate_x=False)
                correction_backup[j] = test.y

        index_min_backup = int(
            myf.find_nearest(wave, myf.doppler_r(wave[0], berv.max() * 1000)[0])[0]
        )
        index_max_backup = int(
            myf.find_nearest(wave, myf.doppler_r(wave[-1], berv.min() * 1000)[0])[0]
        )
        correction_backup[:, 0:index_min_backup] = 1
        correction_backup[:, index_max_backup:] = 1
        index_hole_right = int(
            myf.find_nearest(wave, hole_right + 1)[0]
        )  # correct 1 angstrom band due to stange artefact at the border of the gap
        index_hole_left = int(
            myf.find_nearest(wave, hole_left - 1)[0]
        )  # correct 1 angstrom band due to stange artefact at the border of the gap
        correction_backup[:, index_hole_left : index_hole_right + 1] = 1

        del flux_err
        del weights
        del correction

        ratio2_backup = ratio_backup - correction_backup + 1

        del correction_backup
        del ratio_backup

        new_conti = conti * flux / (ref * ratio2_backup + epsilon)
        new_continuum = new_conti.copy()
        new_continuum[flux == 0] = conti[flux == 0]

        del ratio2_backup

        diff2_backup = flux * conti / new_continuum - ref

        # plot end

        idx_min = 0
        idx_max = len(wave)

        if wave_min is not None:
            idx_min = myf.find_nearest(wave, wave_min)[0]
        if wave_max is not None:
            idx_max = myf.find_nearest(wave, wave_max)[0] + 1

        new_wave = wave[int(idx_min) : int(idx_max)]

        fig = plt.figure(figsize=(21, 9))

        plt.axes([0.05, 0.66, 0.90, 0.25])
        myf.my_colormesh(
            new_wave,
            np.arange(len(diff_backup)),
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
        myf.my_colormesh(
            new_wave,
            np.arange(len(diff_backup)),
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
        myf.my_colormesh(
            new_wave,
            np.arange(len(diff_backup)),
            100 * (diff_backup - diff2_backup)[:, int(idx_min) : int(idx_max)],
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

        plt.savefig(self.dir_root + "IMAGES/Correction_oxygen.png")

        pre_map = np.zeros(np.shape(diff_backup))
        if sub_dico == "matching_oxygen":
            spec = self.import_spectrum()
            sub_dico = spec[sub_dico]["parameters"]["sub_dico_used"]
            step -= 1
            pre_map = pd.read_pickle(self.dir_root + "CORRECTION_MAP/map_matching_oxygen.p")[
                "correction_map"
            ]

        correction_oxygen = diff_backup - diff2_backup
        to_be_saved = {"wave": wave, "correction_map": correction_oxygen + pre_map}
        io.pickle_dump(
            to_be_saved,
            open(self.dir_root + "CORRECTION_MAP/map_matching_oxygen.p", "wb"),
        )

        print("\nComputation of the new continua, wait ... \n")
        time.sleep(0.5)

        i = -1
        for j in tqdm(files):
            i += 1
            file = pd.read_pickle(j)
            output = {"continuum_" + continuum: new_continuum[i]}
            file["matching_oxygen"] = output
            file["matching_oxygen"]["parameters"] = {
                "reference_spectrum": reference,
                "sub_dico_used": sub_dico,
                "step": step + 1,
            }
            io.save_pickle(j, file)

        self.dico_actif = "matching_oxygen"

    # =============================================================================
    # TELLRUCI CORRECTION V2
    # =============================================================================

    # telluric
    def yarara_correct_telluric_gradient(
        self,
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

        """
        Display the time-series spectra with proxies and its correlation

        Parameters
        ----------
        sub_dico : The sub_dictionnary used to  select the continuum
        continuum : The continuum to select (either linear or cubic)
        wave_min : Minimum x axis limit
        wave_max : Maximum x axis limit
        smooth_map = int-type, smooth the 2D plot by gaussian 2D convolution
        berv_shift : True/False to move in terrestrial rest-frame
        cmap : cmap of the 2D plot
        low_cmap : vmin cmap colorbar
        high_cmap : vmax cmap colorbar

        """

        myf.print_box("\n---- RECIPE : CORRECTION TELLURIC PCA ----\n")

        directory = self.directory

        zoom = self.zoom
        smooth_map = self.smooth_map
        cmap = self.cmap
        planet = self.planet
        low_cmap = self.low_cmap * 100
        high_cmap = self.high_cmap * 100
        self.import_material()
        load = self.material

        epsilon = 1e-12

        kw = "_planet" * planet
        if kw != "":
            print("\n---- PLANET ACTIVATED ----")

        if sub_dico_correction is None:
            sub_dico_correction = self.dico_actif
        print("\n---- DICO %s used ----\n" % (sub_dico_correction))

        files = glob.glob(directory + "RASSI*.p")
        files = np.sort(files)

        snr = []
        jdb = []
        berv = []
        rv_shift = []
        flux_backup = []
        flux_corr = []
        flux_det = []
        conti = []
        flux_err = []

        for i, j in enumerate(files):
            file = pd.read_pickle(j)
            if not i:
                wave = file["wave"]
                hole_left = file["parameters"]["hole_left"]
                hole_right = file["parameters"]["hole_right"]
            snr.append(file["parameters"]["SNR_5500"])
            flux_backup.append(file["flux" + kw])
            flux_det.append(file["flux" + kw] / file[sub_dico_detection]["continuum_" + continuum])

            f = file["flux" + kw]
            f_std = file["flux_err"]
            c = file[sub_dico_correction]["continuum_" + continuum]
            c_std = file["continuum_err"]
            f_norm, f_norm_std = myf.flux_norm_std(f, f_std, c, c_std)
            flux_err.append(f_norm_std)
            flux_corr.append(f_norm)
            conti.append(c)
            try:
                jdb.append(file["parameters"]["jdb"])
            except:
                jdb.append(i)
            if type(berv_shift) != np.ndarray:
                try:
                    berv.append(file["parameters"][berv_shift])
                except:
                    berv.append(0)
            else:
                berv = berv_shift
            try:
                rv_shift.append(file["parameters"]["RV_shift"])
            except:
                rv_shift.append(0)

        step = file[sub_dico_correction]["parameters"]["step"]

        wave = np.array(wave)
        flux = np.array(flux_det)
        flux_backup = np.array(flux_backup)
        flux_to_correct = np.array(flux_corr)
        flux_err = np.array(flux_err) + calib_std
        conti = np.array(conti)
        snr = np.array(snr)
        jdb = np.array(jdb)
        rv_shift = np.array(rv_shift)
        berv = np.array(berv)
        mean_berv = np.mean(berv)
        berv = berv - mean_berv - rv_shift

        if len(snr) < nb_pca_comp:
            nb_pca_comp = len(snr) - 1
            print(
                "Nb component too high compared to number of observations, nc reduced to %.0f"
                % (len(snr) - 2)
            )

        def idx_wave(wavelength):
            return int(myf.find_nearest(wave, wavelength)[0])

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

        diff = myf.smooth2d(flux, smooth_map)
        diff_ref = myf.smooth2d(flux - ref, smooth_map)
        diff_ref_to_correct = myf.smooth2d(flux_to_correct - ref, smooth_map)
        ratio_ref = myf.smooth2d(flux_to_correct / (ref + epsilon), smooth_map)
        diff_backup = diff_ref.copy()
        ratio_backup = ratio_ref.copy()

        del flux_to_correct

        if np.sum(abs(berv)) != 0:
            for j in tqdm(np.arange(len(flux))):
                test = myc.tableXY(wave, diff[j], 0 * wave)
                test.x = myf.doppler_r(test.x, berv[j] * 1000)[1]
                test.interpolate(new_grid=wave, method="cubic", replace=True, interpolate_x=False)
                diff[j] = test.y
            for j in tqdm(np.arange(len(flux))):
                test = myc.tableXY(wave, diff_ref[j], 0 * wave)
                test.x = myf.doppler_r(test.x, berv[j] * 1000)[1]
                test.interpolate(new_grid=wave, method="cubic", replace=True, interpolate_x=False)
                diff_ref[j] = test.y
            for j in tqdm(np.arange(len(flux))):
                test = myc.tableXY(wave, ratio_ref[j], flux_err[j] / (ref + epsilon))
                test.x = myf.doppler_r(test.x, berv[j] * 1000)[1]
                test.interpolate(new_grid=wave, method="cubic", replace=True, interpolate_x=False)
                ratio_ref[j] = test.y
                flux_err[j] = test.yerr

        med_wave_gradient_f = np.median(np.gradient(diff)[1], axis=0)
        med_time_gradient_f = np.median(np.gradient(diff)[0], axis=0)
        mad_time_gradient_f = np.median(np.gradient(diff)[0] - med_time_gradient_f, axis=0)

        med_wave_gradient_df = np.median(np.gradient(diff_ref)[1], axis=0)
        med_time_gradient_df = np.median(np.gradient(diff_ref)[0], axis=0)
        mad_time_gradient_df = np.median(np.gradient(diff_ref)[0] - med_time_gradient_df, axis=0)

        med_df = np.median(diff_ref, axis=0)
        med_f = np.median(diff, axis=0)

        par1 = np.log10(abs(med_time_gradient_f + med_wave_gradient_f) + 1e-6)
        par2 = np.log10(abs(med_time_gradient_df + med_wave_gradient_df) + 1e-6)

        par3 = np.log10(abs(med_df) + 1e-6)
        par4 = np.log10(1 - abs(med_f) + 1e-6)

        par5 = np.log10(abs(med_wave_gradient_f / (med_time_gradient_f + 1e-4) + 1e-2))
        par6 = np.log10(abs(med_wave_gradient_f / (mad_time_gradient_f + 1e-4) + 1e-2))

        par7 = np.log10(abs(med_wave_gradient_df / (med_time_gradient_df + 1e-4) + 1e-6))
        par8 = np.log10(abs(med_wave_gradient_df / (mad_time_gradient_df + 1e-4) + 1e-6))

        par9 = np.log10(abs(med_time_gradient_f / (mad_time_gradient_f + 1e-4) + 1e-2))
        par10 = np.log10(abs(med_time_gradient_df / (mad_time_gradient_df + 1e-4) + 1e-2))

        table = pd.DataFrame(
            {
                "wave": wave,
                "med_df": med_df,
                "med_f": med_f,
                "gt_f": med_time_gradient_f,
                "gl_f": med_wave_gradient_f,
                "gt_df": med_time_gradient_df,
                "gl_df": med_wave_gradient_df,
                "par1": par1,
                "par2": par2,
                "par3": par3,
                "par4": par4,
                "par5": par5,
                "par6": par6,
                "par7": par7,
                "par8": par8,
                "par9": par9,
                "par10": par10,
            }
        )

        table["classes"] = "full_spectrum"
        table.loc[
            (table["wave"] > wave_min_train) & (table["wave"] < wave_max_train),
            "classes",
        ] = "telluric_free"

        med1 = np.median(10 ** np.array(table.loc[table["classes"] == "telluric_free", "par2"]))
        mad1 = 1.48 * np.median(
            abs(med1 - 10 ** np.array(table.loc[table["classes"] == "telluric_free", "par2"]))
        )

        med2 = np.median(10 ** np.array(table.loc[table["classes"] == "telluric_free", "par3"]))
        mad2 = 1.48 * np.median(
            abs(med1 - 10 ** np.array(table.loc[table["classes"] == "telluric_free", "par3"]))
        )

        med3 = np.median(10 ** np.array(table.loc[table["classes"] == "telluric_free", "par8"]))
        mad3 = 1.48 * np.median(
            abs(med1 - 10 ** np.array(table.loc[table["classes"] == "telluric_free", "par8"]))
        )

        crit1 = 10**par2 - med1
        crit2 = 10**par3 - med2
        crit3 = 10**par8 - med3

        z1 = crit1 / mad1
        z2 = crit2 / mad2
        z3 = crit3 / mad3

        ztot = z1 + z2 + z3
        ztot -= np.median(ztot)

        dint = int(3 / np.mean(np.diff(wave)))
        criterion = myc.tableXY(wave, ztot)
        criterion.rolling(window=dint)
        iq = np.percentile(ztot, 75) - np.percentile(ztot, 25)

        # telluric detection

        inside = (
            myf.smooth(ztot, 5, shape="savgol") > 1.5 * iq
        )  # &(criterion.y>criterion.roll_median) # hyperparameter
        pos_peaks = (inside > np.roll(inside, 1)) & (inside > np.roll(inside, -1))
        inside = inside * (1 - pos_peaks)
        neg_peaks = (inside < np.roll(inside, 1)) & (inside < np.roll(inside, -1))
        inside = inside + neg_peaks

        # comparison with Molecfit

        model = pd.read_pickle(root + "/Python/Material/model_telluric.p")
        wave_model = myf.doppler_r(model["wave"], mean_berv * 1000)[0]
        telluric_model = model["flux_norm"]
        model = myc.tableXY(wave_model, telluric_model, 0 * wave_model)
        model.interpolate(new_grid=wave, interpolate_x=False)
        model.find_min(vicinity=5)
        mask_model = np.zeros(len(wave))
        mask_model[model.index_min.astype("int")] = 1 - model.y_min
        mask_model[mask_model < 1e-5] = 0  # hyperparameter

        inside_model = (1 - model.y) > 3e-4
        pos_peaks = (inside_model > np.roll(inside_model, 1)) & (
            inside_model > np.roll(inside_model, -1)
        )
        inside_model = inside_model * (1 - pos_peaks)

        completness = mask_model * inside
        completness = completness[completness != 0]
        completness_max = mask_model[mask_model != 0]
        completness2 = mask_model * (1 - inside)
        completness2 = completness2[completness2 != 0]

        # self.debug1 = completness, completness_max, completness2, inside

        plt.figure()
        val = plt.hist(
            np.log10(completness_max),
            bins=np.linspace(-5, 0, 50),
            cumulative=-1,
            histtype="step",
            lw=3,
            color="k",
            label="telluric model",
        )
        val2 = plt.hist(
            np.log10(completness),
            bins=np.linspace(-5, 0, 50),
            cumulative=-1,
            histtype="step",
            lw=3,
            color="r",
            label="telluric detected",
        )
        val3 = plt.hist(
            np.log10(completness2),
            bins=np.linspace(-5, 0, 50),
            cumulative=1,
            histtype="step",
            lw=3,
            color="g",
            label="telluric undetected",
        )
        plt.close()

        # comp_percent = 100*(1 - (val[0]-val2[0])/(val[0]+1e-12)) #update 10.06.21 to complicated metric
        comp_percent = val3[0] * 100 / np.max(val3[0])
        tel_depth_grid = val[1][0:-1] + 0.5 * (val[1][1] - val[1][0])

        plt.figure(12, figsize=(8.5, 7))
        plt.plot(tel_depth_grid, comp_percent, color="k")
        plt.axhline(y=100, color="b", ls="-.")
        plt.grid()
        if len(np.where(comp_percent == 100)[0]) > 0:
            plt.axvline(
                x=tel_depth_grid[np.where(comp_percent == 100)[0][0]],
                color="b",
                label="100%% Completeness : %.2f [%%]"
                % (100 * 10 ** (tel_depth_grid[np.where(comp_percent == 100)[0][0]])),
            )
            plt.axvline(
                x=tel_depth_grid[myf.find_nearest(comp_percent, 90)[0]],
                color="b",
                ls=":",
                label="90%% Completeness : %.2f [%%]"
                % (100 * 10 ** (tel_depth_grid[myf.find_nearest(comp_percent, 90)[0]])),
            )
        plt.ylabel("Completness [%]", fontsize=16)
        plt.xlabel(r"$\log_{10}$(Telluric depth)", fontsize=16)
        plt.title("Telluric detection completeness versus MolecFit model", fontsize=16)
        plt.ylim(-5, 105)
        plt.legend(prop={"size": 14})
        plt.savefig(self.dir_root + "IMAGES/telluric_detection.pdf")

        # extraction telluric

        telluric_location = inside.copy()
        telluric_location[wave < wave_min_correction] = 0  # reject shorter wavelength
        telluric_location[wave > wave_max_correction] = 0  # reject band

        # self.debug = telluric_location

        # extraction of uncontaminated telluric

        plateau, cluster = myf.clustering(telluric_location, 0.5, 1)
        plateau = np.array([np.product(j) for j in plateau]).astype("bool")
        cluster = cluster[plateau]
        # med_width = np.median(cluster[:,-1])
        # mad_width = np.median(abs(cluster[:,-1] - med_width))*1.48
        telluric_kept = cluster  # cluster[(cluster[:,-1]>med_width-mad_width)&(cluster[:,-1]<med_width+mad_width),:]
        telluric_kept[:, 1] += 1
        # telluric_kept = np.hstack([telluric_kept,wave[telluric_kept[:,0],np.newaxis]])
        # plt.figure();plt.hist(telluric_kept[:,-1],bins=100)
        telluric_kept = telluric_kept[
            telluric_kept[:, -1]
            > np.nanmedian(telluric_kept[:, -1]) - myf.mad(telluric_kept[:, -1])
        ]
        min_telluric_size = wave / inst_resolution / np.gradient(wave)
        telluric_kept = telluric_kept[
            min_telluric_size[telluric_kept[:, 0]] < telluric_kept[:, -1]
        ]

        if debug:
            plt.figure(1)
            plt.subplot(3, 2, 1)
            plt.plot(wave, ref, color="k")
            (l4,) = plt.plot(5500 * np.ones(2), [0, 1], color="r")
            ax = plt.gca()
            plt.subplot(2, 2, 2, sharex=ax)
            plt.plot(wave, ztot, color="k")
            ax = plt.gca()
            border_y = ax.get_ylim()
            (l,) = plt.plot(5500 * np.ones(2), border_y, color="r")
            idx = myf.find_nearest(wave, 5500)[0].astype("int")
            plt.ylim(border_y)
            for j in range(len(telluric_kept)):
                plt.axvspan(
                    xmin=wave[telluric_kept[j, 0].astype("int")],
                    xmax=wave[telluric_kept[j, 1].astype("int")],
                    alpha=0.3,
                    color="r",
                )
            plt.subplot(2, 2, 4, sharex=ax)
            plt.imshow(
                ratio_ref,
                aspect="auto",
                cmap="plasma",
                vmin=0.99,
                vmax=1.01,
                extent=[wave[0], wave[-1], 0, len(jdb)],
            )

            (l2,) = plt.plot(5500 * np.ones(2), [0, len(jdb)], color="k")

            plt.subplot(3, 2, 5)
            l5, (), (bars5,) = plt.errorbar(
                jdb % 365.25, ratio_ref[:, idx], 0.001 * np.ones(len(jdb)), fmt="ko"
            )
            plt.ylim(0.99, 1.01)
            ax3 = plt.gca()

            plt.subplot(3, 2, 3)
            l3, (), (bars3,) = plt.errorbar(
                jdb, ratio_ref[:, idx], 0.001 * np.ones(len(jdb)), fmt="ko"
            )
            plt.ylim(0.99, 1.01)
            ax4 = plt.gca()

            class Index:
                def update_data(self, newx, newy):
                    idx = myf.find_nearest(wave, newx)[0].astype("int")
                    l.set_xdata(newx * np.ones(len(l.get_xdata())))
                    l2.set_xdata(newx * np.ones(len(l.get_xdata())))
                    l4.set_xdata(newx * np.ones(len(l.get_xdata())))
                    l3.set_ydata(ratio_ref[:, idx])
                    l5.set_ydata(ratio_ref[:, idx])
                    new_segments = [
                        np.array([[x, yt], [x, yb]])
                        for x, yt, yb in zip(
                            jdb, ratio_ref[:, idx] + 0.001, ratio_ref[:, idx] - 0.001
                        )
                    ]
                    bars3.set_segments(new_segments)
                    bars5.set_segments(new_segments)
                    ax3.set_ylim(
                        np.min(ratio_ref[:, idx]) - 0.002,
                        np.max(ratio_ref[:, idx]) + 0.002,
                    )
                    ax4.set_ylim(
                        np.min(ratio_ref[:, idx]) - 0.002,
                        np.max(ratio_ref[:, idx]) + 0.002,
                    )
                    plt.gcf().canvas.draw_idle()

            t = Index()

            def onclick(event):
                newx = event.xdata
                newy = event.ydata
                if event.dblclick:
                    print(newx)
                    t.update_data(newx, newy)

            plt.gcf().canvas.mpl_connect("button_press_event", onclick)
        else:
            plt.figure(1)
            plt.subplot(3, 1, 1)
            plt.plot(ref, color="k")
            ax = plt.gca()
            plt.subplot(3, 1, 2, sharex=ax)
            plt.plot(ztot, color="k")
            for j in range(len(telluric_kept)):
                plt.axvspan(
                    xmin=telluric_kept[j, 0].astype("int"),
                    xmax=telluric_kept[j, 1].astype("int"),
                    alpha=0.3,
                    color="r",
                )
            plt.subplot(3, 1, 3, sharex=ax)
            plt.imshow(diff_ref, aspect="auto", vmin=-0.005, vmax=0.005)

        telluric_extracted_ratio_ref = []
        telluric_extracted_ratio_ref_std = []

        ratio_ref_std2 = flux_err**2
        for j in range(len(telluric_kept)):

            norm = telluric_kept[j, 1] + 1 - telluric_kept[j, 0]
            val = np.nanmean(ratio_ref[:, telluric_kept[j, 0] : telluric_kept[j, 1] + 1], axis=1)
            val_std = (
                np.sqrt(
                    np.nansum(
                        ratio_ref_std2[:, telluric_kept[j, 0] : telluric_kept[j, 1] + 1],
                        axis=1,
                    )
                )
                / norm
            )

            telluric_extracted_ratio_ref.append(val)
            telluric_extracted_ratio_ref_std.append(val_std)

        telluric_extracted_ratio_ref = np.array(telluric_extracted_ratio_ref).T
        telluric_extracted_ratio_ref_std = np.array(telluric_extracted_ratio_ref_std).T
        telluric_extracted_ratio_ref -= np.median(telluric_extracted_ratio_ref, axis=0)

        plt.figure(2, figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(telluric_extracted_ratio_ref, aspect="auto", vmin=-0.005, vmax=0.005)
        plt.title("Water lines")
        plt.xlabel("Pixels extracted", fontsize=14)
        plt.ylabel("Time", fontsize=14)

        plt.subplot(1, 2, 2)
        plt.imshow(
            telluric_extracted_ratio_ref / np.std(telluric_extracted_ratio_ref, axis=0),
            aspect="auto",
            vmin=-0.005,
            vmax=0.005,
        )
        plt.title("Water lines")
        plt.xlabel("Pixels extracted", fontsize=14)
        plt.ylabel("Time", fontsize=14)

        c = int(equal_weight)

        X_train = (
            telluric_extracted_ratio_ref
            / ((1 - c) + c * np.std(telluric_extracted_ratio_ref, axis=0))
        ).T
        X_train_std = (
            telluric_extracted_ratio_ref_std
            / ((1 - c) + c * np.std(telluric_extracted_ratio_ref, axis=0))
        ).T

        # self.debug = (X_train, X_train_std)
        # io.pickle_dump({'jdb':np.array(self.table.jdb),'ratio_flux':X_train,'ratio_flux_std':X_train_std},open(root+'/Python/datasets/telluri_cenB.p','wb'))

        test2 = myc.table(X_train)

        test2.WPCA("wpca", weight=1 / X_train_std**2, comp_max=nb_pca_comp)

        phase_mod = np.arange(365)[
            np.argmin(
                np.array(
                    [np.max((jdb - k) % 365.25) - np.min((jdb - k) % 365.25) for k in range(365)]
                )
            )
        ]

        plt.figure(4, figsize=(10, 14))
        plt.subplot(3, 1, 1)
        plt.xlabel("# PCA components", fontsize=13)
        plt.ylabel("Variance explained", fontsize=13)
        plt.plot(np.arange(1, len(test2.phi_components) + 1), test2.var_ratio)
        plt.scatter(np.arange(1, len(test2.phi_components) + 1), test2.var_ratio)
        plt.subplot(3, 1, 2)
        plt.xlabel("# PCA components", fontsize=13)
        plt.ylabel("Z score", fontsize=13)
        plt.plot(np.arange(1, len(test2.phi_components) + 1), test2.zscore_components)
        plt.scatter(np.arange(1, len(test2.phi_components) + 1), test2.zscore_components)
        z_max = test2.zscore_components[-10:].max()
        z_min = test2.zscore_components[-10:].min()
        vec_relevant = np.arange(len(test2.zscore_components)) * (
            (test2.zscore_components > z_max) | (test2.zscore_components < z_min)
        )
        pca_comp_kept2 = int(np.where(vec_relevant != np.arange(len(vec_relevant)))[0][0])

        plt.axhspan(ymin=z_min, ymax=z_max, alpha=0.2, color="k")
        plt.axhline(y=0, color="k")
        plt.subplot(3, 1, 3)
        plt.xlabel("# PCA components", fontsize=13)
        plt.ylabel(r"$\Phi(0)$", fontsize=13)
        plt.plot(np.arange(1, len(test2.phi_components) + 1), test2.phi_components)
        plt.scatter(np.arange(1, len(test2.phi_components) + 1), test2.phi_components)
        plt.axhline(y=0.5, color="k")
        phi_max = test2.phi_components[-10:].max()
        phi_min = test2.phi_components[-10:].min()
        plt.axhspan(ymin=phi_min, ymax=phi_max, alpha=0.2, color="k")
        vec_relevant = np.arange(len(test2.phi_components)) * (
            (test2.phi_components > phi_max) | (test2.phi_components < phi_min)
        )
        pca_comp_kept = int(np.where(vec_relevant != np.arange(len(vec_relevant)))[0][0])
        pca_comp_kept = np.max([pca_comp_kept, pca_comp_kept2])

        if nb_pca_comp_kept is not None:
            pca_comp_kept = nb_pca_comp_kept

        if pca_comp_kept > nb_pca_max_kept:
            pca_comp_kept = nb_pca_max_kept

        print(" [INFO] Nb PCA comp kept : %.0f" % (pca_comp_kept))

        plt.savefig(self.dir_root + "IMAGES/telluric_PCA_variances.pdf")

        plt.figure(figsize=(15, 10))
        for j in range(pca_comp_kept):
            plt.subplot(pca_comp_kept, 2, 2 * j + 1)
            plt.scatter(jdb, test2.vec[:, j])
            plt.subplot(pca_comp_kept, 2, 2 * j + 2)
            plt.scatter((jdb - phase_mod) % 365.25, test2.vec[:, j])
        plt.subplots_adjust(top=0.95, bottom=0.07, left=0.07, right=0.95, hspace=0)
        plt.savefig(self.dir_root + "IMAGES/telluric_PCA_vectors.pdf")

        to_be_fit = ratio_ref / (np.std(ratio_ref, axis=0) + epsilon)

        rcorr = np.zeros(len(wave))
        for j in range(pca_comp_kept):
            proxy1 = test2.vec[:, j]
            rslope1 = np.median(
                (to_be_fit - np.mean(to_be_fit, axis=0))
                / ((proxy1 - np.mean(proxy1))[:, np.newaxis]),
                axis=0,
            )

            rcorr1 = abs(rslope1 * np.std(proxy1) / (np.std(to_be_fit, axis=0) + epsilon))
            rcorr = np.nanmax([rcorr1, rcorr], axis=0)
        rcorr[np.isnan(rcorr)] = 0
        rcorr_telluric_free = rcorr[
            int(myf.find_nearest(wave, 4800)[0]) : int(myf.find_nearest(wave, 5000)[0])
        ]
        rcorr_telluric = rcorr[
            int(myf.find_nearest(wave, 5800)[0]) : int(myf.find_nearest(wave, 6000)[0])
        ]

        plt.figure(figsize=(8, 6))
        bins_contam, bins, dust = plt.hist(
            rcorr_telluric,
            label="contaminated region",
            bins=np.linspace(0, 1, 100),
            alpha=0.5,
        )
        bins_control, bins, dust = plt.hist(
            rcorr_telluric_free,
            bins=np.linspace(0, 1, 100),
            label="free region",
            alpha=0.5,
        )
        plt.legend()
        plt.yscale("log")
        bins = bins[0:-1] + np.diff(bins) * 0.5
        sum_a = np.sum(bins_contam[bins > 0.40])
        sum_b = np.sum(bins_control[bins > 0.40])
        crit = int(sum_a > (2 * sum_b))
        check = ["r", "g"][crit]  # five times more correlation than in the control group
        plt.xlabel(r"|$\mathcal{R}_{pearson}$|", fontsize=14, fontweight="bold", color=check)
        plt.title("Density", color=check)
        myf.plot_color_box(color=check)

        plt.savefig(self.dir_root + "IMAGES/telluric_control_check.pdf")
        print(" [INFO] %.0f versus %.0f" % (sum_a, sum_b))

        if crit:
            print(" [INFO] Control check sucessfully performed: telluric")
        else:
            print(
                Fore.YELLOW
                + " [WARNING] Control check failed. Correction may be poorly performed for: telluric"
                + Fore.RESET
            )

        collection = myc.table(
            ratio_ref.T[telluric_location.astype("bool")]
        )  # do fit only on flag position

        weights = 1 / (flux_err) ** 2
        weights = weights.T[telluric_location.astype("bool")]
        IQ = myf.IQ(collection.table, axis=1)
        Q1 = np.nanpercentile(collection.table, 25, axis=1)
        Q3 = np.nanpercentile(collection.table, 75, axis=1)
        sup = Q3 + 1.5 * IQ
        inf = Q1 - 1.5 * IQ
        out = (collection.table > sup[:, np.newaxis]) | (collection.table < inf[:, np.newaxis])
        weights[out] = np.min(weights) / 100

        # base_vec = np.vstack([np.ones(len(flux)), jdb-np.median(jdb), test2.vec[:,0:pca_comp_kept].T])
        base_vec = np.vstack([np.ones(len(flux)), test2.vec[:, 0:pca_comp_kept].T])
        collection.fit_base(base_vec, weight=weights, num_sim=1)
        # collection.coeff_fitted[:,3] = 0 #supress the linear trend fitted

        del weights
        del flux_err

        correction = np.zeros((len(wave), len(jdb)))
        correction[telluric_location.astype("bool")] = collection.coeff_fitted.dot(base_vec)
        correction = np.transpose(correction)

        correction[correction == 0] = 1
        correction_backup = correction.copy()

        if np.sum(abs(berv)) != 0:
            for j in tqdm(np.arange(len(flux))):
                test = myc.tableXY(wave, correction[j], 0 * wave)
                test.x = myf.doppler_r(test.x, berv[j] * 1000)[0]
                test.interpolate(new_grid=wave, method="cubic", replace=True, interpolate_x=False)
                correction_backup[j] = test.y

        index_min_backup = int(
            myf.find_nearest(wave, myf.doppler_r(wave[0], berv.max() * 1000)[0])[0]
        )
        index_max_backup = int(
            myf.find_nearest(wave, myf.doppler_r(wave[-1], berv.min() * 1000)[0])[0]
        )
        correction_backup[:, 0:index_min_backup] = 1
        correction_backup[:, index_max_backup:] = 1
        index_hole_right = int(
            myf.find_nearest(wave, hole_right + 1)[0]
        )  # correct 1 angstrom band due to stange artefact at the border of the gap
        index_hole_left = int(
            myf.find_nearest(wave, hole_left - 1)[0]
        )  # correct 1 angstrom band due to stange artefact at the border of the gap
        correction_backup[:, index_hole_left : index_hole_right + 1] = 1

        del correction

        ratio2_backup = ratio_backup - correction_backup + 1

        del correction_backup

        new_conti = flux_backup / (ref * ratio2_backup + epsilon)
        new_continuum = new_conti.copy()
        new_continuum[flux == 0] = conti[flux == 0]
        new_continuum[np.isnan(new_continuum)] = conti[np.isnan(new_continuum)]
        new_continuum[new_continuum == 0] = conti[new_continuum == 0]

        del ratio2_backup
        del ratio_backup

        diff2_backup = flux_backup / new_continuum - ref

        idx_min = myf.find_nearest(wave, 5700)[0]
        idx_max = myf.find_nearest(wave, 5900)[0] + 1

        new_wave = wave[int(idx_min) : int(idx_max)]

        fig = plt.figure(figsize=(21, 9))
        plt.axes([0.05, 0.55, 0.90, 0.40])
        ax = plt.gca()
        myf.my_colormesh(
            new_wave,
            np.arange(len(diff_ref)),
            100 * diff_backup[:, int(idx_min) : int(idx_max)],
            zoom=zoom,
            vmin=low_cmap,
            vmax=high_cmap,
            cmap=cmap,
        )
        plt.tick_params(direction="in", top=True, right=True, labelbottom=False)
        plt.ylabel("Spectra  indexes (time)", fontsize=16)
        plt.ylim(0, None)
        ax = plt.gca()
        cbaxes = fig.add_axes([0.95, 0.55, 0.01, 0.40])
        ax1 = plt.colorbar(cax=cbaxes)
        ax1.ax.set_ylabel(r"$\Delta$ flux normalised [%]", fontsize=14)

        plt.axes([0.05, 0.1, 0.90, 0.40], sharex=ax, sharey=ax)
        myf.my_colormesh(
            new_wave,
            np.arange(len(diff_ref)),
            100 * diff2_backup[:, int(idx_min) : int(idx_max)],
            zoom=zoom,
            vmin=low_cmap,
            vmax=high_cmap,
            cmap=cmap,
        )
        plt.tick_params(direction="in", top=True, right=True, labelbottom=True)
        plt.ylabel("Spectra  indexes (time)", fontsize=16)
        plt.xlabel(r"Wavelength [$\AA$]", fontsize=16)
        plt.ylim(0, None)
        ax = plt.gca()
        cbaxes2 = fig.add_axes([0.95, 0.1, 0.01, 0.40])
        ax2 = plt.colorbar(cax=cbaxes2)
        ax2.ax.set_ylabel(r"$\Delta$ flux normalised [%]", fontsize=14)

        plt.savefig(self.dir_root + "IMAGES/Correction_telluric.png")

        correction_pca = diff_ref_to_correct - diff2_backup
        to_be_saved = {"wave": wave, "correction_map": correction_pca}
        io.pickle_dump(
            to_be_saved, open(self.dir_root + "CORRECTION_MAP/map_matching_pca.p", "wb")
        )

        print("Computation of the new continua, wait ... \n")
        time.sleep(0.5)

        i = -1
        for j in tqdm(files):
            i += 1
            file = pd.read_pickle(j)
            output = {"continuum_" + continuum: new_continuum[i]}
            file["matching_pca"] = output
            file["matching_pca"]["parameters"] = {
                "reference_spectrum": reference,
                "sub_dico_used": sub_dico_correction,
                "nb_pca_component": pca_comp_kept,
                "step": step + 1,
            }
            io.save_pickle(j, file)

        self.dico_actif = "matching_pca"

        plt.show(block=False)

    # =============================================================================
    #  ACTIVITY CORRECTION
    # =============================================================================

    # activity
    def yarara_correct_activity(
        self,
        sub_dico="matching_telluric",
        continuum="linear",
        wave_min=3900,
        wave_max=4400,
        smooth_corr=5,
        reference="median",
        rv_shift="none",
        proxy_corr=["CaII"],
    ):

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

        myf.print_box("\n---- RECIPE : CORRECTION ACTIVITY (CCF MOMENTS) ----\n")

        directory = self.directory

        zoom = self.zoom
        smooth_map = self.smooth_map
        cmap = self.cmap
        planet = self.planet
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

        #        proxy = myc.tableXY(jdb,proxy)
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

        diff = myf.smooth2d(flux - ref, smooth_map)
        diff_backup = diff.copy()

        if np.sum(rv) != 0:
            for j in tqdm(np.arange(len(flux))):
                test = myc.tableXY(wave, diff[j], 0 * wave)
                test.x = myf.doppler_r(test.x, rv[j] * 1000)[1]
                test.interpolate(new_grid=wave, method="cubic", replace=True, interpolate_x=False)
                diff[j] = test.y

        collection = myc.table(diff.T)
        base_vec = np.vstack([np.ones(len(flux)), proxy])
        collection.fit_base(base_vec, num_sim=1)

        collection.coeff_fitted[:, 1] = myf.smooth(
            collection.coeff_fitted[:, 1], smooth_corr, shape="savgol"
        )

        correction = collection.coeff_fitted.dot(base_vec)
        correction = np.transpose(correction)

        correction_backup = correction.copy()
        if np.sum(rv) != 0:
            for j in tqdm(np.arange(len(flux))):
                test = myc.tableXY(wave, correction[j], 0 * wave)
                test.x = myf.doppler_r(test.x, rv[j] * 1000)[0]
                test.interpolate(new_grid=wave, method="cubic", replace=True, interpolate_x=False)
                correction_backup[j] = test.y

        index_min_backup = int(
            myf.find_nearest(wave, myf.doppler_r(wave[0], rv.max() * 1000)[0])[0]
        )
        index_max_backup = int(
            myf.find_nearest(wave, myf.doppler_r(wave[-1], rv.min() * 1000)[0])[0]
        )
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
            idx_min = myf.find_nearest(wave, wave_min)[0]
        if wave_max is not None:
            idx_max = myf.find_nearest(wave, wave_max)[0] + 1

        if (idx_min == 0) & (idx_max == 1):
            idx_max = myf.find_nearest(wave, np.min(wave) + 500)[0] + 1

        new_wave = wave[int(idx_min) : int(idx_max)]

        fig = plt.figure(figsize=(21, 9))

        plt.axes([0.05, 0.66, 0.90, 0.25])
        myf.my_colormesh(
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
        myf.my_colormesh(
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
        myf.my_colormesh(
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

    # =============================================================================
    # SUPRESS VARIATION RELATIF TO MEDIAN-MAD SPECTRUM (COSMIC PEAK WITH VALUE > 1)
    # =============================================================================

    # outliers
    def yarara_correct_cosmics(
        self,
        sub_dico="matching_diff",
        continuum="linear",
        k_sigma=3,
        bypass_warning=True,
    ):

        """
        Supress flux value outside k-sigma mad clipping

        Parameters
        ----------
        sub_dico : The sub_dictionnary used to  select the continuum
        continuum : The continuum to select (either linear or cubic)

        """

        myf.print_box("\n---- RECIPE : CORRECTION COSMICS ----\n")

        directory = self.directory
        planet = self.planet
        self.import_dico_tree()
        epsilon = 1e-12

        reduction_accepted = True

        if not bypass_warning:
            if "matching_smooth" in list(self.dico_tree["dico"]):
                logging.warn(
                    "Launch that recipes will remove the smooth correction of the previous loop iteration."
                )
                answer = myf.sphinx("Do you want to purchase (y/n) ?", rep=["y", "n"])
                if answer == "n":
                    reduction_accepted = False

        if reduction_accepted:
            kw = "_planet" * planet
            if kw != "":
                print("\n---- PLANET ACTIVATED ----")

            if sub_dico is None:
                sub_dico = self.dico_actif
            print("---- DICO %s used ----" % (sub_dico))

            files = glob.glob(directory + "RASSI*.p")
            files = np.sort(files)

            all_flux = []
            conti = []
            all_flux_norm = []
            all_snr = []
            jdb = []

            for i, j in enumerate(files):
                file = pd.read_pickle(j)
                if not i:
                    grid = file["wave"]
                all_flux.append(file["flux" + kw])
                conti.append(file[sub_dico]["continuum_" + continuum])
                all_flux_norm.append(file["flux" + kw] / file[sub_dico]["continuum_" + continuum])
                all_snr.append(file["parameters"]["SNR_5500"])
                jdb.append(file["parameters"]["jdb"])

            step = file[sub_dico]["parameters"]["step"]
            all_flux = np.array(all_flux)
            conti = np.array(conti)
            all_flux_norm = np.array(all_flux_norm)
            all_snr = np.array(all_snr)
            jdb = np.array(jdb)

            med = np.median(all_flux_norm, axis=0)
            mad = 1.48 * np.median(abs(all_flux_norm - med), axis=0)
            all_flux_corrected = all_flux_norm.copy()
            level = (med + k_sigma * mad) * np.ones(len(jdb))[:, np.newaxis]
            mask = (all_flux_norm > 1) & (all_flux_norm > level)

            print(
                "\n [INFO] Percentage of cosmics detected with k-sigma %.0f : %.2f%% \n"
                % (k_sigma, 100 * np.sum(mask) / len(mask.T) / len(mask))
            )

            med_map = med * np.ones(len(jdb))[:, np.newaxis]

            plt.figure(figsize=(10, 10))
            plt.scatter(
                all_snr,
                np.sum(mask, axis=1) * 100 / len(mask.T),
                edgecolor="k",
                c=jdb,
                cmap="brg",
            )
            ax = plt.colorbar()
            plt.yscale("log")
            plt.ylim(0.001, 100)
            plt.xlabel("SNR", fontsize=13)
            plt.ylabel("Percent of the spectrum flagged as cosmics [%]", fontsize=13)
            plt.grid()
            ax.ax.set_ylabel("Jdb", fontsize=13)

            plt.figure(figsize=(20, 5))
            all_flux_corrected[mask] = med_map[mask]
            for j in range(len(jdb)):
                plt.plot(grid, all_flux_corrected[j] - 1.5, color="b", alpha=0.3)
                plt.plot(grid, all_flux_norm[j], color="k", alpha=0.3)
            plt.ylim(-2, 2)
            plt.xlabel(r"Wavelength [$\AA$]", fontsize=14)
            plt.ylabel(r"Flux normalised", fontsize=14)
            plt.savefig(self.dir_root + "IMAGES/Correction_cosmics.png")

            correction_cosmics = all_flux_norm - all_flux_corrected
            to_be_saved = {"wave": grid, "correction_map": correction_cosmics}
            io.pickle_dump(
                to_be_saved,
                open(self.dir_root + "CORRECTION_MAP/map_matching_cosmics.p", "wb"),
            )

            new_continuum = all_flux / (all_flux_corrected + epsilon)
            new_continuum[all_flux == 0] = conti[all_flux == 0]
            new_continuum[new_continuum != new_continuum] = conti[
                new_continuum != new_continuum
            ]  # to supress mystic nan appearing

            print("\nComputation of the new continua, wait ... \n")
            time.sleep(0.5)
            count_file = -1
            for j in tqdm(files):
                count_file += 1
                file = pd.read_pickle(j)
                output = {"continuum_" + continuum: new_continuum[count_file]}
                file["matching_cosmics"] = output
                file["matching_cosmics"]["parameters"] = {
                    "sub_dico_used": sub_dico,
                    "k_sigma": k_sigma,
                    "step": step + 1,
                }
                io.save_pickle(j, file)

            self.dico_actif = "matching_cosmics"

    # =============================================================================
    # SUPRESS VARIATION RELATIF TO MEDIAN-MAD SPECTRUM (OUTLIERS CORRECTION + ECCENTRIC PLANET)
    # =============================================================================

    # outliers
    def yarara_correct_mad(
        self,
        sub_dico="matching_diff",
        continuum="linear",
        k_sigma=2,
        k_mad=2,
        n_iter=1,
        ext="0",
    ):

        """
        Supress flux value outside k-sigma mad clipping

        Parameters
        ----------
        sub_dico : The sub_dictionnary used to  select the continuum
        continuum : The continuum to select (either linear or cubic)

        """

        myf.print_box("\n---- RECIPE : CORRECTION MAD ----\n")

        directory = self.directory
        planet = self.planet

        kw = "_planet" * planet
        if kw != "":
            print("\n---- PLANET ACTIVATED ----")

        if sub_dico is None:
            sub_dico = self.dico_actif
        print("\n---- DICO %s used ----\n" % (sub_dico))

        files = glob.glob(directory + "RASSI*.p")
        files = np.sort(files)

        all_flux = []
        all_flux_std = []
        all_snr = []
        jdb = []

        for i, j in enumerate(files):
            file = pd.read_pickle(j)
            if not i:
                grid = file["wave"]

            f = file["flux" + kw]
            f_std = file["flux_err"]
            c = file[sub_dico]["continuum_" + continuum]
            c_std = file["continuum_err"]
            f_norm, f_norm_std = myf.flux_norm_std(f, f_std, c, c_std)
            all_flux.append(f_norm)
            all_flux_std.append(f_norm_std)
            all_snr.append(file["parameters"]["SNR_5500"])
            jdb.append(file["parameters"]["jdb"])

        step = file[sub_dico]["parameters"]["step"]

        all_flux = np.array(all_flux)
        all_flux_std = np.array(all_flux_std)
        all_snr = np.array(all_snr)
        snr_max = all_snr.argmax()
        jdb = np.array(jdb)

        # plt.subplot(3,1,1)
        # for j in range(len(all_flux)):
        #    plt.plot(grid,all_flux[j])
        # ax = plt.gca()
        # plt.plot(grid,np.median(all_flux,axis=0),color='k',zorder=1000)

        all_flux[np.isnan(all_flux)] = 0

        all_flux2 = all_flux.copy()
        # all_flux2[np.isnan(all_flux2)] = 0

        med = np.median(all_flux2.copy(), axis=0).copy()
        mean = np.mean(all_flux2.copy(), axis=0).copy()
        sup = np.percentile(all_flux2.copy(), 84, axis=0).copy()
        inf = np.percentile(all_flux2.copy(), 16, axis=0).copy()
        ref = all_flux2[snr_max].copy()

        ok = "y"

        save = []
        count = 0
        while ok == "y":

            mad = 1.48 * np.median(
                abs(all_flux2 - np.median(all_flux2, axis=0)), axis=0
            )  # mad transformed in sigma
            mad[mad == 0] = 100
            counter_removed = []
            cum_curve = []
            for j in tqdm(range(len(all_flux2))):
                sigma = myc.tableXY(
                    grid, (abs(all_flux2[j] - med) - all_flux_std[j] * k_sigma) / mad
                )
                sigma.smooth(box_pts=6, shape="rectangular")
                # sigma.rolling(window=100,quantile=0.50)
                # sig = (sigma.y>(sigma.roll_Q1+3*sigma.roll_IQ))
                # sig = sigma.roll_Q1+3*sigma.roll_IQ

                # sigma.y *= -1
                # sigma.find_max(vicinity=5)
                # loc_min = sigma.index_max.copy()
                # sigma.y *= -1

                mask = sigma.y > k_mad

                # sigma.find_max(vicinity=5)
                # loc_max = np.array([sigma.y_max, sigma.index_max]).T
                # loc_max = loc_max[loc_max[:,0]>k_mad] # only keep sigma higher than k_sigma
                # loc_max = loc_max[:,-1]

                # diff = loc_max - loc_min[:,np.newaxis]
                # diff1 = diff.copy()
                # #diff2 = diff.copy()
                # diff1[diff1<0] = 1000 #arbitrary large value
                # #diff2[diff2>0] = -1000 #arbitrary small value
                # left = np.argmin(diff1,axis=1)
                # left = np.unique(left)
                # mask = np.zeros(len(grid)).astype('bool')
                # for k in range(len(left)-1):
                #     mask[int(sigma.index_max[left[k]]):int(sigma.index_max[left[k]+1])+1] = True

                # all_flux2[j][sigma.y>3] = med[sigma.y>3]

                all_flux2[j][mask] = med[mask]
                counter_removed.append(100 * np.sum(mask * (ref < 0.9)) / np.sum(ref < 0.9))
                cum_curve.append(100 * np.cumsum(mask * (ref < 0.9)) / np.sum(ref < 0.9))

            self.counter_mad_removed = np.array(counter_removed)
            self.cum_curves = np.array(cum_curve)
            self.cum_curves[self.cum_curves[:, -1] == 0, -1] = 1

            med2 = np.median(all_flux2, axis=0)
            mean2 = np.mean(all_flux2, axis=0)
            sup2 = np.percentile(all_flux2, 84, axis=0)
            inf2 = np.percentile(all_flux2, 16, axis=0)
            ref2 = all_flux2[snr_max].copy()

            save.append((mean - mean2).copy())

            if n_iter is None:
                plt.subplot(3, 1, 1)
                plt.plot(grid, med, color="k")
                plt.plot(grid, ref, color="k", alpha=0.4)
                plt.fill_between(grid, sup, y2=inf, alpha=0.5, color="b")
                ax = plt.gca()
                plt.subplot(3, 1, 2, sharex=ax, sharey=ax)
                plt.plot(grid, med2, color="k")
                plt.plot(grid, ref2, color="k", alpha=0.4)
                plt.fill_between(grid, sup2, y2=inf2, alpha=0.5, color="g")
                plt.subplot(3, 1, 3, sharex=ax)
                plt.plot(grid, ref - ref2, color="k", alpha=0.4)
                for k in range(len(save)):
                    plt.plot(grid, save[k])
                plt.axhline(y=0, color="r")

                plt.show(block=False)
                ok = myf.sphinx(
                    " Do you want to iterate one more time (y), quit (n) or save (s) ? (y/n/s)",
                    rep=["y", "n", "s"],
                )
                plt.close()
            else:
                if n_iter == 1:
                    ok = "s"
                else:
                    n_iter -= 1
                    ok = "y"

            if ok != "y":
                break
            else:
                count += 1
        if ok == "s":
            plt.figure(figsize=(23, 16))
            plt.subplot(2, 3, 1)
            plt.axhline(
                y=0.15,
                color="k",
                ls=":",
                label="rejection criterion  (%.0f)" % (sum(self.counter_mad_removed > 0.15)),
            )
            plt.legend()
            plt.scatter(jdb, self.counter_mad_removed, c=jdb, cmap="jet")
            plt.xlabel("Time", fontsize=13)
            plt.ylabel("Percent of the spectrum removed [%]", fontsize=13)
            ax = plt.colorbar()
            ax.ax.set_ylabel("Time")
            plt.subplot(2, 3, 4)
            plt.axhline(
                y=0.15,
                color="k",
                ls=":",
                label="rejection criterion (%.0f)" % (sum(self.counter_mad_removed > 0.15)),
            )
            plt.scatter(all_snr, self.counter_mad_removed, c=jdb, cmap="jet")
            plt.xlabel("SNR", fontsize=13)
            plt.ylabel("Percent of the spectrum removed [%]", fontsize=13)
            ax = plt.colorbar()
            ax.ax.set_ylabel("Time")

            jet = plt.get_cmap("jet")
            vmin = np.min(jdb)
            vmax = np.max(jdb)

            cNorm = mplcolors.Normalize(vmin=vmin, vmax=vmax)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

            plt.subplot(2, 3, 2)
            for j in range(len(jdb)):
                colorVal = scalarMap.to_rgba(jdb[j])
                plt.plot(grid[::500], self.cum_curves[j][::500], color=colorVal, alpha=0.5)
            plt.xlabel("Wavelength", fontsize=13)
            plt.ylabel("Cumulative of spectrum removed [%]", fontsize=13)

            plt.subplot(2, 3, 5)
            for j in range(len(jdb)):
                colorVal = scalarMap.to_rgba(jdb[j])
                plt.plot(
                    grid[::500],
                    self.cum_curves[j][::500] / self.cum_curves[j][-1] * 100,
                    color=colorVal,
                    alpha=0.3,
                )
            plt.xlabel("Wavelength", fontsize=13)
            plt.ylabel("Normalised cumulative spectrum removed [%]", fontsize=13)

            plt.subplot(2, 3, 3)
            for j in range(len(jdb)):
                plt.plot(grid[::500], self.cum_curves[j][::500], color="k", alpha=0.3)
            plt.xlabel("Wavelength", fontsize=13)
            plt.ylabel("Cumulative of spectrum removed [%]", fontsize=13)

            plt.subplot(2, 3, 6)
            for j in range(len(jdb)):
                colorVal = scalarMap.to_rgba(jdb[j])
                plt.plot(
                    grid[::500],
                    self.cum_curves[j][::500] / self.cum_curves[j][-1] * 100,
                    color="k",
                    alpha=0.3,
                )
            plt.xlabel("Wavelength", fontsize=13)
            plt.ylabel("Normalised cumulative spectrum removed [%]", fontsize=13)
            plt.subplots_adjust(left=0.07, right=0.97)
            plt.savefig(self.dir_root + "IMAGES/mad_statistics_iter_%s.png" % (ext))

            correction_mad = all_flux - all_flux2
            to_be_saved = {"wave": grid, "correction_map": correction_mad}
            io.pickle_dump(
                to_be_saved,
                open(self.dir_root + "CORRECTION_MAP/map_matching_mad.p", "wb"),
            )

            print("\nComputation of the new continua, wait ... \n")
            time.sleep(0.5)
            count_file = -1
            self.debug = (all_flux, all_flux2)

            for j in tqdm(files):
                count_file += 1
                file = pd.read_pickle(j)
                all_flux2[count_file][file["flux" + kw] == 0] = 1
                all_flux2[count_file][all_flux2[count_file] == 0] = 1
                new_flux = file["flux" + kw] / all_flux2[count_file]
                new_flux[(new_flux == 0) | (new_flux != new_flux)] = 1
                mask = yarara_artefact_suppressed(
                    file[sub_dico]["continuum_" + continuum],
                    new_flux,
                    larger_than=50,
                    lower_than=-50,
                )
                new_flux[mask] = file[sub_dico]["continuum_" + continuum][mask]
                output = {"continuum_" + continuum: new_flux}
                file["matching_mad"] = output
                file["matching_mad"]["parameters"] = {
                    "iteration": count + 1,
                    "sub_dico_used": sub_dico,
                    "k_sigma": k_sigma,
                    "k_mad": k_mad,
                    "step": step + 1,
                }
                io.save_pickle(j, file)

            self.dico_actif = "matching_mad"

    # =============================================================================
    # CORRECTION OF FROG (GHOST AND STITCHING)
    # =============================================================================

    # instrument
    def yarara_produce_mask_contam(self, frog_file=root + "/Python/Material/Contam_HARPN.p"):

        """
        Creation of the stitching mask on the spectrum

        Parameters
        ----------
        frog_file : files containing the wavelength of the stitching
        """

        directory = self.directory
        kw = "_planet" * self.planet
        if kw != "":
            print("\n---- PLANET ACTIVATED ----")

        myf.print_box("\n---- RECIPE : PRODUCTION CONTAM MASK ----\n")

        print("\n [INFO] FROG file used : %s" % (frog_file))
        self.import_table()
        self.import_material()
        load = self.material

        grid = np.array(load["wave"])

        # extract frog table
        frog_table = pd.read_pickle(frog_file)
        # stitching

        print("\n [INFO] Producing the contam mask...")

        wave_contam = np.hstack(frog_table["wave"])
        contam = np.hstack(frog_table["contam"])

        vec = myc.tableXY(wave_contam, contam)
        vec.order()
        vec.interpolate(new_grid=np.array(load["wave"]), method="linear")

        load["contam"] = vec.y.astype("int")
        io.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))

    # instrument
    def yarara_produce_mask_frog(self, frog_file=root + "/Python/Material/Ghost_HARPS03.p"):

        """
        Correction of the stitching/ghost on the spectrum by PCA fitting

        Parameters
        ----------
        sub_dico : The sub_dictionnary used to  select the continuum
        continuum : The continuum to select (either linear or cubic)
        reference : 'median', 'snr' or 'master' to select the reference normalised spectrum used in the difference
        extended : extension of the cluster size
        frog_file : files containing the wavelength of the stitching
        """

        myf.print_box("\n---- RECIPE : MASK GHOST/STITCHING/THAR WITH FROG ----\n")

        directory = self.directory
        kw = "_planet" * self.planet
        if kw != "":
            print("\n---- PLANET ACTIVATED ----")

        self.import_table()
        self.import_material()
        load = self.material

        file_test = self.import_spectrum()
        grid = file_test["wave"]

        berv_max = self.table["berv" + kw].max()
        berv_min = self.table["berv" + kw].min()
        imin = (
            myf.find_nearest(grid, myf.doppler_r(grid[0], np.max(abs(self.table.berv)) * 1000)[0])[
                0
            ][0]
            + 1
        )
        imax = myf.find_nearest(
            grid, myf.doppler_r(grid[-1], np.max(abs(self.table.berv)) * 1000)[1]
        )[0][0]

        # extract frog table
        frog_table = pd.read_pickle(frog_file)
        berv_file = self.yarara_get_berv_value(frog_table["jdb"])

        # ghost
        for correction in ["stitching", "ghost_a", "ghost_b", "thar"]:
            if correction in frog_table.keys():
                if correction == "stitching":
                    print("\n [INFO] Producing the stitching mask...")

                    wave_stitching = np.hstack(frog_table["wave"])
                    gap_stitching = np.hstack(frog_table["stitching"])

                    vec = myc.tableXY(wave_stitching, gap_stitching)
                    vec.order()
                    stitching = vec.x[vec.y != 0]

                    stitching_b0 = myf.doppler_r(stitching, 0 * berv_file * 1000)[0]
                    # all_stitch = myf.doppler_r(stitching_b0, berv*1000)[0]

                    match_stitching = myf.match_nearest(grid, stitching_b0)
                    indext = match_stitching[:, 0].astype("int")

                    wavet_delta = np.zeros(len(grid))
                    wavet_delta[indext] = 1

                    wavet = grid[indext]
                    max_t = wavet * ((1 + 1.55e-8) * (1 + (berv_max - 0 * berv_file) / 299792.458))
                    min_t = wavet * ((1 + 1.55e-8) * (1 + (berv_min - 0 * berv_file) / 299792.458))

                    mask_stitching = np.sum(
                        (grid > min_t[:, np.newaxis]) & (grid < max_t[:, np.newaxis]),
                        axis=0,
                    ).astype("bool")
                    self.stitching_zones = mask_stitching

                    mask_stitching[0:imin] = 0
                    mask_stitching[imax:] = 0

                    load["stitching"] = mask_stitching.astype("int")
                    load["stitching_delta"] = wavet_delta.astype("int")
                else:
                    if correction == "ghost_a":
                        print("\n [INFO] Producing the ghost mask A...")
                    elif correction == "ghost_b":
                        print("\n [INFO] Producing the ghost mask B...")
                    elif correction == "thar":
                        print("\n [INFO] Producing the thar mask...")

                    contam = frog_table[correction]
                    mask = np.zeros(len(grid))
                    wave_s2d = []
                    order_s2d = []
                    for order in np.arange(len(contam)):
                        vec = myc.tableXY(
                            myf.doppler_r(frog_table["wave"][order], 0 * berv_file * 1000)[0],
                            contam[order],
                            0 * contam[order],
                        )
                        vec.order()
                        vec.y[0:2] = 0
                        vec.y[-2:] = 0
                        begin = int(myf.find_nearest(grid, vec.x[0])[0])
                        end = int(myf.find_nearest(grid, vec.x[-1])[0])
                        sub_grid = grid[begin:end]
                        vec.interpolate(new_grid=sub_grid, method="linear", interpolate_x=False)
                        model = np.array(
                            load["reference_spectrum"][begin:end]
                            * load["correction_factor"][begin:end]
                        )
                        model[model == 0] = 1
                        contam_cumu = vec.y / model
                        if sum(contam_cumu != 0) != 0:
                            mask[begin:end] += np.nanmean(contam_cumu[contam_cumu != 0]) * (
                                contam_cumu != 0
                            )
                            order_s2d.append((vec.y != 0) * (1 + order / len(contam) / 20))
                            wave_s2d.append(sub_grid)

                    mask[0:imin] = 0
                    mask[imax:] = 0
                    load[correction] = mask
            else:
                load[correction] = np.zeros(len(grid))

        io.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))

    # instrument
    def yarara_correct_frog(
        self,
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
    ):

        """
        Correction of the stitching/ghost on the spectrum by PCA fitting

        Parameters
        ----------
        sub_dico : The sub_dictionnary used to  select the continuum
        continuum : The continuum to select (either linear or cubic)
        reference : 'median', 'snr' or 'master' to select the reference normalised spectrum used in the difference
        extended : extension of the cluster size
        """

        myf.print_box("\n---- RECIPE : CORRECTION %s WITH FROG ----\n" % (correction.upper()))

        directory = self.directory
        self.import_table()
        self.import_material()
        load = self.material

        cmap = self.cmap
        planet = self.planet
        low_cmap = self.low_cmap * 100
        high_cmap = self.high_cmap * 100

        kw = "_planet" * planet
        if kw != "":
            print("\n---- PLANET ACTIVATED ----")

        epsilon = 1e-12

        if sub_dico is None:
            sub_dico = self.dico_actif
        print("\n---- DICO %s used ----\n" % (sub_dico))

        files = glob.glob(directory + "RASSI*.p")
        files = np.sort(files)

        all_flux = []
        all_flux_std = []
        snr = []
        jdb = []
        conti = []
        berv = []

        for i, j in enumerate(files):
            file = pd.read_pickle(j)
            if not i:
                grid = file["wave"]
                hole_left = file["parameters"]["hole_left"]
                hole_right = file["parameters"]["hole_right"]
            f = file["flux" + kw]
            f_std = file["flux_err"]
            c = file[sub_dico]["continuum_" + continuum]
            c_std = file["continuum_err"]
            f_norm, f_norm_std = myf.flux_norm_std(f, f_std, c, c_std)
            all_flux.append(f_norm)
            all_flux_std.append(f_norm_std)
            conti.append(c)
            jdb.append(file["parameters"]["jdb"])
            snr.append(file["parameters"]["SNR_5500"])
            if type(berv_shift) != np.ndarray:
                try:
                    berv.append(file["parameters"][berv_shift])
                except:
                    berv.append(0)
            else:
                berv = berv_shift

        step = file[sub_dico]["parameters"]["step"]

        all_flux = np.array(all_flux)
        all_flux_std = np.array(all_flux_std)
        conti = np.array(conti)
        jdb = np.array(jdb)
        snr = np.array(snr)
        berv = np.array(berv)

        if reference == "snr":
            ref = all_flux[snr.argmax()]
        elif reference == "median":
            print(" [INFO] Reference spectrum : median")
            ref = np.median(all_flux, axis=0)
        elif reference == "master":
            print(" [INFO] Reference spectrum : master")
            ref = np.array(load["reference_spectrum"])
        elif type(reference) == int:
            print(" [INFO] Reference spectrum : spectrum %.0f" % (reference))
            ref = all_flux[reference]
        else:
            ref = 0 * np.median(all_flux, axis=0)

        berv_max = self.table["berv" + kw].max()
        berv_min = self.table["berv" + kw].min()

        diff = all_flux - ref

        diff_backup = diff.copy()

        if np.sum(abs(berv)) != 0:
            for j in tqdm(np.arange(len(all_flux))):
                test = myc.tableXY(grid, diff[j], all_flux_std[j])
                test.x = myf.doppler_r(test.x, berv[j] * 1000)[1]
                test.interpolate(new_grid=grid, method="cubic", replace=True, interpolate_x=False)
                diff[j] = test.y
                all_flux_std[j] = test.yerr

        # extract frog table
        # frog_table = pd.read_pickle(frog_file)
        berv_file = 0  # self.yarara_get_berv_value(frog_table['jdb'])

        mask = np.array(load[correction])

        loc_ghost = mask != 0

        # mask[mask<treshold_contam] = 0
        val, borders = myf.clustering(loc_ghost, 0.5, 1)
        val = np.array([np.product(v) for v in val])
        borders = borders[val == 1]

        min_t = grid[borders[:, 0]] * (
            (1 + 1.55e-8) * (1 + (berv_min - 0 * berv_file) / 299792.458)
        )
        max_t = grid[borders[:, 1]] * (
            (1 + 1.55e-8) * (1 + (berv_max - 0 * berv_file) / 299792.458)
        )

        if (correction == "ghost_a") | (correction == "ghost_b"):
            for j in range(3):
                if np.sum(mask > treshold_contam) < 200:
                    print(
                        Fore.YELLOW
                        + " [WARNING] Not enough wavelength in the mask, treshold contamination reduced down to %.2f"
                        % (treshold_contam)
                        + Fore.RESET
                    )
                    treshold_contam *= 0.75

        mask_ghost = np.sum(
            (grid > min_t[:, np.newaxis]) & (grid < max_t[:, np.newaxis]), axis=0
        ).astype("bool")
        mask_ghost_extraction = (
            mask_ghost
            & (mask > treshold_contam)
            & (ref < 1)
            & (np.array(1 - load["activity_proxies"]).astype("bool"))
            & (grid < wave_max_train)
            & (grid > wave_min_train)
        )  # extract everywhere

        if correction == "stitching":
            self.stitching = mask_ghost
            self.stitching_extracted = mask_ghost_extraction
        elif correction == "ghost_a":
            self.ghost_a = mask_ghost
            self.ghost_a_extracted = mask_ghost_extraction
        elif correction == "ghost_b":
            self.ghost_b = mask_ghost
            self.ghost_b_extracted = mask_ghost_extraction
        elif correction == "thar":
            self.thar = mask_ghost
            self.thar_extracted = mask_ghost_extraction
        elif correction == "contam":
            self.contam = mask_ghost
            self.contam_extracted = mask_ghost_extraction

        # compute pca

        if correction == "stitching":
            print(" [INFO] Computation of PCA vectors for stitching correction...")
            diff_ref = diff[:, mask_ghost]
            subflux = diff[:, (mask_ghost) & (np.array(load["ghost_a"]) == 0)]
            subflux_std = all_flux_std[:, (mask_ghost) & (np.array(load["ghost_a"]) == 0)]
            lab = "Stitching"
            name = "stitching"
        elif correction == "ghost_a":
            print(" [INFO] Computation of PCA vectors for ghost correction...")
            diff_ref = diff[:, mask_ghost]
            subflux = diff[:, (np.array(load["stitching"]) == 0) & (mask_ghost_extraction)]
            subflux_std = all_flux_std[
                :, (np.array(load["stitching"]) == 0) & (mask_ghost_extraction)
            ]
            lab = "Ghost_a"
            name = "ghost_a"
        elif correction == "ghost_b":
            print(" [INFO] Computation of PCA vectors for ghost correction...")
            diff_ref = diff[:, mask_ghost]
            subflux = diff[:, (load["thar"] == 0) & (mask_ghost_extraction)]
            subflux_std = all_flux_std[:, (load["thar"] == 0) & (mask_ghost_extraction)]
            lab = "Ghost_b"
            name = "ghost_b"
        elif correction == "thar":
            print(" [INFO] Computation of PCA vectors for thar correction...")
            diff_ref = diff.copy()
            subflux = diff[:, mask_ghost_extraction]
            subflux_std = all_flux_std[:, mask_ghost_extraction]
            lab = "Thar"
            name = "thar"
        elif correction == "contam":
            print(" [INFO] Computation of PCA vectors for contam correction...")
            diff_ref = diff[:, mask_ghost]
            subflux = diff[:, mask_ghost_extraction]
            subflux_std = all_flux_std[:, mask_ghost_extraction]
            lab = "Contam"
            name = "contam"

        subflux_std = subflux_std[:, np.std(subflux, axis=0) != 0]
        subflux = subflux[:, np.std(subflux, axis=0) != 0]

        if not len(subflux[0]):
            subflux = diff[:, 0:10]
            subflux_std = all_flux_std[:, 0:10]

        plt.figure(2, figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(subflux, aspect="auto", vmin=-0.005, vmax=0.005)
        plt.title(lab + " lines")
        plt.xlabel("Pixels extracted", fontsize=14)
        plt.ylabel("Time", fontsize=14)
        ax = plt.gca()
        plt.subplot(1, 2, 2, sharex=ax, sharey=ax)
        plt.imshow(
            subflux / (epsilon + np.std(subflux, axis=0)),
            aspect="auto",
            vmin=-0.005,
            vmax=0.005,
        )
        plt.title(lab + " lines equalized")
        plt.xlabel("Pixels extracted", fontsize=14)
        plt.ylabel("Time", fontsize=14)

        c = int(equal_weight)

        X_train = (subflux / ((1 - c) + epsilon + c * np.std(subflux, axis=0))).T
        X_train_std = (subflux_std / ((1 - c) + epsilon + c * np.std(subflux, axis=0))).T

        # io.pickle_dump({'jdb':np.array(self.table.jdb),'ratio_flux':X_train,'ratio_flux_std':X_train_std},open(root+'/Python/datasets/telluri_cenB.p','wb'))

        test2 = myc.table(X_train)
        test2.WPCA(algo_pca, weight=1 / X_train_std**2, comp_max=nb_pca_comp)

        phase_mod = np.arange(365)[
            np.argmin(
                np.array(
                    [np.max((jdb - k) % 365.25) - np.min((jdb - k) % 365.25) for k in range(365)]
                )
            )
        ]

        plt.figure(4, figsize=(10, 14))
        plt.subplot(3, 1, 1)
        plt.xlabel("# PCA components", fontsize=13)
        plt.ylabel("Variance explained", fontsize=13)
        plt.plot(np.arange(1, len(test2.phi_components) + 1), test2.var_ratio)
        plt.scatter(np.arange(1, len(test2.phi_components) + 1), test2.var_ratio)
        plt.subplot(3, 1, 2)
        plt.xlabel("# PCA components", fontsize=13)
        plt.ylabel("Z score", fontsize=13)
        plt.plot(np.arange(1, len(test2.phi_components) + 1), test2.zscore_components)
        plt.scatter(np.arange(1, len(test2.phi_components) + 1), test2.zscore_components)
        z_max = test2.zscore_components[-5:].max()
        z_min = test2.zscore_components[-5:].min()
        vec_relevant = np.arange(len(test2.zscore_components)) * (
            (test2.zscore_components > z_max) | (test2.zscore_components < z_min)
        )
        plt.axhspan(ymin=z_min, ymax=z_max, alpha=0.2, color="k")
        pca_comp_kept2 = int(np.where(vec_relevant != np.arange(len(vec_relevant)))[0][0])
        plt.subplot(3, 1, 3)
        plt.xlabel("# PCA components", fontsize=13)
        plt.ylabel(r"$\Phi(0)$", fontsize=13)
        plt.plot(np.arange(1, len(test2.phi_components) + 1), test2.phi_components)
        plt.scatter(np.arange(1, len(test2.phi_components) + 1), test2.phi_components)
        plt.axhline(y=0.5, color="k")
        phi_max = test2.phi_components[-5:].max()
        phi_min = test2.phi_components[-5:].min()
        plt.axhspan(ymin=phi_min, ymax=phi_max, alpha=0.2, color="k")
        vec_relevant = np.arange(len(test2.phi_components)) * (
            (test2.phi_components > phi_max) | (test2.phi_components < phi_min)
        )
        if pca_comp_kept is None:
            pca_comp_kept = int(np.where(vec_relevant != np.arange(len(vec_relevant)))[0][0])
            pca_comp_kept = np.max([pca_comp_kept, pca_comp_kept2])

        plt.savefig(self.dir_root + "IMAGES/" + name + "_PCA_variances.pdf")

        plt.figure(figsize=(15, 10))
        for j in range(pca_comp_kept):
            if j == 0:
                plt.subplot(pca_comp_kept, 2, 2 * j + 1)
                ax = plt.gca()
            else:
                plt.subplot(pca_comp_kept, 2, 2 * j + 1, sharex=ax)
            plt.scatter(jdb, test2.vec[:, j])
            plt.subplot(pca_comp_kept, 2, 2 * j + 2)
            plt.scatter((jdb - phase_mod) % 365.25, test2.vec[:, j])
        plt.subplots_adjust(top=0.95, bottom=0.07, left=0.07, right=0.95, hspace=0)
        plt.savefig(self.dir_root + "IMAGES/" + name + "_PCA_vectors.pdf")

        if correction == "stitching":
            self.vec_pca_stitching = test2.vec[:, 0:pca_comp_kept]
        elif correction == "ghost_a":
            self.vec_pca_ghost_a = test2.vec[:, 0:pca_comp_kept]
        elif correction == "ghost_b":
            self.vec_pca_ghost_b = test2.vec[:, 0:pca_comp_kept]
        elif correction == "thar":
            self.vec_pca_thar = test2.vec[:, 0:pca_comp_kept]
        elif correction == "contam":
            self.vec_pca_contam = test2.vec[:, 0:pca_comp_kept]

        to_be_fit = diff / (np.std(diff, axis=0) + epsilon)

        rcorr = np.zeros(len(grid))
        for j in range(pca_comp_kept):
            proxy1 = test2.vec[:, j]
            rslope1 = np.median(
                (to_be_fit - np.mean(to_be_fit, axis=0))
                / ((proxy1 - np.mean(proxy1))[:, np.newaxis]),
                axis=0,
            )
            rcorr1 = abs(rslope1 * np.std(proxy1) / (np.std(to_be_fit, axis=0) + epsilon))
            rcorr = np.nanmax([rcorr1, rcorr], axis=0)
        rcorr[np.isnan(rcorr)] = 0

        val, borders = myf.clustering(mask_ghost, 0.5, 1)
        val = np.array([np.product(j) for j in val])
        borders = borders[val.astype("bool")]
        borders = myf.merge_borders(borders)
        flat_mask = myf.flat_clustering(len(grid), borders, extended=50).astype("bool")
        rcorr_free = rcorr[~flat_mask]
        rcorr_contaminated = rcorr[flat_mask]

        if correction == "thar":
            mask_ghost = np.ones(len(grid)).astype("bool")

        plt.figure(figsize=(8, 6))
        bins_contam, bins, dust = plt.hist(
            rcorr_contaminated,
            label="contaminated region",
            bins=np.linspace(0, 1, 100),
            alpha=0.5,
            density=True,
        )
        bins_control, bins, dust = plt.hist(
            rcorr_free,
            bins=np.linspace(0, 1, 100),
            label="free region",
            alpha=0.5,
            density=True,
        )
        plt.yscale("log")
        plt.legend()
        bins = bins[0:-1] + np.diff(bins) * 0.5
        sum_a = np.sum(bins_contam[bins > 0.40])
        sum_b = np.sum(bins_control[bins > 0.40])
        crit = int(sum_a > (2 * sum_b))
        check = ["r", "g"][crit]  # three times more correlation than in the control group
        plt.xlabel(r"|$\mathcal{R}_{pearson}$|", fontsize=14, fontweight="bold", color=check)
        plt.title("Density", color=check)
        myf.plot_color_box(color=check)

        plt.savefig(self.dir_root + "IMAGES/" + name + "_control_check.pdf")
        print(" [INFO] %.0f versus %.0f" % (sum_a, sum_b))

        if crit:
            print(" [INFO] Control check sucessfully performed: %s" % (name))
        else:
            print(
                Fore.YELLOW
                + " [WARNING] Control check failed. Correction may be poorly performed for: %s"
                % (name)
                + Fore.RESET
            )

        diff_ref[np.isnan(diff_ref)] = 0

        idx_min = myf.find_nearest(grid, wave_min)[0]
        idx_max = myf.find_nearest(grid, wave_max)[0] + 1

        new_wave = grid[int(idx_min) : int(idx_max)]

        if complete_analysis:
            plt.figure(figsize=(18, 12))
            plt.subplot(pca_comp_kept // 2 + 1, 2, 1)
            myf.my_colormesh(
                new_wave,
                np.arange(len(diff)),
                diff[:, int(idx_min) : int(idx_max)],
                vmin=low_cmap / 100,
                vmax=high_cmap / 100,
                cmap=cmap,
            )
            ax = plt.gca()
            for nb_vec in tqdm(range(1, pca_comp_kept)):
                correction2 = np.zeros((len(grid), len(jdb)))
                collection = myc.table(diff_ref.T)
                base_vec = np.vstack([np.ones(len(diff)), test2.vec[:, 0:nb_vec].T])
                collection.fit_base(base_vec, num_sim=1)
                correction2[mask_ghost] = collection.coeff_fitted.dot(base_vec)
                correction2 = np.transpose(correction2)
                diff_ref2 = diff - correction2
                plt.subplot(pca_comp_kept // 2 + 1, 2, nb_vec + 1, sharex=ax, sharey=ax)
                plt.title("Vec PCA fitted = %0.f" % (nb_vec))
                myf.my_colormesh(
                    new_wave,
                    np.arange(len(diff)),
                    diff_ref2[:, int(idx_min) : int(idx_max)],
                    vmin=low_cmap / 100,
                    vmax=high_cmap / 100,
                    cmap=cmap,
                )
            plt.subplots_adjust(top=0.95, bottom=0.07, left=0.07, right=0.95, hspace=0.3)
            plt.subplot(pca_comp_kept // 2 + 1, 2, pca_comp_kept + 1, sharex=ax)
            plt.plot(new_wave, mask[int(idx_min) : int(idx_max)])
            plt.plot(new_wave, mask_ghost_extraction[int(idx_min) : int(idx_max)], color="k")
            if correction == "stitching":
                plt.plot(new_wave, ref[int(idx_min) : int(idx_max)], color="gray")
        else:
            correction = np.zeros((len(grid), len(jdb)))
            collection = myc.table(diff_ref.T)
            base_vec = np.vstack([np.ones(len(diff)), test2.vec[:, 0:pca_comp_kept].T])
            collection.fit_base(base_vec, num_sim=1)
            correction[mask_ghost] = collection.coeff_fitted.dot(base_vec)
            correction = np.transpose(correction)
            correction[:, rcorr < rcorr_min] = 0

            if np.sum(abs(berv)) != 0:
                for j in tqdm(np.arange(len(all_flux))):
                    test = myc.tableXY(grid, correction[j], 0 * grid)
                    test.x = myf.doppler_r(test.x, berv[j] * 1000)[0]
                    test.interpolate(
                        new_grid=grid, method="cubic", replace=True, interpolate_x=False
                    )
                    correction[j] = test.y

                index_min_backup = int(myf.find_nearest(grid, myf.doppler_r(grid[0], 30000)[0])[0])
                index_max_backup = int(
                    myf.find_nearest(grid, myf.doppler_r(grid[-1], -30000)[0])[0]
                )
                correction[:, 0 : index_min_backup * 2] = 0
                correction[:, index_max_backup * 2 :] = 0
                index_hole_right = int(
                    myf.find_nearest(grid, hole_right + 1)[0]
                )  # correct 1 angstrom band due to stange artefact at the border of the gap
                index_hole_left = int(
                    myf.find_nearest(grid, hole_left - 1)[0]
                )  # correct 1 angstrom band due to stange artefact at the border of the gap
                correction[:, index_hole_left : index_hole_right + 1] = 0

            diff_ref2 = diff_backup - correction

            new_conti = conti * (diff_backup + ref) / (diff_ref2 + ref + epsilon)
            new_continuum = new_conti.copy()
            new_continuum[all_flux == 0] = conti[all_flux == 0]
            new_continuum[new_continuum != new_continuum] = conti[
                new_continuum != new_continuum
            ]  # to supress mystic nan appearing
            new_continuum[np.isnan(new_continuum)] = conti[np.isnan(new_continuum)]
            new_continuum[new_continuum == 0] = conti[new_continuum == 0]
            new_continuum = self.uncorrect_hole(new_continuum, conti)

            # plot end

            if (name == "thar") | (name == "stitching"):
                max_var = grid[np.std(correction, axis=0).argsort()[::-1]]
                if name == "thar":
                    max_var = max_var[max_var < 4400][0]
                else:
                    max_var = max_var[max_var < 6700][0]
                wave_min = myf.find_nearest(grid, max_var - 15)[1]
                wave_max = myf.find_nearest(grid, max_var + 15)[1]

                idx_min = myf.find_nearest(grid, wave_min)[0]
                idx_max = myf.find_nearest(grid, wave_max)[0] + 1

            new_wave = grid[int(idx_min) : int(idx_max)]

            fig = plt.figure(figsize=(21, 9))

            plt.axes([0.05, 0.66, 0.90, 0.25])
            myf.my_colormesh(
                new_wave,
                np.arange(len(diff)),
                100 * diff_backup[:, int(idx_min) : int(idx_max)],
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
            myf.my_colormesh(
                new_wave,
                np.arange(len(diff)),
                100 * diff_ref2[:, int(idx_min) : int(idx_max)],
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
            myf.my_colormesh(
                new_wave,
                np.arange(len(diff)),
                100 * diff_backup[:, int(idx_min) : int(idx_max)]
                - 100 * diff_ref2[:, int(idx_min) : int(idx_max)],
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

            plt.savefig(self.dir_root + "IMAGES/Correction_" + name + ".png")

            if name == "ghost_b":
                diff_backup = []
                self.import_dico_tree()
                sub = np.array(
                    self.dico_tree.loc[self.dico_tree["dico"] == "matching_ghost_a", "dico_used"]
                )[0]
                for i, j in enumerate(files):
                    file = pd.read_pickle(j)
                    f = file["flux" + kw]
                    f_std = file["flux_err"]
                    c = file[sub]["continuum_" + continuum]
                    c_std = file["continuum_err"]
                    f_norm, f_norm_std = myf.flux_norm_std(f, f_std, c, c_std)
                    diff_backup.append(f_norm - ref)
                diff_backup = np.array(diff_backup)

                fig = plt.figure(figsize=(21, 9))

                plt.axes([0.05, 0.66, 0.90, 0.25])
                myf.my_colormesh(
                    new_wave,
                    np.arange(len(diff)),
                    100 * diff_backup[:, int(idx_min) : int(idx_max)],
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
                myf.my_colormesh(
                    new_wave,
                    np.arange(len(diff)),
                    100 * diff_ref2[:, int(idx_min) : int(idx_max)],
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
                myf.my_colormesh(
                    new_wave,
                    np.arange(len(diff)),
                    100 * diff_backup[:, int(idx_min) : int(idx_max)]
                    - 100 * diff_ref2[:, int(idx_min) : int(idx_max)],
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

                plt.savefig(self.dir_root + "IMAGES/Correction_ghost.png")

            to_be_saved = {"wave": grid, "correction_map": correction}
            io.pickle_dump(
                to_be_saved,
                open(self.dir_root + "CORRECTION_MAP/map_matching_" + name + ".p", "wb"),
            )

            print("\nComputation of the new continua, wait ... \n")
            time.sleep(0.5)
            i = -1
            for j in tqdm(files):
                i += 1
                file = pd.read_pickle(j)
                output = {"continuum_" + continuum: new_continuum[i]}
                file["matching_" + name] = output
                file["matching_" + name]["parameters"] = {
                    "reference_spectrum": reference,
                    "sub_dico_used": sub_dico,
                    "equal_weight": equal_weight,
                    "pca_comp_kept": pca_comp_kept,
                    "step": step + 1,
                }
                io.save_pickle(j, file)

            self.yarara_analyse_summary()

            self.dico_actif = "matching_" + name

            plt.show(block=False)

    # instrument
    def yarara_correct_borders_pxl(self, pixels_to_reject=[2, 4095], min_shift=-30, max_shift=30):
        """Produce a brute mask to flag lines crossing pixels according to min-max shift

        Parameters
        ----------
        pixels_to_reject : List of pixels
        min_shift : min shist value in km/s
        max_shift : max shist value in km/s
        """

        myf.print_box("\n---- RECIPE : CREATE PIXELS BORDERS MASK ----\n")

        self.import_material()
        load = self.material

        wave = np.array(load["wave"])
        dwave = np.mean(np.diff(wave))
        pxl = self.yarara_get_pixels()
        orders = self.yarara_get_orders()

        pxl *= orders != 0

        pixels_rejected = np.array(pixels_to_reject)

        pxl[pxl == 0] = np.max(pxl) * 2

        dist = np.zeros(len(pxl)).astype("bool")
        for i in np.arange(np.shape(pxl)[1]):
            dist = dist | (np.min(abs(pxl[:, i] - pixels_rejected[:, np.newaxis]), axis=0) == 0)

        # idx1, dust, dist1 = myf.find_nearest(pixels_rejected,pxl[:,0])
        # idx2, dust, dist2 = myf.find_nearest(pixels_rejected,pxl[:,1])

        # dist = (dist1<=1)|(dist2<=1)

        f = np.where(dist == 1)[0]
        plt.figure()
        for i in np.arange(np.shape(pxl)[1]):
            plt.scatter(pxl[f, i], orders[f, i])

        val, cluster = myf.clustering(dist, 0.5, 1)
        val = np.array([np.product(v) for v in val])
        cluster = cluster[val.astype("bool")]

        left = np.round(wave[cluster[:, 0]] * min_shift / 3e5 / dwave, 0).astype("int")
        right = np.round(wave[cluster[:, 1]] * max_shift / 3e5 / dwave, 0).astype("int")
        # length = right-left+1

        # wave_flagged = wave[f]
        # left = myf.doppler_r(wave_flagged,min_shift*1000)[0]
        # right = myf.doppler_r(wave_flagged,max_shift*1000)[0]

        # idx_left = myf.find_nearest(wave,left)[0]
        # idx_right = myf.find_nearest(wave,right)[0]

        idx_left = cluster[:, 0] + left
        idx_right = cluster[:, 1] + right

        flag_region = np.zeros(len(wave)).astype("int")

        for l, r in zip(idx_left, idx_right):
            flag_region[l : r + 1] = 1

        load["borders_pxl"] = flag_region.astype("int")
        io.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))

    # outliers
    def yarara_correct_brute(
        self,
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
    ):

        """
        Brutal suppression of flux value with variance to high (final solution)

        Parameters
        ----------
        sub_dico : The sub_dictionnary used to  select the continuum
        continuum : The continuum to select (either linear or cubic)
        reference : 'median', 'snr' or 'master' to select the reference normalised spectrum used in the difference
        win_roll : window size of the rolling algorithm
        min_length : minimum cluster length to be flagged
        k_sigma : k_sigma of the rolling mad clipping
        extended : extension of the cluster size
        low : lowest cmap value
        high : highest cmap value
        cmap : cmap of the 2D plot
        """

        myf.print_box("\n---- RECIPE : CORRECTION BRUTE ----\n")

        directory = self.directory

        cmap = self.cmap
        planet = self.planet
        low_cmap = self.low_cmap
        high_cmap = self.high_cmap

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

        all_flux = []
        snr = []
        jdb = []
        conti = []

        for i, j in enumerate(files):
            file = pd.read_pickle(j)
            if not i:
                grid = file["wave"]
            all_flux.append(file["flux" + kw] / file[sub_dico]["continuum_" + continuum])
            conti.append(file[sub_dico]["continuum_" + continuum])
            jdb.append(file["parameters"]["jdb"])
            snr.append(file["parameters"]["SNR_5500"])

        step = file[sub_dico]["parameters"]["step"]
        all_flux = np.array(all_flux)
        conti = np.array(conti)

        if reference == "snr":
            ref = all_flux[snr.argmax()]
        elif reference == "median":
            print("[INFO] Reference spectrum : median")
            ref = np.median(all_flux, axis=0)
        elif reference == "master":
            print("[INFO] Reference spectrum : master")
            ref = np.array(load["reference_spectrum"])
        elif type(reference) == int:
            print("[INFO] Reference spectrum : spectrum %.0f" % (reference))
            ref = all_flux[reference]
        else:
            ref = 0 * np.median(all_flux, axis=0)

        all_flux = all_flux - ref
        metric = np.std(all_flux, axis=0)
        smoothed_med = np.ravel(
            pd.DataFrame(metric).rolling(win_roll, center=True, min_periods=1).quantile(0.5)
        )
        smoothed_mad = np.ravel(
            pd.DataFrame(abs(metric - smoothed_med))
            .rolling(win_roll, center=True, min_periods=1)
            .quantile(0.5)
        )
        mask = (metric - smoothed_med) > smoothed_mad * 1.48 * k_sigma

        clus = myf.clustering(mask, 0.5, 1)[0]
        clus = np.array([np.product(j) for j in clus])
        cluster = myf.clustering(mask, 0.5, 1)[-1]
        cluster = np.hstack([cluster, clus[:, np.newaxis]])
        cluster = cluster[cluster[:, 3] == 1]
        cluster = cluster[cluster[:, 2] >= min_length]

        cluster2 = cluster.copy()
        sum_mask = []
        all_flat = []
        for j in tqdm(range(200)):
            cluster2[:, 0] -= extended
            cluster2[:, 1] += extended
            flat_vec = myf.flat_clustering(len(grid), cluster2[:, 0:2])
            flat_vec = flat_vec >= 1
            all_flat.append(flat_vec)
            sum_mask.append(np.sum(flat_vec))
        sum_mask = 100 * np.array(sum_mask) / len(grid)
        all_flat = np.array(all_flat)

        loc = myf.find_nearest(sum_mask, np.arange(5, 26, 5))[0]

        plt.figure(figsize=(16, 16))

        plt.subplot(3, 1, 1)
        plt.plot(grid, metric - smoothed_med, color="k")
        plt.plot(grid, smoothed_mad * 1.48 * k_sigma, color="r")
        plt.ylim(0, 0.01)
        ax = plt.gca()

        plt.subplot(3, 1, 2, sharex=ax)
        for i, j, k in zip(["5%", "10%", "15%", "20%", "25%"], loc, [1, 1.05, 1.1, 1.15, 1.2]):
            plt.plot(grid, all_flat[j] * k, label=i)
        plt.legend()

        plt.subplot(3, 2, 5)
        b = myc.tableXY(np.arange(len(sum_mask)) * 5, sum_mask)
        b.null()
        b.plot()
        plt.xlabel("Extension of rejection zones", fontsize=14)
        plt.ylabel("Percent of the spectrum rejected [%]", fontsize=14)

        for j in loc:
            plt.axhline(y=b.y[j], color="k", ls=":")

        ax = plt.gca()
        plt.subplot(3, 2, 6, sharex=ax)
        b.diff(replace=False)
        b.deri.plot()
        for j in loc:
            plt.axhline(y=b.deri.y[j], color="k", ls=":")

        if percent_removed is None:
            percent_removed = myf.sphinx("Select the percentage of spectrum removed")

        percent_removed = int(percent_removed)

        loc_select = myf.find_nearest(sum_mask, percent_removed)[0]

        final_mask = np.ravel(all_flat[loc_select]).astype("bool")

        if borders_pxl:
            borders_pxl_mask = np.array(load["borders_pxl"]).astype("bool")
        else:
            borders_pxl_mask = np.zeros(len(final_mask)).astype("bool")

        if ghost2:
            g = pd.read_pickle(root + "/Python/Material/Ghost2_" + ghost2 + ".p")
            ghost = myc.tableXY(g["wave"], g["ghost2"], 0 * g["wave"])
            ghost.interpolate(new_grid=grid, replace=True, method="linear", interpolate_x=False)
            ghost_brute_mask = ghost.y.astype("bool")
        else:
            ghost_brute_mask = np.zeros(len(final_mask)).astype("bool")
        load["ghost2"] = ghost_brute_mask.astype("int")
        io.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))

        final_mask = final_mask | ghost_brute_mask | borders_pxl_mask
        self.brute_mask = final_mask

        load["mask_brute"] = final_mask
        io.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))

        all_flux2 = all_flux.copy()
        all_flux2[:, final_mask] = 0

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(all_flux, aspect="auto", vmin=low_cmap, vmax=high_cmap, cmap=cmap)
        ax = plt.gca()
        plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
        plt.imshow(all_flux2, aspect="auto", vmin=low_cmap, vmax=high_cmap, cmap=cmap)
        ax = plt.gca()

        new_conti = conti * (all_flux + ref) / (all_flux2 + ref + epsilon)
        new_continuum = new_conti.copy()
        new_continuum[all_flux == 0] = conti[all_flux == 0]
        new_continuum[new_continuum == 0] = conti[new_continuum == 0]
        new_continuum[np.isnan(new_continuum)] = conti[np.isnan(new_continuum)]

        print("\nComputation of the new continua, wait ... \n")
        time.sleep(0.5)

        i = -1
        for j in tqdm(files):
            i += 1
            file = pd.read_pickle(j)
            output = {"continuum_" + continuum: new_continuum[i]}
            file["matching_brute"] = output
            file["matching_brute"]["parameters"] = {
                "reference_spectrum": reference,
                "sub_dico_used": sub_dico,
                "k_sigma": k_sigma,
                "rolling_window": win_roll,
                "minimum_length_cluster": min_length,
                "percentage_removed": percent_removed,
                "step": step + 1,
            }
            io.save_pickle(j, file)

        self.dico_actif = "matching_brute"

    # =============================================================================
    # AIRMASS
    # =============================================================================

    # processing
    def uncorrect_hole(self, conti, conti_ref, values_forbidden=[0, np.inf]):
        file_test = self.import_spectrum()
        wave = np.array(file_test["wave"])
        hl = file_test["parameters"]["hole_left"]
        hr = file_test["parameters"]["hole_right"]

        if hl != -99.9:
            i1 = int(myf.find_nearest(wave, hl)[0])
            i2 = int(myf.find_nearest(wave, hr)[0])
            conti[:, i1 - 1 : i2 + 2] = conti_ref[:, i1 - 1 : i2 + 2].copy()

        for l in values_forbidden:
            conti[conti == l] = conti_ref[conti == l].copy()

        return conti
