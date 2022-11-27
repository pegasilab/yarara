from __future__ import annotations

import glob as glob
import os
from typing import Any, Dict, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..io import pickle_dump
from ..paths import root


class SIF_float_telluric(TypedDict, total=False):
    fixed: float  #: Generic value
    YARARA: float  #: YARARA value
    telluric: float


class SIF_float(TypedDict, total=False):
    fixed: float  #: Generic value
    YARARA: float  #: YARARA value


class SIF_str(TypedDict, total=False):
    fixed: str  #: Generic value
    YARARA: str  #: YARARA value


class SIF_int(TypedDict, total=False):
    fixed: int  #: Generic value
    YARARA: int  #: YARARA value


class StarInfo(TypedDict, total=False):
    Name: str
    Simbad_name: SIF_str
    Sp_type: SIF_str
    Ra: SIF_str
    Dec: SIF_str
    Pma: SIF_float
    Pmd: SIF_float
    Rv_sys: SIF_float
    Mstar: SIF_float
    Rstar: SIF_float
    magU: SIF_float
    magB: SIF_float
    magV: SIF_float
    magR: SIF_float
    UB: SIF_float
    BV: SIF_float
    VR: SIF_float
    Dist_pc: SIF_float
    Teff: SIF_float
    Log_g: SIF_float
    FeH: SIF_float
    Vsini: SIF_float
    Vmicro: SIF_float
    Prot: SIF_float
    Pmag: SIF_float
    FWHM: SIF_float_telluric
    Contrast: SIF_float_telluric
    CCF_delta: SIF_float
    stellar_template: SIF_str


class spec_time_series(object):
    from .activity import yarara_correct_activity
    from .bridge import (
        continuum_error,
        dace_statistic,
        flux_error,
        import_dace_sts,
        scale_cmap,
        snr_statistic,
        suppress_low_snr_spectra,
        suppress_time_RV,
        yarara_berv_summary,
        yarara_check_fwhm,
        yarara_check_rv_sys,
        yarara_color_template,
        yarara_correct_secular_acc,
        yarara_flux_constant,
        yarara_get_first_wave,
        yarara_map_1d_to_2d,
        yarara_simbad_query,
        yarara_suppress_doubtful_spectra,
        yarara_transit_def,
    )
    from .ccf.analysis import yarara_ccf
    from .ccf.io import read_ccf_mask, yarara_ccf_save
    from .ccf.master import yarara_master_ccf
    from .extract import yarara_get_berv_value, yarara_get_orders, yarara_get_pixels
    from .instrument.correct_borders_pxl import yarara_correct_borders_pxl
    from .instrument.correct_frog import yarara_correct_frog
    from .instrument.correct_pattern import yarara_correct_pattern
    from .instrument.produce_mask_contam import yarara_produce_mask_contam
    from .instrument.produce_mask_frog import yarara_produce_mask_frog
    from .io.ccf import import_ccf, yarara_add_ccf_entry
    from .io.dico import import_dico_chain, import_dico_tree, yarara_add_step_dico
    from .io.flux import import_sts_flux
    from .io.material import import_material
    from .io.obs_info import yarara_obs_info
    from .io.pickle import yarara_exploding_pickle
    from .io.rassine import import_rassine_output
    from .io.reduction import import_info_reduction, update_info_reduction
    from .io.spectrum import import_spectrum, spectrum, yarara_get_bin_length
    from .io.star_info import import_star_info, yarara_star_info
    from .io.summary import yarara_analyse_summary
    from .io.suppress import suppress_time_RV, suppress_time_spectra
    from .io.table import import_table
    from .outliers.brute import yarara_correct_brute
    from .outliers.cosmics import yarara_correct_cosmics
    from .outliers.mad import yarara_correct_mad
    from .outliers.smooth import yarara_correct_smooth
    from .plots import yarara_comp_all, yarara_plot_all
    from .processing.activity_index import yarara_activity_index
    from .processing.map import yarara_map
    from .processing.retropropagation_correction import yarara_retropropagation_correction
    from .processing.uncorrect_hole import uncorrect_hole
    from .telluric.analysis import yarara_telluric
    from .telluric.correct import yarara_correct_telluric_proxy
    from .telluric.gradient import yarara_correct_telluric_gradient
    from .telluric.oxygen import yarara_correct_oxygen
    from .util.cut_spectrum import yarara_cut_spectrum
    from .util.map import yarara_add_map, yarara_substract_map
    from .util.map_all import yarara_map_all
    from .util.median_master import yarara_median_master
    from .util.non_zero_flux import yarara_non_zero_flux
    from .util.poissonian_noise import yarara_poissonian_noise

    def __init__(self, directory: str) -> None:
        if len(directory.split("/")) == 1:
            directory = root + "/spectra/" + directory + "/data/s1d/HARPS03/WORKSPACE/"
        if directory[-1] != "/":
            directory = directory + "/"

        self.directory = directory
        self.starname = directory.split("/spectra/")[-1].split("/")[0].split("=")[0]
        self.dir_root = directory.split("WORKSPACE/")[0]
        self.dir_yarara = directory.split("spectra/")[0] + "spectra/"
        self.cmap = "plasma"

        # color code of the spec time series residuals in flux normalized units
        self.low_cmap = -0.005
        self.high_cmap = 0.005

        self.zoom = 1
        self.smooth_map = 1
        self.planet: bool = False  #: If a planet has been injected
        self.sp_type = None
        self.rv_sys = None
        self.teff = None
        self.log_g = None
        self.bv = None
        self.infos = {}
        self.ram = []

        #: Full width half maximum of the CCF
        self.fwhm: float = np.NaN

        self.info_reduction: Dict[str, Any] = None  # type: ignore

        #: If info_reduction has been loaded, modification time of the loaded file
        self.info_reduction_ut: float = 0.0

        self.instrument = ""  #: Instrument used for the measurements

        # TODO: document the columns
        #: Table of relations between outputs produced at different stages of the pipeline
        self.dico_tree: pd.DataFrame = None  # type: ignore

        # TODO: document the columns
        #: Table with elements having a spectral wavelength dimension
        self.material: pd.DataFrame = None  # type: ignore

        # TODO: document the columns
        # table["filename"] is the spectrum file <- RASSINE
        # table["rv_shift"] is in km/s
        # berv
        self.table: pd.DataFrame = None  # type: ignore

        self.star_info: StarInfo = {}  #: Information about the star

        self.table_ccf: Dict[str, Any] = {}

        # All of these fields are timestamps of file read
        self.table_ut = 0.0
        self.material_ut = 0.0
        self.info_reduction_ut = 0.0
        self.star_info_ut = 0.0
        self.table_snr_ut = 0.0
        self.table_ccf_ut = 0.0
        self.table_ccf_saved_ut = 0.0
        self.lbl_ut = 0.0
        self.dbd_ut = 0.0
        self.lbl_iter_ut = 0.0
        self.wbw_ut = 0.0

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
            pickle_dump(ccf_summary, open(self.directory + "Analyse_ccf.p", "wb"))

        if not os.path.exists(self.directory + "Analyse_material.p"):
            file = pd.read_pickle(glob.glob(self.directory + "RASSI*.p")[0])
            wave = file["wave"]
            dico_material = {
                "wave": wave,
                "correction_factor": np.ones(len(wave)),
                "reference_spectrum": np.ones(len(wave)),
                "color_template": np.ones(len(wave)),
                "blaze_correction": np.ones(len(wave)),
                "rejected": np.zeros(len(wave)),
            }
            dico_material = pd.DataFrame(dico_material)
            pickle_dump(dico_material, open(self.directory + "Analyse_material.p", "wb"))

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

        if not os.path.exists(self.directory + "/CONTINUUM"):
            os.system("mkdir " + self.directory + "/CONTINUUM")

        if not os.path.exists(self.directory + "/FLUX"):
            os.system("mkdir " + self.directory + "/FLUX")

        if not os.path.exists(self.dir_root + "IMAGES/"):
            os.system("mkdir " + self.dir_root + "IMAGES/")

        if not os.path.exists(self.dir_root + "CCF_MASK/"):
            os.system("mkdir " + self.dir_root + "CCF_MASK/")

        if not os.path.exists(self.dir_root + "PCA/"):
            os.system("mkdir " + self.dir_root + "PCA/")

        if not os.path.exists(self.dir_root + "REDUCTION_INFO/"):
            os.system("mkdir " + self.dir_root + "REDUCTION_INFO/")

        if not os.path.exists(self.dir_root + "REDUCTION_INFO/Info_reduction.p"):
            info = {}
            pickle_dump(info, open(self.dir_root + "REDUCTION_INFO/Info_reduction.p", "wb"))

        if not os.path.exists(self.dir_root + "STAR_INFO/"):
            os.system("mkdir " + self.dir_root + "STAR_INFO/")

        if not os.path.exists(self.dir_root + "STAR_INFO/Stellar_info_" + self.starname + ".p"):
            dico: StarInfo = {
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

            pickle_dump(
                dico,
                open(
                    self.dir_root + "STAR_INFO/Stellar_info_" + self.starname + ".p",
                    "wb",
                ),
            )

        self.import_star_info()
        sp = self.star_info["Sp_type"]["fixed"][0]
        self.mask_harps = ["G2", "K5", "M2"][int((sp == "K") | (sp == "M")) + int(sp == "M")]
        if "YARARA" in self.star_info["FWHM"]:
            self.fwhm = self.star_info["FWHM"]["YARARA"]
        else:
            self.fwhm = self.star_info["FWHM"]["fixed"]

        if "YARARA" in self.star_info["Contrast"]:
            self.contrast = self.star_info["Contrast"]["YARARA"]
        else:
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

        #: warning, indexing convention has changed during YARARA lifetime

        # this is assigned by import_dico_chain
        self.dico_chain: Any = None

        # this is assigned by yarara_ccf
        self.ccf_timeseries: Any = None
        self.all_ccf_saved: Dict[
            Any, Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        ] = {}
        self.svrad_phot: Any = None
        self.ccf_rv: Any = None
        self.ccf_centers: Any = None
        self.ccf_contrast: Any = None
        self.ccf_depth: Any = None
        self.ccf_fwhm: Any = None
        self.ccf_vspan: Any = None
        self.ccf_ew: Any = None
        self.ccf_rv_shift: Any = None

        # this is assigned by yarara_telluric
        self.berv_offset: Any = None

        # this is assigned by yarara_correct_frog
        self.stitching: Any = None
        self.stitching_extracted: Any = None
        self.ghost_a: Any = None
        self.ghost_a_extracted: Any = None
        self.ghost_b: Any = None
        self.ghost_b_extracted: Any = None
        self.thar: Any = None
        self.thar_extracted: Any = None
        self.contam: Any = None
        self.contam_extracted: Any = None
        self.vec_pca_stitching: Any = None
        self.vec_pca_ghost_a: Any = None
        self.vec_pca_ghost_b: Any = None
        self.vec_pca_thar: Any = None
        self.vec_pca_contam: Any = None

        # this is assigned by yarara_correct_pattern
        self.fft_output: Any = None
        self.index_corrected_pattern_red: Any = None
        self.index_corrected_pattern_blue: Any = None

        # this is assigned by yarara_produce_mask_frog
        self.stitching_zones: Any = None

        # this is assigned by get_orders
        self.orders: Any = None

        # this is assigned by get_pixels
        self.pixels: Any = None

        # this is assigned by activity_index
        self.all_kernels: Any = None
        self.all_proxies: Any = None
        self.all_proxies_name: Any = None
        self.ca1: Any = None
        self.ca2k: Any = None
        self.ca2h: Any = None
        self.ca2: Any = None
        self.rhk: Any = None
        self.mg1: Any = None
        self.mga: Any = None
        self.mgb: Any = None
        self.mgc: Any = None
        self.nad: Any = None
        self.nad1: Any = None
        self.nad2: Any = None
        self.ha: Any = None
        self.hb: Any = None
        self.hc: Any = None
        self.hd: Any = None
        self.heps: Any = None
        self.hed3: Any = None

        # assigned by yarara_map
        self.map: Tuple[NDArray[np.float64], NDArray[np.float64]] = None  # type: ignore

        # assigned by yarara_median_master
        self.wave: NDArray[np.float64] = None  # type: ignore

        # assigned by median_master
        self.reference: Tuple[NDArray[np.float64], NDArray[np.float64]] = None  # type: ignore

        # assigned by suppress_time_spectra
        self.table_snr: Dict[str, Any] = None  # type: ignore

        # from yarara_check_fwhm
        self.warning_rv_borders: bool = None  # type: ignore
