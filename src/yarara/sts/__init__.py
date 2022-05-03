from __future__ import annotations

import glob as glob
import os
from typing import Any, Dict, TypedDict

import numpy as np
import pandas as pd

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
    from .ccf import yarara_ccf, yarara_master_ccf
    from .extract import yarara_get_berv_value, yarara_get_orders, yarara_get_pixels
    from .instrument import (
        yarara_correct_borders_pxl,
        yarara_correct_frog,
        yarara_correct_pattern,
        yarara_produce_mask_contam,
        yarara_produce_mask_frog,
    )
    from .io import (
        import_dico_tree,
        import_info_reduction,
        import_material,
        import_rassine_output,
        import_spectrum,
        import_star_info,
        import_sts_flux,
        import_table,
        spectrum,
        supress_time_spectra,
        update_info_reduction,
        yarara_add_step_dico,
        yarara_analyse_summary,
        yarara_exploding_pickle,
        yarara_obs_info,
        yarara_star_info,
    )
    from .outliers import (
        yarara_correct_brute,
        yarara_correct_cosmics,
        yarara_correct_mad,
        yarara_correct_smooth,
    )
    from .processing import (
        uncorrect_hole,
        yarara_activity_index,
        yarara_map,
        yarara_retropropagation_correction,
    )
    from .telluric import (
        yarara_correct_oxygen,
        yarara_correct_telluric_gradient,
        yarara_correct_telluric_proxy,
        yarara_telluric,
    )
    from .util import (
        yarara_cut_spectrum,
        yarara_median_master,
        yarara_non_zero_flux,
        yarara_poissonian_noise,
    )

    def __init__(self, directory: str) -> None:
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
        self.planet: bool = False  #: If a planet has been injected
        self.sp_type = None
        self.rv_sys = None
        self.teff = None
        self.log_g = None
        self.bv = None
        self.fwhm = None
        self.wave = None
        self.infos = {}
        self.ram = []

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
        self.table: pd.DataFrame = None  # type: ignore

        self.star_info: StarInfo = {}  #: Information about the star

        #: Datetime of corresponding info read from file, to avoid reading twice
        self.table_ut = 0.0
        #: Datetime of corresponding info read from file, to avoid reading twice
        self.material_ut = 0.0
        #: Datetime of corresponding info read from file, to avoid reading twice
        self.info_reduction_ut = 0.0
        #: Datetime of corresponding info read from file, to avoid reading twice
        self.star_info_ut = 0.0
        #: Datetime of corresponding info read from file, to avoid reading twice
        self.table_snr_ut = 0.0
        #: Datetime of corresponding info read from file, to avoid reading twice
        self.table_ccf_ut = 0.0
        #: Datetime of corresponding info read from file, to avoid reading twice
        self.table_ccf_saved_ut = 0.0
        #: Datetime of corresponding info read from file, to avoid reading twice
        self.lbl_ut = 0.0
        #: Datetime of corresponding info read from file, to avoid reading twice
        self.dbd_ut = 0.0
        #: Datetime of corresponding info read from file, to avoid reading twice
        self.lbl_iter_ut = 0.0
        #: Datetime of corresponding info read from file, to avoid reading twice
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
        self.ccf_timeseries = None
