from typing import Optional

import matplotlib.pylab as plt
import numpy as np

from ..sts import spec_time_series
from . import todo


def load_and_adapt_input_data(sts: spec_time_series) -> None:
    # ADAPT INPUT DATA
    sts.yarara_analyse_summary(rm_old=True)
    sts.yarara_add_step_dico("matching_diff", 0, sub_dico_used="matching_anchors")
    sts.yarara_exploding_pickle()


def matching_cosmics(sts: spec_time_series, reference: Optional[str], close_figure: bool) -> None:
    # needs mask_telluric_telluric.txt, type MaskCCF
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


def matching_telluric(sts: spec_time_series, reference: Optional[str], close_figure: bool) -> None:
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


def matching_oxygen(sts: spec_time_series, reference: Optional[str], close_figure: bool) -> None:
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


def matching_contam(
    sts: spec_time_series, reference: Optional[str], frog_file: str, close_figure: bool
) -> None:

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

        sts.yarara_produce_mask_contam(frog_file)
        # sts.yarara_produce_mask_telluric(telluric_tresh=0.001)

        sts.yarara_correct_frog(
            correction="contam",
            sub_dico="matching_oxygen",
            reference="master",
            wave_min=5710.0,
            wave_max=5840.0,
            wave_max_train=7000.0,
            pca_comp_kept_=3,
            algo_pca="pca",
            threshold_contam=0.5,
            equal_weight=True,
            complete_analysis=False,
            rcorr_min=0,
        )

        if close_figure:
            plt.close("all")


def matching_pca(sts: spec_time_series, reference: Optional[str], close_figure: bool) -> None:
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
        rvs_=np.array(sts.table["berv"]) * 1000.0,
    )

    sts.yarara_telluric(sub_dico="matching_cosmics", reference="norm", ratio=True)

    sts.yarara_master_ccf(
        sub_dico="matching_cosmics",
        name_ext="_telluric",
        rvs_=np.array(sts.table["berv"]) * 1000.0,
    )

    if close_figure:
        plt.close("all")
