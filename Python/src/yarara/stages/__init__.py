from typing import Literal, Optional, Union

import matplotlib.pylab as plt
import numpy as np

from yarara.iofun import pickle_dump

from ..sts import spec_time_series


def load_and_adapt_input_data(sts: spec_time_series) -> None:
    # ADAPT INPUT DATA
    sts.yarara_analyse_summary(rm_old=True)
    sts.yarara_add_step_dico(sub_dico="matching_diff", step=0, sub_dico_used="matching_anchors")
    sts.yarara_exploding_pickle()


def matching_cosmics(sts: spec_time_series, reference: Optional[str], close_figure: bool) -> None:
    # needs mask_telluric_telluric.txt, type MaskCCF
    sub_dico = "matching_diff"

    sts.yarara_correct_cosmics(sub_dico=sub_dico, k_sigma=5)

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


def matching_activity(
    sts: spec_time_series,
    reference: Optional[str],
    ref: Union[int, Literal["snr"], Literal["median"], Literal["master"], Literal["zeros"]],
    close_figure: bool,
    input_dico: str,
) -> None:
    # CORRECT ACTIVITY WITH PROXIES + CCF MOMENT

    # ACTIVITY
    print("\n Compute activity proxy")
    sts.yarara_activity_index(
        sub_dico="matching_pca",
    )

    sts.yarara_ccf(
        mask=sts.read_ccf_mask(sts.mask_harps),
        mask_name=sts.mask_harps,
        plot=True,
        sub_dico=input_dico,
        ccf_oversampling=1,
        rv_range=None,
    )

    proxy = ["Kernel_CaII", "CaII"][reference is None]  # sts.yarara_determine_optimal_Sindex()
    print("\n Optimal proxy of activity : %s" % (proxy))

    sts.yarara_correct_activity(
        sub_dico=input_dico,
        proxy_corr=[proxy, "ccf_fwhm", "ccf_contrast"],
        smooth_corr=1,
        reference=ref,
    )

    if close_figure:
        plt.close("all")


def matching_ghost_a(
    sts: spec_time_series,
    frog_file: str,
    ref: Union[int, Literal["snr"], Literal["median"], Literal["master"], Literal["zeros"]],
    close_figure: bool,
) -> None:
    # Material/Ghost_{instrument}.p type InstrumentGhost

    # CORRECT FROG
    sts.dico_actif = "matching_activity"

    sts.yarara_correct_borders_pxl(
        pixels_to_reject=np.hstack([np.arange(1, 6), np.arange(4092, 4097)])
    )
    sts.yarara_produce_mask_frog(frog_file=frog_file)

    sts.yarara_correct_frog(
        correction="ghost_a",
        sub_dico="matching_activity",
        reference=ref,
        threshold_contam=1,
        equal_weight=False,
        pca_comp_kept_=3,
        complete_analysis=False,
    )

    if close_figure:
        plt.close("all")


def matching_ghost_b(
    sts: spec_time_series,
    ref: Union[int, Literal["snr"], Literal["median"], Literal["master"], Literal["zeros"]],
    close_figure: bool,
) -> None:
    # CORRECT GHOST B CONTAM
    sts.yarara_correct_frog(
        correction="ghost_b",
        sub_dico="matching_ghost_a",
        reference=ref,
        wave_max_train=4100,
        pca_comp_kept_=2,
        threshold_contam=1,
        equal_weight=True,
        complete_analysis=False,
    )

    if close_figure:
        plt.close("all")


def matching_fourier(
    sts: spec_time_series,
    reference: Optional[str],
    close_figure: bool,
) -> None:
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
        reference="master",
        correct_blue=True,
        correct_red=True,
        width_range=(2.5, 3.5),
        jdb_range=(0, 100000),
    )  # all the time a pattern on HARPN

    if close_figure:
        plt.close("all")


def matching_smooth(
    sts: spec_time_series,
    reference: Optional[str],
    ref: Union[int, Literal["snr"], Literal["median"], Literal["master"], Literal["zeros"]],
    close_figure: bool,
) -> None:
    # CORRECT CONTINUUM

    sts.yarara_correct_smooth(sub_dico="matching_fourier", reference=ref, window_ang=5)

    if reference is None:
        sts.yarara_retropropagation_correction(
            correction_map="matching_smooth",
            sub_dico="matching_cosmics",
        )

    if close_figure:
        plt.close("all")
    pass


def matching_mad(sts: spec_time_series, reference: Optional[str], close_figure: bool) -> None:
    # CORRECT MAD
    counter_mad_removed = sts.yarara_correct_mad(
        sub_dico="matching_smooth",
        k_sigma=2,
        k_mad=2,
        n_iter=1,
        ext=["0", "1"][int(reference == "master")],
    )
    # spectrum_removed = counter_mad_removed > [0.15, 0.15][int(reference == "master")]
    spectrum_removed = counter_mad_removed > 0.15
    sts.suppress_time_spectra(mask=spectrum_removed)

    sts.yarara_ccf(
        mask=sts.read_ccf_mask(sts.mask_harps),
        mask_name=sts.mask_harps,
        plot=True,
        sub_dico="matching_mad",
        ccf_oversampling=1,
        rv_range=None,
    )

    if close_figure:
        plt.close("all")


def stellar_atmos1(sts: spec_time_series, reference: Optional[str], close_figure: bool) -> None:
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
        mask=sts.read_ccf_mask(sts.mask_harps),
        mask_name=sts.mask_harps,
        plot=True,
        sub_dico="matching_pca",
        ccf_oversampling=1,
        rv_range=None,
    )
    sts.yarara_ccf(
        mask=sts.read_ccf_mask(sts.mask_harps),
        mask_name=sts.mask_harps,
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


def matching_brute(sts: spec_time_series, reference: Optional[str], close_figure: bool) -> None:
    # CORRECT BRUTE
    sts.yarara_correct_brute(
        sub_dico="matching_mad",
        min_length=3,
        k_sigma=2,
        percent_removed=10,  # None if sphinx
        borders_pxl=True,
    )

    if close_figure:
        plt.close("all")
