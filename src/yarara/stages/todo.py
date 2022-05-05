from typing import Optional

import matplotlib.pylab as plt

from ..sts import spec_time_series

# def matching_activity(
#     sts: spec_time_series, reference: Optional[str], close_figure: bool, input_dico: str
# ) -> None:
#     # CORRECT ACTIVITY WITH PROXIES + CCF MOMENT

#     # ACTIVITY
#     print("\n Compute activity proxy")
#     sts.yarara_activity_index(sub_dico="matching_pca", continuum="linear")

#     sts.yarara_ccf(
#         mask=sts.mask_harps,
#         plot=True,
#         sub_dico=input_dico,
#         ccf_oversampling=1,
#         rv_range=None,
#     )

#     proxy = ["Kernel_CaII", "CaII"][reference is None]  # sts.yarara_determine_optimal_Sindex()
#     print("\n Optimal proxy of activity : %s" % (proxy))

#     sts.yarara_correct_activity(
#         sub_dico=input_dico,
#         proxy_corr=[proxy, "ccf_fwhm", "ccf_contrast"],
#         smooth_corr=1,
#         reference=ref,
#     )

#     if close_figure:
#         plt.close("all")


# def matching_ghost_a(sts: spec_time_series, reference: Optional[str], close_figure: bool) -> None:
#     # Material/Ghost_{instrument}.p type InstrumentGhost

#     # CORRECT FROG
#     sts.dico_actif = "matching_activity"

#     sts.yarara_correct_borders_pxl(
#         pixels_to_reject=np.hstack([np.arange(1, 6), np.arange(4092, 4097)])
#     )
#     sts.yarara_produce_mask_frog(frog_file=root + "/Python/Material/Ghost_" + ins + ".p")

#     sts.yarara_correct_frog(
#         correction="ghost_a",
#         sub_dico="matching_activity",
#         reference=ref,
#         berv_shift_="berv",
#         threshold_contam=1,
#         equal_weight=False,
#         pca_comp_kept_=3,
#         complete_analysis=False,
#     )

#     if close_figure:
#         plt.close("all")


# def matching_ghost_b(sts: spec_time_series, reference: Optional[str], close_figure: bool) -> None:
#     # CORRECT GHOST B CONTAM
#     sts.yarara_correct_frog(
#         correction="ghost_b",
#         sub_dico="matching_ghost_a",
#         reference=reference,
#         berv_shift_="berv",
#         wave_max_train=4100,
#         pca_comp_kept_=2,
#         threshold_contam=1,
#         equal_weight=True,
#         complete_analysis=False,
#     )

#     if close_figure:
#         plt.close("all")


# def matching_fourier(sts: spec_time_series, reference: Optional[str], close_figure: bool) -> None:
#     # CORRECT INTERFERENCE PATTERN

#     # MEDIAN MASTER
#     if reference is None:
#         sts.yarara_median_master(
#             sub_dico="matching_ghost_b",
#             method="mean",  # if smt else than max, the classical weighted average of v1.0
#             jdb_range=[0, 100000, 1],
#         )

#     # CORRECT PATTERN IN FOURIER SPACE
#     sts.yarara_correct_pattern(
#         sub_dico="matching_ghost_b",
#         continuum="linear",
#         reference="master",
#         correct_blue=True,
#         correct_red=True,
#         width_range=[2.5, 3.5],
#         jdb_range=[0, 100000],
#     )  # all the time a pattern on HARPN

#     if close_figure:
#         plt.close("all")


# def matching_smooth(sts: spec_time_series, reference: Optional[str], close_figure: bool) -> None:
#     # CORRECT CONTINUUM

#     sts.yarara_correct_smooth(
#         sub_dico="matching_fourier", continuum="linear", reference=ref, window_ang=5
#     )

#     if reference is None:
#         sts.yarara_retropropagation_correction(
#             correction_map="matching_smooth",
#             sub_dico="matching_cosmics",
#             continuum="linear",
#         )

#     if close_figure:
#         plt.close("all")
#     pass


# def matching_mad(sts: spec_time_series, reference: Optional[str], close_figure: bool) -> None:
#     # CORRECT MAD
#     counter_mad_removed = sts.yarara_correct_mad(
#         sub_dico="matching_smooth",
#         continuum="linear",
#         k_sigma=2,
#         k_mad=2,
#         n_iter=1,
#         ext=["0", "1"][int(reference == "master")],
#     )

#     spectrum_removed = counter_mad_removed > [0.15, 0.15][int(reference == "master")]
#     sts.supress_time_spectra(liste=spectrum_removed)

#     sts.yarara_ccf(
#         mask=sts.mask_harps,
#         plot=True,
#         sub_dico="matching_mad",
#         ccf_oversampling=1,
#         rv_range=None,
#     )

#     if close_figure:
#         plt.close("all")
#     pass


# def stellar_atmos1(sts: spec_time_series, reference: Optional[str], close_figure: bool) -> None:
#     # TEMPERATURE + MODEL FIT

#     sts.yarara_median_master(
#         sub_dico="matching_mad",
#         method="median",  # if smt else than max, the classical weighted average of v1.0
#         suppress_telluric=False,
#         shift_spectrum=False,
#     )

#     if reference != "master":
#         sts.import_material()
#         load = sts.material

#         load["reference_spectrum_backup"] = load["reference_spectrum"].copy()
#         pickle_dump(load, open(sts.directory + "Analyse_material.p", "wb"))

#     sts.yarara_cut_spectrum(wave_min=None, wave_max=6834)

#     #    sts.yarara_stellar_atmos(sub_dico="matching_diff", reference="master", continuum="linear")

#     #    sts.yarara_correct_continuum_absorption(model=None, T=None, g=None)

#     sts.yarara_ccf(
#         mask=sts.mask_harps,
#         plot=True,
#         sub_dico="matching_pca",
#         ccf_oversampling=1,
#         rv_range=None,
#     )
#     sts.yarara_ccf(
#         mask=sts.mask_harps,
#         plot=True,
#         sub_dico="matching_mad",
#         ccf_oversampling=1,
#         rv_range=None,
#     )

#     #    sts.yarara_snr_curve()
#     #    sts.snr_statistic(version=2)
#     #    sts.yarara_kernel_caii(
#     #     contam=True,
#     #     mask="CaII_HD128621",
#     #     power_snr=None,
#     #     noise_kernel="unique",
#     #     wave_max=None,
#     #     doppler_free=False,
#     # )

#     if close_figure:
#         plt.close("all")
#     pass


# def matching_brute(sts: spec_time_series, reference: Optional[str], close_figure: bool) -> None:
#     # CORRECT BRUTE
#     sts.yarara_correct_brute(
#         sub_dico="matching_mad",
#         continuum="linear",
#         min_length=3,
#         k_sigma=2,
#         percent_removed=10,  # None if sphinx
#         ghost2=False,
#         borders_pxl=True,
#     )

#     if close_figure:
#         plt.close("all")
#     pass
