from typing import Optional

import matplotlib.pylab as plt

from ..sts import spec_time_series


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
