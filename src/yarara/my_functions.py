"""
Legacy module, kept to be able to cut'n'paste code
"""


# reexport previous stuff
from . import pickle_protocol_version
from .constants import (
    G_cst,
    Mass_earth,
    Mass_jupiter,
    Mass_sun,
    au_m,
    c_lum,
    h_planck,
    k_boltz,
    radius_earth,
    radius_sun,
)
from .io import pickle_dump
from .mathfun import gaussian, lorentzian, parabole, sinus, voigt
from .plots import auto_axis, my_colormesh, plot_color_box
from .stats import (
    IQ,
    clustering,
    combination,
    find_nearest,
    flat_clustering,
    identify_nearest,
    local_max,
    mad,
    match_nearest,
    merge_borders,
    rm_outliers,
    smooth,
    smooth2d,
)
from .util import (
    ccf,
    doppler_r,
    flux_norm_std,
    get_phase,
    map_rnr,
    my_ruler,
    print_box,
    ratio_line,
)
