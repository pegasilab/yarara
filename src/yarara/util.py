"""
This modules does XXX
"""
import numpy as np
from colorama import Fore
from scipy.interpolate import interp1d
from tqdm import tqdm


def print_iter(verbose):
    if verbose == -1:
        print(
            Fore.BLUE
            + " ==============================================================================\n [INFO] Extracting data with RASSINE...\n ==============================================================================\n"
            + Fore.RESET
        )
    elif verbose == 0:
        print(
            Fore.BLUE
            + " ==============================================================================\n [INFO] Preprocessing data with RASSINE...\n ==============================================================================\n"
            + Fore.RESET
        )
    elif verbose == 1:
        print(
            Fore.GREEN
            + " ==============================================================================\n [INFO] First iteration is beginning...\n ==============================================================================\n"
            + Fore.RESET
        )
    elif verbose == 2:
        print(
            Fore.YELLOW
            + " ==============================================================================\n [INFO] Second iteration is beginning...\n ==============================================================================\n"
            + Fore.RESET
        )
    elif verbose == 42:
        print(
            Fore.YELLOW
            + " ==============================================================================\n [INFO] Merging is beginning...\n ==============================================================================\n"
            + Fore.RESET
        )
    else:
        hours = verbose // 3600 % 24
        minutes = verbose // 60 % 60
        seconds = verbose % 60
        print(
            Fore.RED
            + " ==============================================================================\n [INFO] Intermediate time : %.0fh %.0fm %.0fs \n ==============================================================================\n"
            % (hours, minutes, seconds)
            + Fore.RESET
        )


def yarara_artefact_suppressed(old_continuum, new_continuum, larger_than=50, lower_than=-50):
    ratio = (new_continuum / old_continuum - 1) * 100
    mask = (ratio > larger_than) | (ratio < lower_than)
    return mask


def get_phase(array, period):
    new_array = np.sort((array % period))
    j0 = np.min(new_array) + (period - np.max(new_array))
    diff = np.diff(new_array)
    if np.max(diff) > j0:
        return 0.5 * (new_array[np.argmax(diff)] + new_array[np.argmax(diff) + 1])
    else:
        return 0


def my_ruler(mini, maxi, dmini, dmaxi):
    """make a list from mini to maxi with initial step dmini linearly growing to dmaxi"""
    m = (dmaxi - dmini) / (maxi - mini)
    p = dmini - m * mini

    a = [mini]
    b = mini
    while b < maxi:
        b = a[-1] + (p + m * a[-1])
        a.append(b)
    a = np.array(a)
    a[-1] = maxi
    return a


# util
def map_rnr(array, val_max=None, n=2):
    """val_max must be strictly larger than all number in the array, n smaller than 10"""
    if type(array) != np.ndarray:
        array = np.hstack([array])

    if val_max is not None:
        if sum(array > val_max):
            print("The array cannot be higher than %.s" % (str(val_max)))

        # sort = np.argsort(abs(array))[::-1]
        # array = array[sort]
        array = (array / val_max).astype("str")
        min_len = np.max([len(k.split(".")[-1]) for k in array])
        array = np.array([k.split(".")[-1] for k in array])
        array = np.array([k + "0" * (min_len - len(k)) for k in array])
        if len(array) < n:
            array = np.hstack([array, ["0" * min_len] * (n - len(array))])

        new = ""
        for k in range(min_len):
            for l in range(len(array)):
                new += array[l][k]

        concat = str(n) + str(val_max) + "." + new

        return np.array([concat]).astype("float64")
    else:
        decoded = []
        for i in range(len(array)):
            string = str(array[i])
            code, num = string.split(".")
            n = int(code[0])
            val_max = np.float(code[1:])
            vec = []
            for k in range(n):
                vec.append("0." + num[k::n])
            vec = np.array(vec).astype("float")
            vec *= val_max
            # vec = np.sort(vec)
            decoded.append(vec)
        decoded = np.array(decoded)
        return decoded


# util
def flux_norm_std(flux, flux_std, continuum, continuum_std):
    flux_norm = flux / continuum
    flux_norm_std = np.sqrt(
        (flux_std / continuum) ** 2 + (flux * continuum_std / continuum**2) ** 2
    )
    mask = flux_norm_std > flux_norm
    flux_norm_std[mask] = abs(
        flux_norm[mask]
    )  # impossible to get larger error than the point value
    return flux_norm, flux_norm_std


# util
def ccf(wave, spec1, spec2, extended=1500, rv_range=45, oversampling=10, spec1_std=None):
    "CCF for a equidistant grid in log wavelength spec1 = spectrum, spec2 =  binary mask"
    dwave = np.median(np.diff(wave))

    if spec1_std is None:
        spec1_std = np.zeros(np.shape(spec1))

    if len(np.shape(spec1)) == 1:
        spec1 = spec1[:, np.newaxis].T
    if len(np.shape(spec1_std)) == 1:
        spec1_std = spec1_std[:, np.newaxis].T
    # spec1 = np.hstack([np.ones(extended),spec1,np.ones(extended)])

    spec1 = np.hstack([np.ones((len(spec1), extended)), spec1, np.ones((len(spec1), extended))])
    spec2 = np.hstack([np.zeros(extended), spec2, np.zeros(extended)])
    spec1_std = np.hstack(
        [
            np.zeros((len(spec1_std), extended)),
            spec1_std,
            np.zeros((len(spec1_std), extended)),
        ]
    )
    wave = np.hstack(
        [
            np.arange(-extended * dwave + wave.min(), wave.min(), dwave),
            wave,
            np.arange(wave.max() + dwave, (extended + 1) * dwave + wave.max(), dwave),
        ]
    )
    shift = np.linspace(0, dwave, oversampling + 1)[:-1]
    shift_save = []
    sum_spec = np.nansum(spec2)
    convolution = []
    convolution_std = []

    rv_max = int(np.log10((rv_range / 299.792e3) + 1) / dwave)
    for j in tqdm(shift):
        new_spec = interp1d(
            wave + j, spec2, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )(wave)
        for k in np.arange(-rv_max, rv_max + 1, 1):
            new_spec2 = np.hstack([new_spec[-k:], new_spec[:-k]])
            convolution.append(np.nansum(new_spec2 * spec1, axis=1) / sum_spec)
            convolution_std.append(
                np.sqrt(np.abs(np.nansum(new_spec2 * spec1_std**2, axis=1))) / sum_spec
            )
            shift_save.append(j + k * dwave)
    shift_save = np.array(shift_save)
    sorting = np.argsort(shift_save)
    return (
        (299.792e6 * 10 ** shift_save[sorting]) - 299.792e6,
        np.array(convolution)[sorting],
        np.array(convolution_std)[sorting],
    )


# util
def ratio_line(l1, l2, grid, spectrei, continuum, window=3):
    """index  of the  grid element ofline  1,2 plus the  grid the spectrumand the continuum"""

    subgrid1 = grid[l1 - window : l1 + window + 1]
    subgrid2 = grid[l2 - window : l2 + window + 1]
    subspectre1 = spectrei[l1 - window : l1 + window + 1]
    subspectre2 = spectrei[l2 - window : l2 + window + 1]
    subcont1 = continuum[l1 - window : l1 + window + 1]
    subcont2 = continuum[l2 - window : l2 + window + 1]

    coeff = np.polyfit(subgrid1 - np.mean(subgrid1), subspectre1 / subcont1, 2)
    coeff2 = np.polyfit(subgrid2 - np.mean(subgrid2), subspectre2 / subcont2, 2)
    d1 = 1 - np.polyval(coeff, -coeff[1] * 0.5 / coeff[0])
    d2 = 1 - np.polyval(coeff2, -coeff2[1] * 0.5 / coeff2[0])

    mini1 = np.argmin(subspectre1)
    mini2 = np.argmin(subspectre2)

    std_d1 = (
        np.sqrt(subspectre1[mini1] * (1 + (subspectre1[mini1] / subcont1[mini1]) ** 2))
        / subcont1[mini1]
    )
    std_d2 = (
        np.sqrt(subspectre2[mini2] * (1 + (subspectre2[mini2] / subcont2[mini2]) ** 2))
        / subcont2[mini2]
    )

    l3, l4 = np.min([l1, l2]), np.max([l1, l2])
    std_cont = 1 / np.sqrt(np.percentile(spectrei[l3 - 8 * window : l4 + 8 * window], 95))

    return (
        d1,
        d2,
        std_d1,
        std_d2,
        d1 / d2,
        std_cont * np.sqrt(2 + (1.0 / (1 - d1)) ** 2 + (1.0 / (1 - d2)) ** 2),
        np.sqrt((std_d1 / d2) ** 2 + (std_d2 * d1 / d2**2) ** 2),
    )


def print_box(sentence):
    print("\n")
    print("L" * len(sentence))
    print(sentence)
    print("T" * len(sentence))
    print("\n")


def doppler_r(lamb, v):
    """Relativistic Doppler. Take (wavelength, velocity in [m/s]) and return lambda observed and lambda source"""
    c = 299.792e6
    button = False
    factor = np.sqrt((1 + v / c) / (1 - v / c))
    if type(factor) != np.ndarray:
        button = True
        factor = np.array([factor])
    lambo = lamb * factor[:, np.newaxis]
    lambs = lamb * (factor ** (-1))[:, np.newaxis]
    if button:
        return lambo[0], lambs[0]
    else:
        return lambo, lambs
