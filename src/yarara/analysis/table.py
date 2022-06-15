"""
This modules does XXX
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from wpca import EMPCA, PCA, WPCA

from ..util import assert_never


def rm_diagonal(array, k=0, put_nan=False):
    diag = np.diag(np.ones(len(array) - abs(k)), k=k)
    if put_nan:
        diag[diag == 1] = np.nan
    mat = np.ones(np.shape(array)) - diag
    return array * mat


def rm_sym_diagonal(array2, k, put_nan=False):
    array = array2.copy()
    for j in range(abs(k) + 1):
        if not j:
            array = rm_diagonal(array, k=j, put_nan=put_nan)
            array = rm_diagonal(array, k=-j, put_nan=put_nan)
        else:
            array = rm_diagonal(array, k=j, put_nan=put_nan)
    return array


class table(object):
    """this classe has been establish with pandas DataFrame"""

    def __init__(self, array: NDArray[np.float64]) -> None:
        self.table: NDArray[np.float64] = array
        self.dim: Tuple[int, ...] = np.shape(array)

        self.rms: Any = None

        self.components: Any = None
        self.vec_fitted: Any = None
        self.vec: Any = None
        self.phi_components: Any = None
        self.zscore_components: Any = None
        self.var: Any = None
        self.var_ratio: Any = None
        self.wpca_model: Any = None

    def r_matrix(
        self,
        name=None,
        absolute=True,
        cmap="seismic",
        vmin=None,
        vmax=None,
        light_plot=False,
        angle=45,
        Plot=True,
        rm_diagonal=False,
        paper_plot=False,
        unit=1,
        deg_detrend=0,
        vec_x=None,
    ):

        if deg_detrend:
            if vec_x is None:
                vec_x = np.arange(len(self.table))
            vec_x -= np.nanmedian(vec_x)

            self.fit_base(np.array([vec_x**i for i in range(deg_detrend + 1)]))
            matrix = self.vec_residues
        else:
            matrix = self.table
        if type(matrix) == pd.core.frame.DataFrame:
            if name is None:
                name = list(matrix.columns)
            matrix = matrix[name]
            matrix = np.array(matrix)

            self.name_liste = name
        else:
            if (len(matrix.T) > 20) & (len(matrix) < 20):
                matrix = matrix.T

        if name is None:
            name = np.arange(len(matrix.T))

        r_matrix = np.array(pd.DataFrame(matrix, columns=name, dtype="float").corr())

        if absolute:
            self.matrix_corr = abs(r_matrix.copy())
        else:
            self.matrix_corr = r_matrix.copy()

        if rm_diagonal:
            self.matrix_corr = rm_sym_diagonal(self.matrix_corr, k=0, put_nan=True)

        if Plot:
            if len(r_matrix) > 20:
                light_plot = True

            if absolute:
                cmap = "Reds"
                r_matrix = abs(r_matrix)

            if vmin is None:
                vmin = -1 + int(absolute)

            if vmax is None:
                vmax = 1

            plt.imshow(r_matrix, vmin=vmin, vmax=vmax, cmap=cmap)
            ax = plt.colorbar(pad=[0.05, 0][int(light_plot)])
            ax.ax.set_ylabel(r"$\mathcal{R}_{pearson}$", fontsize=[15, 20][int(paper_plot)])
            ax.ax.set_yticklabels(
                np.round(np.arange(0, 1.1, 0.2), 2), fontsize=[15, 20][int(paper_plot)]
            )
            plt.xticks(
                ticks=np.arange(len(r_matrix)),
                labels=list(name),
                rotation=[angle, 90][int(paper_plot)],
                fontsize=[14, 18][int(paper_plot)],
            )
            plt.yticks(
                ticks=np.arange(len(r_matrix)),
                labels=list(name),
                fontsize=[14, 18][int(paper_plot)],
            )
            plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.20)

            for i in range(len(r_matrix)):
                for j in range(len(r_matrix)):
                    if not light_plot:
                        if unit == 1:
                            plt.annotate(
                                "%.2f" % (r_matrix[i, j]),
                                (i, j),
                                fontsize=[13, 16][int(paper_plot)],
                                va="center",
                                ha="center",
                            )
                        else:
                            plt.annotate(
                                "%.0f" % (r_matrix[i, j] * 100),
                                (i, j),
                                fontsize=[13, 16][int(paper_plot)],
                                va="center",
                                ha="center",
                            )
                    else:
                        if (abs(r_matrix[i, j]) > 0.4) & (r_matrix[i, j] != 1):
                            plt.annotate(
                                "%.0f" % (abs(r_matrix[i, j]) * 100),
                                (i - 0.25, j),
                                fontsize=8,
                                va="center",
                            )

    def rms_w(self, weights: np.ndarray, axis: Union[Literal[0], Literal[1]] = 1) -> None:
        average = np.average(self.table, weights=weights, axis=axis)

        data_recentered: np.ndarray = np.array([])
        if axis == 1:
            data_recentered = self.table - average[:, np.newaxis]
        elif axis == 0:
            data_recentered = (self.table.T - average[:, np.newaxis]).T
        else:
            assert_never(axis)

        variance = np.average((data_recentered) ** 2, weights=weights, axis=axis)
        self.rms = np.sqrt(variance)

    def WPCA(
        self,
        pca_type: Union[Literal["pca"], Literal["wpca"], Literal["empca"]],
        weight: Optional[np.ndarray] = None,
        comp_max: Optional[int] = None,
    ) -> None:
        """from https://github.com/jakevdp/wpca/blob/master/WPCA-Example.ipynb
        enter which pca do yo want either 'pca', 'wpca' or 'empca'
        empca slower than wpca
        """

        # self.replace_outliers(m=m, kind=kind)

        Signal = self.table

        R = Signal.copy()

        if pca_type == "pca":
            ThisPCA = PCA
        elif pca_type == "wpca":
            ThisPCA = WPCA
        elif pca_type == "empca":
            ThisPCA = EMPCA
        else:
            assert_never(pca_type)

        if (weight is None) or (pca_type == "pca"):
            kwds = {}
        else:
            kwds = {"weights": np.sqrt(weight)}  # defined as 1/sigma

        if comp_max == None:
            comp_max = len(R.T)

        # Compute the PCA vectors & variance
        pca = ThisPCA(n_components=comp_max).fit(R, **kwds)

        # Reconstruct the data using the PCA model
        self.components = ThisPCA(n_components=comp_max).fit_transform(R, **kwds).T
        self.vec_fitted = ThisPCA(n_components=comp_max).fit_reconstruct(R, **kwds)
        self.vec = pca.components_.T

        norm = np.sign(np.nanmedian(self.vec, axis=0))
        self.vec = self.vec / norm
        self.components = self.components * norm[:, np.newaxis]

        # self.s_values = pca.singular_values_

        # components = abs_coeff*abs(self.components)+(1-abs_coeff)*self.components

        self.phi_components = np.sum(self.components < 0, axis=1) / len(self.components.T)
        self.zscore_components = np.mean(self.components, axis=1) / np.std(self.components, axis=1)

        self.var = pca.explained_variance_
        self.var_ratio = pca.explained_variance_ratio_

        self.wpca_model = pca

    def fit_base(
        self, base_vec: np.ndarray, weight: Optional[np.ndarray] = None, num_sim: int = 1
    ) -> None:
        """weights define as 1/sigma**2 self.table = MxT, base_vec = NxT, N the number of basis element"""

        if np.shape(base_vec)[1] != np.shape(self.table)[0]:
            base_vec = np.array(
                [
                    base_vec[i] * np.ones(np.shape(self.table)[0])[:, np.newaxis]
                    for i in range(len(base_vec))
                ]
            )

        if (np.shape(self.table)[0] == np.shape(self.table)[1]) & (len(np.shape(self.table)) == 2):
            base_vec = np.array(
                [
                    base_vec[i] * np.ones(np.shape(self.table)[0])[:, np.newaxis]
                    for i in range(len(base_vec))
                ]
            )

        if weight is None:
            weight = np.ones(np.shape(self.table))

        coeff = np.array(
            [
                np.linalg.lstsq(
                    base_vec[:, i, :].T * np.sqrt(weight[i])[:, np.newaxis],
                    (self.table[i]) * np.sqrt(weight[i]),
                    rcond=None,
                )[0]
                for i in range(len(self.table))
            ]
        )

        vec_bootstrap = np.array(
            [
                (self.table[i] + np.random.randn(num_sim, len(weight[i])) / np.sqrt(weight[i]))
                for i in range(len(self.table))
            ]
        )

        coeff_test = np.array(
            [
                np.linalg.lstsq(
                    base_vec[:, i, :].T * np.sqrt(weight[i])[:, np.newaxis],
                    (vec_bootstrap[i] * np.sqrt(weight[i])).T,
                    rcond=None,
                )[0]
                for i in range(len(self.table))
            ]
        )
        coeff_mean = np.mean(coeff_test, axis=2)
        coeff_std = np.std(coeff_test, axis=2)

        self.vec_resampling = vec_bootstrap
        self.coeff_resampling = coeff_test
        self.coeff_mean = coeff_mean
        self.coeff_std = coeff_std

        vec_fitted = np.array(
            [np.sum(coeff[j] * base_vec[:, j, :].T, axis=1) for j in range(len(self.table))]
        )
        all_vec_fitted = np.array([coeff[j] * base_vec[:, j, :].T for j in range(len(self.table))])
        self.coeff_fitted = coeff
        self.vec_fitted = vec_fitted
        self.all_vec_fitted = all_vec_fitted
        vec_residues = self.table - vec_fitted
        vec_residues[self.table == 0] = 0
        self.vec_residues = vec_residues
        self.weights = weight
        coeff_fitted_std = []
        for i in range(len(self.table)):
            X = np.mat(base_vec[:, i, :]).T
            XX = np.linalg.inv(X.T * X)
            V = np.mat(np.diag(weight[i] ** -1))
            cov = XX * X.T * V * X * XX
            coeff_fitted_std.append(np.sqrt(np.diag(cov)))

        self.coeff_fitted_std = np.array(coeff_fitted_std)

        self.chi2 = np.sum(vec_residues**2 * weight, axis=1)
        coeff_pos = coeff * np.sign(np.median(coeff, axis=0))
        mean_coeff = np.mean(coeff_pos, axis=0)
        if sum(mean_coeff != 0):
            epsilon = 1e-6 * np.min(abs(mean_coeff[mean_coeff != 0]))
        else:
            epsilon = 1e-6
        self.zscore_base = mean_coeff / (np.std(coeff_pos, axis=0) + epsilon)
        self.phi_base = np.sum(coeff_pos < 0, axis=0) / len(coeff_pos)
