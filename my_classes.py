"""
@author: Cretignier Michael 
@university University of Geneva
"""

import os
import sys

cwd = os.getcwd()
root = "/".join(cwd.split("/")[:-1])

sys.path.append(root + "/Python/l1periodogram/l1periodogram_codes")
sys.path.append(root + "/Python/keplerian")
sys.path.append(root + "/Python/fit_ccf")

try:
    import rvmodel as rvm

    rvmodel = rvm.rvModel
    spleaf_version = "old"
except:
    from kepmodel import rv as rvm  # updated 20.07.21

    rvmodel = rvm.RvModel
    from kepmodel import tools as ktools
    from spleaf import term

    spleaf_version = "new"

import itertools
import pickle

import astropy.time as Time
import astropy.visualization.hist as astrohist
import corner

# import l1periodogram_v1 TODO: restore
# import covariance_matrices TODO: restore
# import fastlinsquare_cholesky as l1_cholesky TODO: restore
# import functions_py3 as f3 TODO: restore
import matplotlib
import matplotlib as mpl
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pylab as plt

# import my_MCMC as mym
import numpy as np
import pandas as pd
import scipy.odr as odr
import scipy.optimize as opt
import scipy.stats as stats

# import significance as l1_sig
import statsmodels.api as sm
from astropy.modeling.models import Voigt1D
from descartes import PolygonPatch
from lmfit import Model, Parameters
from matplotlib.ticker import FormatStrFormatter
from matplotlib.widgets import Button, RadioButtons, Slider
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, interp1d
from scipy.optimize import curve_fit, minimize
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde, kde
from scipy.stats import norm as norm_gauss
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from sklearn import linear_model, metrics
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.weightstats import DescrStatsW
from tqdm import tqdm
from wpca import EMPCA
from wpca import PCA as PCA_BACK
from wpca import WPCA
from xgboost import XGBClassifier, XGBRegressor, plot_importance

import my_functions as myf
from my_functions import rm_outliers as rm_out

plt_version = float("".join(matplotlib.__version__.split(".")))

# import fit3 as fit3


# import statsmodels.formula.api as sm

# exec(open('import_libraries.py').read())


class identification(object):
    """Make an identification between two matrices and give the position of the elements of matrice 1 in matrice 2."""

    def __init__(self, table1, table2, list1, list2):
        self.liste1 = list1
        self.liste2 = list2
        try:
            np.shape(table1)[1]
            self.table1 = table1
        except IndexError:
            self.table1 = table1[:, np.newaxis]
        try:
            np.shape(table2)[1]
            self.table2 = table2
        except IndexError:
            self.table2 = table2[:, np.newaxis]
        if len(list1) != len(list2):
            print("List1 and List2 have not the same number of element")

    def switch(self):
        self.table1, self.table2 = self.table2, self.table1

    def Identification(self):
        Position = []
        for j in self.table1[:, self.liste1]:
            Match = np.where(
                np.product(self.table2[:, self.liste2] == j, axis=1) == True
            )[0]
            if len(Match) == 1:
                Position.append(Match)

            else:
                Position.append(-99.9)
        self.position = np.hstack(np.array(Position))
        mask = self.position == -99.9
        if np.sum(mask) != 0:
            self.positiont1 = np.arange(len(self.position))[~mask]
            self.positiont2 = self.position[~mask]
            self.positiont1 = self.positiont1.astype("int")
            self.positiont2 = self.positiont2.astype("int")

    def reduction_overlapping(self, replace=False):
        copy = identification(self.table1, self.table2, self.liste1, self.liste2)
        copy.Identification()
        delete_liste = np.where(copy.position == -99.9)[0]
        matrice1_reduced = np.delete(copy.table1, delete_liste, axis=0)
        inversion = identification(
            copy.table2, matrice1_reduced, copy.liste2, copy.liste1
        )
        inversion.Identification()
        delete_liste = np.where(inversion.position == -99.9)[0]
        matrice2_reduced = np.delete(inversion.table1, delete_liste, axis=0)
        if replace == False:
            self.table1_reduced = matrice1_reduced
            self.table2_reduced = matrice2_reduced
        if replace == True:
            self.table1 = matrice1_reduced
            self.table2 = matrice2_reduced


class projectedTable(object):
    """Allow to project parameter in parameter space to reduce the dimension. Lists must be given as list of list : for example [[0,1],[0,3,2]] and [[2,3],[1]]. Any columns of list1 must be found in list2."""

    def __init__(self, table, list1, list2):
        self.table = table
        self.liste1 = list1
        self.liste2 = list2
        if len(np.shape(list1)) != 2:
            print("Please recheck the format of the list1")
        if len(np.shape(list2)) != 2:
            print("Please recheck the format of the list2")
        if len(list1) != len(list2):
            print("List1 and List2 have not the same number of element")

    def projection(self):
        Mean2 = []
        Min2 = []
        Std2 = []
        Max2 = []
        for j in range(len(self.liste1)):
            Mean = []
            Min = []
            Std = []
            Max = []
            self.unprojected_par = self.liste1[j]
            self.recorded_par = self.liste2[j]
            exclusion = float("nan")
            values = [
                np.unique(self.table[:, j]).tolist() for j in self.unprojected_par
            ]
            combination = np.array(list(itertools.product(*values)))
            for item in combination:
                position = np.where(
                    np.product((self.table[:, self.unprojected_par] == item), axis=1)
                    == 1
                )[0]
                Mean.append(
                    item.tolist()
                    + [np.mean(self.table[position, k]) for k in self.recorded_par]
                )
                Std.append(
                    item.tolist()
                    + [np.std(self.table[position, k]) for k in self.recorded_par]
                )
                try:
                    Min.append(
                        item.tolist()
                        + [np.min(self.table[position, k]) for k in self.recorded_par]
                    )
                except ValueError:
                    Min.append(item.tolist() + [exclusion for k in self.recorded_par])
                try:
                    Max.append(
                        item.tolist()
                        + [np.max(self.table[position, k]) for k in self.recorded_par]
                    )
                except ValueError:
                    Max.append(item.tolist() + [exclusion for k in self.recorded_par])
            Max2.append(np.array(Max))
            Mean2.append(np.array(Mean))
            Min2.append(np.array(Min))
            Std2.append(np.array(Std))
        self.Mean = Mean2
        self.Min = Min2
        self.Std = Std2
        self.Max = Max2


class tableXYZ(object):
    """Class to help plot of 3D or 4D dimensions. TableXYZ(x, y, z,*h) with x, y, z three vectors of the same size. h also the same size is the fourth variables which play the role of color."""

    def __init__(self, x, y, z, *h):
        self.x = x
        self.y = y
        self.z = z
        check_len = np.unique([len(x), len(y), len(z)])
        if len(h) != 0:
            self.h = h[0]
            check_len = np.unique([len(x), len(y), len(z), len(h[0])])
        if len(check_len) != 1:
            print("X,Y,Z or H have not the same size !")

    def supress_nan(self):
        mask = np.isnan(self.x) | np.isnan(self.y) | np.isnan(self.z)
        self.x = self.x[~mask]
        self.y = self.y[~mask]
        self.z = self.z[~mask]

    def ternary(self, cmap="seismic"):
        plt.plot([0, 1], [0, 0], c="k")
        plt.plot([0, 0.5], [0, np.sqrt(3) / 2], c="k")
        plt.plot([0.5, 1], [np.sqrt(3) / 2, 0], c="k")
        plt.scatter(
            0.5 * (2 * self.y + self.z) / (self.x + self.y + self.z),
            np.sqrt(3) * self.z / 2 / (self.x + self.y + self.z),
        )

    def plot3D(
        self,
        Show=False,
        fraction=1,
        xlabel="",
        ylabel="",
        zlabel="",
        color="k",
        cmap="Greys",
        marker="o",
        ax=None,
        s=20,
        alpha_p=1,
        alpha_c=0.5,
        proj=False,
        nbins=50,
        n_levels=["2d", [1, 2]],
    ):
        if ax == None:
            fig = plt.figure()
            ax = Axes3D(fig)
            self.ax = ax
        else:
            ax = ax
        mask = np.random.choice(
            np.arange(len(self.x)), int(fraction * len(self.x)), replace=False
        )
        ax.scatter(
            self.x[mask],
            self.y[mask],
            self.z[mask],
            color=color,
            marker=marker,
            s=s,
            alpha=alpha_p,
        )
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.set_zlabel(zlabel)
        if proj:
            for permu in [
                [self.x, self.y, "z", zlim[0]],
                [self.x, self.z, "y", ylim[1]],
                [self.z, self.y, "x", xlim[0]],
            ]:
                x, y, direction, lim = permu
                DX = x.max() - x.min()
                DY = y.max() - y.min()
                nbins = nbins
                k = kde.gaussian_kde(np.vstack([x, y]))
                xi1, yi1 = np.mgrid[
                    x.min() - 0.5 * DX : x.max() + 0.5 * DX : nbins * 1j,
                    y.min() - 0.5 * DY : y.max() + 0.5 * DY : nbins * 1j,
                ]
                z = k(np.vstack([xi1.flatten(), yi1.flatten()]))

                if type(n_levels) != int:
                    if n_levels[0] == "1d":
                        niveaux = np.sort(
                            [
                                np.hstack(z)[
                                    np.hstack(z).argsort()[::-1][
                                        myf.find_nearest(
                                            np.cumsum(np.sort(np.hstack(z))[::-1]),
                                            (1 - 2 * (1 - norm_gauss.cdf(j)))
                                            * np.sum(z),
                                        )
                                    ]
                                ]
                                for j in n_levels[1]
                            ]
                        )
                    elif n_levels[0] == "2d":
                        niveaux = np.sort(
                            [
                                np.hstack(z)[
                                    np.hstack(z).argsort()[::-1][
                                        myf.find_nearest(
                                            np.cumsum(np.sort(np.hstack(z))[::-1]),
                                            (1 - np.exp(-0.5 * j**2)) * np.sum(z),
                                        )[0]
                                    ]
                                ]
                                for j in n_levels[1]
                            ]
                        )
                    niveaux = np.hstack(niveaux)
                    niveaux = np.append(niveaux, 2 * z.max())
                    niveaux = np.sort(niveaux)
                    self.niveaux = niveaux
                    n_levels = niveaux
                else:
                    niveaux = None

                zi1 = z.reshape(xi1.shape)
                if direction == "y":
                    yi1, zi1 = zi1, yi1
                if direction == "x":
                    zi1, xi1 = xi1, zi1
                ax.contour(
                    xi1,
                    yi1,
                    zi1,
                    zdir=direction,
                    offset=lim,
                    levels=niveaux,
                    colors=color,
                )
                ax.contourf(
                    xi1,
                    yi1,
                    zi1,
                    zdir=direction,
                    offset=lim,
                    levels=niveaux,
                    cmap=cmap,
                    alpha=alpha_c,
                )

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
        if Show == True:
            plt.show()

    def plot4Dto3Dscatter(
        self,
        Show=False,
        vmax=None,
        vmin=None,
        cmap="coolwarm",
        marker="s",
        s=30,
        xlabel="",
        ylabel="",
        zlabel="",
    ):
        """WARNING : no colorbar possible with 3D axis."""
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(
            self.x,
            self.y,
            self.z,
            c=self.h,
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
            marker=marker,
            s=s,
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.set_zlabel(zlabel)
        if Show == True:
            plt.show()

    def plot3Dto2Dscatter(
        self,
        Show=False,
        Showbar=True,
        Use=2,
        vmax=None,
        vmin=None,
        fontsizetext=12,
        cmap="coolwarm",
        marker="s",
        s=30,
        xlabel="",
        ylabel="",
        colorlabel="",
    ):
        if Use == 2:
            col_vec = self.z
        else:
            col_vec = self.h
        plt.scatter(
            self.x,
            self.y,
            c=col_vec,
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
            marker=marker,
            s=s,
        )
        plt.xlabel(xlabel, fontsize=fontsizetext)
        plt.ylabel(ylabel, fontsize=fontsizetext)
        if Showbar == True:
            ax1 = plt.colorbar()
            ax1.ax.set_ylabel(colorlabel, fontsize=fontsizetext)
        if Show == True:
            plt.show()

    def plot4Dto2D(
        self,
        Show=False,
        Showbar=True,
        Nb=10,
        vmax=None,
        vmin=None,
        alpha=0.8,
        cmap="coolwarm",
        xlabel="",
        ylabel="",
        colorlabel="",
        fontsizetext=12,
        inline=1,
        fontsize=10,
        linewidth=0.5,
        fmt="%1.1f",
    ):
        dim1 = len(np.unique(self.x))
        dim2 = int(len(self.x) / dim1)
        X = np.reshape(self.x, (dim1, dim2))
        Y = np.reshape(self.y, (dim1, dim2))
        Z = np.reshape(self.z, (dim1, dim2))
        H = np.reshape(self.h, (dim1, dim2))
        if vmax == None:
            vmax = self.h.max()
        if vmin == None:
            vmin = self.h.min()
        plt.contourf(X, Y, H, alpha=alpha, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xlabel(xlabel, fontsize=fontsizetext)
        plt.ylabel(ylabel, fontsize=fontsizetext)
        if Showbar == True:
            ax, _ = mpl.colorbar.make_axes(plt.gca())
            ax1 = mpl.colorbar.ColorbarBase(
                ax, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            )
            ax1.ax.set_ylabel(colorlabel, fontsize=fontsizetext)
        C = plt.contour(X, Y, Z, Nb, colors="black", linewidth=linewidth)
        plt.clabel(C, inline=inline, fontsize=fontsize, fmt=fmt)
        if Show == True:
            plt.show()

    def plot4Dto2Dscatter(
        self,
        Show=False,
        Showbar=True,
        Nb=10,
        vmax=None,
        vmin=None,
        marker="s",
        s=30,
        cmap="coolwarm",
        xlabel="",
        ylabel="",
        colorlabel="",
        fontsizetext=12,
        inline=1,
        fontsize=10,
        linewidth=0.5,
        fmt="%1.1f",
    ):
        dim1 = len(np.unique(self.x))
        dim2 = int(len(self.x) / dim1)
        X = np.reshape(self.x, (dim1, dim2))
        Y = np.reshape(self.y, (dim1, dim2))
        Z = np.reshape(self.z, (dim1, dim2))
        plt.scatter(
            self.x,
            self.y,
            c=self.h,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            marker=marker,
            s=s,
        )
        plt.xlabel(xlabel, fontsize=fontsizetext)
        plt.ylabel(ylabel, fontsize=fontsizetext)
        if Showbar == True:
            ax1 = plt.colorbar()
            ax1.ax.set_ylabel(colorlabel, fontsize=fontsizetext)
        C = plt.contour(X, Y, Z, Nb, colors="black", linewidth=linewidth)
        plt.clabel(C, inline=inline, fontsize=fontsize, fmt=fmt)
        if Show == True:
            plt.show()


class histo(object):
    def __init__(self, x, histo=None, xlabel=None, bins="scott"):
        self.x = x
        self.label = histo
        self.nom = xlabel
        self.bin_method = bins

    def plot1(self, col, Show=False, color="r", alpha=0.7, normed=False, fontsize=14):
        if self.label == None:
            self.label = [None] * len(self.x)
        astrohist(
            self.x[col],
            bins=self.bin_method,
            color=color,
            alpha=alpha,
            ec="black",
            label=self.label[col],
            density=normed,
        )
        plt.xlabel(self.nom, fontsize=fontsize)
        if Show == True:
            plt.show()
        return astrohist(
            self.x[col],
            bins=self.bin_method,
            color=color,
            alpha=alpha,
            ec="black",
            label=self.label[col],
            density=normed,
        )

    def plotmany(
        self,
        cols,
        Show=True,
        colors=None,
        alpha=0.5,
        normed=False,
        fontsize=14,
        loc="best",
        vline=False,
    ):
        bins = astrohist(
            np.hstack([self.x[j] for j in cols]),
            bins=self.bin_method,
            color="white",
            density=normed,
        )[1]
        if colors == None:
            colors = [None] * len(self.x)
        colors = np.array(colors)
        ec = np.where((colors == "white") | (colors == "w"), "white", "black")
        if loc == "best":
            loc = ["best"] * len(self.x)
        if self.label == None:
            self.label = [None] * len(self.x)
        for j in colors[cols].argsort()[::-1]:
            if (colors[j] == "w") | (colors[j] == "white"):
                plt.hist(
                    self.x[j],
                    bins=bins,
                    color=colors[j],
                    alpha=alpha,
                    ec=ec[j],
                    density=normed,
                )
            else:
                plt.hist(
                    self.x[j],
                    bins=bins,
                    color=colors[j],
                    alpha=alpha,
                    ec=ec[j],
                    label=self.label[j],
                    density=normed,
                )
            plt.hist(
                self.x[j],
                bins=bins,
                color=ec[j],
                density=normed,
                histtype="step",
                linewidth=0.5,
            )
            if (self.label != [None] * len(self.x)) & (
                self.label != [""] * len(self.x)
            ):
                plt.legend(loc=loc[j])
        plt.xlabel(self.nom, fontsize=fontsize)
        if vline:
            for j in colors[cols].argsort()[::-1]:
                plt.axvline(x=np.mean(self.x[j]), color=colors[j], linestyle=":")
        if Show == True:
            plt.show()


class histoXY(object):
    def __init__(self, x, y, histo1=None, histo2=None, xlabel=None, bins="scott"):
        self.x = x
        self.y = y
        self.xlabel = histo1
        self.ylabel = histo2
        self.nom = xlabel
        self.bin_method = bins

    def plot1(self, Show=False, color="r", alpha=0.7, normed=False):
        astrohist(
            self.x,
            bins=self.bin_method,
            color=color,
            alpha=alpha,
            ec="black",
            label=self.xlabel,
            density=normed,
        )
        plt.xlabel(self.nom)
        if Show == True:
            plt.show()

    def plot2(self, Show=False, color="g", alpha=0.7):
        astrohist(
            self.y,
            bins=self.bin_method,
            color=color,
            alpha=alpha,
            ec="black",
            label=self.ylabel,
        )
        plt.xlabel(self.nom)
        if Show == True:
            plt.show()

    def plotboth(
        self, Show=True, color1="r", color2="g", alpha=0.5, normed=False, fontsize=14
    ):
        bins = astrohist(
            np.hstack((self.x, self.y)),
            bins=self.bin_method,
            color="white",
            density=normed,
        )[1]
        plt.hist(
            self.x,
            bins=bins,
            color=color1,
            alpha=alpha,
            ec="black",
            label=self.xlabel,
            density=normed,
        )
        plt.hist(
            self.y,
            bins=bins,
            color=color2,
            alpha=alpha,
            ec="black",
            label=self.ylabel,
            density=normed,
        )
        plt.hist(
            self.y, bins=bins, color="k", density=normed, histtype="step", linewidth=0.5
        )
        plt.hist(
            self.x, bins=bins, color="k", density=normed, histtype="step", linewidth=0.5
        )
        plt.legend()
        plt.xlabel(self.nom, fontsize=fontsize)
        if Show == True:
            plt.show()


class vectorX(object):
    def __init__(self, x):
        self.x = x  # vector of x

    def find_nearest(self, value):
        idx = (np.abs(self.x - value)).argmin()
        self.idx_nearest = idx
        self.val_nearest = self.x[idx]
        self.dist_nearest = abs(self.x[idx] - value)

    def doppler_c(self, v):
        """Classic Doppler. Take (wavelenght, velocity[m/s]) and return lambda observed and lambda source"""
        c = 3e8
        factor = 1 + v / c
        lambo = self.x * factor
        lambs = self.x * factor ** (-1)
        self.lambo = lambo
        self.lambs = lambs

    def doppler_r(self, v):
        """Relativistic Doppler. Take (wavelenght, velocity[m/s]) and return lambda observed and lambda source"""
        c = 3e8
        factor = np.sqrt((1 + v / c) / (1 - v / c))
        lambo = self.x * factor
        lambs = self.x * factor ** (-1)
        self.lambo = lambo
        self.lambs = lambs


class table(object):
    """this classe has been establish with pandas DataFrame"""

    def __init__(self, array):
        self.table = array
        self.dim = np.shape(array)

    def copy(self):
        new_table = table(np.array(self.table).copy())
        return new_table

    def fit_poly(self, deg_detrend, vec_x, replace=False):
        vec_x -= np.nanmedian(vec_x)
        new_tab = table(np.array(self.table).T)
        new_tab.fit_unique_base(np.array([vec_x**i for i in range(deg_detrend + 1)]))
        if replace:
            if type(self.table) == pd.core.frame.DataFrame:
                self.table.values = pd.DataFrame(
                    new_tab.vec_residues.T, columns=self.table.keys()
                )
            else:
                self.table = new_tab.vec_residues.T
        else:
            if type(self.table) == pd.core.frame.DataFrame:
                self.table_poly_detrended = pd.DataFrame(
                    new_tab.vec_residues.T, columns=self.table.keys()
                )
            else:
                self.table_poly_detrended = new_tab.vec_residues.T

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
            self.matrix_corr = myf.rm_sym_diagonal(self.matrix_corr, k=0, put_nan=True)

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
            ax.ax.set_ylabel(
                r"$\mathcal{R}_{pearson}$", fontsize=[15, 20][int(paper_plot)]
            )
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

    def cross_validation(
        self,
        cross_valid_size,
        nb_comp,
        r_min=0.6,
        frac_affected=0.01,
        cv_rm=20,
        debug=False,
        offset=0,
        fig_num=1,
        algo_block=2,
        algo_borders=3,
    ):
        """For a NxT matrix with T the number of time observations and N the number of vectors (N=CxS with C the number of component and S the number of independent simulations)"""

        vecs = self.table

        all_vec_name = ["v%.0f" % (i) for i in range(1, len(vecs) + 1)]

        table_raw = pd.DataFrame(vecs.T, columns=all_vec_name)
        table_array = np.array(table_raw)

        def highlight_comp(var, legend=False, Plot=False):
            var = np.array(var)
            l = [
                np.where(var == "v%.0f" % (i))[0][0] for i in np.arange(1, nb_comp + 1)
            ]
            l_comp = var[l]
            l_comp = [int(v[1:]) - 1 for v in l_comp]

            if Plot:
                for n, p in enumerate(l):
                    plt.axvline(x=p, color="r", alpha=0.5)
                    plt.axhline(y=p, color="r", alpha=0.5)
                    plt.scatter(
                        p,
                        p,
                        color=None,
                        label="%.0f" % (n + 1),
                        zorder=100,
                        edgecolor="k",
                    )
                if legend:
                    plt.legend(loc=1)

            return l, l_comp

        coeff = table(table_raw)

        if debug:
            plt.figure()

        coeff.r_matrix(
            name=list(coeff.table.keys()),
            absolute=False,
            light_plot=True,
            angle=90,
            Plot=debug,
        )

        if debug:
            plt.figure(figsize=(24, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(abs(coeff.matrix_corr), vmin=0, vmax=1)
        l, l_comp = highlight_comp(list(coeff.table.keys()))

        coeff.matrix_corr = myf.rm_sym_diagonal(coeff.matrix_corr, k=offset)

        var_kept = np.array(list(coeff.table.keys()))[
            np.where(np.sum(abs(coeff.matrix_corr) > r_min, axis=0) != 0)[0]
        ]

        if debug:
            plt.figure()
        coeff.r_matrix(
            name=var_kept,
            absolute=False,
            light_plot=True,
            angle=90,
            vmin=-1.2,
            vmax=1.2,
            Plot=debug,
        )

        if debug:
            plt.subplot(1, 3, 2)
            plt.imshow(abs(coeff.matrix_corr), vmin=0, vmax=1)
        l, l_comp = highlight_comp(var_kept, Plot=debug)
        coeff.matrix_corr = myf.rm_sym_diagonal(coeff.matrix_corr, k=offset)

        if algo_block == 1:
            block_order, dust = myf.block_matrix(coeff.matrix_corr.copy(), r_min=r_min)
        else:
            self.debug3 = coeff.matrix_corr.copy()
            block_order, dust = myf.block_matrix2(coeff.matrix_corr.copy())

        if debug:
            plt.figure()
        coeff.r_matrix(
            name=var_kept[block_order],
            absolute=False,
            light_plot=True,
            angle=90,
            vmin=-1.2,
            vmax=1.2,
            Plot=debug,
        )

        if debug:
            plt.subplot(1, 3, 3)
            plt.imshow(abs(coeff.matrix_corr), vmin=0, vmax=1)
        l, l_comp = highlight_comp(var_kept[block_order], Plot=debug)
        # coeff.matrix_corr = myf.rm_sym_diagonal(coeff.matrix_corr,k=offset)

        binary_matrix = abs(coeff.matrix_corr) > r_min
        cluster_loc = []
        for j in range(1, len(binary_matrix)):
            if not np.sum(binary_matrix[j, 0:j]):
                cluster_loc.append(j)
                if debug:
                    plt.axvline(x=j - 0.5, color="k")
                    plt.axhline(y=j - 0.5, color="k")

        cluster_loc = [0] + cluster_loc + [len(binary_matrix)]

        mask_kept = np.ones(len(binary_matrix))
        for i, j in zip(cluster_loc[0:-1], cluster_loc[1:]):
            for k in np.arange(i, j):
                if k != i:
                    if not np.sum(binary_matrix[k, i:k]):
                        mask_kept[k] = False
                else:
                    if not np.sum(binary_matrix[k, k : j + 1]):
                        mask_kept[k] = False
        mask_kept = mask_kept.astype("bool")

        if debug:
            plt.figure()
        coeff.r_matrix(
            name=var_kept[block_order][mask_kept],
            absolute=False,
            light_plot=True,
            angle=90,
            vmin=-1.2,
            vmax=1.2,
            Plot=debug,
        )
        # coeff.matrix_corr = myf.rm_sym_diagonal(coeff.matrix_corr,k=offset)

        vec_ordered = list(var_kept[block_order][mask_kept]) + list(
            np.array(all_vec_name)[
                ~np.in1d(all_vec_name, var_kept[block_order][mask_kept])
            ]
        )

        if debug:
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        plt.figure(fig_num, figsize=(10, 12))
        plt.title(
            "R_min=%.3f, Nb_comp=%.0f, Nb_draw=%.0f, Min_size_cluster=%.0f%%, CV_rm=%.0f%%, Algo=%.0f"
            % (
                r_min,
                nb_comp,
                cross_valid_size,
                frac_affected * 100,
                cv_rm,
                algo_borders,
            )
        )
        coeff.r_matrix(
            name=vec_ordered,
            absolute=False,
            light_plot=True,
            angle=90,
            vmin=-1.2,
            vmax=1.2,
            Plot=debug,
        )
        plt.imshow(abs(coeff.matrix_corr), vmin=0.5, vmax=1)
        dust = highlight_comp(vec_ordered, legend=True, Plot=True)
        # coeff.matrix_corr = myf.rm_sym_diagonal(coeff.matrix_corr,k=offset)
        self.loc_comp = dust[0]
        final_coeff = coeff.matrix_corr.copy()
        self.cv_matrix = final_coeff

        cluster_min_size = int(np.round(cross_valid_size * frac_affected, 0))

        if algo_borders == 1:  # based on 1 diagonal-off algorithm
            binary_matrix = (abs(final_coeff) > r_min).astype("int")
            cluster_loc = []
            for j in range(1, len(binary_matrix)):
                if not np.sum(binary_matrix[j, 0:j]):
                    cluster_loc.append(j)
            cluster_loc = np.array([0] + cluster_loc + [len(binary_matrix)])
        elif algo_borders == 2:  # based on derivative (failed cause too noisy)
            grad = np.diff((abs(final_coeff) > r_min).astype("int"), axis=0)
            # grad = np.diff((abs(final_coeff)),axis=0)
            border_left = np.sum(grad > 0, axis=1) / cross_valid_size
            border_left = np.hstack([1, border_left])
            border_right = np.sum(grad < 0, axis=1) / cross_valid_size
            border_right = np.hstack([0, border_right])
            loc_comp = dust[0]
            loc_comp = np.hstack([np.sort(loc_comp), [len(border_left)]])
            left = [0]
            right = []
            for i in np.arange(0, len(loc_comp) - 1):
                if i:
                    left.append(
                        np.argmax(border_left[loc_comp[i - 1] : loc_comp[i] + 1])
                        + loc_comp[i - 1]
                    )
                right.append(
                    np.argmax(border_right[loc_comp[i] : loc_comp[i + 1]]) + loc_comp[i]
                )

            self.cv_borders_alter = np.array([border_left, border_right])
            # self.cluster_loc = cluster_loc

            cluster_loc = np.unique(left + right)
        elif algo_borders == 3:  # based on integrant (seems to be okay)
            binary_matrix = (abs(final_coeff) > r_min).astype("int")
            borders = []
            i = 0
            for j in np.arange(1, len(binary_matrix)):
                # if np.sum(binary_matrix[j,i:j])<(j-i)*0.5:
                if (
                    np.sum(binary_matrix[j : j + cluster_min_size, i:j])
                    < (j - i) * 0.5 * cluster_min_size
                ):
                    # if (j-i)>cluster_min_size:
                    borders.append([i, j])
                    i = j
            borders.append([borders[-1][1], len(binary_matrix) - 1])
            cluster_loc = np.unique(np.hstack(borders))

        comp_inside_cluster = []
        clusters = []
        for j, k in zip(cluster_loc[0:-1] - 0.5, cluster_loc[1:] - 0.5):
            if (k - j) > cluster_min_size:
                plt.plot([j, j], [j, k], color="w")
                plt.plot([k, k], [j, k], color="w")
                plt.plot([j, k], [j, j], color="w")
                plt.plot([j, k], [k, k], color="w")
                components = np.where((l > j) & (l < k))[0]
                if len(components):
                    m, n = int(j + 0.5), int(k + 0.5)
                    clusters.append([m, n])
                    comp_inside_cluster.append(list(components))
                    for n, c in enumerate(components):
                        vec_ordered[int(j + 0.5) + n], vec_ordered[l[c]] = (
                            vec_ordered[l[c]],
                            vec_ordered[int(j + 0.5) + n],
                        )
                        coeff.matrix_corr[int(j + 0.5) + n], coeff.matrix_corr[l[c]] = (
                            coeff.matrix_corr[l[c]],
                            coeff.matrix_corr[int(j + 0.5) + n],
                        )
                        (
                            coeff.matrix_corr[:, int(j + 0.5) + n],
                            coeff.matrix_corr[:, l[c]],
                        ) = (
                            coeff.matrix_corr[:, l[c]],
                            coeff.matrix_corr[:, int(j + 0.5) + n],
                        )

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        if algo_borders == 2:
            plt.figure(fig_num + 2, figsize=(10, 4))
            plt.plot(
                np.arange(len(final_coeff)) - 0.5, border_left, color="k", zorder=100
            )
            plt.plot(
                np.arange(len(final_coeff)) - 0.5,
                border_right - 1,
                color="gray",
                zorder=100,
            )
            plt.axhline(y=frac_affected, ls=":", color="k")
            plt.axhline(y=frac_affected - 1, ls=":", color="k")
            for j in np.array(dust[0]):
                plt.plot([j, j], [-1, 1], color="b")
            for j, k in zip(cluster_loc[0:-1] - 0.5, cluster_loc[1:] - 0.5):
                if (k - j) > cluster_min_size:
                    plt.plot([j, j], [-1, 1], color="r")
                    plt.plot([k, k], [-1, 1], color="r")

        mini = np.argsort(
            [np.min(i) for i in comp_inside_cluster]
        )  # TBD to order cluster by the index of the component (semble prise de tete)

        self.cv_clusters = clusters
        self.cv_comp_inside = comp_inside_cluster
        self.cv_nb_comp_inside = [len(i) for i in comp_inside_cluster]

        cluster_min_size = np.round(cross_valid_size * frac_affected, 0)
        new_vec_order = np.array([int(i[1:]) - 1 for i in vec_ordered])

        temp = table_array.copy()[:, new_vec_order]

        components_stacked = []
        percentage = []
        r_med = []
        counter = 0
        plt.figure(fig_num + 1, figsize=(12, 8))
        for c in clusters:
            counter += 1
            n, m = c[0], c[1]
            loc_sign = np.argmax(abs(temp[:, n]))
            indep_comp = temp[:, n:m]
            indep_comp /= myf.mad(indep_comp, axis=0)
            indep_comp *= np.sign(indep_comp[loc_sign])
            percent = (
                (m - n - self.cv_nb_comp_inside[counter - 1]) * 100 / cross_valid_size
            )
            percentage.append(percent)
            components_stacked.append(np.median(indep_comp, axis=1))
            med_r = np.nanmedian(abs(final_coeff[n:m, n:m]))
            r_med.append(med_r)
            plt.subplot(
                nb_comp // 2 + nb_comp % 2, 2, comp_inside_cluster[counter - 1][0] + 1
            )
            plt.title(
                "Components : "
                + "+".join([str(i + 1) for i in comp_inside_cluster[counter - 1]])
                + "  |  frequency = %.0f %%  |  median(R)=%.3f" % (percent, med_r)
            )
            plt.plot(indep_comp, color="k", alpha=0.1)
            plt.plot(
                np.median(indep_comp, axis=1), color="r", lw=1.5
            )  # ,label='frequency = %.0f %%\nmedian(R)=%.3f'%(percent,med_r))
            plt.plot(
                np.sign(temp[loc_sign, n]) * temp[:, n] / myf.mad(temp[:, n]), color="b"
            )
            # plt.legend()

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.45)

        self.cv_clusters = clusters
        self.cv_vec_ordered = new_vec_order
        self.cv_percentage = percentage
        self.cv_components = components_stacked
        self.cv_rmed = r_med

        self.cv_percentage_norm = np.array(
            [
                self.cv_percentage[i] / len(self.cv_comp_inside[i])
                for i in np.arange(len(self.cv_comp_inside))
            ]
        )

    def cross_validation_it(
        self,
        cross_valid_size,
        nb_comp,
        frac_affected=0.33,
        iteration=5,
        fig_num=1,
        cv_rm=10,
        algo_borders=[3, 3],
    ):

        borders = np.array([0.0, 1.0])
        r_min_it = 0.25
        good_r_min = []
        sizes = []
        highest_percentage = []
        sizes_binary = []
        print("Cross-validation optimal threshold determination")
        r_mins = np.linspace(0.25, 0.50, iteration)
        for condition in range(iteration):
            r_min_it = r_mins[condition]
            plt.close("all")
            print(
                "Iteration %.0f, threshold fixed at R=%.3f" % (condition + 1, r_min_it)
            )
            try:
                self.cross_validation(
                    cross_valid_size,
                    nb_comp,
                    r_min=r_min_it,
                    frac_affected=frac_affected,
                    debug=False,
                    cv_rm=cv_rm,
                    fig_num=fig_num,
                    algo_borders=algo_borders[0],
                )
                good_r_min.append(r_min_it)
                sizes.append(np.sum(self.cv_percentage_norm))
                highest_percentage.append(-abs(100 - np.max(self.cv_percentage_norm)))
                if np.max(self.cv_percentage_norm) <= 100 - cv_rm:
                    tresh = np.max(self.cv_percentage_norm)
                else:
                    tresh = 100 - cv_rm
                crit = np.sum(
                    (np.array(self.cv_percentage_norm) >= tresh)
                    & (np.array(self.cv_percentage_norm) <= (105))
                )
                sizes_binary.append(crit)
                print(self.cv_percentage_norm.astype("int"), crit)
            except IndexError:
                pass

        plt.pause(0.1)
        plt.close("all")

        liste = [
            (i, j, k) for i, j, k in zip(sizes_binary, highest_percentage, good_r_min)
        ]
        # print(liste)
        liste.sort()
        r_min_it = liste[-1][-1]

        print("Best R found to be : %.3f" % (r_min_it))
        self.cross_validation(
            cross_valid_size,
            nb_comp,
            r_min=r_min_it,
            frac_affected=frac_affected,
            debug=False,
            cv_rm=cv_rm,
            fig_num=fig_num,
            algo_borders=algo_borders[1],
        )
        plt.pause(0.1)

        mini = np.argsort([np.min(i) for i in self.cv_comp_inside])

        components_stacked = []
        comp_inside_cluster = []
        nb_comp_inside_cluster = []
        clusters = []

        percentage = np.zeros(nb_comp)
        percentage_norm = np.zeros(nb_comp)
        r_med = np.zeros(nb_comp)
        new_vec_order = []

        for i, elem in enumerate(self.cv_comp_inside):
            for j in elem:
                percentage[j] = self.cv_percentage[i]
                percentage_norm[j] = percentage[j] / len(self.cv_comp_inside[i])
                r_med[j] = self.cv_rmed[i]

        b = 0
        for i, j in enumerate(mini):
            i1, i2 = self.cv_clusters[j]
            comp_inside_cluster.append(self.cv_comp_inside[j])
            nb_comp_inside_cluster.append(self.cv_nb_comp_inside[j])
            components_stacked.append(self.cv_components[j])
            new_vec_order.append(self.cv_vec_ordered[i1:i2])
            clusters.append([b, b + (i2 - i1)])
            b += i2 - i1

        components_stacked = np.array(components_stacked)
        new_vec_order = np.hstack(new_vec_order)

        self.cv_comp_inside = comp_inside_cluster
        self.cv_nb_comp_inside = nb_comp_inside_cluster
        self.cv_clusters = clusters
        self.cv_vec_ordered = new_vec_order  # TBD check if new order is the good one
        self.cv_percentage = percentage
        self.cv_percentage_norm = percentage_norm

        self.cv_components = components_stacked
        self.cv_rmed = r_med

    def cross_validation2(
        self,
        nb_comp,
        r_min=0.6,
        frac_affected=0.01,
        cv_rm=20,
        fig_num=1,
        overselection=2,
    ):
        """For a NxT matrix with T the number of time observations and N the number of vectors (N=CxS with C the number of component and S the number of independent simulations)"""

        vecs = self.table

        cross_valid_size = int((len(vecs) - nb_comp) / nb_comp)
        selection = overselection * nb_comp  # number of cluster kept for the new step

        all_vec_name = ["v%.0f" % (i) for i in range(1, len(vecs) + 1)]

        table_raw = pd.DataFrame(vecs.T, columns=all_vec_name)
        table_array = np.array(table_raw)

        coeff = table(table_raw)

        coeff.r_matrix(
            name=list(coeff.table.keys()), absolute=True, Plot=False, rm_diagonal=True
        )
        # only kept relevant rows
        var_kept = np.array(list(coeff.table.keys()))[
            np.where(np.sum(coeff.matrix_corr > r_min, axis=0) != 0)[0]
        ]
        coeff.r_matrix(name=var_kept, absolute=True, Plot=False, rm_diagonal=True)

        plt.figure(fig_num + 2, figsize=(9, 8))

        plt.subplot(3, 3, 1)
        plt.title("Iteration 0")
        plt.imshow(coeff.matrix_corr.copy())
        # block algo
        block_order, dust = myf.block_matrix2(coeff.matrix_corr.copy())
        coeff.r_matrix(
            name=var_kept[block_order], absolute=True, Plot=False, rm_diagonal=True
        )
        var_kept1 = var_kept[block_order]
        coeff1 = coeff.copy()

        for itera in np.arange(1, 8):
            print("Iteration %.0f : " % (itera))
            loc_comp, dust = myf.highlight_comp(
                var_kept1, nb_comp, legend=True, Plot=False
            )
            plt.subplot(3, 3, 1 + itera)
            plt.title("Iteration %.0f" % (itera))
            coeff1, var_kept1 = myf.block_matrix_iter(
                coeff,
                cross_valid_size,
                frac_affected,
                cv_rm,
                var_kept1,
                selection=selection,
                loc_comp=loc_comp,
            )

        plt.subplot(3, 3, 9)
        plt.title("Iteration 8")
        (
            borders,
            cluster_loc,
            mask_cluster,
            med_cluster,
            cluster_signi,
            mask_cluster_complete,
        ) = myf.find_borders_it(
            coeff1.matrix_corr.copy(),
            cross_valid_size,
            frac_affected,
            cv_rm,
            Draw=True,
            selection=selection,
        )
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        final_vec_ordered = var_kept1.copy()

        # =============================================================================
        #         plot
        # =============================================================================

        plt.figure(fig_num, figsize=(9, 7))
        plt.title(
            "Nb_comp=%.0f, Nb_draw=%.0f, Min_size_cluster=%.0f%%, CV_rm=%.0f%%"
            % (nb_comp, cross_valid_size, frac_affected * 100, cv_rm),
            fontsize=15,
        )
        coeff.r_matrix(name=final_vec_ordered, absolute=True, Plot=False)
        l, dust = myf.highlight_comp(final_vec_ordered, nb_comp, legend=True, Plot=True)
        (
            borders,
            cluster_loc,
            mask_cluster,
            med_cluster,
            cluster_signi,
            mask_cluster_complete,
        ) = myf.find_borders_it(
            coeff.matrix_corr.copy(),
            cross_valid_size,
            frac_affected,
            cv_rm,
            Draw=True,
            selection=selection,
        )
        plt.imshow(coeff.matrix_corr.copy())
        ax = plt.colorbar(pad=0)
        ax.ax.set_ylabel(r"$|\mathcal{R}_{Pearson}|$", fontsize=15)
        # coeff.matrix_corr = myf.rm_sym_diagonal(coeff.matrix_corr,k=offset)
        final_coeff = coeff.matrix_corr.copy()

        cluster_min_size = np.round(cross_valid_size * frac_affected, 0)

        comp_inside_cluster = []
        nb_comp_inside = []
        clusters = []
        for j, k in zip(cluster_loc[0:-1] - 0.5, cluster_loc[1:] - 0.5):
            if (k - j) > cluster_min_size:
                plt.plot([j, j], [j, k], color="w")
                plt.plot([k, k], [j, k], color="w")
                plt.plot([j, k], [j, j], color="w")
                plt.plot([j, k], [k, k], color="w")
                components = np.where((l > j) & (l < k))[0]
                if len(components):
                    m, n = int(j + 0.5), int(k + 0.5)
                    clusters.append([m, n])
                    comp_inside_cluster.append(list(components))
                    nb_comp_inside.append(len(list(components)))
                    for n, c in enumerate(components):
                        final_vec_ordered[int(j + 0.5) + n], final_vec_ordered[l[c]] = (
                            final_vec_ordered[l[c]],
                            final_vec_ordered[int(j + 0.5) + n],
                        )
                        coeff.matrix_corr[int(j + 0.5) + n], coeff.matrix_corr[l[c]] = (
                            coeff.matrix_corr[l[c]],
                            coeff.matrix_corr[int(j + 0.5) + n],
                        )
                        (
                            coeff.matrix_corr[:, int(j + 0.5) + n],
                            coeff.matrix_corr[:, l[c]],
                        ) = (
                            coeff.matrix_corr[:, l[c]],
                            coeff.matrix_corr[:, int(j + 0.5) + n],
                        )

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # =============================================================================
        # plot components
        # =============================================================================

        mini = np.argsort([np.min(i) for i in comp_inside_cluster])
        new_vec_order = np.array([int(i[1:]) - 1 for i in final_vec_ordered])
        temp = np.array(coeff.table).copy()[:, new_vec_order]

        components = []
        percentage = []
        r_med = []
        counter = 0
        plt.figure(fig_num + 1, figsize=(12, 8))
        for c in clusters:
            counter += 1
            n, m = c[0], c[1]
            loc_sign = np.argmax(abs(temp[:, n]))
            indep_comp = temp[:, n:m]
            indep_comp /= myf.mad(indep_comp, axis=0)
            indep_comp *= np.sign(indep_comp[loc_sign])

            intermediate = table(indep_comp)
            nb_out = intermediate.count_outliers(m=1.5, transpose=False)
            mask_out = nb_out > (len(indep_comp) * 0.5)
            percent = (
                (m - n - nb_comp_inside[counter - 1] - sum(mask_out))
                * 100
                / cross_valid_size
            )
            percentage.append(percent)
            components.append(np.median(indep_comp, axis=1))
            med_r = np.nanmedian(abs(final_coeff[n:m, n:m]))
            r_med.append(med_r)
            plt.subplot(
                nb_comp // 2 + nb_comp % 2, 2, comp_inside_cluster[counter - 1][0] + 1
            )
            plt.title(
                "Components : "
                + "+".join([str(i + 1) for i in comp_inside_cluster[counter - 1]])
                + "  |  frequency = %.0f %%  |  median(R)=%.3f" % (percent, med_r)
            )
            plt.plot(indep_comp[:, ~mask_out], color="k", alpha=0.1)
            plt.plot(
                np.median(indep_comp[:, ~mask_out], axis=1), color="r", lw=1.5
            )  # ,label='frequency = %.0f %%\nmedian(R)=%.3f'%(percent,med_r))
            plt.plot(
                np.sign(temp[loc_sign, n]) * temp[:, n] / myf.mad(temp[:, n]), color="b"
            )
            # plt.legend()

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.45)

        # =============================================================================
        # stack component
        # =============================================================================

        components_stacked = []
        comp_inside_cluster_stacked = []
        nb_comp_inside_cluster = []
        clusters_stacked = []

        percentage_stacked = np.zeros(nb_comp)
        percentage_stacked_norm = np.zeros(nb_comp)
        r_med_stacked = np.zeros(nb_comp)
        new_vec_order = []

        for i, elem in enumerate(comp_inside_cluster):
            for j in elem:
                percentage_stacked[j] = percentage[i]
                percentage_stacked_norm[j] = percentage[i] / len(comp_inside_cluster[i])
                r_med_stacked[j] = r_med[i]

        b = 0
        for i, j in enumerate(mini):
            i1, i2 = clusters[j]
            comp_inside_cluster_stacked.append(comp_inside_cluster[j])
            nb_comp_inside_cluster.append(nb_comp_inside[j])
            components_stacked.append(components[j])
            new_vec_order.append(final_vec_ordered[i1:i2])
            clusters_stacked.append([b, b + (i2 - i1)])
            b += i2 - i1

        components_stacked = np.array(components_stacked)
        new_vec_order = np.hstack(new_vec_order)

        self.cv_comp_inside = comp_inside_cluster_stacked
        self.cv_nb_comp_inside = nb_comp_inside_cluster
        self.cv_clusters = clusters
        self.cv_vec_ordered = new_vec_order  # TBD check if new order is the good one
        self.cv_percentage = percentage_stacked
        self.cv_percentage_norm = percentage_stacked_norm

        self.cv_components = components_stacked
        self.cv_rmed = r_med_stacked

        self.cv_matrix = final_coeff
        self.loc_comp = l

    def count_outliers(self, m=1.5, transpose=False):
        matrix = np.array(self.table)
        if transpose:
            matrix = matrix.T
        IQ = myf.IQ(matrix, axis=1)
        mask_sup = (
            matrix > (np.nanpercentile(matrix, 75, axis=1) + (m * IQ))[:, np.newaxis]
        )
        mask_inf = (
            matrix < (np.nanpercentile(matrix, 25, axis=1) - (m * IQ))[:, np.newaxis]
        )
        counter = np.sum(mask_sup, axis=0) + np.sum(mask_inf, axis=0)
        return counter

    def gram_schimdt_modified(self, vector, base_std=None, Plot=True):
        """Insert a new vector in a basis already orthogonal, the new vector become the reference"""

        tab = self.table

        if len(vector) == len(tab.T):
            old_table = table(np.vstack([vector, tab]))

            new_basis = [vector]
            if base_std is None:
                base_std = np.ones(np.shape(tab))
            for j in np.arange(len(tab)):
                base_to_fit = np.vstack([np.ones(len(vector)), new_basis])
                a = tableXY(np.arange(len(vector)), tab[j], base_std[j])

                a.fit_base(base_to_fit)
                new_basis.append(a.vec_residues.y)

            new_table = table(np.array(new_basis))

            if Plot:
                plt.figure(figsize=(18, 6))
                plt.subplot(1, 2, 1)
                plt.title("Before Gram-schimdt")
                old_table.r_matrix()
                plt.subplot(1, 2, 2)
                plt.title("After Gram-schimdt")
                new_table.r_matrix()

            return new_table.table[1:]

        else:
            print(
                "Shape of the vector (%.0f) is different from the basis shape (%.0f)"
                % (len(vector), len(self.table.T))
            )

    def rms_w(self, weights, axis=1):
        average = np.average(self.table, weights=weights, axis=axis)

        if axis == 1:
            data_recentered = self.table - average[:, np.newaxis]
        if axis == 0:
            data_recentered = (self.table.T - average[:, np.newaxis]).T

        variance = np.average((data_recentered) ** 2, weights=weights, axis=axis)
        self.rms = np.sqrt(variance)

    def export_to_dace(self, file_name, convert_to_kms=1, convert_to_jdb=0):
        if len(file_name.split(".")) > 1:
            if file_name.split(".")[-1] != "rdb":
                file_name = file_name.split(".")[0] + ".rdb"
        else:
            file_name += ".rdb"

        all_kw = [
            "fwhm",
            "sig_fwhm",
            "contrast",
            "sig_contrast",
            "bis_span",
            "sig_bis_span",
            "s_mw",
            "sig_s",
            "ha",
            "sig_ha",
            "na",
            "sig_na",
            "ca",
            "sig_ca",
            "rhk",
            "sig_rhk",
            "berv",
            "sn_caii",
            "model",
            "ins_name",
        ]

        self.table["jdb"] = self.table["jdb"] + convert_to_jdb
        self.table["vrad"] = self.table["vrad"] * convert_to_kms
        self.table["svrad"] = self.table["svrad"] * convert_to_kms

        for s in all_kw:
            if s not in self.table.keys():
                self.table[s] = 0

        all_name = ["jdb", "vrad", "svrad"] + all_kw

        self.table = self.table[all_name]

        matrice = np.array(self.table)

        f1 = "\t".join(all_name)
        f2 = "\t".join(["-" * len(i) for i in all_name])

        np.savetxt(
            file_name,
            matrice,
            delimiter="\t",
            header=f1 + "\n" + f2,
            fmt=["%.6f"] + ["%.8e"] * (len(all_name) - 2) + ["%s"],
        )

        f = open(file_name, "r")
        lines = f.readlines()
        lines[0] = lines[0][2:]
        lines[1] = lines[1][2:]
        lines[-1] = lines[-1][:-1]
        f.close()
        f = open(file_name, "w")
        f.writelines(lines)
        f.close()

    def rv_subselection(self, rv_std=None, selection=None):
        """Only if the table if a time-series of several lines"""

        if selection is None:
            selection = np.ones(len(self.table)).astype("bool")
        matrix_rv = self.table
        matrix_rv_std = rv_std

        mean_rv = np.sum(
            matrix_rv[selection] / matrix_rv_std[selection] ** 2, axis=0
        ) / np.sum(1 / matrix_rv_std[selection] ** 2, axis=0)
        mean_rv_std = 1 / np.sqrt(np.sum(1 / matrix_rv_std[selection] ** 2, axis=0))

        return tableXY(np.arange(len(mean_rv)), mean_rv, mean_rv_std)

    def slilder_plot(
        self, x, color="k", cut_index_min=0, cut_index_max=-1, logx=False, logy=False
    ):

        if len(np.shape(self.table)) != 2:
            print("The table must be a NxM matrix")
        else:
            if len(x) != len(self.table[0, :]):
                self.table = self.table.T

            x = x[cut_index_min:cut_index_max]
            table = self.table[:, cut_index_min:cut_index_max]

            index1 = 0
            index2 = np.shape(table)[0] - 1
            dindex = 1
            fig = plt.figure()
            plt.subplots_adjust(left=0.10, bottom=0.25, top=0.95, hspace=0.30)

            if logx:
                plt.xscale("log")
            if logy:
                plt.yscale("log")

            (l1,) = plt.plot(x, table[0, :], color=color)

            axcolor = "whitesmoke"
            axsmoothing = plt.axes([0.1, 0.1, 0.8, 0.1], facecolor=axcolor)
            ssmoothing = Slider(
                axsmoothing, "Index", index1, index2, valinit=index1, valstep=dindex
            )

            class Index:
                def update(self, val):
                    smoothing = ssmoothing.val
                    if smoothing < 0:
                        smoothing = 0
                    elif smoothing > index2:
                        smoothing = index2
                    l1.set_ydata(table[int(smoothing), :])
                    fig.canvas.draw_idle()

            callback = Index()
            ssmoothing.on_changed(callback.update)

    def ICA(self, comp_max=None, m=2, kind="inter", num_sim=1000, algorithm="parallel"):
        """perform an independent component analysis with comp_max components. Replacing outliers performed with kind and m parameters."""
        # self.replace_outliers(m=2,kind=kind)

        Signal = self.table
        # mean_vec = np.mean(Signal,axis=1)
        # Signal = Signal - mean_vec[:,np.newaxis]

        R = Signal.copy()

        self.vec_init = R

        ica = FastICA(n_components=comp_max, max_iter=num_sim, algorithm=algorithm)

        if False:
            ica.fit(R.T)
            self.mixing = ica.mixing_.T  # coefficient of the ica fit
            self.ica_vec = ica.transform(self.table.T)  # vector basis
        else:
            ica.fit(R)
            self.mixing = ica.transform(self.table).T  # coefficient of the ica fit
            self.ica_vec = ica.mixing_  # vector basis
            scaling = np.std(self.ica_vec, axis=0)
            self.ica_vec /= scaling
            self.mixing *= scaling[:, np.newaxis]

        self.ica = ica  # model saved, model is saved again later but i let this line in case it is used in some other codes

        mean_coeff = np.mean(ica.mixing_, axis=0)
        inverse_vector = np.where(mean_coeff < 0)[0]

        self.mixing[inverse_vector, :] *= -1
        self.ica_vec[:, inverse_vector] *= -1

        sort = np.std(self.mixing, axis=1).argsort()[::-1]

        self.ica_vec = self.ica_vec[:, sort]
        self.mixing = self.mixing[sort, :]
        self.var_mixing = np.std(self.mixing, axis=1)

        norm = np.sign(np.nanmedian(self.ica_vec, axis=0))
        self.ica_vec = self.ica_vec / norm
        self.mixing = self.mixing * norm[:, np.newaxis]

        self.var_ratio = (self.var_mixing) / np.sum(self.var_mixing)

        self.ica_vec_fitted = np.dot(self.ica_vec, self.mixing).T

        self.phi_mixing = np.sum(self.mixing < 0, axis=1) / len(self.mixing.T)
        self.zscore_mixing = np.mean(self.mixing, axis=1) / np.std(self.mixing, axis=1)

        self.ica_model = ica  # saved a second time due to the new name convention

    def UAPCA(self, weight=None, m=2, kind="inter", scaling=1, comp_max=None):
        """Uncertainties-Aware PCA Görtler"""
        # self.replace_outliers(m=m, kind=kind)

        Signal = self.table

        if weight is None:
            w = np.ones(np.shape(Signal))
        else:
            w = weight.copy()
            w = np.where(w == 0, np.min(w[w != 0]), w)

        if len(np.shape(w)) < 3:
            new_w = []
            for i in range(len(w)):
                new_w.append(np.diag(w[i]))
            w = np.array(new_w)

        w_no_cov = np.array([np.diag(w[i]) for i in range(len(w))])

        mu = np.sum(Signal * w_no_cov, axis=0) / np.sum(w_no_cov, axis=0)

        cov = np.zeros((np.shape(Signal)[1], np.shape(Signal)[1]))
        for j in range(len(Signal)):
            m = Signal[j]
            inv = 1 / w[j]
            inv[inv == np.inf] = 0
            cov += m * m[:, np.newaxis] + (scaling**2) * inv - mu * mu[:, np.newaxis]
        cov /= len(Signal)

        eigen_val, eigen_vec = np.linalg.eig(cov)
        eigen_vec = eigen_vec[:, abs(eigen_val).argsort()]
        eigen_val = eigen_val[abs(eigen_val).argsort()]
        self.uapca_vec = eigen_vec[:, ::-1][:, 0:comp_max]
        self.uapca_lambda = eigen_val[::-1][0:comp_max]

        self.var_ratio = (self.uapca_lambda) / np.sum(self.uapca_lambda)

        norm = np.sign(np.nanmedian(self.uapca_vec, axis=0))
        self.uapca_vec = self.uapca_vec / norm

        test = table(Signal)
        test.fit_base(base_vec=self.uapca_vec.T, weight=w_no_cov)
        self.uapca_vec_fitted = test.vec_fitted
        self.uapca_coeff = test.coeff_fitted.T
        self.phi_uapca = np.sum(self.uapca_coeff < 0, axis=1) / len(self.uapca_coeff.T)
        self.zscore_uapca = np.mean(self.uapca_coeff, axis=1) / np.std(
            self.uapca_coeff, axis=1
        )

    def SVD(self, comp_max=None, weight=None):
        Signal = self.table
        if weight is None:
            w = np.ones(np.shape(Signal))
        else:
            w = weight.copy()

        self.svd_vec, self.svd_eig_val, dust = np.linalg.svd(
            np.dot(self.table.T, self.table)
        )

        norm = np.sign(np.nanmedian(self.svd_vec, axis=0))

        self.var_ratio = (self.svd_eig_val / np.sum(self.svd_eig_val))[0:comp_max]
        self.svd_vec = self.svd_vec / norm

        test = table(Signal)
        test.fit_base(base_vec=self.svd_vec[:, 0:comp_max].T, weight=w)
        self.svd_vec_fitted = test.vec_fitted
        self.svd_coeff = test.coeff_fitted.T
        self.phi_svd = np.sum(self.svd_coeff < 0, axis=1) / len(self.svd_coeff.T)
        self.zscore_svd = np.mean(self.svd_coeff, axis=1) / np.std(
            self.svd_coeff, axis=1
        )

    def PCA(
        self,
        weight=None,
        m=2,
        kind="inter",
        num_sim=3000,
        comp_max=None,
        abs_coeff=False,
    ):
        """For the PCA enter a n x m array, with n the number of observations and m 'time' axis"""

        # self.replace_outliers(m=m, kind=kind)
        # scaler = StandardScaler()

        Signal = self.table
        if weight is None:
            w = np.ones(np.shape(Signal))
        else:
            w = weight

        # mean_vec = np.sum(Signal*w,axis=1)/np.sum(w,axis=1)
        # Signal = Signal - mean_vec[:,np.newaxis]

        # scaler.fit(Signal)
        # Signal = scaler.transform(Signal)

        R = Signal.copy()

        if weight is None:
            pca = PCA(n_components=comp_max, whiten=False)

            if (
                False
            ):  # old version which was wrong (producing good result but the model cannot be applied on other test sample, I keep it until other algo as ICA and UMPCA are corrected in a same way)
                pca.fit(R.T)
                self.components = pca.components_  # coefficient of the pca fit
                self.vec = pca.transform(self.table.T)  # vector basis
            else:
                pca.fit(R)
                self.vec = pca.components_.T  # vector basis
                self.components = pca.transform(
                    self.table
                ).T  # coefficient of the pca fit
            self.vec_fitted = np.dot(
                self.vec, self.components
            ).T  # sample projected on the basis

            self.var = pca.explained_variance_  # variance explained of the pca fit
            self.var_ratio = (
                pca.explained_variance_ratio_
            )  # variance_ratio explained of the pca fit
            self.s_values = pca.singular_values_  # s values explained of the pca fit

            norm = np.sign(np.nanmedian(self.vec, axis=0))
            self.vec = self.vec / norm
            self.components = self.components * norm[:, np.newaxis]

            components = (
                abs_coeff * abs(self.components) + (1 - abs_coeff) * self.components
            )

            self.phi_components = np.sum(components < 0, axis=1) / len(components.T)
            self.zscore_components = np.mean(components, axis=1) / np.std(
                components, axis=1
            )
            self.pca_model = pca
        else:
            # PCA with error bars following the Paper Tamuz, Mazeh 2013
            a_save = []
            c_save = []
            chi2 = [np.sum(w * R**2)]
            a = R[0, :]  # initial guess for the correlation

            if comp_max == None:
                max_c = len(R)
            else:
                max_c = comp_max
            for i in tqdm(range(max_c)):
                for j in range(num_sim):
                    c = np.nansum(w * R * a, axis=1) / np.nansum(w * a**2, axis=1)
                    c = c[:, np.newaxis]
                    c = c / np.sqrt(np.sum(c**2))
                    a = np.nansum(w * R * c, axis=0) / np.nansum(w * c**2, axis=0)
                    if (
                        np.nanmean(c) < 0
                    ):  # leave degeneracy +/- for the vector direction
                        c *= -1
                        a *= -1
                    # c = c * (2*(np.sum(np.sign(a))>=0)-1)
                    # a =  a * (2*(np.sum(np.sign(a))>=0)-1)
                    a2 = a.copy()
                    a = (
                        a + np.random.randn(len(a)) * 0.1
                    )  # fix the liberty degree of c*a by posing the std of a at 1
                chi2.append(np.sum(w * (R - a2 * c) ** 2))
                a_save.append(a2)
                c_save.append(c)
                R = R - a2 * c
                R = R - np.mean(R, axis=1)[:, np.newaxis]

            a = np.array(a_save).T
            c = np.array(c_save).T
            chi2 = np.array(chi2)

            self.my_vec = a
            self.chi = chi2
            self.coeff = c[0]
            self.coeff = self.coeff.T
            self.phi_coeff = np.sum(self.coeff < 0, axis=1) / len(self.coeff.T)
            self.zscore_coeff = np.mean(self.coeff, axis=1) / np.std(self.coeff, axis=1)

    def WPCA(self, pca, weight=None, comp_max=None, m=2, kind="inter"):
        """from https://github.com/jakevdp/wpca/blob/master/WPCA-Example.ipynb
        enter which pca do yo want either 'pca', 'wpca' or 'empca'
        empca slower than wpca

        """

        # self.replace_outliers(m=m, kind=kind)

        Signal = self.table

        R = Signal.copy()

        if pca == "pca":
            ThisPCA = PCA_BACK
        elif pca == "wpca":
            ThisPCA = WPCA
        elif pca == "empca":
            ThisPCA = EMPCA
        elif pca == "pca_backup":
            # ThisPCA = PCA
            pass

        if (weight is None) | (pca == "pca"):
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

        self.phi_components = np.sum(self.components < 0, axis=1) / len(
            self.components.T
        )
        self.zscore_components = np.mean(self.components, axis=1) / np.std(
            self.components, axis=1
        )

        self.var = pca.explained_variance_
        self.var_ratio = pca.explained_variance_ratio_

        self.wpca_model = pca

    def dim_reduction(self, reduction, nb_comp, weights):
        """The matrix should be format as S x D where S is the number of examples and D the dimension axis of the samples. Example : For 1000 sinusoid (random phase and amplitudes) sampled on a grid of 50 time elements, M = 1000 x 50"""
        matrix = self
        if weights is None:
            w = np.ones(np.shape(self.table))
        else:
            w = weights.copy()
        if reduction == "pca_scikit":
            matrix.PCA(comp_max=nb_comp)
            zscore = matrix.zscore_components
            phi = matrix.phi_components
            base_vec = matrix.vec

        elif reduction == "pca":
            matrix.WPCA("pca", comp_max=nb_comp)
            zscore = matrix.zscore_components
            phi = matrix.phi_components
            base_vec = matrix.vec

        elif reduction == "wpca":
            matrix.WPCA("wpca", comp_max=nb_comp, weight=w)
            zscore = matrix.zscore_components
            phi = matrix.phi_components
            base_vec = matrix.vec

        elif reduction == "empca":
            matrix.WPCA("empca", comp_max=nb_comp, weight=w)
            zscore = matrix.zscore_components
            phi = matrix.phi_components
            base_vec = matrix.vec

        elif reduction == "ica":
            matrix.ICA(comp_max=nb_comp)
            zscore = matrix.zscore_mixing
            phi = matrix.phi_mixing
            base_vec = matrix.ica_vec

        elif reduction == "uapca":
            matrix.UAPCA(comp_max=nb_comp, weight=w)
            zscore = matrix.zscore_uapca
            phi = matrix.phi_uapca
            base_vec = matrix.uapca_vec

        elif reduction == "svd":
            matrix.SVD(comp_max=nb_comp)
            zscore = matrix.zscore_svd
            phi = matrix.phi_svd
            base_vec = matrix.svd_vec
        else:
            print(
                "\n [ERROR] This algorithm of dimensionnal reduction does not exist\n"
            )
            myf.make_sound("Error")
            zscore = None
            phi = None
            base_vec = None

        return zscore, phi, base_vec

    def eval_model(self, model_fitted, table):
        coeffs = model_fitted.transform(table).T
        vec_fitted = np.dot(self.vec, coeffs).T
        self.eval_components = coeffs
        self.eval_vec_fitted = vec_fitted
        self.eval_vec_residues = table - vec_fitted

        rslope = np.median(
            (table - np.median(table, axis=0))
            / ((vec_fitted - np.median(vec_fitted, axis=0)) + 1e-6),
            axis=1,
        )
        rcorr = rslope * np.std(vec_fitted, axis=1) / (np.std(table, axis=1) + 1e-6)
        rcorr[abs(rcorr) > 1.3] = np.nan
        self.eval_r_corr = rcorr

    def fit_odr(self, weight=None):

        if np.shape(self.table)[1] != 2:
            print("The second dimension of your table should be 2")
        else:
            Signal = self.table

            if weight is None:
                w = np.ones(np.shape(Signal))
            else:
                w = weight

        mean_vec = np.sum(Signal * w, axis=1) / np.sum(w, axis=1)
        Signal = Signal - mean_vec[:, np.newaxis]
        R = Signal.copy()

        cloud = tableXY(R[:, 0], R[:, 1], w[:, 0], w[:, 1])
        cloud.fit_line_odr()
        vec1 = np.array([1, cloud.odr_slope])
        vec1 = vec1 / np.sqrt(np.sum(vec1**2))
        vec2 = vec1.copy()
        vec2[0], vec2[1] = -vec2[1], vec2[0]
        self.odr_eigen_vec = np.array([vec1, vec2])

    def fit_base(self, base_vec, weight=None, num_sim=1):
        """weights define as 1/sigma**2 self.table = MxT, base_vec = NxT, N the number of basis element"""

        if np.shape(base_vec)[1] != np.shape(self.table)[0]:
            base_vec = np.array(
                [
                    base_vec[i] * np.ones(np.shape(self.table)[0])[:, np.newaxis]
                    for i in range(len(base_vec))
                ]
            )

        if (np.shape(self.table)[0] == np.shape(self.table)[1]) & (
            len(np.shape(self.table)) == 2
        ):
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
                (
                    self.table[i]
                    + np.random.randn(num_sim, len(weight[i])) / np.sqrt(weight[i])
                )
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
            [
                np.sum(coeff[j] * base_vec[:, j, :].T, axis=1)
                for j in range(len(self.table))
            ]
        )
        all_vec_fitted = np.array(
            [coeff[j] * base_vec[:, j, :].T for j in range(len(self.table))]
        )
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

    def fit_unique_base(self, base_vec, weight=None):
        """weights define as 1/sigma**2"""

        if weight is None:
            weight = np.ones(np.shape(self.table))
            weight = 1

        coeff = np.linalg.lstsq(
            base_vec.T * np.sqrt(weight), ((self.table) * np.sqrt(weight)).T, rcond=None
        )[0].T
        vec_fitted = np.dot(coeff, base_vec)
        self.coeff_fitted = coeff
        self.vec_fitted = vec_fitted
        vec_residues = self.table - vec_fitted
        vec_residues[self.table == 0] = 0
        self.vec_residues = vec_residues

        self.chi2 = np.sum(vec_residues**2 * weight, axis=1)
        coeff_pos = coeff * np.sign(np.median(coeff, axis=0))
        mean_coeff = np.mean(coeff_pos, axis=0)
        if sum(mean_coeff != 0):
            epsilon = 1e-6 * np.min(abs(mean_coeff[mean_coeff != 0]))
        else:
            epsilon = 1e-6
        self.zscore_base = mean_coeff / (np.std(coeff_pos, axis=0) + epsilon)
        self.phi_base = np.sum(coeff_pos < 0, axis=0) / len(coeff_pos)

    def fit_corr2(self, x, perm=1000, rm=False):
        y = self.table[0, :, :]
        w = self.table[1, :, :]
        yerr = 1 / np.sqrt(w)
        rcorr = np.nan * np.ones(len(y))
        std_rcorr = np.nan * np.ones(len(y))
        rslope = np.nan * np.ones(len(y))
        std_rslope = np.nan * np.ones(len(y))
        r0 = np.nan * np.ones(len(y))
        s0 = np.nan * np.ones(len(y))
        rms = np.nan * np.ones(len(y))
        med_err = np.nan * np.ones(len(y))
        inter = np.nan * np.ones(len(y))
        a = -1
        for j in tqdm(range(len(self.table[0, :, :]))):
            a += 1
            if np.product(w[j, :] != 0):
                t = tableXY(x, y[j], yerr[j])
                if rm:
                    t.rm_outliers(who="Y", m=2)
                rms[j] = DescrStatsW(y[j], weights=w[j]).std
                med_err[j] = np.median(yerr[j])
                t.fit_line(perm=perm)
                rcorr[a] = t.r_pearson_w
                std_rcorr[a] = t.r_errpearson_w
                rslope[a] = t.lin_slope_w
                inter[a] = t.lin_intercept_w
                std_rslope[a] = t.lin_errslope_w
                r0[a] = t.r[0]
                s0[a] = t.s[0]
        self.r_pearson = np.array(rcorr)
        self.r_pearson_std = np.array(std_rcorr)
        self.slope = np.array(rslope)
        self.slope_std = np.array(std_rslope)
        self.r0 = r0
        self.s0 = s0
        self.rms = rms
        self.med_err = med_err
        self.inter = inter

    def fit_corr(self, x, weight=True):
        "fit a weighted linear correlation between the table and the x entry"

        if len(np.shape(self.table)) == 2:
            weight = False
            table = np.array([self.table, np.ones(np.shape(self.table))])
        else:
            table = self.table

        Signal = table[0, :, :]
        if len(Signal[0]) != len(x):
            print("x and y not the same size")
        else:
            if weight == False:
                Rerr = np.ones(np.shape(Signal))
            else:
                w = table[1, :, :]
                Rerr = 1 / np.sqrt(w)
            num = 0
            rpearson = np.ones(len(Signal)) * -99.9
            rslope = np.ones(len(Signal)) * -99.9
            rms = np.ones(len(Signal)) * -99.9
            med_err = np.ones(len(Signal)) * -99.9

            for j in tqdm(range(len(Signal))):
                corr = tableXY(x, Signal[j], Rerr[j])
                corr.fit_line_weighted()
                corr.rms_w()
                rpearson[num] = corr.r_pearson_w
                rslope[num] = corr.lin_slope_w
                rms[num] = corr.rms
                med_err[num] = np.nanmedian(corr.yerr)
                num += 1
            self.rpearson, self.rslope, self.rms, self.med_err = (
                rpearson,
                rslope,
                rms,
                med_err,
            )

    def fit_pca(
        self, base_vec, weight=None, ini_par=None, par_over=True, method="Powell"
    ):
        "return the coefficient liste fitting the pca base (there is on parameter in plus to account underestimating of the weights"
        Signal = self.table
        self.base_vec = base_vec

        if weight is None:
            mean_vec = np.mean(Signal, axis=1)
            Rerr = np.ones(np.shape(Signal))
        else:
            w = weight
            mean_vec = np.sum(Signal * w, axis=1) / np.sum(w, axis=1)
            Rerr = 1 / np.sqrt(w)

        Signal = Signal - mean_vec[:, np.newaxis]
        R = Signal.copy()

        def Chicarre(par, y, yerr):
            if par_over == False:
                par[0] = 0
            sn_2 = yerr**2 + par[0] ** 2
            Chi2 = (y - np.sum(base_vec * par[1:], axis=1)) ** 2 / sn_2 + np.log(sn_2)
            return np.sum(Chi2)

        if ini_par == None:
            ini_par = np.ones(np.shape(base_vec)[1] + 1)

        chi_liste = np.nan * np.ones(np.shape(R)[0])
        coefficient = np.nan * np.ones((np.shape(R)[0], np.shape(base_vec)[1] + 1))

        for j in tqdm(range(np.shape(R)[0])):
            result = minimize(
                Chicarre, ini_par, args=(R[j, :], Rerr[j, :]), method=method
            )
            result2 = minimize(
                Chicarre, ini_par, args=(R[j, :], Rerr[j, :]), method="SLSQP"
            )
            if result["fun"] > result2["fun"]:
                result = result2
            ParChi = result["x"]
            chi2 = result["fun"]

            chi_liste[j] = chi2
            coefficient[j, :] = ParChi

        self.coeff_fitting = coefficient
        self.coeff_n = coefficient[:, 1:] * np.std(
            base_vec, axis=0
        )  # normalise the coefficient by the rms of the base vector
        self.coeff_nn = (
            self.coeff_n / np.std(Signal, axis=1)[:, np.newaxis]
        )  # normalise the coefficient by the rms of the vectors decomposed
        self.coeff_nna = (
            abs(self.coeff_nn) / np.sum(abs(self.coeff_nn), axis=1)[:, np.newaxis]
        )
        self.chi2_residues = chi_liste
        self.vec_fitted = self.coeff_fitting[:, 1:].dot(self.base_vec.T)
        self.vec_residues = Signal - self.vec_fitted

    def cross_counter(self, col1, col2, order_index=None, order_col=None):
        """make a cross counter occurence of two columns in a panda table"""
        flavour1 = np.unique(self.table[col1])
        flavour2 = np.unique(self.table[col2])
        count = []
        for j in flavour1:
            for k in flavour2:
                count.append(sum((self.table[col1] == j) & (self.table[col2] == k)))
        count = np.array(count)
        count = np.reshape(count, (len(flavour1), len(flavour2)))
        df = pd.DataFrame(count)
        df.columns = flavour2
        df.index = flavour1

        if order_col != None:
            new_cols = df.columns[order_col].values
            df = df[new_cols]
        if order_index != None:
            df = df.loc[order_index]

        df["Sum"] = df.sum(axis=1)
        df.loc["Sum"] = df.sum(axis=0)

        return df

    def rm_outliers(self, cols=None, remove=True, m=2, kind="inter"):
        mask = np.ones(len(self.table[self.table.keys()[0]])).astype("bool")
        if cols == None:
            for j in self.table.keys():
                mask = mask & rm_out(self.table[j], m=m, kind=kind)[0]
        else:
            for j in cols:
                if j in self.table.keys():
                    mask = mask & rm_out(self.table[j], m=m, kind=kind)[0]
        if remove == True:
            self.table = self.table.loc[mask]
        else:
            return mask

    def drop_col_duplicated(self):
        self.table = self.table.loc[:, ~self.table.columns.duplicated()]

    def replace_outliers(self, m=2, kind="inter"):
        for j in np.arange(np.shape(self.table)[0]):
            test = tableXY(np.arange(np.shape(self.table)[1]), self.table[j, :])
            test.replace_outliers(m=m, kind=kind)
            self.table[j, :] = test.y

    def KDE(
        self,
        cycle=0,
        cols=None,
        nbins=70,
        fraction=1.0,
        edgecolor=None,
        s=10,
        n_levels=["2d", [1, 2]],
        col_species=None,
        liste=["Blues", "Reds", "Greens", "Greys", "Purples", "Oranges"],
        contour_color="k",
        alpha_c=0.7,
        alpha_s=0.8,
        coeff_alpha=[0.2, 0],
    ):
        """cycle=0,1,.. to change the order of plots if species      cols [name1,name2] column of the table to display      nbins=40 for the kde resampling      fraction=[0,1] fraction of data point display randomly    col_species='name' column for the species"""
        species = [np.ones(len(self.table[self.table.keys()[0]])).astype("bool")]
        keys_table = self.table.keys()
        if col_species != None:
            species_name = np.unique(self.table[col_species])
            species = []
            for j in species_name:
                species.append(self.table[col_species] == j)

        keys_table = cols

        indices = np.arange(len(species)).tolist()
        for it in range(cycle):
            indices = indices[1:] + [indices[0]]
        for num, idx in enumerate(indices):
            x = self.table[keys_table[0]].loc[species[idx]]
            DX = x.max() - x.min()
            y = self.table[keys_table[1]].loc[species[idx]]
            DY = y.max() - y.min()

            nbins = nbins
            k = kde.gaussian_kde(np.array([x, y]))
            xi1, yi1 = np.mgrid[
                x.min() - 0.5 * DX : x.max() + 0.5 * DX : nbins * 1j,
                y.min() - 0.5 * DY : y.max() + 0.5 * DY : nbins * 1j,
            ]
            z = k(np.vstack([xi1.flatten(), yi1.flatten()]))

            if type(n_levels) != int:
                if n_levels[0] == "1d":
                    niveaux = np.sort(
                        [
                            np.hstack(z)[
                                np.hstack(z).argsort()[::-1][
                                    myf.find_nearest(
                                        np.cumsum(np.sort(np.hstack(z))[::-1]),
                                        (1 - 2 * (1 - norm_gauss.cdf(j))) * np.sum(z),
                                    )
                                ]
                            ]
                            for j in n_levels[1]
                        ]
                    )
                elif n_levels[0] == "2d":
                    niveaux = np.sort(
                        [
                            np.hstack(z)[
                                np.hstack(z).argsort()[::-1][
                                    myf.find_nearest(
                                        np.cumsum(np.sort(np.hstack(z))[::-1]),
                                        (1 - np.exp(-0.5 * j**2)) * np.sum(z),
                                    )[0]
                                ]
                            ]
                            for j in n_levels[1]
                        ]
                    )
                niveaux = np.hstack(niveaux)
                niveaux = np.append(niveaux, 2 * z.max())
                niveaux = np.sort(niveaux)
                self.niveaux = niveaux
                nlevel = 10
            else:
                niveaux = None
                nlevel = n_levels

            plt.contourf(
                xi1,
                yi1,
                z.reshape(xi1.shape),
                nlevel,
                cmap=liste[idx],
                levels=niveaux,
                alpha=alpha_c + coeff_alpha[num],
            )
            if contour_color != None:
                plt.contour(
                    xi1,
                    yi1,
                    z.reshape(xi1.shape),
                    n_levels,
                    colors=contour_color,
                    levels=niveaux,
                    linewidths=1,
                )

            sample = np.random.choice(
                np.arange(len(x)), np.int(fraction * len(x)), replace=False
            )
            plt.scatter(
                x.iloc[sample],
                y.iloc[sample],
                color=np.array(["b", "r", "g", "k", "purple", "orange"])[idx],
                s=s,
                edgecolor=edgecolor,
                alpha=alpha_s + coeff_alpha[num],
            )

    def backwardElimination(self, col_species, SL, cols=None):
        """linear regression with backward suppresion to remove variable with p-value lower than SL"""
        keys_table = self.table.keys()[self.table.keys() != col_species]
        if cols != None:
            key = np.array([])
            for name in keys_table:
                if cols[0] == "k":
                    if name in cols[1]:
                        key = np.append(key, name)
                if cols[0] == "r":
                    if name not in cols[1]:
                        key = np.append(key, name)
            keys_table = key
        X = self.table[keys_table].copy()
        X = X.reset_index(drop=True)
        y = self.tabkle[col_species].copy()
        y = y.reset_index(drop=True)

        X = np.append(
            arr=np.ones((30, 1)).astype(int), values=X, axis=1
        )  # <---- change this 30
        numVars = len(X[0])
        temp = np.zeros((30, 6)).astype(int)  # <---- change this 30
        for i in range(0, numVars):
            regressor_OLS = sm.OLS(y, X).fit()
            maxVar = max(regressor_OLS.pvalues).astype(float)
            adjR_before = regressor_OLS.rsquared_adj.astype(float)
            if maxVar > SL:
                for j in range(0, numVars - i):
                    if regressor_OLS.pvalues[j].astype(float) == maxVar:
                        temp[:, j] = X[:, j]
                        X = np.delete(X, j, 1)
                        tmp_regressor = sm.OLS(y, X).fit()
                        adjR_after = tmp_regressor.rsquared_adj.astype(float)
            if adjR_before >= adjR_after:
                X_rollback = np.hstack((X, temp[:, [0, j]]))
                X_rollback = np.delete(X_rollback, j, 1)
                print(regressor_OLS.summary())
                return X_rollback
            else:
                continue
                regressor_OLS.summary()
                return X

    def histo_cumu(
        self, density=True, cumulative=True, cols=None, nbins=50, xmin=None, xmax=None
    ):
        keys_table = self.table.keys()
        if cols is not None:
            key = np.array([])
            for name in keys_table:
                if cols[0] == "k":
                    if name in cols[1]:
                        key = np.append(key, name)
                if cols[0] == "r":
                    if name not in cols[1]:
                        key = np.append(key, name)
            keys_table = key
        self.keys_table = keys_table
        self.table = self.table.dropna()
        stats = self.table.describe()
        for num, j in enumerate(self.keys_table):
            median = stats.loc["50%"][j]
            q1 = stats.loc["25%"][j]
            q3 = stats.loc["75%"][j]
            plt.plot(len(self.keys_table), 1, num + 1)
            plt.title(j)
            plt.axvline(x=q1, color="k", ls=":")
            plt.axvline(x=median, color="k", ls="-")
            plt.axvline(x=q3, color="k", ls="-.")
            plt.hist(self.table[j], nbins, cumulative=cumulative, density=density)
            plt.xlim(xmin, xmax)

    def plot3D(
        self,
        cols=None,
        col_species=None,
        Show=False,
        fraction=1,
        xlabel="",
        ylabel="",
        zlabel="",
        marker="o",
        ax=None,
        s=20,
        alpha_p=1,
        alpha_c=0.5,
        proj=False,
        nbins=50,
        n_levels=["2d", [1, 2]],
    ):
        cmap = ["Blues", "Greens", "Reds", "Greys", "Purples", "Oranges"]
        color = ["b", "g", "r", "k", "purple", "orange"]
        if len(cols) != 3:
            print("3 columns must me specified")
        else:
            for enum, species in enumerate(np.unique(self.table[col_species])):
                if enum == 0:
                    test_save = tableXYZ(
                        np.array(
                            self.table.loc[self.table[col_species] == species, cols[0]]
                        ),
                        np.array(
                            self.table.loc[self.table[col_species] == species, cols[1]]
                        ),
                        np.array(
                            self.table.loc[self.table[col_species] == species, cols[2]]
                        ),
                    )
                    test_save.plot3D(
                        Show=False,
                        fraction=fraction,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        zlabel=zlabel,
                        color=color[enum],
                        cmap=cmap[enum],
                        marker=marker,
                        ax=ax,
                        s=s,
                        alpha_p=alpha_p,
                        alpha_c=alpha_c,
                        proj=proj,
                        nbins=nbins,
                        n_levels=n_levels,
                    )
                else:
                    test = tableXYZ(
                        np.array(
                            self.table.loc[self.table[col_species] == species, cols[0]]
                        ),
                        np.array(
                            self.table.loc[self.table[col_species] == species, cols[1]]
                        ),
                        np.array(
                            self.table.loc[self.table[col_species] == species, cols[2]]
                        ),
                    )
                    test.plot3D(
                        Show=Show,
                        fraction=fraction,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        zlabel=zlabel,
                        color=color[enum],
                        cmap=cmap[enum],
                        marker=marker,
                        ax=test_save.ax,
                        s=s,
                        alpha_p=alpha_p,
                        alpha_c=alpha_c,
                        proj=proj,
                        nbins=nbins,
                        n_levels=n_levels,
                    )

    def peculiar_parameter(self, number, m=1.5):
        summarize = self.table.describe()
        IQ = summarize.loc["75%"] - summarize.loc["25%"]
        value = self.table.loc[number, IQ.keys()] - summarize.loc["50%"]

        cond1 = 1.5 * IQ < value
        cond2 = -1.5 * IQ > value
        cond = cond1 | cond2

        columns = [j for j in cond.loc[cond == True].keys()]
        print(columns)
        self.peculiar_columns = columns

    def match_nearest_vector(self, base2, m_out=0, Plot=False):
        """Match two basis of vectors where B = TxN T vector dimension, N number of vectors"""
        base1 = np.array(self.table)
        if np.shape(base2) != np.shape(base1):
            print("Shape of not equivalent")
            pouet

        T, N = np.shape(base1)
        matrix = table(np.hstack([base1, base2]).T)
        matrix.r_matrix(Plot=False, absolute=True)
        rcorr = matrix.matrix_corr[0:N, N:]
        perms = np.array([list(i) for i in set(itertools.permutations(np.arange(N)))])
        perms2 = np.arange(N) * np.ones(len(perms))[:, np.newaxis].astype("int")
        all_sum = np.sum(rcorr[perms2, perms], axis=1)
        best = np.argmax(all_sum)

        rcorr3 = []

        c = -1
        rcorr2 = rcorr[perms2[best], perms[best]].copy()

        for i, j in zip(perms2[best], perms[best]):
            c += 1
            if Plot:
                plt.subplot(len(perms2[best]), 2, 2 * c + 1)
                plt.plot(np.array(matrix.table[0:5][i]))
                plt.subplot(len(perms2[best]), 2, 2 * c + 2)
                plt.plot(np.array(matrix.table[N:][j]))
            if m_out:
                test = tableXY(
                    np.array(matrix.table[0:5][i]), np.array(matrix.table[N:][j])
                )
                test.rm_outliers(who="Y", m=m_out, kind="inter")
                test.yerr /= 100
                test.fit_line(perm=10)
                rcorr2[i] = test.r_pearson_w

        return perms[best], rcorr[perms2[best], perms[best]], rcorr2

    def pairplot(
        self,
        cycle=0,
        cols=None,
        nbins=70,
        bins_method="scott",
        fraction=1.0,
        xline=None,
        yline=None,
        edgecolor=None,
        s=10,
        n_levels=["2d", [1, 2]],
        col_species=None,
        kde_plot=True,
        liste=["Blues", "Greens", "Reds", "Greys", "Purples", "Oranges"],
        contour_color="k",
        alpha_c=0.8,
        alpha_h=0.6,
        alpha_s=0.9,
        color_param=None,
    ):
        """for a table MxN with M the number of dots, N the number of parameters, cycle=0,1,.. to change the order of plots if species      cols ['k' or 'r', [name1,name2]] column of the table to display      nbins=40 for the kde resampling      fraction=[0,1] fraction of data point display randomly    col_species='name' column for the species"""

        if type(self.table) == np.ndarray:
            self.table = pd.DataFrame(self.table)

        self.table_backup = self.table

        species = [np.ones(len(self.table[self.table.keys()[0]])).astype("bool")]
        keys_table = self.table.keys()

        if cols is not None:
            if cols[0] == "k":
                key = np.array(cols[1])
            if cols[0] == "r":
                key = np.array([])
                for name in keys_table:
                    if name not in cols[1]:
                        key = np.append(key, name)
            keys_table = key

        matrix_index = np.arange(1, 1 + len(keys_table) ** 2)
        matrix_index = matrix_index.reshape(len(keys_table), len(keys_table))
        self.keys_table = keys_table
        self.table = self.table[keys_table.tolist()]
        self.table = self.table.dropna()
        self.index_kept = self.table.index

        self.table = self.table_backup.loc[self.index_kept].copy()

        if col_species is not None:
            if type(col_species) != str:
                species_name = col_species[1]
                col_species = col_species[0]
            else:
                species_name = np.unique(self.table[col_species])
            species = []
            for j in species_name:
                species.append(self.table[col_species] == j)
            keys_table = self.keys_table[self.keys_table != col_species]
            matrix_index = np.arange(1, 1 + len(keys_table) ** 2)
            matrix_index = matrix_index.reshape(len(keys_table), len(keys_table))
            histo_label = species_name.astype("str").tolist()
        else:
            self.table["dustbin"] = "trash"
            col_species = "dustbin"
            histo_label = None
            species_name = np.array(["trash"])
            species = [np.ones(len(self.table)).astype("bool")]

        self.table = self.table[keys_table.tolist() + [col_species]]
        self.species = species
        self.histo_label = histo_label
        self.matrix_index = matrix_index

        x_limites = []
        for i, j in enumerate(keys_table):
            table1 = [
                self.table[j].loc[species[items]] for items in range(len(species))
            ]
            Histo = histo(table1, histo=histo_label, xlabel=j, bins=bins_method)
            plt.subplot(len(keys_table), len(keys_table), np.diagonal(matrix_index)[i])
            Histo.plotmany(
                np.arange(len(species)).tolist(),
                alpha=alpha_h,
                colors=list(
                    np.array(["b", "g", "r", "k", "purple", "orange"])[
                        np.arange(len(species)).tolist()
                    ]
                ),
                normed=True,
                Show=False,
            )
            plt.tick_params(axis="y", labelleft=False, left=False)
            histo_label = [""] * len(species_name)
            ax = plt.gca()
            x_limites.append(ax.get_xlim())

        for i, j in enumerate(matrix_index[np.triu_indices(len(keys_table), 1)]):
            plt.subplot(len(keys_table), len(keys_table), j)
            indices = np.arange(len(species)).tolist()
            save = j.copy()
            for it in range(cycle):
                indices = indices[1:] + [indices[0]]
            for idx in indices:
                x = self.table[
                    keys_table[np.triu_indices(len(keys_table), 1)[1][i]]
                ].loc[species[idx]]
                DX = x.max() - x.min()
                y = self.table[
                    keys_table[np.triu_indices(len(keys_table), 1)[0][i]]
                ].loc[species[idx]]
                DY = y.max() - y.min()
                if kde_plot:
                    nbins = nbins
                    k = kde.gaussian_kde(np.array([x, y]))
                    xi1, yi1 = np.mgrid[
                        x.min() - 0.5 * DX : x.max() + 0.5 * DX : nbins * 1j,
                        y.min() - 0.5 * DY : y.max() + 0.5 * DY : nbins * 1j,
                    ]
                    z = k(np.vstack([xi1.flatten(), yi1.flatten()]))
                    self.z = z
                    if type(n_levels) != int:
                        if n_levels[0] == "1d":
                            niveaux = np.sort(
                                [
                                    np.hstack(z)[
                                        np.hstack(z).argsort()[::-1][
                                            myf.find_nearest(
                                                np.cumsum(np.sort(np.hstack(z))[::-1]),
                                                (1 - 2 * (1 - norm_gauss.cdf(j)))
                                                * np.sum(z),
                                            )[0]
                                        ]
                                    ]
                                    for j in n_levels[1]
                                ]
                            )
                        elif n_levels[0] == "2d":
                            niveaux = np.sort(
                                [
                                    np.hstack(z)[
                                        np.hstack(z).argsort()[::-1][
                                            myf.find_nearest(
                                                np.cumsum(np.sort(np.hstack(z))[::-1]),
                                                (1 - np.exp(-0.5 * j**2)) * np.sum(z),
                                            )[0]
                                        ]
                                    ]
                                    for j in n_levels[1]
                                ]
                            )
                        niveaux = np.hstack(niveaux)
                        niveaux = np.sort(niveaux)
                        niveaux = np.append(niveaux, z.max())
                        nlevel = 10
                    else:
                        nlevel = n_levels
                        niveaux = None
                    plt.contourf(
                        xi1,
                        yi1,
                        z.reshape(xi1.shape),
                        nlevel,
                        cmap=liste[idx],
                        levels=niveaux,
                        alpha=alpha_c,
                    )
                    plt.contour(
                        xi1,
                        yi1,
                        z.reshape(xi1.shape),
                        n_levels,
                        colors=contour_color,
                        levels=niveaux,
                        linewidths=1,
                    )

                color = np.array(["b", "g", "r", "k", "purple", "orange"])[idx]
                sample = np.random.choice(
                    np.arange(len(x)), np.int(fraction * len(x)), replace=False
                )
                if color_param is not None:
                    color = np.array(self.table[color_param].iloc[sample])
                    iq = myf.IQ(color)
                    color[color > np.percentile(color, 75) + 1.5 * iq] = (
                        np.percentile(color, 75) + 1.5 * iq
                    )
                    color[color < np.percentile(color, 25) - 1.5 * iq] = (
                        np.percentile(color, 25) - 1.5 * iq
                    )

                plt.scatter(
                    x.iloc[sample],
                    y.iloc[sample],
                    c=color,
                    s=s,
                    edgecolor=edgecolor,
                    alpha=alpha_s,
                )
                plt.tick_params(axis="x", direction="in", labelbottom=False, top=True)
                if xline is not None:
                    plt.axvline(x=xline, color="k")
                if yline is not None:
                    plt.axhline(y=yline, color="k")
            plt.xlim(x_limites[(save - 1) % len(keys_table)])
            plt.ylim(x_limites[(save - 1) // len(keys_table)])
        plt.subplots_adjust(hspace=0, top=0.99, left=0.01, right=0.99, bottom=0.09)
        self.table = self.table_backup

    def splitting_data(
        self,
        col_species,
        cols=None,
        categorical=False,
        seed=0,
        test_size=0.2,
        outliers=["inter", 2],
    ):
        self.table = self.table.sample(frac=1)  # shuffle
        self.table = self.table.reset_index(drop=True)
        keys_table = self.table.keys()[self.table.keys() != col_species]
        if cols != None:
            key = np.array([])
            for name in keys_table:
                if cols[0] == "k":
                    if name in cols[1]:
                        key = np.append(key, name)
                if cols[0] == "r":
                    if name not in cols[1]:
                        key = np.append(key, name)
            keys_table = key
        X = self.table.dropna(axis=0)[keys_table].copy()
        X = X.reset_index(drop=True)
        y = self.table.dropna(axis=0)[col_species].copy()
        y = y.reset_index(drop=True)

        if categorical:
            for num, name in enumerate(np.unique(y)):
                y.loc[y == name] = int(num)

        out = np.ones(len(X)).astype("bool")
        for j in X.keys():
            out = out & rm_out(X[j], m=outliers[1], kind=outliers[0])[0]
        X = X.loc[out]
        y = y.loc[out]
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        if categorical:
            for i, j in enumerate(np.unique(y)):
                print("classe %s : %s elements" % (i + 1, sum(y == j)))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def Machine_learning(
        self,
        col_species,
        xgb=True,
        logi=True,
        pred_full=False,
        objective="binary:logistic",
        plot_imp=False,
        cols=None,
        seed=0,
        outliers=["inter", 2],
        num_splits=3,
        test=[1, "xgb"],
        print_infos=True,
        regular=1,
        n_est=300,
        alpha=0.1,
        ax1=None,
        ax2=None,
        ax3=None,
    ):
        """need a panda Dataframe with np.nan on raws for predictions"""
        if "predictions_xgb" in self.table.keys():
            self.table = self.table.drop(columns=["predictions_xgb"])
        self.table = self.table.sample(frac=1)

        self.table = self.table.reset_index(drop=True)
        keys_table = self.table.keys()[self.table.keys() != col_species]
        if cols != None:
            key = np.array([])
            for name in keys_table:
                if cols[0] == "k":
                    if name in cols[1]:
                        key = np.append(key, name)
                if cols[0] == "r":
                    if name not in cols[1]:
                        key = np.append(key, name)
            keys_table = key

        X = self.table.dropna(axis=0)[keys_table].copy()
        X = X.reset_index(drop=True)
        y = self.table.dropna(axis=0)[col_species].copy()
        y = y.reset_index(drop=True)
        for num, name in enumerate(np.unique(y)):
            y.loc[y == name] = int(num)

        out = np.ones(len(X)).astype("bool")
        for j in X.keys():
            out = out & rm_out(X[j], m=outliers[1], kind=outliers[0])[0]
        X = X.loc[out]
        y = y.loc[out]
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        keys_table = np.setdiff1d(
            keys_table, np.array(list((X.nunique().index)[X.nunique() == 1]))
        )

        X = X[keys_table]

        nb_species_found = len(np.unique(y))
        for i, j in enumerate(np.unique(y)):
            print("classe %s : %s elements" % (i + 1, sum(y == j)))
        # test session
        self.save_sim = []
        self.save_prog = []
        if test[0] != 0:
            accuracy_xgb = []
            accuracy_logi = []
            precision_xgb = []
            precision_logi = []
            recall_xgb = []
            recall_logi = []
            coeff_power_xgb = []
            coeff_power_logi = []
            for j in np.arange(test[0]):
                skf = StratifiedKFold(
                    n_splits=num_splits, shuffle=True, random_state=j + seed
                )
                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    # XGB
                    if test[1] != "logi":
                        model = XGBClassifier(
                            objective=objective, n_estimators=n_est, learning_rate=alpha
                        )
                        model.fit(X_train, y_train)
                        mat = metrics.confusion_matrix(y_test, model.predict(X_test))
                        precision_xgb.append(
                            np.diag(mat) / (np.sum(mat, axis=0) + 0.001)
                        )
                        recall_xgb.append(np.diag(mat) / np.sum(mat, axis=1))
                        accuracy_xgb.append(np.trace(mat) / np.sum(mat))
                        coeff_power_xgb.append(model.feature_importances_.copy())
                        self.matrix_confusion_xgb = mat

                    if (test[0] == 1) & (print_infos == True) & (test[1] != "logi"):
                        eval_set = [(X_train, y_train), (X_test, y_test)]
                        eval_metric = ["error"]
                        model1 = XGBClassifier(
                            objective=objective, n_estimators=n_est, learning_rate=alpha
                        )
                        model1.fit(
                            X_train,
                            y_train,
                            eval_metric=eval_metric,
                            eval_set=eval_set,
                            verbose=False,
                        )
                        # graphic to show how iteration increase precision + diagnostiv of overfitting
                        ax3 = plt.subplot(1, 2, 1)
                        y1 = 100 * (
                            1 - np.array(model1.evals_result_["validation_0"]["error"])
                        )
                        y2 = 100 * (
                            1 - np.array(model1.evals_result_["validation_1"]["error"])
                        )
                        ax3.plot(myf.smooth(y1, 5), color="k")
                        ax3.plot(myf.smooth(y2, 5))
                        self.save_prog.append([y1, y2])
                        ax3.set_ylabel("Accuracy [%]")
                        ax3.set_xlabel("Iteration")
                        # graphic to show how accuracy chance with confidence level
                        model1 = XGBClassifier(
                            objective=objective, n_estimators=n_est, learning_rate=alpha
                        )
                        model1.fit(X_train, y_train, eval_metric="error")
                        proba = model1.predict_proba(X_test)[:, 1]
                        output = []
                        for k in np.arange(0, 0.45, 0.01):
                            keep = abs(proba - 0.5) >= k
                            y_test_small = y_test.loc[keep]
                            proba2 = np.round(proba[keep])
                            CAA = np.sum(proba2 + y_test_small == 0)
                            CBB = np.sum(proba2 * y_test_small == 1)
                            output.append(
                                [CAA, CBB, len(proba2), 100 * (CAA + CBB) / len(proba2)]
                            )
                        output = np.array(output)
                        ax2 = plt.subplot(2, 2, 2)
                        self.save_sim.append(output)
                        ax2.plot(
                            0.5 + np.arange(0, 0.45, 0.01), myf.smooth(output[:, 3], 3)
                        )
                        ax = plt.gca()
                        ax2.set_ylabel("accuracy [%]")
                        ax3 = plt.subplot(2, 2, 4, sharex=ax)
                        ax3.plot(
                            0.5 + np.arange(0, 0.45, 0.01), myf.smooth(output[:, 2], 3)
                        )
                        ax3.set_ylabel("Size sample")
                        ax3.set_xlabel("Confidence level threshold")
                        plt.subplots_adjust(hspace=0)

                        print("--- XGB results ---")
                        print("\n Confusion matrix : \n")
                        print(mat)
                        print("\n Score : " + np.str(model.score(X_test, y_test)))
                        print(
                            "\n Final report : \n "
                            + np.str(
                                metrics.classification_report(
                                    y_test, model.predict(X_test)
                                )
                            )
                            + "\n\n"
                        )

                    # Logistique
                    if test[1] != "xgb":
                        regularization = regular
                        regressor = linear_model.LogisticRegression(
                            penalty="l2", class_weight="balanced", C=regularization
                        )
                        regressor.fit(X_train, y_train)
                        mat = metrics.confusion_matrix(
                            y_test, regressor.predict(X_test)
                        )
                        precision_logi.append(
                            np.diag(mat) / (np.sum(mat, axis=0) + 0.001)
                        )
                        recall_logi.append(np.diag(mat) / np.sum(mat, axis=1))
                        accuracy_logi.append(np.trace(mat) / np.sum(mat))
                        coeff_power_logi.append(
                            regressor.coef_[0] * np.std(X_train, axis=0)
                        )

                    if (test[0] == 1) & (print_infos == True) & (test[1] != "xgb"):
                        print("--- Logi results ---")
                        print("\n Confusion matrix : \n")
                        print(mat)
                        print("\n Score : " + np.str(regressor.score(X_test, y_test)))
                        print(
                            "\n Final report : \n "
                            + np.str(
                                metrics.classification_report(
                                    y_test, regressor.predict(X_test)
                                )
                            )
                        )

            self.acc_xgb = np.array([0])
            self.prec_xgb = np.array([0])
            self.recall_xgb = np.array([0])
            self.acc_logi = np.array([0])
            self.prec_logi = np.array([0])
            self.recall_logi = np.array([0])

            if test[1] != "logi":
                self.acc_xgb = np.array(accuracy_xgb)
                self.prec_xgb = np.array(precision_xgb)
                self.recall_xgb = np.array(recall_xgb)
                self.coeff_power_xgb = pd.DataFrame(
                    np.array(coeff_power_xgb), columns=keys_table
                ).describe(percentiles=[0.025, 0.16, 0.50, 0.84, 0.975])

            if test[1] != "xgb":
                self.acc_logi = np.array(accuracy_logi)
                self.prec_logi = np.array(precision_logi)
                self.recall_logi = np.array(recall_logi)
                self.coeff_power_logi = pd.DataFrame(
                    np.array(coeff_power_logi), columns=keys_table
                ).describe(percentiles=[0.025, 0.16, 0.50, 0.84, 0.975])

        #            self.infos = pd.DataFrame({'acc_xgb':np.mean(self.acc_xgb, axis=0), 'acc_logi':np.mean(self.acc_logi, axis=0),
        #            'recall_xgb':np.mean(self.recall_xgb, axis=0), 'recall_logi':np.mean(self.recall_logi, axis=0),
        #            'prec_xgb':np.mean(self.prec_xgb, axis=0), 'prec_logi':np.mean(self.prec_logi, axis=0)})
        #            self.infos = self.infos.T

        X_train = X.copy()
        y_train = y.copy()

        if logi == True:
            save = self.table.copy()
            self.table = self.table.dropna(axis=0)
            # self.pairplot(col_species=col_species)
            # plt.show()
            self.table = save.copy()
            regularization = regular
            regressor = linear_model.LogisticRegression(penalty="l1", C=regularization)
            regressor.fit(X_train, y_train)
            self.logi_coef = regressor.coef_
            self.logi_intercept = regressor.intercept_
            print("\n\n Regressor coefficient : %s \n\n" % (regressor.coef_))
            print("\n\n Regressor intercept : %s \n\n" % (regressor.intercept_))
            self.model = regressor

            if pred_full:
                X_topred = self.table[keys_table]
            else:
                X_topred = self.table[keys_table][self.table[col_species].isna()]
            y_pred = regressor.predict(X_topred)

            for i, j in enumerate(np.unique(y_pred)):
                print("classe %s : %s elements" % (i + 1, sum(y_pred == j)))
            self.table["predictions_logi"] = self.table[col_species].copy()

            if pred_full:
                self.table["predictions_logi"] = y_pred
            else:
                self.table.loc[
                    self.table[col_species].isna(), "predictions_logi"
                ] = y_pred

            print(
                "\n\n Regressor score on test sample : %s"
                % (regressor.score(X_train, y_train))
            )
            self.Coeff_power = regressor.coef_[0] * np.std(X_train, axis=0)

        if xgb == True:
            model = XGBClassifier(objective=objective, n_estimators=n_est)

            model.fit(X_train, y_train)

            if plot_imp == True:
                if (ax1 is None) & (ax1 is None) & (ax1 is None):
                    plt.figure(figsize=(18, 6))
                    ax1 = plt.subplot(1, 3, 1)
                    ax2 = plt.subplot(1, 3, 2)
                    ax3 = plt.subplot(1, 3, 3)

                m = []
                columns_name = []
                if ax1 is not None:
                    plot_importance(
                        model,
                        ax=ax1,
                        importance_type="gain",
                        title="Feature imp. (gain)",
                    )
                    m1 = np.array(
                        [
                            [ax1.patches[i].get_width() for i in ax1.get_yticks()],
                            [
                                ax1.get_ymajorticklabels()[i].get_text()
                                for i in ax1.get_yticks()
                            ],
                        ]
                    )
                    m1 = m1[:, np.argsort(m1[1])]
                    m.append(m1)
                    columns_name.append("gain")
                if ax2 is not None:
                    plot_importance(
                        model,
                        ax=ax2,
                        importance_type="weight",
                        title="Feature imp. (weight)",
                    )
                    m2 = np.array(
                        [
                            [ax2.patches[i].get_width() for i in ax2.get_yticks()],
                            [
                                ax2.get_ymajorticklabels()[i].get_text()
                                for i in ax2.get_yticks()
                            ],
                        ]
                    )
                    m2 = m2[:, np.argsort(m2[1])]
                    m.append(m2)
                    columns_name.append("weight")
                if ax3 is not None:
                    plot_importance(
                        model,
                        ax=ax3,
                        importance_type="cover",
                        title="Feature imp. (cover)",
                    )
                    m3 = np.array(
                        [
                            [ax3.patches[i].get_width() for i in ax3.get_yticks()],
                            [
                                ax3.get_ymajorticklabels()[i].get_text()
                                for i in ax3.get_yticks()
                            ],
                        ]
                    )
                    m3 = m3[:, np.argsort(m3[1])]
                    m.append(m3)
                    columns_name.append("cover")

                self.feature_importance = pd.DataFrame(
                    (np.array(m)[:, 0, :].astype("float")).T,
                    index=m[0][1],
                    columns=columns_name,
                )

                plt.subplots_adjust(wspace=0.45, left=0.12, right=0.97)
                plt.show()
            # if save_model == True:
            # pickle.dump(model, open(directory_model+'model.p', 'wb'))
            self.model = model

            if pred_full:
                X_topred = self.table[keys_table]
            else:
                X_topred = self.table[keys_table][self.table[col_species].isna()]
            y_pred = model.predict_proba(X_topred)[
                :, 1
            ]  # changed to predict proba 16.11.20

            self.table["predictions_xgb"] = self.table[col_species].copy()

            if pred_full:
                self.table["predictions_xgb"] = y_pred
            else:
                self.table.loc[
                    self.table[col_species].isna(), "predictions_xgb"
                ] = y_pred

            print("\n\nNumber of prediction in each classes: \n")
            if nb_species_found == 2:
                y_pred = (y_pred > 0.5).astype("int")

            for i, j in enumerate(np.unique(y_pred)):
                print("classe %s : %s elements" % (i + 1, sum(y_pred == j)))

    def barplot(self, who=["xgb", "logi"]):
        if ("xgb" in who) & ("logi" not in who):
            sort = self.coeff_power_xgb.loc["50%"].argsort()[::-1]
            plt.bar(
                self.coeff_power_xgb.columns[sort],
                self.coeff_power_xgb.loc["50%"][sort],
                yerr=[
                    self.coeff_power_xgb.loc["50%"][sort]
                    - self.coeff_power_xgb.loc["16%"][sort],
                    self.coeff_power_xgb.loc["84%"][sort]
                    - self.coeff_power_xgb.loc["50%"][sort],
                ],
            )
        elif ("xgb" not in who) & ("logi" in who):
            sort = self.coeff_power_logi.loc["50%"].argsort()[::-1]
            plt.bar(
                self.coeff_power_logi.columns[sort],
                self.coeff_power_logi.loc["50%"][sort],
                yerr=[
                    self.coeff_power_logi.loc["50%"][sort]
                    - self.coeff_power_logi.loc["16%"][sort],
                    self.coeff_power_logi.loc["84%"][sort]
                    - self.coeff_power_logi.loc["50%"][sort],
                ],
            )
        else:
            sort_logi = self.coeff_power_logi.loc["50%"].argsort()[::-1]
            sort_xgb = self.coeff_power_xgb.loc["50%"].argsort()[::-1]
            plt.subplot(1, 2, 2)
            plt.bar(
                self.coeff_power_logi.columns[sort_logi],
                self.coeff_power_logi.loc["50%"][sort_logi],
                yerr=[
                    self.coeff_power_logi.loc["50%"][sort_logi]
                    - self.coeff_power_logi.loc["16%"][sort_logi],
                    self.coeff_power_logi.loc["84%"][sort_logi]
                    - self.coeff_power_logi.loc["50%"][sort_logi],
                ],
            )
            plt.subplot(1, 2, 1)
            plt.bar(
                self.coeff_power_xgb.columns[sort_xgb],
                self.coeff_power_xgb.loc["50%"][sort_xgb],
                yerr=[
                    self.coeff_power_xgb.loc["50%"][sort_xgb]
                    - self.coeff_power_xgb.loc["16%"][sort_xgb],
                    self.coeff_power_xgb.loc["84%"][sort_xgb]
                    - self.coeff_power_xgb.loc["50%"][sort_xgb],
                ],
            )
        plt.ylabel(r"Coefficient power $\times$ std", fontsize=12)


class tableXY(object):
    def __init__(self, x, y, *yerr):
        self.stats = pd.DataFrame({}, index=[0])
        self.y = np.array(y)  # vector of y

        if x is None:  # for a fast table initialisation
            x = np.arange(len(y))
        self.x = np.array(x)  # vector of x

        try:
            np.sum(self.y)
        except:
            self.y = np.zeros(len(self.x))
            self.yerr = np.ones(len(self.y))

        if len(x) != len(y):
            print("X et Y have no the same lenght")

        if len(yerr) != 0:
            if len(yerr) == 1:
                self.yerr = np.array(yerr[0])
                self.xerr = np.zeros(len(self.x))
            elif len(yerr) == 2:
                self.xerr = np.array(yerr[0])
                self.yerr = np.array(yerr[1])
        else:
            if sum(~np.isnan(self.y.astype("float"))):
                self.yerr = np.ones(len(self.x)) * myf.mad(
                    rm_out(self.y.astype("float"), m=2, kind="sigma")[1]
                )
                if not np.sum(abs(self.yerr)):
                    self.yerr = np.ones(len(self.x))
            else:
                self.yerr = np.ones(len(self.x))
            self.xerr = np.zeros(len(self.x))

    def rms_w(self):
        if len(self.x) > 1:
            self.rms = DescrStatsW(self.y, weights=1.0 / self.yerr**2).std
            self.weighted_average = DescrStatsW(
                self.y, weights=1.0 / self.yerr**2
            ).mean
            self.stats["rms"] = self.rms
        else:
            self.rms = 0
            self.weighted_average = self.y[0]

    def zscore(self):
        self.rms_w()
        zscore = self.rms / np.nanmedian(self.yerr)
        self.zs = zscore
        self.stats["zscore"] = self.zs

    def copy(self):
        return tableXY(self.x.copy(), self.y.copy(), self.xerr.copy(), self.yerr.copy())

    def rm_from_plot(self, invers=False):
        axes = plt.gca()
        xmin = axes.get_xlim()[0]
        xmax = axes.get_xlim()[1]
        ymin = axes.get_ylim()[0]
        ymax = axes.get_ylim()[1]
        new = tableXY(self.x, self.y, self.xerr, self.yerr)
        new.clip(min=[xmin, ymin], max=[xmax, ymax], replace=True, invers=invers)
        self.x = new.x
        self.y = new.y
        self.xerr = new.xerr
        self.yerr = new.yerr

    def export(self, name):
        """export in pickle format"""
        myf.pickle_dump(
            pd.DataFrame(
                np.array([self.x, self.y, self.yerr]).T, columns=["x", "y", "yerr"]
            ),
            open(name, "wb"),
        )

    def periodogram_gael(self, bins=20, p_min=0.8, p_max=1000, p_nb=5000, v1=False):

        if v1:
            periods = 1 / np.linspace(1 / p_max, 1 / p_min, p_nb)
        else:
            periods = np.linspace(p_min, p_max, p_nb)

        bins = np.linspace(0, 1, bins)

        all_phases = self.x % periods[:, np.newaxis] / periods[:, np.newaxis]

        bins_cumu = np.sum((all_phases[:, :, np.newaxis] <= bins), axis=1) / len(self.x)
        line = table(bins_cumu)
        base_vec = np.array([np.ones(len(bins)), bins])
        line.fit_base(base_vec)
        scores = np.std(line.vec_residues, axis=1)

        self.gael = tableXY(periods, scores)
        plt.plot(periods, scores)

    def cross_corr(
        self,
        array,
        fill_hole=False,
        dx=1,
        Draw=True,
        save=True,
        periodic=True,
        num_sim=100,
    ):
        self.y -= np.median(self.y)
        array -= np.median(array)

        indices = [
            np.arange(len(self.y))
        ]  # put the real correlation in the bootstrap vector
        for j in range(num_sim):
            indices.append(
                np.random.choice(np.arange(len(self.y)), len(self.y), replace=False)
            )
        indices = np.array(indices)

        x = np.round(self.x / dx, 0) * dx
        new_grid = np.arange(x.min(), x.max() + dx, dx)
        match = myf.match_nearest(x, new_grid)

        if not fill_hole:
            new_y_random = np.nan * np.zeros((num_sim + 1, len(new_grid)))
            new_y2_random = np.nan * np.zeros((num_sim + 1, len(new_grid)))
        else:
            new_y_random = np.nanmean(self.y) * np.zeros((num_sim + 1, len(new_grid)))
            new_y2_random = np.nanmean(array) * np.zeros((num_sim + 1, len(new_grid)))

        random1 = self.y[indices]
        random2 = array[indices]

        new_y_random[:, match[:, 1].astype("int")] = random1[
            :, match[:, 0].astype("int")
        ]
        new_y2_random[:, match[:, 1].astype("int")] = random2[
            :, match[:, 0].astype("int")
        ]

        autocorr_test = []
        for i, k in zip([0, len(new_grid)], [0, 1]):
            slider = new_y2_random.copy()
            if len(autocorr_test) != 0:
                autocorr_test = autocorr_test[::-1][:-1]
            for j in range(len(new_grid)):
                autocorr_test.append(np.nansum(slider * new_y_random, axis=1))
                if not periodic:
                    slider = np.insert(slider, i, np.nan, axis=1)[
                        :, k : np.shape(slider)[1] + k
                    ]
                else:
                    slider = np.roll(slider, 1, axis=0)
        autocorr = np.array(autocorr_test)[:, 0]
        noise = np.array(autocorr_test)[:, 1:]

        noise_level1 = np.nanmedian(noise, axis=1) + 1.5 * (
            np.nanpercentile(noise, 75, axis=1) - np.nanpercentile(noise, 25, axis=1)
        )
        # noise_level2 = np.nanmedian(noise,axis=1) + 2*1.48*(np.median(abs(noise-np.nanmedian(noise,axis=1)[:,np.newaxis]),axis=1)) #alternate definition

        if Draw:
            plt.scatter(
                np.arange(len(autocorr)) * dx - (len(autocorr) // 2) * dx,
                autocorr,
                color="k",
            )
            plt.plot(
                np.arange(len(autocorr)) * dx - (len(autocorr) // 2) * dx,
                noise_level1,
                color="r",
            )

        if save:
            self.crosscorrelation = tableXY(
                np.arange(len(autocorr)) * dx - (len(autocorr) // 2) * dx, autocorr
            )
        else:
            return (
                tableXY(
                    np.arange(len(autocorr)) * dx - (len(autocorr) // 2) * dx,
                    autocorr,
                    0 / np.sqrt(np.ones(len(autocorr))),
                ),
                noise_level1,
            )

    def autocorr(
        self,
        fill_hole=False,
        dx=1,
        Draw=True,
        save=True,
        periodic=False,
        norm=False,
        peak_vicinity=10,
        num_sim=100,
    ):
        if save:

            self.autocorrelation, noise_level = self.cross_corr(
                self.y,
                fill_hole=fill_hole,
                dx=dx,
                Draw=False,
                periodic=periodic,
                save=False,
                num_sim=num_sim,
            )
            # self.autocorrelation.y -= np.median(self.autocorrelation.y)
            self.autocorrelation.y /= self.autocorrelation.y.max()
            noise_level = tableXY(self.autocorrelation.x, noise_level)
            noise_level.y /= noise_level.y.max()
            center = np.where(noise_level.y == 1)[0]
            noise_level.y[center] = 0.5 * (
                noise_level.y[center - 1] + noise_level.y[center + 1]
            )
            noise_level.smooth(box_pts=10)
            noise_level = noise_level.y
            self.autocorrelation_noise_level = noise_level

            if norm:
                self.autocorrelation.clip(min=[0, None], replace=False)
                self.autocorrelation.clipped.find_max(vicinity=peak_vicinity)
                highest = self.autocorrelation.clipped.x_max[
                    np.argmax(self.autocorrelation.clipped.y_max)
                ]
                print("highest_peak : %.1f" % (highest))
                reference = tableXY(self.x, np.sin(2 * np.pi / highest * self.x))
                normalisation, dust = reference.cross_corr(
                    reference.y,
                    fill_hole=fill_hole,
                    dx=dx,
                    Draw=False,
                    periodic=periodic,
                    save=False,
                )
                normalisation.y -= np.median(normalisation.y)
                normalisation.y /= normalisation.y.max()
                normalisation.find_max(vicinity=peak_vicinity)
                maximum = tableXY(normalisation.x_max, normalisation.y_max)
                maximum.interpolate(
                    new_grid=self.autocorrelation.x, replace=True, method="linear"
                )
                self.autocorrelation.y /= maximum.y
                self.autocorrelation.yerr /= maximum.y
                noise_level /= maximum.y

            if Draw:
                plt.figure(figsize=(14, 14))
                plt.subplot(2, 1, 1)
                plt.plot(
                    self.autocorrelation.x, self.autocorrelation.y * 100, color="k"
                )
                plt.plot(
                    self.autocorrelation.x,
                    noise_level * 100,
                    color="r",
                    label="1.5IQ level",
                )
                plt.plot(self.autocorrelation.x, -noise_level * 100, color="r")
                plt.legend()
                plt.ylim(-50, 100)
                plt.xlim(0, None)
                plt.axhline(y=0, color="k", ls=":")
                plt.ylabel("Autocorrelation [%]", fontsize=14)
                plt.xlabel("Period [u.a]", fontsize=14)
                ax = plt.gca()
                plt.subplot(2, 1, 2, sharex=ax)
                plt.plot(
                    self.autocorrelation.x, abs(self.autocorrelation.y * 100), color="k"
                )
                plt.plot(self.autocorrelation.x, noise_level * 100, color="r")
                plt.ylim(1e-2, 100)
                plt.xlim(0, None)
                plt.yscale("log")
                plt.ylabel("Autocorrelation [%]", fontsize=14)
                plt.xlabel("Period [u.a]", fontsize=14)

        else:
            return self.cross_corr(
                self.y,
                fill_hole=fill_hole,
                dx=dx,
                Draw=Draw,
                periodic=periodic,
                save=False,
                num_sim=num_sim,
            )

    def switch(self):
        self.x, self.y = self.y, self.x
        self.xerr, self.yerr = self.yerr, self.xerr

    def intersect(self, table2):
        match = myf.match_nearest(self.x, table2.x)[:, 0:2].astype("int")
        mask1 = np.in1d(np.arange(len(self.x)), match[:, 0])
        mask2 = np.in1d(np.arange(len(table2.x)), match[:, 1])
        return (self.masked(mask1, replace=False), table2.masked(mask2, replace=False))

    def merge(self, table2):
        self.x = np.hstack([self.x, table2.x])
        self.y = np.hstack([self.y, table2.y])
        self.xerr = np.hstack([self.xerr, table2.xerr])
        self.yerr = np.hstack([self.yerr, table2.yerr])
        self.order()

    def switch_for(self, vec="old_values"):
        if type(vec) == str:
            self.y, self.yerr, self.x, self.xerr = (
                self.oldy,
                self.oldyerr,
                self.oldx,
                self.oldxerr,
            )
        else:
            self.oldy, self.oldyerr, self.oldx, self.oldxerr = (
                self.y,
                self.yerr,
                self.x,
                self.xerr,
            )
            self.x, self.xerr = self.y, self.yerr
            self.y, self.yerr = vec.y, vec.yerr

    def stack(self):
        self.x = np.hstack(self.x)
        self.y = np.hstack(self.y)

    def order(self, order=None):
        if order is None:
            order = self.x.argsort()
        self.order_liste = order
        self.x = self.x[order]
        self.y = self.y[order]
        self.xerr = self.xerr[order]
        self.yerr = self.yerr[order]

    def null(self):
        self.yerr = 0 * self.yerr

    def not_null(self):
        if sum(abs(self.yerr)):
            self.yerr[self.yerr == 0] = np.min(self.yerr[self.yerr != 0])
        else:
            self.yerr += 1

    def rv_shift(self, rv, method="linear", xmin=None, xmax=None, replace=True):
        """rv in kms, x wavelength in [\AA]"""
        vec = self.copy()
        x_init = vec.x.copy()
        vec.x = myf.doppler_r(vec.x, rv * 1000)[0]
        vec.interpolate(new_grid=x_init, method=method)
        i1 = 0
        i2 = len(self.x)

        if xmin is not None:
            i1 = myf.find_nearest(self.x, xmin)[0][0]

        if xmax is not None:
            i2 = myf.find_nearest(self.x, xmax)[0][0]

        if replace:
            self.y[i1:i2] = vec.y[i1:i2]
        else:
            self.shifted = self.copy()
            self.shifted.y[i1:i2] = vec.y[i1:i2]

    def clip(self, min=[None, None], max=[None, None], replace=True, invers=False):
        """This function seems sometimes to not work without any reason WARNING"""
        min2 = np.array(min).copy()
        max2 = np.array(max).copy()
        if min2[1] == None:
            min2[1] = np.nanmin(self.y) - 1
        if max2[1] == None:
            max2[1] = np.nanmax(self.y) + 1
        masky = (self.y <= max2[1]) & (self.y >= min2[1])
        if min2[0] == None:
            min2[0] = np.nanmin(self.x) - 1
        if max2[0] == None:
            max2[0] = np.nanmax(self.x) + 1
        maskx = (self.x <= max2[0]) & (self.x >= min2[0])
        mask = maskx & masky
        try:
            self.clip_mask = self.clip_mask & mask
        except:
            self.clip_mask = mask

        if invers:
            mask = ~mask
        self.clipped = tableXY(
            self.x[mask], self.y[mask], self.xerr[mask], self.yerr[mask]
        )
        if replace == True:
            self.x = self.x[mask]
            self.y = self.y[mask]
            self.yerr = self.yerr[mask]
            self.xerr = self.xerr[mask]
        else:
            self.clipx = self.x[mask]
            self.clipy = self.y[mask]
            self.clipyerr = self.yerr[mask]
            self.clipxerr = self.xerr[mask]

    def supress_nan(self):
        mask = (
            ~np.isnan(self.x)
            & ~np.isnan(self.y)
            & ~np.isnan(self.yerr)
            & ~np.isnan(self.xerr)
        )
        if sum(~mask) == len(mask):
            self.replace_nan()
        else:
            self.mask_not_nan = mask
            self.x = self.x[mask]
            self.y = self.y[mask]
            self.xerr = self.xerr[mask]
            self.yerr = self.yerr[mask]

    def supress_zero(self):
        mask = ~(self.x == 0) & ~(self.y == 0)
        if sum(~mask) == len(mask):
            self.replace_zero()
        else:
            self.mask_not_zero = mask
            self.x = self.x[mask]
            self.y = self.y[mask]
            self.xerr = self.xerr[mask]
            self.yerr = self.yerr[mask]

    def replace_nan(self):
        self.y[np.isnan(self.y)] = np.random.randn(sum(np.isnan(self.y)))
        self.x[np.isnan(self.x)] = np.random.randn(sum(np.isnan(self.x)))
        self.yerr[np.isnan(self.yerr)] = np.random.randn(sum(np.isnan(self.yerr)))
        self.xerr[np.isnan(self.xerr)] = np.random.randn(sum(np.isnan(self.xerr)))

    def supress_inf(self):
        mask = (
            ~(abs(self.x) == np.inf)
            & ~(abs(self.y) == np.inf)
            & ~(abs(self.yerr) == np.inf)
            & ~(abs(self.xerr) == np.inf)
        )
        self.mask_not_inf = mask
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.xerr = self.xerr[mask]
        self.yerr = self.yerr[mask]

    def supress_mask(self, mask):
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.xerr = self.xerr[mask]
        self.yerr = self.yerr[mask]

    def frac(self, frac, replace=False):
        loc = self.x
        sel = np.arange(len(loc))
        if frac != 1:
            sel = np.random.choice(
                np.arange(len(loc)), size=int(frac * len(loc)), replace=False
            )
        if replace:
            self.x = self.x[sel]
            self.xerr = self.xerr[sel]
            self.y = self.y[sel]
            self.yerr = self.yerr[sel]
        else:
            self.sub_frac = tableXY(
                self.x[sel], self.y[sel], self.xerr[sel], self.yerr[sel]
            )

    def supress_twins(self, occurence=2):
        match = self.y == self.y[:, np.newaxis]
        mask = np.sum(match, axis=0) < occurence
        self.masked(mask)

    def reject_twin(self):
        table = pd.DataFrame({"x": self.x, "y": self.y})
        mask = np.array(table.duplicated(subset="x", keep="first"))
        self.masked(~mask)

    def min_noise(self):
        med_err = np.median(self.yerr)
        min_noise = np.std(self.y) / 5
        self.yerr = np.sqrt(self.yerr**2 + min_noise**2)

    def recenter(self, who="both", weight=False):
        if (who == "X") | (who == "both") | (who == "x"):
            self.xmean = np.nanmean(self.x)
            self.x = self.x - np.nanmean(self.x)
        if (who == "Y") | (who == "both") | (who == "y"):
            self.ymean = np.nanmean(self.y)
            self.y = self.y - np.nanmean(self.y)

    def znorm(self, who="Y", recenter=False):
        if (who == "X") | (who == "both") | (who == "x"):
            self.xstd = np.nanmean(self.x)
            self.xerr = self.xerr / self.xstd
            self.x = self.x / self.xstd
        if (who == "Y") | (who == "both") | (who == "y"):
            self.ystd = np.nanstd(self.y)
            self.yerr = self.yerr / self.ystd
            self.y = self.y / self.ystd
        if recenter:
            self.recenter(who=who)

    def decenter(self, who="both"):
        if (who == "X") | (who == "both"):
            self.x = self.x + self.xmean
        if (who == "Y") | (who == "both"):
            self.y = self.y + self.ymean

    def inv(self):
        """used to convert frequency to period in periodiogram"""
        self.x = 1 / self.x[::-1]
        self.y = self.y[::-1]
        self.yerr = self.yerr[::-1]
        self.xerr = self.xerr[::-1]

    def fmt(self):
        self.x = np.array(self.x).astype("float")
        self.y = np.array(self.y).astype("float")
        self.xerr = np.array(self.xerr).astype("float")
        self.yerr = np.array(self.yerr).astype("float")

    def backup(self):
        self.x = self.x_backup.copy()
        self.y = self.y_backup.copy()
        self.xerr = self.xerr_backup.copy()
        self.yerr = self.yerr_backup.copy()

    def add_noise(self, noise, noise_x=0):
        "Add gaussian noise on the data"
        self.noisy = tableXY(
            self.x + np.random.randn(len(self.x)) * noise_x,
            self.y + np.random.randn(len(self.y)) * noise,
            np.sqrt(self.xerr**2 + noise_x**2),
            np.sqrt(self.yerr**2 + noise**2),
        )

    def rescale_yerr(self):
        self.yerr = np.nanstd(self.y) * np.ones(len(self.yerr)) + (
            self.yerr - self.yerr
        )

    def read_value(self, value, who="X"):
        if who == "X":
            index = myf.find_nearest(self.x, value)[0][0]
            return self.y[index]
        else:
            index = myf.find_nearest(self.y, value)[0][0]
            return self.x[index]

    def scale_with(self, tab, replace=False, method="fit_line"):
        table_xy = tab.copy()
        new = tableXY(table_xy.y, self.y, table_xy.yerr, self.yerr)
        new.recenter(who="both")
        if method == "fit_line":
            new.fit_line_weighted()
            scaling = new.lin_slope_w
        else:
            scaling = np.std(new.y) / np.std(new.x)

        self.scaling_factor = scaling

        new.y /= scaling
        new.yerr /= scaling
        new.y += np.mean(table_xy.y)
        new.x = self.x
        new.xerr = self.xerr
        if replace:
            self.y = new.y
            self.yerr = new.yerr
        else:
            self.scaled = new

    def scatter(
        self,
        c="none",
        alpha=np.ones(1),
        cmap="viridis",
        zorder=100,
        vmin=None,
        vmax=None,
        edgecolor=None,
    ):
        if type(c) != str:
            c = np.array(c)[self.x.argsort()]
        else:
            c = np.arange(len(self.x))

        c1 = c.copy()
        c1[c1 < np.nanpercentile(c1, 16)] = np.nanpercentile(c1, 16)
        c1[c1 > np.nanpercentile(c1, 84)] = np.nanpercentile(c1, 84)
        c1 -= c1.min()
        c1 = c1 / (c1.max() - c1.min())

        if cmap == "viridis":
            c1 = plt.cm.viridis(c1)  # must be a number between 0 and 1000
        elif cmap == "seismic":
            c1 = plt.cm.seismic(c1)
        if len(alpha) != 1:
            c1[:, -1] = alpha
            plt.scatter(
                self.x[self.x.argsort()], self.y[self.x.argsort()], c=c1, zorder=zorder
            )
        else:
            if vmin is None:
                vmin = np.nanpercentile(c, 16)
            if vmax is None:
                vmax = np.nanpercentile(c, 84)
            plt.scatter(
                self.x[self.x.argsort()],
                self.y[self.x.argsort()],
                c=c,
                cmap=cmap,
                zorder=zorder,
                vmin=vmin,
                vmax=vmax,
                edgecolor=edgecolor,
            )

    def species_recenter(self, species, ref=None, replace=True):

        spe = np.unique(species)
        shift = np.zeros(len(self.y))

        if (len(spe) > 1) & (len(species) == len(self.y)):
            val_median = np.array(
                [np.nanmedian(self.y[np.where(species == s)[0]]) for s in spe]
            )

            if ref is None:
                ref = 0
            else:
                ref = val_median[ref]

            val_median -= ref

            for k, s in enumerate(spe):
                shift[species == s] += val_median[k]
        else:
            shift += np.nanmedian(self.y)

        newy = self.y - shift

        self.species = species
        if replace == True:
            self.y = newy
        else:
            self.species_recentered = tableXY(self.x, newy, self.xerr, self.yerr)

    def substract_calib(self, calib):
        """HARPS_CCF_fwhm , HARPS_CCF_contrast"""
        self.calib = self.copy()

        if calib == "HARPS_CCF_fwhm":
            self.calib.y -= np.polyval(
                np.array([3.73878777e-06, 2.50768767e00]), self.calib.x
            )  # calibrated on HD10700

        elif calib == "HARPS_CCF_contrast":
            self.calib.y -= np.polyval(
                np.array([6.57656028e-07, 3.85548927e-01]), self.calib.x
            )  # calibrated on HD10700

    def two_point_smoothing(self, dx_max=1, replace=True):
        points_to_average = np.diff(self.x) <= dx_max
        newx = []
        newy = []
        newyerr = []
        for j in range(len(self.x) - 1):
            if points_to_average[j]:
                newx.append(0.5 * (self.x[j] + self.x[j + 1]))
                newyerr.append(
                    (1 / self.yerr[j] ** 2 + 1 / self.yerr[j + 1] ** 2) ** (-0.5)
                )
                newy.append(
                    (
                        self.y[j] / self.yerr[j] ** 2
                        + self.y[j + 1] / self.yerr[j + 1] ** 2
                    )
                    / (1 / self.yerr[j] ** 2 + 1 / self.yerr[j + 1] ** 2)
                )
            else:
                newx.append(self.x[j])
                newy.append(self.y[j])
                newyerr.append(self.yerr[j])
        self.sx = newx
        self.sy = newy
        self.syerr = newyerr
        first = self.y[0]
        last = self.y[-1]

        smoothed = interp1d(
            self.sx,
            self.sy,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )(self.x)
        smoothed[0] = first
        smoothed[-1] = last

        self.smoothed = tableXY(self.x, smoothed, self.xerr, self.yerr)

        if replace:
            self.y = interp1d(
                self.sx,
                self.sy,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )(self.x)
            self.y[0] = first
            self.y[-1] = last

        else:
            self.y_smoothed = interp1d(
                self.sx,
                self.sy,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )(self.x)
            self.y_smoothed[0] = first
            self.y_smoothed[-1] = last

    def myscatter(
        self,
        num=True,
        liste=None,
        factor=30,
        color="b",
        alpha=1,
        x_offset=0,
        y_offset=0,
        color_text="k",
        modulo=None,
    ):
        n = np.arange(len(self.x)).astype("str")
        if modulo is not None:
            newx = self.x % modulo
        else:
            newx = self.x
        plt.scatter(np.array(newx), np.array(self.y), color=color, alpha=alpha)
        ax = plt.gca()
        dx = (ax.get_xlim()[1] - ax.get_xlim()[0]) / factor
        dy = (ax.get_ylim()[1] - ax.get_ylim()[0]) / factor
        if num:
            for i, txt in enumerate(n):
                plt.annotate(
                    txt,
                    (np.array(newx)[i] + x_offset, np.array(self.y)[i] + y_offset),
                    color=color_text,
                )
        if liste is not None:
            for i, txt in enumerate(liste):
                plt.annotate(
                    txt,
                    (
                        np.array(newx)[i] + dx + x_offset,
                        np.array(self.y)[i] + dy + y_offset,
                    ),
                    color=color_text,
                )

    def myannotate(
        self,
        liste,
        colors=None,
        colors_line=None,
        rotation=None,
        direction="x",
        frac_shift=0.2,
        draw_line=False,
        fontsize=14,
        reference="data",
        color_cycle=["k", "b", "r", "g"],
        n_first=None,
    ):
        ax = plt.gca()
        values = [self.x, self.y][int(direction != "x")]
        axes = [ax.get_xlim(), ax.get_ylim()][int(direction != "x")]
        span = axes[1] - axes[0]
        if rotation is None:
            rotation = [0, 90][int(direction != "x")]
        if colors is None:
            colors_text = ["k"] * len(values)
        else:
            colors_text = [color_cycle[n] for n in colors]

        if colors_line is None:
            colors_line = ["k"] * len(values)
        else:
            colors_line = [color_cycle[n] for n in colors_line]

        liste = np.array([str(l) for l in liste])

        if n_first is not None:
            order = np.argsort(values)[::-1]
            liste[order[n_first:]] = ""

        for n, x in enumerate(values):
            if str(liste[n]) != "":
                cond = int(x < (axes[0] + 0.5 * (span)))
                if reference == "data":
                    val = x + frac_shift * span * np.sign(cond - 0.5)
                    val_line = x + frac_shift * span * np.sign(cond - 0.5) * 0.9
                else:
                    val = [axes[0], axes[1]][cond] + frac_shift * span * np.sign(
                        cond - 0.5
                    )
                    val_line = [axes[0], axes[1]][cond] + frac_shift * span * np.sign(
                        cond - 0.5
                    ) * 0.9
                if direction == "x":
                    plt.annotate(
                        "%s" % (str(liste[n])),
                        (val, self.y[n]),
                        ha=["right", "left"][cond],
                        va="center",
                        rotation=rotation,
                        fontsize=fontsize,
                        color=colors_text[n],
                    )
                    if draw_line:
                        plt.plot(
                            [val_line, self.x[n]],
                            [self.y[n], self.y[n]],
                            alpha=0.2,
                            color=colors_line[n],
                        )
                else:
                    plt.annotate(
                        "%s" % (str(liste[n])),
                        (self.x[n], val),
                        ha="center",
                        va=["top", "bottom"][cond],
                        rotation=rotation,
                        fontsize=fontsize,
                        color=colors_text[n],
                    )
                    if draw_line:
                        plt.plot(
                            [self.x[n], self.x[n]],
                            [val_line, self.y[n]],
                            alpha=0.2,
                            color=colors_line[n],
                        )

    def modulo(self, mod, phase_mod=0, modulo_norm=False):
        new = tableXY((self.x - phase_mod) % mod, self.y, self.xerr, self.yerr)
        if modulo_norm:
            new.x /= mod
        new.old_index_modulo = np.arange(len(self.y))[new.x.argsort()]
        new.old_x_values = self.x
        new.order()
        self.mod = new

    def slicing(self, ds, s0=0, replace=False):
        x = self.x[s0::ds]
        y = self.y[s0::ds]
        xerr = self.xerr[s0::ds]
        yerr = self.yerr[s0::ds]

        if replace:
            self.x = x
            self.y = y
            self.xerr = xerr
            self.yerr = yerr
        else:
            self.sliced = tableXY(x, y, xerr, yerr)

    def demodulo(self):
        self.demod = tableXY(self.mod.old_x_values, np.zeros(len(self.mod.x)))
        self.demod.y[self.mod.old_index_modulo] = self.mod.y
        self.demod.xerr[self.mod.old_index_modulo] = self.mod.xerr
        self.demod.yerr[self.mod.old_index_modulo] = self.mod.yerr

    def center_symmetrise(self, center, replace=False, Plot=False):
        x = self.x
        kernel = self.copy()
        window = np.min([np.max(x) - center, center - np.min(x)])
        nb_elem = len(x) * 10

        new_grid = np.ravel(
            np.linspace(center - window, center + window, 2 * nb_elem - 1)
        )
        kernel.interpolate(new_grid=new_grid)

        sym_kernel = np.ravel(
            np.mean(
                np.array([kernel.y[0 : nb_elem - 1], kernel.y[nb_elem:][::-1]]), axis=0
            )
        )
        sym_kernel = np.hstack([sym_kernel, kernel.y[nb_elem - 1], sym_kernel[::-1]])
        sym_kernel = tableXY(new_grid, sym_kernel)
        sym_kernel.interpolate(new_grid=x)

        if Plot:
            self.plot(ls="-")
            sym_kernel.plot(color="r", ls="-")

        if replace:
            self.y = sym_kernel.y
        else:
            self.y_sym = sym_kernel.y

    def ccf(
        self,
        mask,
        rv_sys=0,
        rv_range=15,
        weighted=True,
        ccf_oversampling=1,
        wave_min=None,
        wave_max=None,
    ):

        if len(np.shape(mask)) < 2:
            mask = np.hstack([mask[:, np.newaxis], np.ones(len(mask))[:, np.newaxis]])

        grid = self.x
        flux = self.y[:, np.newaxis].T
        flux_err = self.yerr[:, np.newaxis].T

        if rv_sys:
            mask[:, 0] = myf.doppler_r(mask[:, 0], rv_sys)[0]

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

        mask_min = np.min(mask[:, 0])
        mask_max = np.max(mask[:, 0])

        grid_min = int(myf.find_nearest(grid, myf.doppler_r(mask_min, -100000)[0])[0])
        grid_max = int(myf.find_nearest(grid, myf.doppler_r(mask_max, 100000)[0])[0])
        grid = grid[grid_min:grid_max]

        log_grid = np.linspace(np.log10(grid).min(), np.log10(grid).max(), len(grid))
        dgrid = log_grid[1] - log_grid[0]
        # dv = (10**(dgrid)-1)*299.792e6

        used_region = ((10**log_grid) >= mask_shifted[1][:, np.newaxis]) & (
            (10**log_grid) <= mask_shifted[0][:, np.newaxis]
        )
        used_region = (np.sum(used_region, axis=0) != 0).astype("bool")
        print(
            "\n [INFO] Percentage of the spectrum used : %.1f [%%] \n"
            % (100 * sum(used_region) / len(grid))
        )

        mask_wave = np.log10(mask[:, 0])
        mask_contrast = mask[:, 1] * weighted + (1 - weighted)

        log_grid_mask = np.arange(
            log_grid.min() - 10 * dgrid,
            log_grid.max() + 10 * dgrid + dgrid / 10,
            dgrid / 11,
        )
        log_mask = np.zeros(len(log_grid_mask))

        match = myf.identify_nearest(mask_wave, log_grid_mask)
        for j in np.arange(-5, 6, 1):
            log_mask[match + j] = (mask_contrast) ** 2

        all_flux = []
        all_flux.append(
            interp1d(
                np.log10(self.x),
                flux[0],
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )(log_grid)
        )
        flux = np.array(all_flux)

        all_flux_err = []
        all_flux_err.append(
            interp1d(
                np.log10(self.x),
                flux_err[0],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )(log_grid)
        )
        flux_err = np.array(all_flux_err)

        log_template = interp1d(
            log_grid_mask,
            log_mask,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )(log_grid)

        vrad, ccf_power, ccf_power_std = myf.ccf(
            log_grid[used_region],
            flux[:, used_region],
            log_template[used_region],
            rv_range=rv_range,
            oversampling=ccf_oversampling,
            spec1_std=flux_err[:, used_region],
        )  # to compute on all the ccf simultaneously

        self.ccf_profile = tableXY(vrad, np.ravel(ccf_power))
        self.ccf_profile.yerr /= np.max(self.ccf_profile.y)
        self.ccf_profile.y /= np.max(self.ccf_profile.y)

        ccf_profile = self.ccf_profile

        plt.figure(figsize=(18, 6))
        plt.axes([0.05, 0.1, 0.58, 0.75])
        plt.plot(self.x, self.y, color="k")
        plt.xlim(np.min(mask[:, 0]), np.max(mask[:, 0]))
        for j in mask[:, 0]:
            plt.axvline(x=j, color="b", alpha=0.5)
        plt.axes([0.68, 0.1, 0.3, 0.75])
        plt.scatter(ccf_profile.x, ccf_profile.y, color="k", marker="o")
        plt.axvline(x=0, ls=":", color="k")
        plt.axhline(y=1, ls=":", color="k")
        plt.ylim(0, 1.1)

        amp = np.percentile(ccf_profile.y, 95) - np.percentile(ccf_profile.y, 5)
        xmin = np.argmin(ccf_profile.y)
        x1 = myf.find_nearest(ccf_profile.y[0:xmin], 1 - amp / 2)[0][0]
        x2 = myf.find_nearest(ccf_profile.y[xmin:], 1 - amp / 2)[0][0]
        width = ccf_profile.x[xmin:][x2] - ccf_profile.x[0:xmin][x1]
        center = ccf_profile.x[xmin]

        ccf_profile.fit_gaussian(guess=[-amp, center, width, 1], Plot=True)
        plt.axvline(x=ccf_profile.params["cen"].value, color="r", alpha=0.3)

        self.ccf_params = ccf_profile.params

        plt.title(
            "C = %.2f +/- %.2f [%%] \n FWHM = %.2f +/- %.2f [km/s] \n RV = %.2f +/- %.2f [m/s]"
            % (
                100 * (-ccf_profile.params["amp"].value),
                ccf_profile.params["amp"].stderr,
                ccf_profile.params["wid"].value / 1000 * 2.355,
                ccf_profile.params["wid"].stderr / 1000 * 2.355,
                ccf_profile.params["cen"].value,
                ccf_profile.params["cen"].stderr,
            )
        )

    def plot_outliers_cut(self, m=1.5, kind="inter", color=0, label=None):

        ax = plt.gca()
        y1 = ax.get_ylim()[0]
        y2 = ax.get_ylim()[1]

        if not color:
            mask_out = ~myf.rm_outliers(self.y, m=m, kind=kind)[0]
        else:
            mask_out = self.y > (y1 + 0.95 * (y2 - y1))

        self.y[mask_out] = y1 + 0.95 * (y2 - y1)

        cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        plt.plot(self.x, self.y, "-", color=cycle_colors[color], label=label)
        plt.scatter(
            self.x[~mask_out], self.y[~mask_out], marker="o", color=cycle_colors[color]
        )
        plt.scatter(
            self.x[mask_out], self.y[mask_out], marker="x", color=cycle_colors[color]
        )

    def plot_lowest_perio(
        self, table_xy2, color1="b", color2="g", zorder_min=10, alpha=1
    ):
        period = 1 / self.freq[::-1]
        v1 = self.power[::-1]
        v2 = table_xy2.power[::-1]
        ymax = np.max([np.max(v1), np.max(v2)]) * 1.1

        mask_sup = v1 > v2
        mask_inf = v1 < v2

        c1 = v1.copy()
        c1[mask_inf] = np.nan
        c2 = v1.copy()
        c2[mask_sup] = np.nan
        c3 = v2.copy()
        c3[mask_inf] = np.nan
        c4 = v2.copy()
        c4[mask_sup] = np.nan

        for j in range(1, len(c1) - 1):
            if (np.isnan(c1[j])) & (~np.isnan(c1[j + 1])):
                c1[j] = v1[j]
            if (np.isnan(c3[j])) & (~np.isnan(c3[j + 1])):
                c3[j] = v2[j]
            if (np.isnan(c2[j])) & (~np.isnan(c2[j + 1])):
                c2[j] = v1[j]
            if (np.isnan(c4[j])) & (~np.isnan(c4[j + 1])):
                c4[j] = v2[j]

        for j in range(1, len(c1) - 1)[::-1]:
            if (np.isnan(c1[j])) & (~np.isnan(c1[j - 1])):
                c1[j] = v1[j]
            if (np.isnan(c3[j])) & (~np.isnan(c3[j - 1])):
                c3[j] = v2[j]
            if (np.isnan(c2[j])) & (~np.isnan(c2[j - 1])):
                c2[j] = v1[j]
            if (np.isnan(c4[j])) & (~np.isnan(c4[j - 1])):
                c4[j] = v2[j]

        plt.plot(period, c1, color=color1, zorder=zorder_min, alpha=alpha)
        plt.plot(period, c4, color=color2, zorder=zorder_min + 1, alpha=alpha)
        plt.plot(period, c2, color=color1, zorder=zorder_min + 2, alpha=alpha)
        plt.plot(period, c3, color=color2, zorder=zorder_min + 3, alpha=alpha)
        plt.ylim(0, ymax)

    def fill_between(
        self,
        color="k",
        alpha=0.2,
        borders=True,
        label=None,
        oversampling=1,
        scale="lin",
    ):

        x = self.x
        y = self.y
        inf = self.xerr
        sup = self.yerr
        if oversampling > 1:
            self.interpolate(new_grid=oversampling, replace=False, scale=scale)
            x = self.interpolated.x
            y = self.interpolated.y
            inf = self.interpolated.xerr
            sup = self.interpolated.yerr

        p = plt.plot(x, y, color=color, label=label)
        color = p[0].get_color()
        plt.fill_between(x, y + sup, y - inf, alpha=alpha, color=color)
        if borders:
            plt.plot(x, y - inf, color=color, lw=1, alpha=0.3)
            plt.plot(x, y + sup, color=color, lw=1, alpha=0.3)

    def plot(
        self,
        Show=False,
        color="k",
        label="",
        ls="",
        offset=0,
        mask=None,
        capsize=3,
        fmt="o",
        markersize=6,
        zorder=1,
        species=None,
        alpha=1,
        modulo=None,
        modulo_norm=False,
        cmap="viridis",
        new=False,
        phase_mod=0,
        periodic=False,
        frac=1,
        yerr=True,
    ):

        """For the mask give either the first and last index in a list [a,b] or the mask boolean"""

        if (len(self.x) > 10000) & (ls == "") & (modulo is None):
            ls = "-"

        if species is None:
            species = np.ones(len(self.x))

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors_species = [color] + prop_cycle.by_key()["color"]

        for num, selection in enumerate(np.unique(species)):

            color = colors_species[num]

            if num != 0:
                label = None

            if mask is None:
                mask2 = np.ones(len(self.x)).astype("bool")
            elif type(mask[0]) == int:
                mask2 = np.zeros(len(self.x)).astype("bool")
                mask2[mask[0] : mask[1]] = True
            else:
                mask2 = mask

            loc = np.where(species[mask2] == selection)[0]

            sel = np.arange(len(loc))
            if frac != 1:
                sel = np.random.choice(
                    np.arange(len(loc)), size=int(frac * len(loc)), replace=False
                )

            if new:
                plt.figure()

            if ls != "":
                plt.plot(
                    self.x[mask2][loc][sel],
                    self.y[mask2][loc][sel] + offset,
                    ls=ls,
                    zorder=zorder,
                    label=label,
                    color=color,
                    alpha=alpha,
                )
            else:
                if modulo is not None:
                    norm = (1 - modulo_norm) + modulo_norm * modulo
                    new_x = ((self.x[mask2][loc] - phase_mod) % modulo) / norm
                    plt.errorbar(
                        new_x[sel],
                        self.y[mask2][loc][sel] + offset,
                        xerr=self.xerr[mask2][loc][sel],
                        yerr=self.yerr[mask2][loc][sel] * int(yerr),
                        fmt=fmt,
                        color=color,
                        alpha=alpha,
                        capsize=capsize,
                        label=label,
                        markersize=markersize,
                        zorder=zorder,
                    )
                    plt.scatter(
                        new_x[sel],
                        self.y[mask2][loc][sel] + offset,
                        marker="o",
                        c=self.x[mask2][loc][sel],
                        cmap=cmap,
                        s=markersize,
                        zorder=zorder * 100,
                    )
                    if periodic:
                        for i in range(int(np.float(periodic))):
                            plt.errorbar(
                                new_x[sel] - (i + 1) * modulo / norm,
                                self.y[mask2][loc][sel] + offset,
                                xerr=self.xerr[mask2][loc][sel],
                                yerr=self.yerr[mask2][loc][sel],
                                fmt=fmt,
                                color=color,
                                alpha=0.3,
                                capsize=capsize,
                                markersize=markersize,
                                zorder=zorder,
                            )
                            plt.errorbar(
                                new_x[sel] + (i + 1) * modulo / norm,
                                self.y[mask2][loc][sel] + offset,
                                xerr=self.xerr[mask2][loc][sel],
                                yerr=self.yerr[mask2][loc][sel],
                                fmt=fmt,
                                color=color,
                                alpha=0.3,
                                capsize=capsize,
                                markersize=markersize,
                                zorder=zorder,
                            )
                else:
                    plt.errorbar(
                        self.x[mask2][loc][sel],
                        self.y[mask2][loc][sel] + offset,
                        xerr=self.xerr[mask2][loc][sel],
                        yerr=self.yerr[mask2][loc][sel] * int(yerr),
                        fmt=fmt,
                        color=color,
                        alpha=alpha,
                        capsize=capsize,
                        label=label,
                        markersize=markersize,
                        zorder=zorder,
                    )
        if Show == True:
            plt.legend()
            plt.show()

    def binning(self, bin_width, db=0, replace=False, debug=False):
        self.yerr[self.yerr == np.inf] = 1000 * self.yerr[self.yerr != np.inf].max()
        mini = self.x.min()
        maxi = self.x.max()
        binning = np.arange(mini + db, maxi, bin_width)
        if debug:
            print(binning, bin_width)
        self.binx = binning + bin_width / 2
        t = (self.x[:, np.newaxis] < self.binx + bin_width / 2.0) * (
            self.x[:, np.newaxis] > self.binx - bin_width / 2.0
        )
        Y = self.y[:, np.newaxis] * t
        Yerr = self.yerr[:, np.newaxis] * t
        Yerr[Yerr == 0] = np.inf
        Wy = 1 / Yerr**2

        summing = np.sum(Wy, axis=0)
        summing[summing == 0] = np.nan
        new_y = np.sum(Y * Wy, axis=0) / summing
        new_yerr = 1 / np.sqrt(summing)

        self.binxerr = np.ones(len(self.binx)) * bin_width / 2.0
        self.biny = new_y
        self.binyerr = new_yerr

        mask = (
            ~np.isnan(self.binx)
            & ~np.isnan(self.biny)
            & ~np.isnan(self.binyerr)
            & ~np.isnan(self.binxerr)
        )
        self.binx = self.binx[mask]
        self.biny = self.biny[mask]
        self.binxerr = self.binxerr[mask]
        self.binyerr = self.binyerr[mask]

        self.binned = tableXY(self.binx, self.biny, self.binxerr, self.binyerr)

        if replace == True:
            self.x_backup = self.x.copy()
            self.y_backup = self.y.copy()
            self.xerr_backup = self.xerr.copy()
            self.yerr_backup = self.yerr.copy()
            self.x = self.binx
            self.y = self.biny
            self.xerr = self.binxerr
            self.yerr = self.binyerr

    def find_max(self, vicinity=3):
        self.index_max, self.y_max = myf.local_max(self.y, vicinity=vicinity)
        self.index_max = self.index_max.astype("int")
        self.x_max = self.x[self.index_max.astype("int")]
        self.max_extremum = tableXY(self.x_max, self.y_max)

    def find_min(self, vicinity=3):
        self.index_min, self.y_min = myf.local_max(-self.y, vicinity=vicinity)
        self.index_min = self.index_min.astype("int")
        self.y_min *= -1
        self.x_min = self.x[self.index_min.astype("int")]
        self.min_extremum = tableXY(self.x_min, self.y_min)

    def smooth(self, box_pts=5, shape="rectangular", replace=True):
        self.y_smoothed = myf.smooth(self.y, box_pts, shape=shape)

        self.smoothed = tableXY(self.x, self.y_smoothed, self.xerr, self.yerr)

        if replace:
            self.x_backup = self.x.copy()
            self.y_backup = self.y.copy()
            self.xerr_backup = self.xerr.copy()
            self.yerr_backup = self.yerr.copy()
            self.y = self.y_smoothed

    def joint_plot_multi(
        self,
        classes=None,
        columns=None,
        kind="kde",
        space=0,
        stat_func=None,
        shade=True,
        shade_lowest=False,
        n_levels=["2d", [0.5, 1, 2]],
        colors_s=["r", "b", "g"],
        nbins=50,
        s=5,
        alphas=0.5,
        alphac=0.8,
    ):
        import seaborn as sns

        sns.set(style="ticks", color_codes=True)

        if columns == None:
            columns = ["a", "b"]
        colors = ["r", "b", "g"]
        cmaps = ["Reds", "Blues", "Greens"]
        if type(classes) == np.ndarray:
            self.table = pd.DataFrame(
                np.array([self.x, self.y, classes]).T, columns=columns + ["classes"]
            )
        else:
            self.table = pd.DataFrame(
                np.array([self.x, self.y, np.array(["name"] * len(self.x))]).T,
                columns=columns + ["classes"],
            )

        g = sns.JointGrid(columns[0], columns[1], self.table, space=space)
        i = 0
        for day, day_tips in self.table.groupby("classes"):
            x = np.array(day_tips[columns[0]]).astype("float")
            DX = x.max() - x.min()
            y = np.array(day_tips[columns[1]]).astype("float")
            DY = y.max() - y.min()

            nbins = nbins
            k = kde.gaussian_kde(np.array([x, y]))
            xi1, yi1 = np.mgrid[
                x.min() - 0.5 * DX : x.max() + 0.5 * DX : nbins * 1j,
                y.min() - 0.5 * DY : y.max() + 0.5 * DY : nbins * 1j,
            ]
            z = k(np.vstack([xi1.flatten(), yi1.flatten()]))

            if type(n_levels) != int:
                if n_levels[0] == "1d":
                    niveaux = np.sort(
                        [
                            np.hstack(z)[
                                np.hstack(z).argsort()[::-1][
                                    myf.find_nearest(
                                        np.cumsum(np.sort(np.hstack(z))[::-1]),
                                        (1 - 2 * (1 - norm_gauss.cdf(j))) * np.sum(z),
                                    )
                                ]
                            ]
                            for j in n_levels[1]
                        ]
                    )
                elif n_levels[0] == "2d":
                    niveaux = np.sort(
                        [
                            np.hstack(z)[
                                np.hstack(z).argsort()[::-1][
                                    myf.find_nearest(
                                        np.cumsum(np.sort(np.hstack(z))[::-1]),
                                        (1 - np.exp(-0.5 * j**2)) * np.sum(z),
                                    )[0]
                                ]
                            ]
                            for j in n_levels[1]
                        ]
                    )
                niveaux = np.hstack(niveaux)
                niveaux = np.append(niveaux, 10 * z.max())
                niveaux = np.sort(niveaux)
                self.niveaux = niveaux.copy()
            if kind == "kde":
                sns.kdeplot(
                    day_tips[columns[0]],
                    ax=g.ax_marg_x,
                    legend=False,
                    color=colors[i],
                    shade=True,
                    alpha=0.5,
                )
                sns.kdeplot(
                    day_tips[columns[1]],
                    ax=g.ax_marg_y,
                    legend=False,
                    vertical=True,
                    color=colors[i],
                    shade=True,
                    alpha=0.5,
                )
                sns.kdeplot(
                    day_tips[columns[0]],
                    ax=g.ax_marg_x,
                    legend=False,
                    color="k",
                    shade=False,
                )
                sns.kdeplot(
                    day_tips[columns[1]],
                    ax=g.ax_marg_y,
                    legend=False,
                    vertical=True,
                    color="k",
                    shade=False,
                )
            if kind == "hist":
                g.ax_marg_x.hist(
                    np.array(day_tips[columns[0]]).astype("float"),
                    bins=np.linspace(self.x.min(), self.x.max(), 10),
                    color=colors[i],
                    alpha=0.5,
                )
                g.ax_marg_y.hist(
                    np.array(day_tips[columns[1]]).astype("float"),
                    bins=np.linspace(self.y.min(), self.y.max(), 10),
                    color=colors[i],
                    alpha=0.5,
                    orientation="horizontal",
                )
            sns.kdeplot(
                day_tips[columns[0]],
                day_tips[columns[1]],
                ax=g.ax_joint,
                legend=False,
                cmap=cmaps[i],
                alpha=alphac,
                shade=shade,
                shade_lowest=shade_lowest,
                n_levels=niveaux[1:],
            )
            sns.kdeplot(
                day_tips[columns[0]],
                day_tips[columns[1]],
                ax=g.ax_joint,
                legend=False,
                color="k",
                alpha=1,
                shade=False,
                n_levels=niveaux[1:],
            )
            # g.ax_joint.plot(day_tips[columns[0]], day_tips[columns[1]], "o", ms=s,alpha=alphas,color=colors[i],zorder=0, label=day)
            # g.ax_joint.collections[0].set_alpha(0) #remove the first contour
            i += 1
            # self.graphique = sns.jointplot(x=str(self.table.keys()[0]), y=str(self.table.keys()[1]), data=self.table, kind=kind, space=space, stat_func=stat_func, marginal_kws=marginal_kws, joint_kws=joint_kws, contour_color = contour_color, shade=shade, shade_lowest=shade_lowest, n_levels=n_levels)
            # self.graphique.plot_joint(plt.scatter, c="k", s=s,linewidth=1, marker="o",alpha=alphas)
            # self.graphique.ax_joint.collections[0].set_alpha(0) #remove the first contour
        sns.scatterplot(
            columns[0],
            columns[1],
            hue="classes",
            ax=g.ax_joint,
            data=self.table,
            zorder=0,
            palette=colors_s,
        )
        g.ax_joint.set_xlabel(columns[0], fontsize=15)
        g.ax_joint.set_ylabel(columns[1], fontsize=15)
        self.graphique = g

    def joint_plot(
        self,
        classes=None,
        columns=None,
        kind="kde",
        space=0,
        stat_func=None,
        marginal_kws={"color": "black", "lw": 0.5},
        joint_kws={"colors": None, "cmap": "Greys"},
        shade=True,
        shade_lowest=False,
        n_levels=["2d", [0.5, 1, 2, 3]],
        nbins=50,
        s=5,
        alphas=0.5,
    ):
        import seaborn as sns

        sns.set(style="ticks", color_codes=True)

        if columns == None:
            columns = ["a", "b"]
        if classes is not None:
            self.table = pd.DataFrame(
                np.array([self.x, self.y, np.array(["name"] * len(self.x))]).T,
                columns=columns + ["classes"],
            )
        self.table = pd.DataFrame(np.array([self.x, self.y]).T, columns=columns)
        x = self.x
        DX = x.max() - x.min()
        y = self.y
        DY = y.max() - y.min()

        nbins = nbins
        k = kde.gaussian_kde(np.array([x, y]))
        xi1, yi1 = np.mgrid[
            x.min() - 0.5 * DX : x.max() + 0.5 * DX : nbins * 1j,
            y.min() - 0.5 * DY : y.max() + 0.5 * DY : nbins * 1j,
        ]
        z = k(np.vstack([xi1.flatten(), yi1.flatten()]))

        if kind == "scatter":
            self.graphique = sns.jointplot(
                x=str(self.table.keys()[0]),
                y=str(self.table.keys()[1]),
                data=self.table,
                kind="scatter",
                space=space,
            )
        else:
            if type(n_levels) != int:
                if n_levels[0] == "1d":
                    niveaux = np.sort(
                        [
                            np.hstack(z)[
                                np.hstack(z).argsort()[::-1][
                                    myf.find_nearest(
                                        np.cumsum(np.sort(np.hstack(z))[::-1]),
                                        (1 - 2 * (1 - norm_gauss.cdf(j))) * np.sum(z),
                                    )
                                ]
                            ]
                            for j in n_levels[1]
                        ]
                    )
                elif n_levels[0] == "2d":
                    niveaux = np.sort(
                        [
                            np.hstack(z)[
                                np.hstack(z).argsort()[::-1][
                                    myf.find_nearest(
                                        np.cumsum(np.sort(np.hstack(z))[::-1]),
                                        (1 - np.exp(-0.5 * j**2)) * np.sum(z),
                                    )[0]
                                ]
                            ]
                            for j in n_levels[1]
                        ]
                    )
                niveaux = np.hstack(niveaux)
                niveaux = np.sort(niveaux)
                niveaux = np.append(niveaux, 2 * z.max())
                niveaux = niveaux[niveaux <= 1]
                niveaux = np.append(niveaux, 1)
                self.niveaux = niveaux

                n_levels = niveaux
            self.graphique = sns.jointplot(
                x=str(self.table.keys()[0]),
                y=str(self.table.keys()[1]),
                data=self.table,
                kind=kind,
                space=space,
                stat_func=stat_func,
                marginal_kws=marginal_kws,
                joint_kws=joint_kws,
                shade=shade,
                thresh=shade_lowest,
                n_levels=n_levels,
            )
            self.graphique.plot_joint(
                plt.scatter, c="k", s=s, linewidth=1, marker="o", alpha=alphas
            )
            self.graphique.ax_joint.collections[0].set_alpha(
                0
            )  # remove the first contour

    def binned_scatter(
        self,
        bin,
        extend=True,
        Plot=True,
        Show=False,
        color="k",
        size=5,
        xlabel="",
        ylabel="",
        alpha_p=0.5,
        cap=True,
        ls="",
        lw=None,
        replace=False,
    ):
        """allow to binned a scatter plot giving either the number of bins or the list of binned. If extend =True extend, the list of bins to match with the border of the data"""
        xmin = self.x.min()
        xmax = self.x.max()
        bins = 0
        if type(bin) == int:
            bins = np.linspace(xmin, xmax, bin)
        if (type(bin) == np.ndarray) & (extend == True):
            dbin = abs(bin[0] - bin[1])
            num1 = int(abs(xmin - bin[0]) / dbin) + 2
            num2 = int(abs(xmax - bin[-1]) / dbin) + 2
            bins = np.arange(bin[0] - num1 * dbin, bin[-1] + num2 * dbin, dbin)
        if (type(bin) == np.ndarray) & (extend == False):
            bins = bin
        self.bins = bins
        binning = np.zeros(len(bins) - 1) * np.float("nan")
        err_binning_sup = np.zeros(len(bins) - 1)
        err_binning_inf = np.zeros(len(bins) - 1)
        err_binning2_sup = np.zeros(len(bins) - 1)
        err_binning2_inf = np.zeros(len(bins) - 1)
        for j in np.arange(len(bins) - 1):
            if sum((self.x > bins[j]) & (self.x <= bins[j + 1])) > 1:
                binning[j] = np.median(
                    self.y[(self.x > bins[j]) & (self.x <= bins[j + 1])]
                )
            if cap == True:
                if sum((self.x > bins[j]) & (self.x <= bins[j + 1])) > 1:
                    binning_sup = np.percentile(
                        self.y[(self.x > bins[j]) & (self.x <= bins[j + 1])], 84
                    )
                    binning_inf = np.percentile(
                        self.y[(self.x > bins[j]) & (self.x <= bins[j + 1])], 16
                    )
                    bin2sig_sup = np.percentile(
                        self.y[(self.x > bins[j]) & (self.x <= bins[j + 1])], 97.5
                    )
                    bin2sig_inf = np.percentile(
                        self.y[(self.x > bins[j]) & (self.x <= bins[j + 1])], 2.5
                    )
                    err_binning_sup[j] = binning_sup - binning[j]
                    err_binning_inf[j] = binning[j] - binning_inf
                    err_binning2_sup[j] = bin2sig_sup - binning[j]
                    err_binning2_inf[j] = binning[j] - bin2sig_inf
                else:
                    err_binning_sup[j] = 0
                    err_binning_inf[j] = 0
                    err_binning2_sup[j] = 0
                    err_binning2_inf[j] = 0
        self.binx = np.array(
            [np.mean([bins[j], bins[j + 1]]) for j in np.arange(len(bins) - 1)]
        )
        self.biny = binning
        self.binsup = [err_binning_sup, err_binning2_sup]
        self.bininf = [err_binning_inf, err_binning2_inf]

        matrice = np.array([bins[0:-1], bins[1:]])
        bins_idx = (self.x > matrice[0][:, np.newaxis]) & (
            self.x <= matrice[1][:, np.newaxis]
        )
        indices = (np.arange(len(matrice[0])) * np.ones(len(self.x))[:, np.newaxis]).T
        self.binidx = np.sum(bins_idx * indices, axis=0).astype("int")

        if replace:
            self.x = self.binx
            self.y = self.biny
            self.yerr = 0.5 * (self.binsup[0] + self.bininf[0])
            self.xerr = np.zeros(len(self.binx))
        else:
            self.binned_data = tableXY(
                self.binx, self.biny, 0.5 * (self.binsup[0] + self.bininf[0])
            )

        if Plot:
            if alpha_p:
                plt.scatter(
                    self.x,
                    self.y,
                    marker="o",
                    color=color,
                    edgecolor="k",
                    alpha=alpha_p,
                    s=size,
                )
            if cap == True:
                xerr = (bins[1] - bins[0]) / 2
            else:
                xerr = np.zeros(len(binning))
            plt.errorbar(
                self.binx,
                binning,
                xerr=xerr,
                yerr=[err_binning_inf, err_binning_sup],
                c=color,
                marker="o",
                capsize=4,
                markeredgecolor="k",
                ls=ls,
                lw=lw,
            )
            plt.xlabel(xlabel, fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            if Show == True:
                plt.show()

    def binned_area(
        self,
        bin,
        extend=True,
        Show=False,
        color="k",
        size=5,
        xlabel="",
        ylabel="",
        alpha=0.7,
        alpha_p=0.5,
    ):
        """allow to binned a scatter plot giving either the number of bins or the list of binned. If extend =True extend, the list of bins to match with the border of the data"""
        plt.scatter(self.x, self.y, marker="o", color=color, alpha=alpha_p, s=size)
        xmin = self.x.min()
        xmax = self.x.max()
        bins = 0
        if type(bin) == int:
            bins = np.linspace(xmin, xmax, bin)
        if (type(bin) == np.ndarray) & (extend == True):
            dbin = abs(bin[0] - bin[1])
            num1 = int(abs(xmin - bin[0]) / dbin) + 2
            num2 = int(abs(xmax - bin[-1]) / dbin) + 2
            bins = np.arange(bin[0] - num1 * dbin, bin[-1] + num2 * dbin, dbin)
        if (type(bin) == np.ndarray) & (extend == False):
            bins = bin
        self.bins = bins
        binning = np.zeros(len(bins) - 1) * np.float("nan")
        err_binning_sup = np.zeros(len(bins) - 1)
        err_binning_inf = np.zeros(len(bins) - 1)
        for j in np.arange(len(bins) - 1):
            if sum((self.x > bins[j]) & (self.x <= bins[j + 1])) > 3:
                binning[j] = np.median(
                    self.y[(self.x > bins[j]) & (self.x <= bins[j + 1])]
                )
            if sum((self.x > bins[j]) & (self.x <= bins[j + 1])) > 5:
                binning_sup = np.percentile(
                    self.y[(self.x > bins[j]) & (self.x <= bins[j + 1])], 84
                )
                binning_inf = np.percentile(
                    self.y[(self.x > bins[j]) & (self.x <= bins[j + 1])], 16
                )
                err_binning_sup[j] = binning_sup - binning[j]
                err_binning_inf[j] = binning[j] - binning_inf
        plt.plot(
            np.array(
                [np.mean([bins[j], bins[j + 1]]) for j in np.arange(len(bins) - 1)]
            ),
            binning,
            color=color,
        )
        plt.fill_between(
            np.array(
                [np.mean([bins[j], bins[j + 1]]) for j in np.arange(len(bins) - 1)]
            ),
            binning - err_binning_inf,
            err_binning_sup + binning,
            color=color,
            alpha=alpha,
        )
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        if Show == True:
            plt.show()

    def diff(self, replace=True):
        diff = np.diff(self.y) / np.diff(self.x)
        new = tableXY(self.x[0:-1] + np.diff(self.x) / 2, diff)
        new.interpolate(new_grid=self.x, replace=True)

        self.deri = tableXY(self.x, new.y, self.xerr, self.yerr)

        if replace:
            self.y_backup = self.y
            self.y = new.y

    def add(self, obj):
        "obj as to be a tableXY instance"
        self.x = np.hstack([self.x, obj.x])
        self.y = np.hstack([self.y, obj.y])
        self.xerr = np.hstack([self.xerr, obj.xerr])
        self.yerr = np.hstack([self.yerr, obj.yerr])

    def masked(self, mask, replace=True):
        if replace:
            self.x = self.x[mask]
            self.y = self.y[mask]
            self.xerr = self.xerr[mask]
            self.yerr = self.yerr[mask]
        else:
            return tableXY(self.x[mask], self.y[mask], self.xerr[mask], self.yerr[mask])

    def baseline(self):
        return np.nanmax(self.x) - np.nanmin(self.x)

    def split_species(self, species):
        self.species_splited = [
            self.masked(species == s, replace=False) for s in np.unique(species)
        ]
        self.species_name = np.unique(species)
        self.species_mederr = {
            i: np.nanmedian(j.y)
            for i, j in zip(np.unique(species), self.species_splited)
        }

    def split_seasons(self, min_gap=50, Plot=False):
        seasons = myf.detect_obs_season(self.x, min_gap=min_gap)
        self.seasons_splited = [
            self.chunck(seasons[i, 0], seasons[i, 1] + 1)
            for i in np.arange(len(seasons[:, 0]))
        ]
        self.seasons_rms = []
        self.seasons_std = []
        self.seasons_meanx = []
        self.seasons_meany = []
        self.seasons_meany_std = []
        self.seasons_medx = []
        self.seasons_medy = []
        self.seasons_maxy = []
        self.seasons_miny = []
        self.seasons_maxx = []
        self.seasons_minx = []
        self.seasons_nb = []

        for i in range(len(self.seasons_splited)):
            if Plot:
                self.seasons_splited[i].plot(color=None)
            self.seasons_splited[i].rms_w()
            self.seasons_rms.append(self.seasons_splited[i].rms)
            self.seasons_std.append(np.std(self.seasons_splited[i].y))
            self.seasons_meanx.append(np.nanmean(self.seasons_splited[i].x))
            self.seasons_meany.append(
                np.nansum(self.seasons_splited[i].y / self.seasons_splited[i].yerr ** 2)
                / np.nansum(1 / self.seasons_splited[i].yerr ** 2)
            )
            self.seasons_meany_std.append(
                1 / np.sqrt(np.nansum(1 / self.seasons_splited[i].yerr ** 2))
            )
            self.seasons_medx.append(np.nanmedian(self.seasons_splited[i].x))
            self.seasons_medy.append(np.nanmedian(self.seasons_splited[i].y))
            self.seasons_miny.append(np.nanmin(self.seasons_splited[i].y))
            self.seasons_maxy.append(np.nanmax(self.seasons_splited[i].y))
            self.seasons_minx.append(np.nanmin(self.seasons_splited[i].x))
            self.seasons_maxx.append(np.nanmax(self.seasons_splited[i].x))
            self.seasons_nb.append(len(self.seasons_splited[i].x))

    def recenter_seasons(self, min_gap=50, replace=False):
        self.split_seasons(min_gap=min_gap)
        new = self.seasons_splited[0]
        new.y -= self.seasons_meany[0]
        for j in range(1, len(self.seasons_splited)):
            offset = self.seasons_meany[j]
            new_seasons = self.seasons_splited[j]
            new_seasons.y -= offset
            new.merge(new_seasons)

        if replace:
            self = new
        else:
            self.seasons_recentered = new

    def rm_offset_seasons(self, min_gap=40, replace=False):
        self.split_seasons(min_gap=min_gap)

        first = self.seasons_splited[0].copy()
        first.y -= self.seasons_meany[0]
        for mean, elem in zip(self.seasons_meany[1:], self.seasons_splited[1:]):
            elem.y -= mean
            first.add(elem)
        if replace:
            self = first.copy()
        else:
            self.seasons_recentered = first.copy()

    def rm_gap_seasons(self, min_gap=40, replace=False, jump=10):
        self.split_seasons(min_gap=min_gap)
        first = self.seasons_splited[0].copy()
        for elem in self.seasons_splited[1:]:
            elem.x -= np.min(elem.x)
            elem.x += np.max(first.x) + jump
            first.add(elem)

        if replace:
            self = first.copy()

        else:
            self.seasons_removed = first.copy()

    def seasons_periodogram(
        self,
        min_gap=40,
        split_graph=False,
        infos=False,
        deg=0,
        legend=None,
        axis_y_var="p",
        alpha=1,
    ):
        self.split_seasons(min_gap=min_gap)
        maxi = []
        c = 1
        nb = np.array([len(i.x) for i in self.seasons_splited])
        seasons_cuted = list(np.array(self.seasons_splited)[nb > 6])
        for j in range(len(seasons_cuted)):
            try:
                if not j:
                    plt.subplot(len(seasons_cuted), 1, c)
                    ax = plt.gca()
                else:
                    plt.subplot(len(seasons_cuted), 1, c, sharex=ax, sharey=ax)
                mean_time = Time.Time(
                    np.mean(seasons_cuted[j].x), format="mjd"
                ).decimalyear

                seasons_cuted[j].substract_polyfit(deg, replace=False)
                seasons_cuted[j].detrend_poly.periodogram(
                    Norm=True,
                    norm_val=True,
                    color=None,
                    legend=legend,
                    infos=infos,
                    level=0.90,
                    axis_y_var=axis_y_var,
                    alpha=alpha,
                )  # 10%FAP
                maxi.append(
                    np.max(
                        seasons_cuted[j].detrend_poly.power
                        / seasons_cuted[j].detrend_poly.fap
                    )
                )
                plt.ylabel("power")
                plt.tick_params(direction="in", top=True, which="both")

                if not j:
                    if legend is not None:
                        plt.legend()

                ax3 = plt.gca()
                ax2 = ax3.twinx()
                ax2.tick_params(labelright=False)
                ax2.set_ylabel("%.1f" % (mean_time), fontsize=14)
                c += 1
            except:
                pass

        plt.ylim(0, np.max(maxi))
        plt.subplots_adjust(top=0.95, bottom=0.06, hspace=0)

    def species_recontinuity(self, dist_max=100, alpha=1):
        """weights are defined by (1**alpha-x**alpha)**(1/alpha), points farther than dist_max are rejected of the continuity process"""
        alpha = abs(alpha)

        if len(self.species_name) > 1:
            match = myf.match_unique_closest(
                self.species_splited[0].x, self.species_splited[1].x
            )
            match[:, -1] = np.abs(match[:, -1])
            match = match[match[:, -1] != 0]

            if sum(match[:, -1] < dist_max) < 2:
                offset = 0
                print(
                    " [ERROR] No points closer than %s, closest pair found at : %s"
                    % (str(dist_max), str(np.nanmin(match[:, -1])))
                )
            else:
                match = match[match[:, -1] <= dist_max]

                y1 = self.species_splited[0].y[match[:, 0].astype("int")]
                y2 = self.species_splited[1].y[match[:, 1].astype("int")]

                match[:, -1] -= np.nanmin(match[:, -1])
                match[:, -1] /= np.nanmax(match[:, -1])
                match[:, -1] = (1**alpha - match[:, -1] ** alpha) ** (1 / alpha)

                diff = (
                    self.species_splited[1].y[match[:, 1].astype("int")]
                    - self.species_splited[0].y[match[:, 0].astype("int")]
                )
                diff_std = (1 / (match[:, -1] + 0.001) ** 2) * np.sqrt(
                    self.species_splited[1].yerr[match[:, 1].astype("int")] ** 2
                    + self.species_splited[0].yerr[match[:, 0].astype("int")] ** 2
                )

                offset = np.nansum(diff / diff_std**2) / np.nansum(1 / diff_std**2)

            self.species_splited[1].y -= offset

    def merge_species(self, replace=False):
        nb_species = len(self.species_name)
        x_merged = np.hstack([self.species_splited[j].x for j in range(nb_species)])
        xerr_merged = np.hstack(
            [self.species_splited[j].xerr for j in range(nb_species)]
        )
        y_merged = np.hstack([self.species_splited[j].y for j in range(nb_species)])
        yerr_merged = np.hstack(
            [self.species_splited[j].yerr for j in range(nb_species)]
        )
        species_merged = np.hstack(
            [
                [self.species_name[j]] * len(self.species_splited[j].x)
                for j in range(nb_species)
            ]
        )
        order = np.argsort(x_merged)

        self.species = species_merged[order]

        if replace:
            self.x = x_merged[order]
            self.y = y_merged[order]
            self.xerr = xerr_merged[order]
            self.yerr = yerr_merged[order]
        else:
            self.species_merged = tableXY(
                x_merged[order], y_merged[order], xerr_merged[order], yerr_merged[order]
            )

    def convert_x(self, fmt="decimalyear"):
        if fmt == "decimalyear":
            self.x = Time.Time(self.x, format="mjd").decimalyear

    def knee_detection(
        self, Plot=False
    ):  # from Satopää et al., 2011 [2] https://towardsdatascience.com/detecting-knee-elbow-points-in-a-graph-d13fc517a63c
        copy = self.copy()

        copy.x = myf.transform_min_max(copy.x)
        copy.y = myf.transform_min_max(copy.y)

        if copy.y[0] > copy.y[-1]:
            copy.y = 1 - copy.y

        d1 = abs(copy.y - copy.x)
        d2 = abs(copy.x - copy.y) / np.sqrt(2)

        diff_curve = d1 - d2
        knee = np.argmax(diff_curve)

        if Plot:
            plt.subplot(3, 1, 1)
            plt.plot(copy.x, copy.y)
            plt.subplot(3, 1, 2)
            plt.plot(copy.x, diff_curve)
            plt.axvline(x=copy.x[knee])
            plt.subplot(3, 1, 3)
            self.plot()
            plt.axvline(x=self.x[knee])

        return knee, self.x[knee]

    def extend_clustering(self, extend):
        diff = abs(np.diff(self.y))
        mini = np.min(diff[diff != 0])
        val, borders = myf.clustering(self.y, mini, 0)
        val = np.array([np.nanmean(v) for v in val])
        borders = borders[val != 0]
        val = val[val != 0]
        self.y = myf.flat_clustering(
            len(self.x), borders, elevation=val, extended=extend
        )

    def rm_chunck(self, gap_length=10):
        "rm or extract a chunck after using clustering function"
        matrice = myf.clustering(self.x, gap_length, 1)[-1]
        liste = []
        for j in range(len(matrice)):
            liste.append(self.chunck(matrice[j, 0], matrice[j, 1] + 2, inv=True))
        self.chuncks = liste

    def extract_chunck(self, gap_length=10):
        "rm or extract a chunck after using clustering function"
        matrice = myf.clustering(self.x, gap_length, 2)[-1]
        liste = []
        for j in range(len(matrice)):
            liste.append(self.chunck(matrice[j, 0], matrice[j, 1] + 2, inv=False))
        self.chuncks = liste

    def chunck(self, idx1, idx2, inv=False):
        idx1 = int(idx1)
        idx2 = int(idx2)
        if not inv:
            chunk = tableXY(self.x[idx1:idx2], self.y[idx1:idx2], self.yerr[idx1:idx2])
        else:
            chunk = tableXY(
                np.hstack([self.x[0:idx1], self.x[idx2:]]),
                np.hstack([self.y[0:idx1], self.y[idx2:]]),
                np.hstack([self.xerr[0:idx1], self.xerr[idx2:]]),
                np.hstack([self.yerr[0:idx1], self.yerr[idx2:]]),
            )
        return chunk

    def baseline_oversampled(self, oversampling=1000):
        return tableXY(
            np.linspace(np.nanmin(self.x), np.nanmax(self.x), oversampling),
            np.zeros(oversampling),
        )

    def substract_rolling(
        self,
        windows,
        replace=False,
        Draw=False,
        color="r",
        color_p="k",
        lw=1.5,
        alpha_p=0.5,
        zorder=10,
    ):
        model = np.ravel(
            pd.DataFrame(self.y)
            .rolling(windows, center=True, min_periods=1)
            .quantile(0.5)
        )
        sub_model = self.y - model

        self.detrend_roll = tableXY(self.x, sub_model, self.xerr, self.yerr)

        if Draw:
            self.plot(color=color_p, alpha=alpha_p)
            plt.plot(self.x, model, color=color, lw=lw, zorder=zorder)
        if replace:
            self.x_backup = self.x.copy()
            self.y_backup = self.y.copy()
            self.xerr_backup = self.xerr.copy()
            self.yerr_backup = self.yerr.copy()
            self.y = sub_model
        else:
            self.sub_model = sub_model

    def substract_line(self, replace=False, weight=True, Draw=False, param=False):
        """'current model line(a,b)"""
        if param is not False:
            par = param
        else:
            if weight:
                self.fit_line_weighted(Draw=Draw)
                par = [self.lin_slope_w, self.lin_intercept_w]
            else:
                par = self.fit_line(Draw=Draw)
                par = [self.lin_slope, self.lin_intercept]
        a, b = par[0], par[1]
        model = np.polyval([a, b], self.x)
        sub_model = self.y - model
        self.detrend_line = tableXY(self.x, sub_model, self.xerr, self.yerr)

        if replace:
            self.x_backup = self.x.copy()
            self.y_backup = self.y.copy()
            self.xerr_backup = self.xerr.copy()
            self.yerr_backup = self.yerr.copy()
            self.y = sub_model
        else:
            self.sub_model = sub_model

    def substract_polydisc(self, cut, degree=0, replace=False, Draw=False):
        self.fit_discontinuity(cut, degree=degree, Draw=Draw)
        sub_model = self.y - self.discontinuity_fitted
        self.detrend_polydisc = tableXY(self.x, sub_model, self.xerr, self.yerr)
        self.model = self.y - sub_model

        if replace:
            self.x_backup = self.x.copy()
            self.y_backup = self.y.copy()
            self.xerr_backup = self.xerr.copy()
            self.yerr_backup = self.yerr.copy()
            self.y = sub_model
        else:
            self.sub_model = sub_model

    def substract_polymorceau(self, cut, degree=0, replace=False, Draw=False):
        sub_model = self.fit_par_morceau(cut, degree=degree, Draw=Draw)
        self.detrend_polymorceau = tableXY(self.x, sub_model, self.xerr, self.yerr)
        self.model = self.y - sub_model

        if replace:
            self.x_backup = self.x.copy()
            self.y_backup = self.y.copy()
            self.xerr_backup = self.xerr.copy()
            self.yerr_backup = self.yerr.copy()
            self.y = sub_model
        else:
            self.sub_model = sub_model

    def substract_polyfit(self, deg, replace=False, Draw=False):
        model = None
        self.replace_nan()

        if sum(self.yerr):
            w = 1 / (self.yerr + np.max(self.yerr[self.yerr != 0]) * 100) ** 2
            if len(np.unique(self.yerr)) == 1:
                w = np.ones(len(self.yerr))
        else:
            if np.nanstd(self.y):
                w = 1 / (np.nanstd(self.y) / 2 + self.y * 0) ** 2
            else:
                model = 1 + self.y * 0

        if model is None:
            self.poly_coefficient = np.polyfit(self.x, self.y, deg, w=np.sqrt(w))
            model = np.polyval(self.poly_coefficient, self.x)
        sub_model = self.y - model
        self.detrend_poly = tableXY(self.x, sub_model, self.xerr, self.yerr)

        if Draw == True:
            plt.plot(self.x, model)
        if replace:
            self.x_backup = self.x.copy()
            self.y_backup = self.y.copy()
            self.xerr_backup = self.xerr.copy()
            self.yerr_backup = self.yerr.copy()
            self.y = sub_model
        else:
            self.sub_model = sub_model
        self.poly_curve = model
        self.model = model

    def substract_model(self, *par, replace=False):
        """'current model line(a,b)"""
        try:
            a, b = par[0], par[1]
        except IndexError:
            a, b = par[0]
        model = np.polyval([a, b], self.x)
        sub_model = self.y - model

        self.detrend_model = tableXY(self.x, sub_model, self.xerr, self.yerr)

        if replace == True:
            self.x_backup = self.x.copy()
            self.y_backup = self.y.copy()
            self.xerr_backup = self.xerr.copy()
            self.yerr_backup = self.yerr.copy()
            self.y = sub_model
        if replace == False:
            self.sub_model = sub_model

    def my_periodogram(
        self,
        p_min=1,
        p_max=None,
        p_nb=None,
        nb_bins=20,
        dbin=4,
        min_element_in_bins=10,
        bootstrap=100,
        num_sim=10,
        Draw=True,
        freq=False,
    ):
        """
        Compute a periodogram by measuring the standard deviation of binned data phase-folded
        p_min : minimum period of the grid
        p_max : maximum period of the grid
        p_nb : length of the grid
        nb_bins : number of bins  to bins the data
        dbin : number of different phases to test
        min_element_in_bins : mminimum number of element to consider a bin a reliable
        bootstrap : number of shuffling
        num_sim : number of shuffling for the autocorrelation noise level (do not seem to vary regardless the number)

        """

        if p_max is None:
            p_max = int(np.max(self.x) - np.min(self.x)) / 2
            print("maximum period automatically defined at %.0f days" % (p_max))

        if p_nb is None:
            p_nb = p_max - p_min + 1
            print("length grid period automatically defined at %.0f elements" % (p_nb))

        p_nb = int(p_nb)
        nb_bins = int(nb_bins)
        dbin = int(dbin)
        min_element_in_bins = int(min_element_in_bins)

        print(
            "\n Size of the N matrix : %.3e \n "
            % (p_nb * dbin * len(self.x) * bootstrap)
        )

        if freq:
            liste_p = 1 / np.linspace(1 / p_max, 1 / p_min, p_nb)[::-1]
        else:
            liste_p = np.linspace(p_min, p_max, p_nb)

        random = np.array(
            [
                np.random.choice(np.arange(len(self.x)), len(self.x), replace=False)
                for num in range(bootstrap)
            ]
        )

        random_data = self.y[random]

        days = (self.x % liste_p[:, np.newaxis]) / liste_p[:, np.newaxis]
        bins = np.linspace(0, 1.0, nb_bins + 1)
        delta_bin = np.diff(bins)[-1]
        bins = np.array(bins.tolist() + [bins[-1] + delta_bin])

        inf = bins[0:-1]
        sup = bins[1:]

        inf = inf - np.linspace(0, delta_bin, dbin)[:, np.newaxis]
        sup = sup - np.linspace(0, delta_bin, dbin)[:, np.newaxis]

        self.myp_bins = 0.5 * (inf[0] + sup[0])

        conds = []

        for j in range(dbin):
            cond1 = days >= inf[j][:, np.newaxis][:, np.newaxis]
            cond2 = days < sup[j][:, np.newaxis][:, np.newaxis]
            cond = cond1 & cond2
            conds.append(cond)

        conds = np.array(conds)
        nb_days_in_bins = np.sum(conds, axis=3)

        # mean of random variable

        mean_random = []

        # sum_days_random = np.nansum(conds*random_data[:,np.newaxis][:,np.newaxis][:,np.newaxis],axis=4)

        for j in tqdm(range(bootstrap)):
            sum_days_random = np.nansum(conds * random_data[j], axis=3)
            normalisation = nb_days_in_bins
            normalisation[sum_days_random == 0] = 1
            mean = sum_days_random / normalisation
            mean[normalisation < min_element_in_bins] = np.nan
            mean_random.append(mean)  #
        mean_random = np.array(mean_random)

        # mean of data
        sum_days = np.nansum(conds * self.y, axis=3)
        normalisation = nb_days_in_bins
        normalisation[sum_days == 0] = 1
        mean = sum_days / normalisation
        mean[normalisation < min_element_in_bins] = np.nan

        # std of random variable
        std_random = []
        for j in tqdm(range(bootstrap)):
            std_days_random = np.nansum(
                conds * (self.y - mean_random[j][:, :, :, np.newaxis]) ** 2, axis=3
            )
            normalisation = nb_days_in_bins
            normalisation[std_days_random == 0] = 1
            std = std_days_random / normalisation
            std[normalisation < min_element_in_bins] = np.nan
            std_random.append(std)
        std_random = np.array(std_random)

        # std of data

        std_days = np.nansum(conds * (self.y - mean[:, :, :, np.newaxis]) ** 2, axis=3)

        normalisation = nb_days_in_bins
        normalisation[std_days == 0] = 1
        std = std_days / normalisation
        std[normalisation < min_element_in_bins] = np.nan

        std = np.nanmean(std, axis=0)
        mean = np.nanmean(mean, axis=0)

        std_random = np.nanmean(std_random, axis=0)
        mean_random = np.nanmean(mean_random, axis=0)

        mean_random[std_random == 0] = np.nan
        std_random[std_random == 0] = np.nan

        mean[std == 0] = np.nan
        std[std == 0] = np.nan

        std_random_average = np.nanpercentile(std_random, 1, axis=0)

        med_std = np.nanmean(std, axis=0)
        std_std = np.nanstd(std, axis=0)

        med_std_random = np.nanmean(std_random_average, axis=0)
        std_std_random = np.nanstd(std_random_average, axis=0)

        c1, c2 = med_std, std_std
        c3, c4 = med_std_random, std_std_random

        self.myp_stats = liste_p, c1, c2, c3, c4
        self.myp_ts = np.array(
            [(self.myp_bins * np.ones(p_nb)[:, np.newaxis]).T, mean, std]
        )

        if not freq:
            bdiff = tableXY(liste_p[1:-1], c1[1:-1] / c3[1:-1])
            bdiff.autocorr(
                Draw=False, num_sim=num_sim, dx=np.unique(np.diff(liste_p))[0]
            )
            bdiff2 = tableXY(liste_p[1:-1], c2[1:-1] / c4[1:-1])
            bdiff2.autocorr(
                Draw=False, num_sim=num_sim, dx=np.unique(np.diff(liste_p))[0]
            )
        c_ratio1 = c1 / c3
        c_ratio2 = c2 / c4

        if Draw:
            if not freq:
                plt.figure(figsize=(24, 18))
                plt.subplot(3, 3, 1)
                plt.plot(liste_p, c1)
                plt.plot(liste_p, c3)
                ax = plt.gca()
                plt.subplot(3, 3, 4, sharex=ax)
                plt.plot(liste_p, c2)
                plt.plot(liste_p, c4)
                plt.subplot(3, 3, 7, sharex=ax)
                plt.plot(liste_p, (c2 + c1) / (c1 * c2))
                plt.plot(liste_p, (c3 + c4) / (c3 * c4))

                plt.subplot(3, 3, 2, sharex=ax)
                plt.plot(liste_p, c1 / c3, color="k")
                plt.subplot(3, 3, 5, sharex=ax)
                plt.plot(liste_p, c2 / c4, color="k")
                plt.subplot(3, 3, 8, sharex=ax)
                k1 = (c_ratio1 + c_ratio2) / (c_ratio1 * c_ratio2)
                plt.plot(liste_p, k1 / np.nanmax(k1), color="k")
                k2 = (c2 + c1) / (c1 * c2) - (c3 + c4) / (c3 * c4)
                plt.plot(liste_p, k2 / np.nanmax(k2), color="gray")

                plt.subplot(3, 3, 3)
                ax1 = plt.gca()
                plt.plot(bdiff.autocorrelation.x, bdiff.autocorrelation.y, color="k")
                plt.plot(
                    bdiff.autocorrelation.x,
                    bdiff.autocorrelation_noise_level,
                    color="r",
                )
                plt.axhline(y=0, color="k", alpha=0.3)

                plt.subplot(3, 3, 6, sharex=ax1)
                plt.plot(bdiff2.autocorrelation.x, bdiff2.autocorrelation.y, color="k")
                plt.plot(
                    bdiff2.autocorrelation.x,
                    bdiff2.autocorrelation_noise_level,
                    color="r",
                )
                plt.axhline(y=0, color="k", alpha=0.3)

                plt.subplot(3, 3, 9, sharex=ax1)

                test = tableXY(liste_p, k1)
                test.x = test.x[1:-1]
                test.y = test.y[1:-1]
                test.xerr = test.xerr[1:-1]
                test.yerr = test.yerr[1:-1]
                test.autocorr(
                    Draw=False, num_sim=num_sim, dx=np.unique(np.diff(liste_p))[0]
                )

                test2 = tableXY(liste_p, k2)
                test2.x = test2.x[1:-1]
                test2.y = test2.y[1:-1]
                test2.xerr = test2.xerr[1:-1]
                test2.yerr = test2.yerr[1:-1]
                test2.autocorr(
                    Draw=False, num_sim=num_sim, dx=np.unique(np.diff(liste_p))[0]
                )

                plt.plot(test.autocorrelation.x, test.autocorrelation.y, color="k")
                plt.plot(test2.autocorrelation.x, test2.autocorrelation.y, color="gray")
                plt.plot(
                    test.autocorrelation.x, test.autocorrelation_noise_level, color="r"
                )
                plt.axhline(y=0, color="k", alpha=0.3)
            else:
                plt.figure(figsize=(24, 18))
                k1 = (c_ratio1 + c_ratio2) / (c_ratio1 * c_ratio2)
                plt.plot(liste_p, k1 / np.nanmax(k1), color="k")
                k2 = (c2 + c1) / (c1 * c2) - (c3 + c4) / (c3 * c4)
                plt.plot(liste_p, k2 / np.nanmax(k2), color="gray")

        self.myp_periods = liste_p
        self.myp_power1 = k1
        self.myp_power2 = k2
        self.myp_autocorr1 = test.autocorrelation
        self.myp_autocorr2 = test2.autocorrelation
        self.myp_autocorr_noise_level = test.autocorrelation_noise_level

        #        plt.subplot(3,2,4,sharex=ax)
        #        plt.plot(liste_p,c2_harm)
        #        plt.subplot(3,2,6,sharex=ax)
        #        plt.plot(liste_p,(c2_harm+c1_harm)/(c1_harm*c2_harm))
        #        plt.plot(liste_p,(c3_harm+c4_harm)/(c3_harm*c4_harm))

        #        harmonics_max = 5
        #
        #        all_harmonics = np.unique(np.arange(1,harmonics_max+1) / np.arange(1,harmonics_max+1)[:,np.newaxis])
        #        all_harmonics = all_harmonics[all_harmonics!=1]
        #        all_harmonics = np.insert(all_harmonics,0,1)
        #
        #        coeff_harm = np.zeros(len(all_harmonics))
        #        coeff_harm_pos  = coeff_harm.copy()
        #        coeff_harm_neg  = coeff_harm.copy()
        #        coeff_harm_pos[np.where(all_harmonics!=all_harmonics.astype('int').astype('float'))[0]] = 1
        #        coeff_harm_neg[np.where(all_harmonics==all_harmonics.astype('int').astype('float'))[0]] = 1
        #
        #        coeff_harm = coeff_harm_neg - coeff_harm_pos
        #
        #
        #        phase_p = liste_p * all_harmonics[:,np.newaxis]
        #        indices = []
        #        for j in range(len(phase_p)):
        #            indices.append(myf.find_nearest(liste_p,phase_p[j,:])[0].tolist())
        #        indices = np.array(indices)
        #
        #        c1[0] = np.nan ; c1[-1] = np.nan ; c2[0] = np.nan ; c2[-1] = np.nan
        #        c3[0] = np.nan ; c3[-1] = np.nan ; c4[0] = np.nan ; c4[-1] = np.nan
        #
        #        c1_harm = np.nanmean(c1[indices]*(coeff_harm_pos+coeff_harm_neg)[:,np.newaxis],axis=0)
        #        c2_harm = np.nanmean(c2[indices]*(coeff_harm_neg - coeff_harm_pos)[:,np.newaxis],axis=0)
        #
        #        c3_harm = np.nanmean(c3[indices]*(coeff_harm)[:,np.newaxis],axis=0)
        #        c4_harm = np.nanmean(c4[indices]*(coeff_harm)[:,np.newaxis],axis=0)

        #        b1 = tableXY(liste_p[1:-1],c1[1:-1])
        #        b1.autocorr(Draw=False,num_sim=1,dx = np.unique(np.diff(liste_p))[0])
        #        b3 = tableXY(liste_p[1:-1],c3[1:-1])
        #        b3.autocorr(Draw=False,num_sim=1,dx = np.unique(np.diff(liste_p))[0])

        if False:
            plt.subplot(4, 1, 1)
            plt.plot(liste_p, c1)
            plt.plot(liste_p, c3)
            ax = plt.gca()
            plt.subplot(4, 1, 2, sharex=ax)
            plt.plot(liste_p, c2)
            plt.plot(liste_p, c4)
            plt.subplot(4, 1, 3, sharex=ax)
            plt.plot(liste_p, (c1 + c2) / (c1 * c2))
            plt.plot(liste_p, (c3 + c4) / (c3 * c4))
            plt.subplot(4, 1, 4, sharex=ax)
            plt.plot(
                liste_p, ((c1 + c2) / (c1 * c2)) / ((c3 + c4) / (c3 * c4)), color="k"
            )
            plt.axhline(y=1, color="r")
            for i in range(1, 7):
                for j in range(1, 7):
                    plt.axvline(x=365.25 * i / j, color="k", alpha=0.3)
            plt.xlim(p_min, p_max)

    #        plt.figure(figsize=(25,5))
    #        plt.plot(liste_p,((c1+c2)/(c1*c2)) / ((c3+c4)/(c3*c4)), color='k')
    #        plt.axhline(y=1,color='r')
    #        for i in range(1,6):
    #            for j in range(1,6):
    #                plt.axvline(x=365.25*i/j,color='r',alpha=0.3,ls=':')
    #        plt.xlim(p_min,p_max)
    #        plt.xlabel('Period [days]',fontsize=14)
    #        plt.ylabel('Normalised power',fontsize=14)

    # plt.plot(liste_p,med_std_random+std_std_random)

    def substract_bin(
        self,
        windows,
        method="interp",
        degree=3,
        periodic=False,
        replace=False,
        Draw=False,
        color="r",
        color_p="k",
        lw=1.5,
        alpha_p=0.5,
        alpha_bin=1,
        zorder=10,
    ):
        if periodic:
            length = len(self.x)
            vec_before = tableXY(
                self.x - (np.max(self.x) - np.min(self.x)), self.y, self.xerr, self.yerr
            )
            vec_after = tableXY(
                self.x + (np.max(self.x) - np.min(self.x)), self.y, self.xerr, self.yerr
            )
            vec_before.add(self)
            vec_before.add(vec_after)
            self.x = vec_before.x
            self.y = vec_before.y
            self.xerr = vec_before.xerr
            self.yerr = vec_before.yerr

        self.binning(windows, replace=False)
        new = tableXY(self.binx, self.biny, self.binyerr)

        if periodic:
            self.x = self.x[length:-length]
            self.y = self.y[length:-length]
            self.xerr = self.xerr[length:-length]
            self.yerr = self.yerr[length:-length]

        if method == "interp":
            new.interpolate(new_grid=self.x, method=degree)
            new_plot = new.copy()
            new_plot.interpolate(new_grid=1000, method=degree)

        elif method == "polyfit":
            new.fit_poly(d=degree, Draw=False)
            new.y = np.polyval(new.poly_coefficient, new.x)
            new.interpolate(new_grid=self.x, method=3)
            new_plot = new.copy()
            new_plot.interpolate(new_grid=1000, method=3)

        model = new.y

        sub_model = model

        self.detrend_bin = tableXY(self.x, self.y - sub_model, self.xerr, self.yerr)

        if Draw:
            if periodic:
                vec_before.plot(color="gray")
                vec_after.plot(color="gray")
            plt.errorbar(
                self.binx,
                self.biny,
                yerr=self.binyerr,
                xerr=self.binxerr,
                fmt="ro",
                alpha=alpha_bin,
            )
            self.plot(color=color_p, alpha=alpha_p)
            plt.plot(
                new_plot.x[new_plot.x.argsort()],
                new_plot.y[new_plot.x.argsort()],
                color=color,
                lw=lw,
                zorder=zorder,
            )
        if replace:
            self.x_backup = self.x.copy()
            self.y_backup = self.y.copy()
            self.xerr_backup = self.xerr.copy()
            self.yerr_backup = self.yerr.copy()
            self.y -= sub_model
        else:
            self.sub_model = sub_model

    def fit_spearman(self):
        self.rho_spearman, self.pval_spearman = stats.spearmanr(self.x, self.y)
        self.r_pearson, self.pval_pearson = stats.pearsonr(self.x, self.y)
        self.tau_kendall, self.pval_kendall = stats.kendalltau(self.x, self.y)

        self.stats["r_pearson"] = self.r_pearson
        self.stats["pval_pearson"] = self.pval_pearson
        self.stats["rho_spearman"] = self.rho_spearman
        self.stats["pval_spearman"] = self.pval_spearman
        self.stats["tau_kendall"] = self.tau_kendall
        self.stats["pval_kendall"] = self.pval_kendall

    def fit_dependency(self):
        n = len(self.x)

        aij = abs(self.x - self.x[:, np.newaxis])
        ai = np.sum(aij, axis=0)
        a = np.sum(ai)

        bij = abs(self.y - self.y[:, np.newaxis])
        bi = np.sum(bij, axis=0)
        b = np.sum(bi)

        sample_dist_XY = (
            np.sum(aij * bij) / n**2 - np.sum(ai * bi) * 2 / n**3 + a * b / n**4
        )
        sample_dist_XX = (
            np.sum(aij * aij) / n**2 - np.sum(ai * ai) * 2 / n**3 + a * a / n**4
        )
        sample_dist_YY = (
            np.sum(bij * bij) / n**2 - np.sum(bi * bi) * 2 / n**3 + b * b / n**4
        )

        denum = sample_dist_XX * sample_dist_YY
        if denum > 0:
            Rdist = sample_dist_XY / np.sqrt(denum)
        else:
            Rdist = 0

        self.r_dist_dep = Rdist

    def fit_base(self, base_vec, Plot=False, num_sim=1):
        new = table(self.y[:, np.newaxis].T)
        new.fit_base(
            base_vec, weight=(1 / self.yerr[:, np.newaxis].T) ** 2, num_sim=num_sim
        )
        self.coeff_fitted_std = new.coeff_fitted_std[0]
        self.coeff_fitted = new.coeff_fitted[0]
        self.coeff_fitted_std2 = new.coeff_std
        self.coeff_fitted2 = new.coeff_mean
        self.vec_fitted = tableXY(self.x, new.vec_fitted[0])
        self.all_vec_fitted = new.all_vec_fitted[0]
        self.vec_residues = tableXY(self.x, new.vec_residues[0], self.yerr)
        if Plot:
            self.plot()
            plt.plot(self.x, self.vec_fitted.y, color="r")

    def fit_xgb(
        self,
        base_vec,
        column_name=None,
        frac_cv=0.20,
        n_est=None,
        depth_tree=None,
        num_sim=10,
    ):

        if column_name is None:
            column_name = list(np.arange(len(base_vec)).astype(str))

        simulation = False

        if n_est is None:
            n_est = [10, 50, 100]
            simulation = True

        if depth_tree is None:
            depth_tree = [2, 3, 4, 5]
            simulation = True

        tab_all = np.vstack([self.y, base_vec])
        tab_all = pd.DataFrame(tab_all.T, columns=["species"] + column_name)
        tab_all = tab_all.dropna()

        index = np.array(list(tab_all.index))

        y = tab_all["species"]
        X = tab_all[column_name]
        X_to_pred = X.copy()

        def split_train_test():
            mask_cv = np.in1d(
                index,
                np.random.choice(index, size=int(len(index) * frac_cv), replace=False),
            )

            y_test = tab_all[mask_cv]["species"]
            y_train = tab_all[~mask_cv]["species"]

            X_test = tab_all[mask_cv][column_name]
            X_train = tab_all[~mask_cv][column_name]

            return X_train, X_test, y_train, y_test

        # =============================================================================
        #  cross validation
        # =============================================================================

        if simulation:
            plt.figure(99, figsize=(9, 9))
            m1 = []
            m2 = []
            c = -1
            parameters = []
            for depth in depth_tree:
                for n in n_est:
                    c += 1
                    M1 = []
                    M2 = []
                    for j in tqdm(range(num_sim)):

                        X_train, X_test, y_train, y_test = split_train_test()

                        model = XGBRegressor(
                            objective="reg:squarederror",
                            n_estimators=n,
                            max_depth=depth,
                        )
                        model.fit(pd.DataFrame(X_train, columns=column_name), y_train)

                        y_cv_test = model.predict(
                            pd.DataFrame(X_test, columns=column_name)
                        )
                        y_cv_train = model.predict(
                            pd.DataFrame(X_train, columns=column_name)
                        )
                        y_test = np.array(y_test)
                        y_train = np.array(y_train)

                        M1.append(np.std(y_cv_train - y_train) / np.std(y_train))
                        M2.append(np.std(y_cv_test - y_test) / np.std(y_test))
                    m1.append(M1)
                    m2.append(M2)
                    plt.scatter(M1, np.array(M2) - np.array(M1), color=None, alpha=0.1)
                    plt.errorbar(
                        np.median(M1),
                        np.median(M2) - np.median(M1),
                        color="k",
                        xerr=np.std(M1),
                        yerr=np.std(M2),
                        marker="o",
                        mec="k",
                        capsize=6,
                        lw=3,
                    )
                    plt.errorbar(
                        np.median(M1),
                        np.median(M2) - np.median(M1),
                        color=None,
                        xerr=np.std(M1),
                        yerr=np.std(M2),
                        marker="o",
                        mec="k",
                        capsize=0,
                        label="D=%.0f | N=%.0f" % (depth, n),
                    )
                    parameters.append(
                        [
                            depth,
                            n,
                            np.median(M1),
                            np.median(M2) - np.median(M1),
                            np.sqrt(
                                np.median(M1) ** 2
                                + (np.median(M2) - np.median(M1)) ** 2
                            ),
                        ]
                    )
            parameters = np.array(parameters)
            # plt.axvline(x=m1,color='r',alpha=0.4)
            # plt.axhline(y=m2,color='r',alpha=0.4)
            # plt.axvline(x=0,color='k',ls=':') ; plt.axvline(x=1,color='k',ls=':')
            plt.axhline(y=0, ls=":", color="k")
            plt.plot([0, 1], [1, 0], color="k", ls="--")
            plt.legend(ncol=len(depth_tree))
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel(
                r"Underfitting = $R_{train}$ = std($Y_{train}$ - model)/std($Y_{train}$)",
                fontsize=16,
            )
            plt.ylabel(r"Overfitting = $R_{test}$ - $R_{train}$", fontsize=16)

            best_parameters = parameters[np.argmin(parameters[:, -1])]
            depth_tree = int(best_parameters[0])
            n_est = int(best_parameters[1])
            print(
                " [INFO] Best parameters found : D=%.0f | N=%.0f" % (depth_tree, n_est)
            )

        # =============================================================================
        # binary regression
        # =============================================================================

        X_train, X_test, y_train, y_test = split_train_test()

        plt.figure(2, figsize=(15, 7))
        ax1 = plt.subplot(1, 2, 1)
        model_binary = XGBClassifier(
            objective="binary:logistic", n_estimators=n_est, max_depth=depth_tree
        )
        model_binary.fit(X_train, (y_train > np.median(y)).astype("int"))
        plot_importance(
            model_binary,
            ax=ax1,
            importance_type="gain",
            title="Binary feature imp. (gain)",
        )

        y_pred_binary = model_binary.predict_proba(X_to_pred)[:, 1]
        y_cv_binary = model_binary.predict_proba(X_test)[:, 1]
        y_cv_binary = (y_cv_binary > 0.5).astype("int")
        y_test_binary = (y_test > np.median(y)).astype("int")
        mat = metrics.confusion_matrix(y_test_binary, model_binary.predict(X_test))
        print("\n Confusion matrix :")
        print(mat)
        accuracy = int(np.sum(np.diag(mat)) / np.sum(mat) * 100)
        print("\n Accuracy : ", accuracy, "%")
        self.xgb_accuracy = accuracy

        # =============================================================================
        # linear regression
        # =============================================================================

        plt.figure(4)
        table4 = table(
            pd.DataFrame(
                np.vstack([y_train, X_train.T]).T, columns=["species"] + column_name
            )
        )
        table4.pairplot(color_param="species", kde_plot=False)

        plt.figure(2)
        ax1 = plt.subplot(1, 2, 2)
        model = XGBRegressor(
            objective="reg:squarederror", n_estimators=n_est, max_depth=depth_tree
        )
        model.fit(pd.DataFrame(X_train, columns=["a", "b", "c", "d"]), y_train)
        plot_importance(
            model, ax=ax1, importance_type="gain", title="Regressor feature imp. (gain)"
        )

        y_cv_test = model.predict(pd.DataFrame(X_test, columns=["a", "b", "c", "d"]))
        y_cv_train = model.predict(pd.DataFrame(X_train, columns=["a", "b", "c", "d"]))
        y_test = np.array(y_test)
        y_train = np.array(y_train)

        m1 = np.std(y_cv_train - y_train) / np.std(y_train) * 100
        m2 = np.std(y_cv_test - y_test) / np.std(y_test) * 100

        print("\n Underfitting : %.0f %%" % (m1))
        print("\n Overfitting : %.0f %%" % (m2 - m1))

        y_pred = model.predict(pd.DataFrame(X_to_pred, columns=["a", "b", "c", "d"]))

        plt.figure(6)
        table5 = table(
            pd.DataFrame(
                np.vstack([y_pred, X_to_pred.T]).T, columns=["species"] + column_name
            )
        )
        table5.pairplot(color_param="species", kde_plot=False)

        self.vec_fitted = tableXY(self.x, y_pred)
        self.vec_residues = tableXY(self.x, self.y - y_pred, self.yerr)

    def plus(self, new_tableXY, const=1):
        self.y = self.y + new_tableXY.y * const
        self.yerr = np.sqrt(self.yerr**2 + (const * new_tableXY.yerr) ** 2)

    def minus(self, new_tableXY, const=1):
        self.y = self.y - new_tableXY.y * const
        self.yerr = np.sqrt(self.yerr**2 + (const * new_tableXY.yerr) ** 2)

    def product(self, new_tableXY, const=1):
        self.y = self.y * new_tableXY.y * const
        self.yerr = const * np.sqrt(
            (self.yerr * new_tableXY.y) ** 2 + (self.y * new_tableXY.yerr) ** 2
        )

    def divide(self, new_tableXY, const=1):

        divider = new_tableXY.copy()
        divider.y[divider.y == 0] = 1e-8

        self.y = self.y / (divider.y * const)
        self.yerr = (
            np.sqrt(
                (self.yerr / divider.y) ** 2
                + (self.y * divider.yerr / divider.y**2) ** 2
            )
            / const
        )

    def duplicate(self, nb_duplication):
        new_x = []
        for j in range(1, 1 + nb_duplication):
            new_x.append(
                list(self.x + (np.min(self.x) + (np.max(self.x) - np.min(self.x)) * j))
            )
        self.x = np.hstack(new_x)
        self.y = np.array(list(self.y) * nb_duplication)
        self.xerr = np.array(list(self.xerr) * nb_duplication)
        self.yerr = np.array(list(self.yerr) * nb_duplication)

    def integrate(self, x0, x1, output=False):

        i0 = myf.find_nearest(self.x, x0)[0][0]
        i1 = myf.find_nearest(self.x, x1)[0][0]
        if i0 < i1:
            self.integrated = np.nansum(self.y[i0 : i1 + 1])
        else:
            self.integrated = np.nansum(self.y[i1 : i0 + 1])

        print(self.integrated)
        if output:
            return self.integrated

    def rolling(self, window=1, quantile=None, median=True, iq=True):
        if median:
            self.roll_median = np.ravel(
                pd.DataFrame(self.y)
                .rolling(window, min_periods=1, center=True)
                .quantile(0.50)
            )
        if iq:
            self.roll_Q1 = np.ravel(
                pd.DataFrame(self.y)
                .rolling(window, min_periods=1, center=True)
                .quantile(0.25)
            )
            self.roll_Q3 = np.ravel(
                pd.DataFrame(self.y)
                .rolling(window, min_periods=1, center=True)
                .quantile(0.75)
            )
            self.roll_IQ = self.roll_Q3 - self.roll_Q1
        if quantile is not None:
            self.roll = np.ravel(
                pd.DataFrame(self.y)
                .rolling(window, min_periods=1, center=True)
                .quantile(quantile)
            )

    def rolling_quantile(self, window=1, quantile=0.50):
        return np.ravel(
            pd.DataFrame(self.y)
            .rolling(window, min_periods=1, center=True)
            .quantile(quantile)
        )

    def draw_alpha_shape(self, alpha, only_outer=True):
        x = self.x / np.std(self.x)
        y = self.y / np.std(self.y)

        kde1 = gaussian_kde(x)
        kde2 = gaussian_kde(y)

        genx = np.random.choice(
            np.linspace(x.min(), x.max(), 100),
            size=len(x * 10),
            p=kde1.pdf(np.linspace(x.min(), x.max(), 100))
            / np.sum(kde1.pdf(np.linspace(x.min(), x.max(), 100))),
        )
        geny = np.random.choice(
            np.linspace(y.min(), y.max(), 100),
            size=len(y * 10),
            p=kde2.pdf(np.linspace(y.min(), y.max(), 100))
            / np.sum(kde2.pdf(np.linspace(y.min(), y.max(), 100))),
        )

        points = np.vstack([x, y]).T
        points_random = np.vstack([genx, geny]).T

        edges = myf.alpha_shape(points, alpha=alpha, only_outer=only_outer)
        edges_random = myf.alpha_shape(
            points_random, alpha=alpha, only_outer=only_outer
        )

        self.edges_random = edges_random
        self.points_random = points_random
        self.edges = edges
        self.points = points

        plt.figure()
        plt.subplot(1, 2, 1).plot(points[:, 0], points[:, 1], ".")
        plt.axis("equal")
        for i, j in edges:
            plt.plot(points[[i, j], 0], points[[i, j], 1])
        plt.subplot(1, 2, 2).plot(points_random[:, 0], points_random[:, 1], ".")
        plt.axis("equal")
        for i, j in edges_random:
            plt.plot(points_random[[i, j], 0], points_random[[i, j], 1])
        plt.show()

    def kde(
        self,
        levels=["2d", [1, 2]],
        alpha=0.8,
        cmap="Greys",
        vmin=0.1,
        vmax=0.9,
        zorder=1,
        Plot=True,
        contour_color=None,
        lw=2,
    ):
        x = self.x
        y = self.y
        points = np.vstack([x, y])

        kde = gaussian_kde(points)

        dx = x.max() - x.min()
        dy = y.max() - y.min()
        xx, yy = np.meshgrid(
            np.linspace(x.min() - dx * 0.2, x.max() + dx * 0.2, 100),
            np.linspace(y.min() - dy * 0.2, y.max() + 0.2 * dy, 100),
        )
        zz = np.ones(np.shape(xx))
        for i in range(np.shape(zz)[0]):
            for j in range(np.shape(zz)[1]):
                zz[i, j] = kde((xx[i, j], yy[i, j]))

        zz = zz / np.sum(zz)
        zz_sort = np.hstack(zz)[np.hstack(zz).argsort()[::-1]]
        zz_cumsum = np.cumsum(zz_sort)
        if levels[0] == "1d":
            zz_curve = [
                zz_sort[
                    myf.find_nearest(zz_cumsum, (1 - 2 * (1 - norm_gauss.cdf(j))))[0]
                ][0]
                for j in levels[1]
            ]
        if levels[0] == "2d":
            zz_curve = [
                zz_sort[myf.find_nearest(zz_cumsum, (1 - np.exp(-0.5 * j**2)))[0]][0]
                for j in levels[1]
            ]

        # plt.scatter(x,y)

        colorlin = np.linspace(vmin, vmax, len(levels[1]) + 1)
        colorlvl = plt.cm.get_cmap(cmap)(colorlin)
        if alpha != 0:
            self.cs = plt.contourf(
                xx,
                yy,
                zz,
                levels=np.hstack([zz_curve[::-1], zz_sort[0]]),
                colors=colorlvl,
                alpha=alpha,
                zorder=zorder,
            )
        if contour_color != None:
            plt.contour(
                xx,
                yy,
                zz,
                levels=np.hstack([zz_curve[::-1], zz_sort[0]]),
                colors=contour_color,
                linewidths=lw,
            )

        # self.cs = plt.contourf(xx,yy,zz,levels = np.hstack([zz_curve[::-1],zz_sort[0]]),alpha=alpha,cmap=cmap)

        vertices = self.cs.collections[0].get_paths()[0].vertices

        self.vertices_curve = [
            self.cs.collections[i].get_paths()[0].vertices.copy()
            for i in range(len(self.cs.collections))
        ]
        polygon = Polygon([(k, l) for k, l in zip(vertices[:, 0], vertices[:, 1])])

        x = np.random.uniform(
            low=vertices[:, 0].min(), high=vertices[:, 0].max(), size=3000
        )
        y = np.random.uniform(
            low=vertices[:, 1].min(), high=vertices[:, 1].max(), size=3000
        )
        vertices = list(vertices)
        for j in range(len(x)):
            point = Point(x[j], y[j])
            if polygon.contains(point):
                vertices.append([x[j], y[j]])
        self.vertices = np.array(vertices)

    #    def fit_kde(self, levels= ['2d',[2]],alpha=0.8):
    #        x = self.x/np.std(self.x)
    #        y = self.y/np.std(self.y)
    #
    #        kde1 = gaussian_kde(x)
    #        kde2 = gaussian_kde(y)
    #
    #        genx = np.random.choice(np.linspace(x.min(),x.max(),100), size=len(x*10), p=kde1.pdf(np.linspace(x.min(),x.max(),100))/np.sum(kde1.pdf(np.linspace(x.min(),x.max(),100))))
    #        geny = np.random.choice(np.linspace(y.min(),y.max(),100), size=len(y*10), p=kde2.pdf(np.linspace(y.min(),y.max(),100))/np.sum(kde2.pdf(np.linspace(y.min(),y.max(),100))))
    #
    #        points = np.vstack([x, y]).T
    #        points_random = np.vstack([genx, geny]).T
    #
    #        points = tableXY(points[:,0],points[:,1])
    #        points_random = tableXY(points_random[:,0],points_random[:,1])
    #
    #        self.points_random = points_random
    #        self.points = points
    #
    #        points.kde(levels = levels, alpha = alpha, cmap = 'Blues')
    #        collec = points.cs.collections
    #
    #        points_random.kde(levels = levels, alpha = alpha, cmap = 'Greens')
    #        collec_random = points_random.cs.collections
    #
    #        if len(collec_random[0].get_paths())>1:
    #            print('[WARNING] several cluster for the independent joint distribution')
    #
    #        polygon_random = Polygon()
    #        for num,i in enumerate(range(len(collec_random[0].get_paths()))):
    #            p = collec_random[0].get_paths()[i]
    #            v = p.vertices
    #            x = v[:,0]
    #            y = v[:,1]
    #            polygon_random = polygon_random.union(Polygon([(k, l) for k,l in zip(x,y)]))
    #        self.poly_random = polygon_random
    #        self.area_random = polygon_random.area
    #
    #        polygon = Polygon()
    #        for num,i in enumerate(range(len(collec[0].get_paths()))):
    #            p = collec[0].get_paths()[i]
    #            v = p.vertices
    #            x = v[:,0]
    #            y = v[:,1]
    #            polygon = polygon.union(Polygon([(k, l) for k,l in zip(x,y)]))
    #        self.poly = polygon
    #        self.area = polygon.area
    #
    #        #polygon_intersection = polygon_random.intersection(polygon)
    #        #polygon_union = polygon.union(polygon_random)
    #
    #        #self.area_intersection = polygon_intersection.area
    #        #self.area_union = polygon_union.area
    #
    #        self.r1_kde = 1 - self.area/self.area_random
    #        #self.r2_kde = 1 - self.area_intersection/self.area_union

    def fit_entropy(self):
        x = self.x
        y = self.y

        kde1 = gaussian_kde(x)
        kde2 = gaussian_kde(y)

        points = np.vstack([x, y])

        x_liste = np.linspace(x.min(), x.max(), 100)
        y_liste = np.linspace(y.min(), y.max(), 100)

        xx, yy = np.meshgrid(x_liste, y_liste)

        px = kde1.pdf(x_liste) / np.sum(kde1.pdf(x_liste))
        py = kde2.pdf(y_liste) / np.sum(kde2.pdf(y_liste))

        genx = np.random.choice(x_liste, size=len(x * 10), p=px)
        geny = np.random.choice(y_liste, size=len(y * 10), p=py)

        points_random = np.vstack([genx, geny])

        kde = gaussian_kde(points)
        kde_random = gaussian_kde(points_random)

        entropy_x = -np.sum(px * np.log(px))
        entropy_y = -np.sum(py * np.log(py))

        pxy = np.ones(np.shape(xx))
        pxy_random = np.ones(np.shape(xx))
        for i in range(np.shape(xx)[0]):
            for j in range(np.shape(xx)[1]):
                pxy[i, j] = kde((xx[i, j], yy[i, j]))
                pxy_random[i, j] = kde_random((xx[i, j], yy[i, j]))
        pxy = pxy / np.sum(pxy)
        pxy_random = pxy_random / np.sum(pxy_random)
        pxy[pxy == 0] = pxy[pxy != 0].min()
        pxy_random[pxy_random == 0] = pxy_random[pxy_random != 0].min()

        entropy_xy = -np.sum(pxy * np.log(pxy))
        entropy_xy_random = -np.sum(pxy_random * np.log(pxy_random))
        self.px = px
        self.py = py
        self.pxy = pxy
        self.pxy_random = pxy_random
        self.entropy_x = entropy_x
        self.entropy_y = entropy_y
        self.entropy_xy = entropy_xy
        self.entropy_xy_random = entropy_xy_random

        a = 1 / (np.min([entropy_x, entropy_y]) - entropy_xy_random)
        b = -entropy_xy_random * a
        self.r_entropy = a * self.entropy_xy + b
        if self.r_entropy < 0:
            self.r_entropy = 0
        if self.r_entropy > 1:
            self.r_entropy = 1

    def inside_kde(self, kde_curve):

        cluster = []

        polygon = Polygon([(k, l) for k, l in zip(kde_curve[:, 0], kde_curve[:, 1])])
        for j in np.arange(len(self.x)):
            point = Point(self.x[j], self.y[j])
            if polygon.contains(point):
                cluster.append(1)
            else:
                cluster.append(0)

        cluster = np.array(cluster)

        return cluster

    def fit_alpha_shape(self, alpha=0.2, Plot=True, kde=["1d", [1]]):

        x = self.x / np.std(self.x)
        y = self.y / np.std(self.y)

        kde1 = gaussian_kde(x)
        kde2 = gaussian_kde(y)

        genx = np.random.choice(
            np.linspace(x.min(), x.max(), 100),
            size=len(x * 10),
            p=kde1.pdf(np.linspace(x.min(), x.max(), 100))
            / np.sum(kde1.pdf(np.linspace(x.min(), x.max(), 100))),
        )
        geny = np.random.choice(
            np.linspace(y.min(), y.max(), 100),
            size=len(y * 10),
            p=kde2.pdf(np.linspace(y.min(), y.max(), 100))
            / np.sum(kde2.pdf(np.linspace(y.min(), y.max(), 100))),
        )

        points = np.vstack([x, y]).T
        points_random = np.vstack([genx, geny]).T

        if len(kde) != 0:
            points = tableXY(points[:, 0], points[:, 1])
            points_random = tableXY(points_random[:, 0], points_random[:, 1])
            points.kde(levels=kde, cmap="Blues", alpha=0.2)
            points = points.vertices

            points_random.kde(levels=kde, cmap="Greens", alpha=0.1)
            points_random = points_random.vertices

        edges = myf.alpha_shape(points, alpha=alpha, only_outer=True)
        edges_random = myf.alpha_shape(points_random, alpha=alpha, only_outer=True)

        self.edges_random = edges_random
        self.points_random = points_random
        self.edges = edges
        self.points = points

        number_cluster = []
        save = []
        for edge, point in zip([edges, edges_random], [points, points_random]):
            clusters_liste = []
            liste = np.array(list(edge))
            number = len(liste)
            first = liste[0]
            cluster = 0
            init = []
            tobreak = True
            for k in range(number + 2):
                if tobreak:
                    where = np.where(liste[:, 0] == first[1])[0]
                    if len(where) != 0:
                        first = liste[where[0]]
                        init.append([point[first[0], 0], point[first[0], 1]])
                        liste = np.delete(liste, where[0], axis=0)
                    else:
                        cluster += 1
                        clusters_liste.append(init.copy())
                        if len(liste) != 0:
                            first = liste[0]
                            init = []
                        else:
                            tobreak = False
            number_cluster.append(cluster)
            save.append(clusters_liste)
        self.liste_cluster = save
        self.nb_cluster_random = number_cluster[1]
        self.nb_cluster = number_cluster[0]

        if self.nb_cluster_random != 1:
            print(
                "[WARNING] : the independent joint distribution has more than 1 cluster, increase alpha"
            )

        nb_pts_in_cluster = [
            len(self.liste_cluster[0][j]) for j in range(len(self.liste_cluster[0]))
        ]
        nb_pts_in_cluster_random = [
            len(self.liste_cluster[1][j]) for j in range(len(self.liste_cluster[1]))
        ]

        main_cluster = np.argmax(nb_pts_in_cluster)
        sub_cluster = np.setdiff1d(np.arange(len(nb_pts_in_cluster)), main_cluster)
        env = np.array(save[0][main_cluster])
        env_random = np.array(save[1][np.argmax(nb_pts_in_cluster_random)])

        self.env = env
        self.env_random = env_random

        sign_cluster = []
        polygon = Polygon([(k, l) for k, l in zip(env[:, 0], env[:, 1])])
        polygon_random = Polygon(
            [(k, l) for k, l in zip(env_random[:, 0], env_random[:, 1])]
        )
        for j in list(sub_cluster):
            point = Point(save[0][j][0][0], save[0][j][0][1])
            if polygon.contains(point):
                sign_cluster.append(-1)
            else:
                sign_cluster.append(1)

        area = myf.PolyArea(env[:, 0], env[:, 1])
        area_random = myf.PolyArea(env_random[:, 0], env_random[:, 1])

        polygon_positif = polygon
        polygon_negatif = Polygon()
        for num, j in enumerate(sub_cluster):
            if sign_cluster[num] == 1:
                polygon_positif = polygon_positif.union(
                    Polygon(
                        [
                            (k, l)
                            for k, l in zip(
                                np.array(save[0][j])[:, 0], np.array(save[0][j])[:, 1]
                            )
                        ]
                    )
                )
            if sign_cluster[num] == -1:
                polygon_negatif = polygon_negatif.union(
                    Polygon(
                        [
                            (k, l)
                            for k, l in zip(
                                np.array(save[0][j])[:, 0], np.array(save[0][j])[:, 1]
                            )
                        ]
                    )
                )
            area += sign_cluster[num] * myf.PolyArea(
                np.array(save[0][j])[:, 0], np.array(save[0][j])[:, 1]
            )

        self.area = area
        self.area_random = area_random

        self.poly1 = polygon_positif
        self.poly2 = polygon_negatif

        polygon_intersection_pos = polygon_random.intersection(polygon_positif)
        polygon_intersection_neg = polygon_random.intersection(polygon_negatif)

        polygon_union_pos = polygon_random.union(polygon_positif)

        self.area_intersection = (
            polygon_intersection_pos.area - polygon_intersection_neg.area
        )
        self.area_union = (
            polygon_union_pos.area
            + polygon_intersection_neg.area
            - polygon_negatif.area
        )

        self.r1_alpha = 1 - self.area / self.area_random
        self.r2_alpha = 1 - self.area_intersection / self.area_union

        if Plot:
            plt.figure()
            plt.subplot(1, 3, 1).plot(
                points[:, 0], points[:, 1], ".", label="area %.1f" % (self.area)
            )
            plt.axis("equal")
            for i, j in edges:
                plt.plot(points[[i, j], 0], points[[i, j], 1])
            plt.legend()
            plt.subplot(1, 3, 2).plot(
                points_random[:, 0],
                points_random[:, 1],
                ".",
                label="area %.1f" % (self.area_random),
            )
            plt.axis("equal")
            for i, j in edges_random:
                plt.plot(points_random[[i, j], 0], points_random[[i, j], 1])
            plt.legend()
            ax = plt.subplot(1, 3, 3)
            plt.title(r"$R_1 =$ %.2f, $R_2 =$ %.2f" % (self.r1_alpha, self.r2_alpha))
            plt.plot(
                points[:, 0],
                points[:, 1],
                ".",
                color="b",
                alpha=0.3,
                label="area %.1f" % (self.area),
            )
            plt.plot(
                points_random[:, 0],
                points_random[:, 1],
                ".",
                color="g",
                alpha=0.3,
                label="area %.1f" % (self.area_random),
            )

            patch = PolygonPatch(
                polygon_union_pos,
                facecolor="k",
                edgecolor="k",
                alpha=0.20,
                label="union : area %.1f" % (self.area_union),
            )
            patch2 = PolygonPatch(
                polygon_intersection_pos,
                facecolor="r",
                edgecolor="r",
                alpha=0.20,
                label="intersection : area %.1f" % (self.area_intersection),
            )
            ax.add_patch(patch)
            ax.add_patch(patch2)

            plt.legend()
            plt.show()

    def fit_def(
        self,
        Draw=False,
        Mcmc=False,
        Plot=False,
        Corner=True,
        True_param="",
        color="r",
        kind="fill1",
        name="",
        infos=False,
        color_truth="b",
        mean=0,
    ):
        x = self.x
        y = self.y
        yerr = self.yerr
        self.best_par = mym.BestChi(x, y, yerr, mym.my_func, Draw)
        self.best_chi = mym.Chicarre(self.best_par, x, y, yerr, mym.my_func)
        if Mcmc == True:
            self.samples = mym.MCMC(
                x,
                y,
                yerr,
                mym.my_func,
                True_param=True_param,
                name=name,
                infos=infos,
                savegraph=Corner,
                color=color_truth,
            )
            self.med_samples = np.median(self.samples, axis=0)
            self.sup_samples = np.percentile(self.samples, 84, axis=0)
            self.inf_samples = np.percentile(self.samples, 16, axis=0)
        if Plot == True:
            mym.FinalPlot(
                x, y, yerr, mym.my_func, kind=kind, color=color, name=name, mean=mean
            )

    def fit_line(
        self, perm=1000, Draw=False, color="k", info=False, fontsize=13, label=True
    ):
        k = perm
        self.yerr[self.yerr == 0] = [np.min(self.yerr), 0.1][
            np.min(self.yerr) == 0
        ]  # to avoid 0 value

        w = 1 / self.yerr**2
        A = np.array([(self.x - np.mean(self.x)), np.ones(len(self.x))]).T
        A = A * np.sqrt(w)[:, np.newaxis]
        B = np.array([self.y] * (k + 1)).T
        noise = (
            np.random.randn(np.shape(B)[0], np.shape(B)[1]) / np.sqrt(w)[:, np.newaxis]
        )
        noise[:, 0] = 0
        B = B + noise
        Bmean = np.sum(B * w[:, np.newaxis], axis=0) / np.sum(w)
        Brms = np.sqrt(
            np.sum(((B - Bmean) ** 2 * w[:, np.newaxis]), axis=0) / np.sum(w)
        )
        B = B * np.sqrt(w)[:, np.newaxis]
        Cmean = np.sum(self.x * w, axis=0) / np.sum(w)
        Crms = np.sqrt(np.sum(((self.x - Cmean) ** 2 * w), axis=0) / np.sum(w))

        self.s = np.linalg.lstsq(A, B, rcond=None)[0][0]
        self.i = np.linalg.lstsq(A, B, rcond=None)[0][1]
        self.r = self.s * Crms / Brms

        self.r_pearson_w = np.mean(self.r)
        self.r_errpearson_w = np.std(self.r)

        self.lin_slope_w = np.mean(self.s)
        self.lin_errslope_w = np.std(self.s)

        self.lin_intercept_w = np.mean(self.i)
        self.lin_errintercept_w = np.std(self.i)

        self.stats["r_pearson_w"] = self.r_pearson_w
        self.stats["r_pearson_w_std"] = self.r_errpearson_w
        self.stats["lin_slope_w"] = self.lin_slope_w
        self.stats["lin_slope_w_std"] = self.lin_errslope_w
        self.stats["lin_intercept_w"] = self.lin_intercept_w
        self.stats["lin_intercept_w_std"] = self.lin_errintercept_w

        temp = tableXY(
            self.x,
            self.y
            - ((self.x - np.mean(self.x)) * self.lin_slope_w + self.lin_intercept_w),
            self.yerr,
        )
        temp.rms_w()
        self.fit_line_rms = temp.rms

        if Draw:
            if label:
                plt.plot(
                    self.x,
                    (self.x - np.mean(self.x)) * self.lin_slope_w
                    + self.lin_intercept_w,
                    color=color,
                    ls="-.",
                    label=" $\\mathcal{R}$ = %.2f $\\pm$ %.2f \n S = %.2f $\\pm$ %.2f \n rms : %.2f"
                    % (
                        self.r_pearson_w,
                        self.r_errpearson_w,
                        self.lin_slope_w,
                        self.lin_errslope_w,
                        temp.rms,
                    ),
                )
            else:
                plt.plot(
                    self.x,
                    (self.x - np.mean(self.x)) * self.lin_slope_w
                    + self.lin_intercept_w,
                    color=color,
                    ls="-.",
                )

        if info & Draw:
            plt.legend(fontsize=fontsize)

    def fit_line_weighted(self, Draw=False, Infos=False, color="k"):
        X = self.x
        Y = self.y
        X = sm.add_constant(X)
        if sum(1 / self.yerr**2) == 0:  # arbitrar weights uniform if no error bar
            if np.std(self.y) != 0:
                self.yerr = np.std(self.y) * np.ones(len(self.yerr))
            else:
                self.yerr = np.ones(len(self.yerr))
        wls_model = sm.WLS(Y, X, weights=1 / self.yerr**2)
        results = wls_model.fit()
        self.lin_slope_w = results.params[1]
        self.lin_intercept_w = results.params[0]
        self.r_pearson_w = np.sqrt(results.rsquared) * np.sign(results.params[1])
        self.p_val_w = results.pvalues[1]
        self.lin_errslope_w = results.bse[1]
        self.lin_errintercept_w = results.bse[0]
        self.rms_w()
        rms1 = self.rms
        rms2 = np.std(self.x)
        self.r_err_w = self.lin_errslope_w * rms2 / rms1
        if Draw == True:
            plt.plot(
                [np.min(self.x), np.max(self.x)],
                [
                    self.lin_intercept_w + self.lin_slope_w * np.min(self.x),
                    self.lin_intercept_w + self.lin_slope_w * np.max(self.x),
                ],
                linestyle="-.",
                color=color,
                linewidth=1,
                label=r"S = %.2f $\pm$ %.2f" % (self.lin_slope_w, self.lin_errslope_w)
                + "\n"
                + r"$\mathcal{R}$ = %.2f " % (self.r_pearson_w),
            )
        if Infos == True:
            print(results.summary())
            plt.legend()

    def fit_line_odr(self, Draw=False, Infos=False, color="r"):
        def line_func(p, x):
            m, c = p
            return m * x + c

        quad_model = odr.Model(line_func)
        data = odr.RealData(self.x, self.y, sx=self.xerr, sy=self.yerr)
        model = odr.ODR(data, quad_model, beta0=[0.0, 1.0])
        out = model.run()

        self.odr_slope, self.odr_intercept = out.beta
        self.odr_errslope, self.odr_errintercept = out.sd_beta
        self.odr_res_var = out.res_var

        if Draw:
            x_fit = np.linspace(np.min(self.x), np.max(self.x), 1000)
            y_fit = line_func(out.beta, x_fit)
            self.plot()
            plt.plot(x_fit, y_fit, color=color, zorder=10)

    def fit_poly(self, Draw=False, d=2, color="r", cov=True):
        if np.sum(self.yerr) != 0:
            weights = self.yerr
        else:
            weights = np.ones(len(self.x))
        if cov:
            coeff, V = np.polyfit(self.x, self.y, d, w=1 / weights, cov=cov)
            self.cov = V
            self.err = np.sqrt(np.diag(V))
        else:
            coeff = np.polyfit(self.x, self.y, d, w=1 / weights, cov=cov)
        self.poly_coefficient = coeff
        self.chi2 = np.sum((self.y - np.polyval(coeff, self.x)) ** 2) / np.sum(
            self.yerr**2
        )
        self.bic = self.chi2 + (d + 1) * np.log(len(self.x))
        if Draw == True:
            new_x = np.linspace(self.x.min(), self.x.max(), 10000)
            plt.plot(
                new_x,
                np.polyval(coeff, new_x),
                linestyle="-.",
                color=color,
                linewidth=1,
            )
            # uncertainty = np.sqrt(np.sum([(err[j]*new_x**j)**2 for j in range(len(err))],axis=0))
            # plt.fill_between(new_x,np.polyval(coeff, new_x)-uncertainty/2,np.polyval(coeff, new_x)+uncertainty/2,alpha=0.4,color=color)

    def fit_gauss_proba(self, Draw=False, guess=[0, 1, 1], report=False, c="r"):
        gmodel = Model(myf.gaussian)
        fit_params = Parameters()
        fit_params.add("amp", value=guess[1], min=0)
        fit_params.add("cen", value=guess[0])
        fit_params.add("wid", value=guess[2], min=0)
        fit_params.add("offset", value=0)
        fit_params["offset"].vary = False

        result2 = gmodel.fit(self.y, fit_params, x=self.x)

        if Draw:
            newx = np.linspace(self.x.min(), self.x.max(), 1000)
            y_fit = gmodel.eval(result2.params, x=newx)
            plt.plot(newx, y_fit, color=c)

        if report:
            print(result2.fit_report())

        self.lmfit = result2
        self.params = fit_params

    def fit_lorentzian_proba(self, Draw=False, guess=[0, 1, 1], report=False, c="r"):
        gmodel = Model(myf.lorentzian)
        fit_params = Parameters()
        fit_params.add("amp", value=guess[1], min=0)
        fit_params.add("cen", value=guess[0])
        fit_params.add("wid", value=guess[2], min=0)
        fit_params.add("offset", value=0)
        fit_params["offset"].vary = False

        result2 = gmodel.fit(self.y, fit_params, x=self.x)

        if Draw:
            newx = np.linspace(self.x.min(), self.x.max(), 1000)
            y_fit = gmodel.eval(result2.params, x=newx)
            plt.plot(newx, y_fit, color=c)

        if report:
            print(result2.fit_report())

        self.lmfit = result2
        self.params = fit_params

    def fit_par_morceau(self, cut, degree=1, Draw=False):
        model = np.zeros(len(self.x))
        residues = np.zeros(len(self.x))
        cut = np.sort(np.array(cut))
        x_fine = np.linspace(self.x.min(), self.x.max(), 1000)
        y_fine = np.nan * np.zeros(len(x_fine))

        if self.x.min() < cut[0]:
            cut = np.insert(cut, 0, self.x.min() - 1)
        if self.x.max() > cut[-1]:
            cut = np.insert(cut, len(cut), self.x.max() + 1)
        k = 0
        for j in range(len(cut) - 1):
            self.clip(min=[cut[j], None], max=[cut[j + 1], None], replace=False)
            self.clipped.substract_polyfit(degree, replace=False)
            left = myf.find_nearest(x_fine, self.clipped.x.min())[0]
            right = myf.find_nearest(x_fine, self.clipped.x.max())[0]
            y_fine[int(left) : int(right) + 1] = np.polyval(
                self.clipped.poly_coefficient, x_fine[int(left) : int(right) + 1]
            )

            res = self.clipped.sub_model
            sub_model = self.clipped.y - res

            model[k : k + len(self.clipped.x)] = sub_model
            residues[k : k + len(self.clipped.x)] = res

            k += len(self.clipped.x)

        if Draw:
            plt.plot(x_fine, y_fine, color="r")
        return residues

    def fit_discontinuity(
        self, cut, degree=0, guess=np.zeros(8), Draw=False, report=False, c="r"
    ):
        """maximum polynome degree 4 with 4 cuts"""
        try:
            len(cut)
        except TypeError:
            cut = [cut]
        cut = np.array(cut + (4 - len(cut)) * [-99.9])

        def poly_disc(x, x0, x1, x2, x3, x4, offset1, offset2, offset3, offset4):
            vec = x4 * x**4 + x3 * x**3 + x2 * x**2 + x1 * x + x0
            vec[x > cut[0]] += offset1
            vec[x > cut[1]] += offset2
            vec[x > cut[2]] += offset3
            vec[x > cut[3]] += offset4
            return vec

        gmodel = Model(poly_disc)
        fit_params = Parameters()
        fit_params.add("offset1", value=guess[0])
        fit_params.add("offset2", value=guess[1])
        fit_params.add("offset3", value=guess[2])
        fit_params.add("offset4", value=guess[2])
        fit_params.add("x0", value=guess[3])
        fit_params.add("x1", value=guess[4])
        fit_params.add("x2", value=guess[5])
        fit_params.add("x3", value=guess[6])
        fit_params.add("x4", value=guess[7])
        if degree < 4:
            fit_params["x4"].vary = False
        if degree < 3:
            fit_params["x3"].vary = False
        if degree < 2:
            fit_params["x2"].vary = False
        if degree < 1:
            fit_params["x1"].vary = False

        if (np.sum(np.array(cut) == -99.9)) > 0:
            fit_params["offset4"].vary = False
        if (np.sum(np.array(cut) == -99.9)) > 1:
            fit_params["offset3"].vary = False
        if (np.sum(np.array(cut) == -99.9)) > 2:
            fit_params["offset2"].vary = False

        result2 = gmodel.fit(self.y, fit_params, x=self.x)

        if Draw:
            newx = np.linspace(self.x.min(), self.x.max(), 1000)
            y_fit = gmodel.eval(result2.params, x=newx)
            plt.plot(newx, y_fit, color=c)

        if report:
            print(result2.fit_report())
        self.discontinuity_fitted = gmodel.eval(result2.params, x=self.x)

        self.lmfit = result2
        self.params = result2.params

    def fit_sinus(
        self, Draw=False, d=0, guess=[0, 1, 0, 0, 0, 0], p_max=500, report=False, c="r"
    ):
        gmodel = Model(myf.sinus)
        fit_params = Parameters()
        fit_params.add("amp", value=guess[0])
        fit_params.add("period", value=guess[1], min=0, max=p_max)
        fit_params.add("phase", value=guess[2], min=-0.1, max=np.pi + 0.1)
        fit_params.add("c", value=guess[3])
        fit_params.add("a", value=guess[4])
        fit_params["a"].vary = False
        fit_params.add("b", value=guess[5])
        fit_params["b"].vary = False
        if d == 1:
            fit_params["b"].vary = True
        if d > 1:
            fit_params["b"].vary = True
            fit_params["a"].vary = True

        result2 = gmodel.fit(self.y, fit_params, x=self.x)

        if Draw:
            newx = np.linspace(self.x.min(), self.x.max(), 1000)
            y_fit = gmodel.eval(result2.params, x=newx)
            plt.plot(newx, y_fit, color=c)

        if report:
            print(result2.fit_report())
        self.sinus_fitted = gmodel.eval(result2.params, x=self.x)

        self.lmfit = result2
        self.params = result2.params

    def fit_2dbasis(self, z, zerr=None, basis="legendre", deg=3):
        basis = myf.poly2D_basis(self.x, self.y, deg=deg, basis=basis)
        self.z = z
        if zerr is not None:
            self.zerr = zerr
        else:
            self.zerr = np.ones(len(self.z))
        vec = tableXY(np.arange(len(z)), z, zerr)
        vec.fit_base(basis)

    def fit_poly2d(
        self,
        z,
        zerr=None,
        expo1=3,
        expo2=3,
        maximum=True,
        Draw=False,
        cmap="seismic",
        ax_label="",
        vmin=None,
        vmax=None,
        alpha_p=1,
    ):
        """make the 2d map of the sum_i,j x**i*y**j maximum cut the mixed coefficient such i+j<max(i,j)"""
        self.z = z
        if zerr is not None:
            self.zerr = zerr
        else:
            self.zerr = np.ones(len(self.z))
        nb_par, Surface_test, S_test = myf.make_poly2D(expo1, expo2, maxi=maximum)
        p0 = np.zeros(nb_par)
        par, popt = curve_fit(Surface_test, (self.x, self.y), z, p0)
        self.poly2_coefficient = par
        self.poly_model = S_test
        self.z_fitted = np.ravel(
            S_test(self.x[:, np.newaxis], self.y[:, np.newaxis], par)
        )
        self.z_res = self.z - self.z_fitted
        self.chi2 = np.sum(self.z_res**2 / self.zerr**2)
        self.bic = self.chi2 + len(popt) * np.log(len(self.z))

        if Draw:
            if vmin is None:
                vmin = z.min()
            if vmax is None:
                vmax = z.max()
            xnew = np.linspace(self.x.min(), self.x.max(), 200)
            ynew = np.linspace(self.y.min(), self.y.max(), 200)
            X, Y = np.meshgrid(xnew, ynew)
            Z_fit = S_test(X, Y, par)
            if plt_version > 330:
                plt.pcolor(X, Y, Z_fit, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
            else:
                plt.pcolor(X, Y, Z_fit, vmin=vmin, vmax=vmax, cmap=cmap)
            ax = plt.colorbar()
            ax.ax.set_ylabel(ax_label, fontsize=14)
            plt.scatter(
                self.x,
                self.y,
                c=z,
                edgecolor="k",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                alpha=alpha_p,
            )

    def fit_gauss(self, Draw=False):
        self.best_par = mym.BestChi(self.x, self.y, self.yerr, mym.gauss, Draw)
        self.best_chi = mym.Chicarre(
            self.best_par, self.x, self.y, self.yerr, mym.gauss
        )

    def fit_line2(
        self,
        Draw=False,
        Mcmc=False,
        Plot=False,
        True_param="",
        kind="fill1",
        color="r",
        name="",
        infos=False,
        output_dir=None,
    ):
        self.best_par = mym.BestChi(self.x, self.y, self.yerr, mym.line, Draw)
        self.best_chi = mym.Chicarre(self.best_par, self.x, self.y, self.yerr, mym.line)
        if Mcmc == True:
            self.samples = mym.MCMC(
                self.x,
                self.y,
                self.yerr,
                mym.line,
                True_param=True_param,
                name=name,
                infos=infos,
                output_dir=output_dir,
            )
            self.med_samples = np.median(self.samples, axis=0)
            self.sup_samples = np.percentile(self.samples, 84, axis=0)
            self.inf_samples = np.percentile(self.samples, 16, axis=0)
        if Plot == True:
            mym.FinalPlot(
                self.x, self.y, self.yerr, mym.line, kind=kind, color=color, name=name
            )

    def rm_outliers(self, who="Y", m=2, kind="inter", bin_length=0, replace=True):
        vec = self.copy()
        if bin_length:
            self.night_stack(bin_length=bin_length, replace=False)
            vec_binned = self.stacked.copy()
        else:
            vec_binned = self.copy()

        if who == "Xerr":
            mask = rm_out(self.xerr, m=m, kind=kind)[0]
            vec_binned = vec_binned.xerr
            vec = vec.xerr
        if who == "Yerr":
            mask = rm_out(self.yerr, m=m, kind=kind)[0]
            vec_binned = vec_binned.yerr
            vec = vec.yerr
        if who == "Y":
            mask = rm_out(self.y, m=m, kind=kind)[0]
            vec_binned = vec_binned.y
            vec = vec.y
        if who == "X":
            mask = rm_out(self.x, m=m, kind=kind)[0]
            vec_binned = vec_binned.x
            vec = vec.x

        if bin_length:
            outputs = rm_out(vec_binned, m=m, kind=kind, return_borders=True)
            mask = (vec >= outputs[-1]) & (vec <= outputs[-2])

        if who == "both":
            mask = rm_out(self.x, m=m, kind=kind)[0] & rm_out(self.y, m=m, kind=kind)[0]
        self.mask = mask
        if replace:
            self.x = self.x[mask]
            self.y = self.y[mask]
            self.yerr = self.yerr[mask]
            self.xerr = self.xerr[mask]

    def replace_outliers(self, m=2, kind="inter", mask=None):
        if mask is None:
            mask = rm_out(self.y, m=m, kind=kind)[0]
        self.mask = mask
        num = np.arange(len(self.x))[~mask]
        for j in num:
            if j == 0:
                self.y[0] = 2 * self.y[1] - self.y[2]
            elif j == len(self.x) - 1:
                self.y[-1] = 2 * self.y[-2] - self.y[-3]
            else:
                self.y[j] = 0.5 * (self.y[j + 1] + self.y[j - 1])

    def fft(self, Plot=False, verbose=True):
        dx = np.unique(np.round(np.diff(self.x), 4))
        self.power = np.fft.fft(self.y)
        dstep = np.unique(np.diff(self.x))
        if len(dstep) != 1:
            if verbose:
                print("WARNING not equidistant in x")
            dstep = np.median(np.diff(self.x))
        self.freq = np.fft.fftfreq(len(self.y)) / dstep
        self.sig_fft = tableXY(
            1 / self.freq[1 : len(self.freq) // 2],
            abs(self.power)[1 : len(self.freq) // 2],
        )

        if Plot:
            plt.plot(self.sig_fft.x, self.sig_fft.y)
            plt.ylim(np.min(self.sig_fft.y))
            plt.xscale("log")

    def fft_extract_split(self, nb_comp, debug=False):
        self.fft(Plot=False)
        self.sig_fft.find_max()
        if debug:
            plt.figure()
            plt.plot(
                np.arange(1, 1 + len(self.sig_fft.y_max)),
                np.sort(self.sig_fft.y_max)[::-1],
                "ko-",
            )
            plt.xscale("log")
            plt.yscale("log")

        loc = self.sig_fft.index_max[np.argsort(self.sig_fft.y_max)[::-1][0:nb_comp]]
        loc_left = []
        loc_right = []
        for j in loc:
            loc_left.append(
                myf.find_nearest(self.sig_fft.index_max[self.sig_fft.index_max < j], j)[
                    1
                ][0]
            )
            loc_right.append(
                myf.find_nearest(self.sig_fft.index_max[self.sig_fft.index_max > j], j)[
                    1
                ][0]
            )

        half_width = (
            np.min(np.array([abs(loc - loc_left), abs(loc - loc_right)]), axis=0) / 2
        )
        loc2 = loc + 1
        signal_extracted = {}
        for l, h in zip(loc2.astype("int"), half_width.astype("int")):
            mask = np.zeros(len(self.power))
            mask[l - h : l + h + 1] = 1
            mask[len(self.power) - (l + h) : len(self.power) - (l - h) + 1] = 1
            signal_extracted[self.sig_fft.x[l]] = tableXY(
                self.x, np.fft.ifft(self.power * mask)
            )
        self.fft_signals = signal_extracted

    def pattern(self, width_range=[0.1, 20], new=True, color="k", label=None):
        """If the vector is in angstrom"""
        diff_pattern = tableXY(2 / self.x[::-1], self.y[::-1], 0 * self.x)

        new_grid = np.linspace(
            diff_pattern.x.min(), diff_pattern.x.max(), len(diff_pattern.x)
        )
        diff_pattern.interpolate(new_grid=new_grid, interpolate_x=False)

        dl = np.diff(new_grid)[0]

        fft_pattern = np.fft.fft(diff_pattern.y)

        diff_fourrier = np.abs(fft_pattern)
        diff_fourrier_pos = diff_fourrier[0 : int(len(diff_fourrier) / 2) + 1]

        width = (
            1e-7
            * abs(np.fft.fftfreq(len(diff_fourrier)))[
                0 : int(len(diff_fourrier) / 2) + 1
            ]
            / dl
        )  # transformation of the frequency in mm of the material
        maximum = np.argmax(
            diff_fourrier_pos[(width > width_range[0]) & (width < width_range[1])]
        )
        freq_maximum_ref = width[(width > width_range[0]) & (width < width_range[1])][
            maximum
        ]

        if new:
            plt.figure()
        diff_fourrier_pos[0:20] = 0
        plt.plot(width, diff_fourrier_pos, color=color, label=label)
        self.fourier = tableXY(width, diff_fourrier_pos, 0 * width)
        print(
            "\n[INFO] The interference pattern is produced by material with a width of %.3f mm"
            % (freq_maximum_ref)
        )
        plt.xscale("log")

    def circular_corr(
        self,
        grid,
        border=5,
        dx=1,
        fill_hole=0,
        complete_analysis=False,
        Draw=False,
        min_pts=10,
    ):
        """Check for linear correlation changing the phase of the two signals. Need to provide a grid of days as integer type."""

        holes = np.diff(grid)
        dgrid = np.median(holes)

        locations = np.where((holes > dgrid) & (holes <= fill_hole + 1))[0]
        new_grid = grid
        if len(locations) > 0:
            for j in locations:
                new_grid = np.hstack(
                    [
                        new_grid,
                        np.arange(grid[j] + dgrid, grid[j] + dgrid * holes[j], dgrid),
                    ]
                )

        vec1 = tableXY(grid, self.x, self.xerr)
        vec2 = tableXY(grid, self.y, self.yerr)

        new_grid = np.sort(new_grid)

        vec1.interpolate(new_grid=new_grid, method="linear", replace=True)
        vec2.interpolate(new_grid=new_grid, method="linear", replace=True)
        grid = new_grid

        self.x_backup, self.y_backup, self.xerr_backup, self.yerr_backup = (
            self.x,
            self.y,
            self.xerr,
            self.yerr,
        )
        self.x, self.y, self.xerr, self.yerr = vec1.y, vec2.y, vec1.yerr, vec2.yerr

        if 2 * (len(self.x) - border) <= 0:
            print("value border too high, maximum value  : %.0f" % (len(self.x)))

        if ((2 * (len(self.x) - border)) > min_pts) & (complete_analysis):
            print(
                "number of subplots too high, border = fixed at %.0f"
                % (len(self.x) - 5)
            )
            border = len(self.x) - 5

        correlation = np.zeros(2 * (len(self.x) - border))
        errcorrelation = np.zeros(2 * (len(self.x) - border))
        rho = np.zeros(2 * (len(self.x) - border))
        slopes = np.zeros(2 * (len(self.x) - border))
        errslopes = np.zeros(2 * (len(self.x) - border))
        points = np.zeros(2 * (len(self.x) - border))
        num = np.arange(grid.min(), grid.max() + dx, dx)

        vec = np.in1d(num, grid)
        index = []
        c = 0
        for i in vec:
            if i:
                index.append(c)
                c += 1
            else:
                index.append(-1)
        index = np.array(index)

        if complete_analysis:
            plt.figure(figsize=(15, 3))
        for i in range(len(self.x) - border):
            vec1 = np.array(vec[i:].tolist() + i * [0])
            index1 = np.array(index[i:].tolist() + i * [0])
            new_vec = (vec * vec1).astype("bool")
            test = tableXY(
                self.x[index[new_vec]],
                self.y[index1[new_vec]],
                self.xerr[index[new_vec]],
                self.yerr[index1[new_vec]],
            )
            if len(test.x) > min_pts:  # minimum 10 points to make the correlation
                test.fit_line()
                test.fit_spearman()
            else:
                test.rho_spearman = np.nan
                test.r_pearson_w = np.nan
                test.r_errpearson_w = np.nan
                test.lin_slope_w = np.nan
                test.lin_errslope_w = np.nan
            rho[i] = test.rho_spearman
            correlation[i] = test.r_pearson_w
            errcorrelation[i] = test.r_errpearson_w
            slopes[i] = test.lin_slope_w
            errslopes[i] = test.lin_errslope_w
            points[i] = sum(new_vec)
            if complete_analysis:
                plt.subplot(
                    1, 2 * (len(self.x) - border) - 1, (len(self.x) - border) - (i)
                )
                test.plot(label="num_points : %.0f" % (len(test.x)))
                test.fit_line(Draw=True, info=True)
        # new_x = self.x[::-1] ; new_y = self.y[::-1] ; new_xerr = self.xerr[::-1] ; new_yerr = self.yerr[::-1]
        for i in range(len(self.x) - border - 1):
            i += 1
            # vec1 = np.array(vec[i:].tolist()+i*[0])
            vec1 = np.array(i * [0] + vec[:-i].tolist())
            index1 = np.array(i * [0] + index[:-i].tolist())
            new_vec = (vec * vec1).astype("bool")
            # test = tableXY(new_x[index[new_vec]],new_y[index1[new_vec]],new_xerr[index[new_vec]],new_yerr[index1[new_vec]])
            test = tableXY(
                self.x[index[new_vec]],
                self.y[index1[new_vec]],
                self.xerr[index[new_vec]],
                self.yerr[index1[new_vec]],
            )
            if len(test.x) > min_pts:  # minimum 10 points to make the correlation
                test.fit_line()
                test.fit_spearman()
            else:
                test.rho_spearman = np.nan
                test.r_pearson_w = np.nan
                test.r_errpearson_w = np.nan
                test.lin_slope_w = np.nan
                test.lin_errslope_w = np.nan
            rho[len(self.x) - border + i] = test.rho_spearman
            correlation[len(self.x) - border + i] = test.r_pearson_w
            errcorrelation[len(self.x) - border + i] = test.r_errpearson_w
            slopes[len(self.x) - border + i] = test.lin_slope_w
            errslopes[len(self.x) - border + i] = test.lin_errslope_w
            points[len(self.x) - border + i] = sum(new_vec)
            if (complete_analysis) & (i != 0):
                plt.subplot(
                    1, 2 * (len(self.x) - border) - 1, i + (len(self.x) - border)
                )
                test.plot(label="num_points : %.0f" % (len(test.x)))
                test.fit_line(Draw=True, info=True)

        rho = np.hstack(
            [rho[len(self.x) - border + 1 :][::-1], rho[0 : len(self.x) - border]]
        )
        correlation = np.hstack(
            [
                correlation[len(self.x) - border + 1 :][::-1],
                correlation[0 : len(self.x) - border],
            ]
        )
        errcorrelation = np.hstack(
            [
                errcorrelation[len(self.x) - border + 1 :][::-1],
                errcorrelation[0 : len(self.x) - border],
            ]
        )
        slopes = np.hstack(
            [slopes[len(self.x) - border + 1 :][::-1], slopes[0 : len(self.x) - border]]
        )
        errslopes = np.hstack(
            [
                errslopes[len(self.x) - border + 1 :][::-1],
                errslopes[0 : len(self.x) - border],
            ]
        )
        points = np.hstack(
            [points[len(self.x) - border + 1 :][::-1], points[0 : len(self.x) - border]]
        )

        self.circ_rho = rho
        self.circ_corr = correlation
        self.circ_errcorr = errcorrelation
        self.circ_slope = slopes
        self.circ_errslope = errslopes
        self.circ_nbpoints = points

        if Draw:
            plt.figure(figsize=(10, 10))
            plt.subplot(4, 1, 1)
            plt.axhline(y=0, color="gray", alpha=0.5)
            plt.axvline(x=0, color="k", ls=":")
            plt.plot(np.arange(len(rho)) * dx - (len(rho) // 2) * dx, rho)
            plt.ylabel(r"$\rho$ spearman", fontsize=13)
            ax = plt.gca()
            plt.subplot(4, 1, 2, sharex=ax)
            plt.axhline(y=0, color="gray", alpha=0.5)
            plt.axvline(x=0, color="k", ls=":")
            plt.ylabel(r"$\mathcal{R}$ pearson", fontsize=13)
            plt.errorbar(
                np.arange(len(rho)) * dx - (len(rho) // 2) * dx,
                correlation,
                yerr=errcorrelation,
                fmt="ko",
            )
            plt.subplot(4, 1, 3, sharex=ax)
            plt.axvline(x=0, color="k", ls=":")
            plt.errorbar(
                np.arange(len(rho)) * dx - (len(rho) // 2) * dx,
                slopes,
                yerr=errslopes,
                fmt="ko",
            )
            plt.ylabel("slope", fontsize=13)
            plt.subplot(4, 1, 4, sharex=ax)
            plt.axvline(x=0, color="k", ls=":")
            plt.plot(
                (np.arange(len(rho)) * dx - (len(rho) // 2) * dx)[points > min_pts],
                points[points > min_pts],
            )
            plt.ylabel("nb_points", fontsize=13)
        self.backup()

    def binned_rms_curve(self, nb_time=25, nb_db=50, num_sim=100, floor_noise=0):

        vec = self.copy()
        simu = np.random.randn(num_sim, len(vec.x))
        matrix = table(np.vstack([vec.y, simu]))
        weight = 1 / vec.yerr**2

        matrix.rms_w(vec.yerr)
        rms = matrix.rms
        coeff = rms / rms[0]

        matrix.table /= coeff[:, np.newaxis]

        time = vec.x

        tmax = (np.max(time) - np.min(time)) / 4
        all_t = np.linspace(0, np.log10(tmax), nb_time)

        v = []
        v_std = []
        sim_mean = []
        sim_std = []
        for j in tqdm(10**all_t):
            v1 = []
            sim1_mean = []
            sim1_std = []
            for i in range(nb_db):
                modulo = ((time + (nb_db * i / j)) // j).astype("int")
                modulo -= np.min(modulo)
                indice = myf.unique_indexing(modulo)
                mat = indice == np.unique(indice)[:, np.newaxis]
                mat = mat.astype("int") * weight
                new_weight = 1 / np.sqrt(np.sum(mat, axis=1))
                new_weight = np.sqrt(new_weight**2 + floor_noise**2)

                binned = table(np.dot(matrix.table, mat.T) / np.sum(mat.T, axis=0))
                binned.rms_w(new_weight)
                v1.append(binned.rms[0])
                sim1_mean.append(np.mean(binned.rms[1:]))
                sim1_std.append(np.std(binned.rms[1:]))
            v.append(np.mean(v1))
            v_std.append(np.std(v1))
            sim_mean.append(np.mean(sim1_mean))
            sim_std.append(np.mean(sim1_std))

        output = np.array(v)
        output_std = np.array(v_std)
        sim = np.array(sim_mean)
        sim_std = np.array(sim_std)

        return 10**all_t, output, output_std, sim, sim_std

    def rolling_regression(
        self,
        box=5,
        min_freq=None,
        weighted=True,
        center=False,
        outliers=[True, 1.5, "inter"],
    ):
        if weighted:
            if center:
                if box % 2 == 0:
                    box = box + 1
                list = [[np.nan] * 6] * int(box / 2)
            else:
                list = [[np.nan] * 6] * box
            num = len(self.x) - box
            for j in range(num):
                if outliers[0] == True:
                    mask = rm_out(self.y[j : box + j], m=outliers[1], kind=outliers[2])[
                        0
                    ]
                    X = self.x[j : box + j][mask]
                    Y = self.y[j : box + j][mask]
                    YERR = (self.yerr[j : box + j][mask]) ** 2
                else:
                    X = self.x[j : box + j]
                    Y = self.y[j : box + j]
                    YERR = self.yerr[j : box + j] ** 2
                X = sm.add_constant(X)
                wls_model = sm.WLS(Y, X, weights=1 / YERR)
                results = wls_model.fit()
                list.append(
                    [
                        results.params[1],
                        results.bse[1],
                        results.params[0],
                        results.bse[0],
                        np.sqrt(results.rsquared) * np.sign(results.params[1]),
                        results.pvalues[1],
                    ]
                )
            if center:
                list = list + [[np.nan] * 6] * int(box / 2)
            self.rolling_values = pd.DataFrame(
                np.array(list),
                columns=[
                    "slope",
                    "err_slope",
                    "intercept",
                    "err_intercept",
                    "Rcorr",
                    "pval",
                ],
            )
        else:
            if center:
                if box % 2 == 0:
                    box = box + 1
                list = [[np.nan] * 5] * int(box / 2)
            else:
                list = [[np.nan] * 5] * box
            num = len(self.x) - box
            for j in range(num):
                if outliers[0] == True:
                    mask = rm_out(self.y[j : box + j], m=outliers[1], kind=outliers[2])[
                        0
                    ]
                    X = self.x[j : box + j][mask]
                    Y = self.y[j : box + j][mask]
                else:
                    X = self.x[j : box + j]
                    Y = self.y[j : box + j]
                list.append(stats.linregress(X, Y))
            if center:
                list = list + [[np.nan] * 5] * int(box / 2)
            self.rolling_values = pd.DataFrame(
                np.array(list),
                columns=["slope", "intercept", "Rcorr", "pval", "err_slope"],
            )

    def periodogram_l1(
        self,
        starname="",
        dataset_name="",
        photon_noise=0,
        max_n_significant=9,
        p_min=2,
        Plot=True,
        text_output=1,
        fap_min=-3,
        method_signi=["fap", "evidence_laplace"],
        sort_val="log10faps",
        species=None,
        verbose=True,
    ):
        """Based on Nathan code"""
        c = l1periodogram_v1.l1p_class(self.x, self.y)
        sigmaW = photon_noise  # We add in quadrature 1 m/s to the nominal HARPS errors
        sigmaR, tau = 0.0, 0.0  # No red noise
        sigma_calib = 0.0  # No calibration noise
        V = covariance_matrices.covar_mat(
            self.x, self.yerr, sigmaW, sigmaR, sigma_calib, tau
        )

        if species is None:
            species = np.ones(len(self.x))

        nb_species = len(np.unique(species))
        offsets = np.array([(species == s).astype("int") for s in np.unique(species)]).T

        c.starname = starname
        c.dataset_names = dataset_name
        c.offsets = offsets
        # To set the model use the method:
        c.set_model(
            omegamax=2
            * np.pi
            / p_min,  # We will search frequencies up to 1.5 cycles/day = 3*np.pi rad/day
            V=V,
            MH0=offsets,
        )  # The only vector we assumed to be in the data by default is the offset

        # Now the noise model is defined, let us plot the data
        # The error bars are the square root of the diagonal of V
        # c.plot_input_data()

        # c.update_model(omegamax = 1.9*np.pi)

        c.l1_perio(
            numerical_method="lars",
            significance_evaluation_methods=method_signi,
            max_n_significance_tests=max_n_significant,
            verbose=text_output,
            plot_output=Plot,
        )

        signi = len(c.significance["log10faps"] < -3)

        # check signi and sort

        A = l1_sig.create_om_mat(c.t, c.MH0, c.omega_peaks[:signi])

        Chi2, PL = l1_cholesky.fastchi2lin_V(A, V, c.y_init, full_chi2=False)

        PL = PL[:-nb_species]

        amp = np.sqrt(PL[0:-1:2] ** 2 + PL[1::2] ** 2)
        phi = np.arctan2(PL[1::2], PL[0:-1:2])

        c.omega_peaks = c.omega_peaks[:signi]
        c.peakvalues = c.peakvalues[:signi]
        c.significance["log10_bayesf_laplace"] = c.significance["log10_bayesf_laplace"]
        c.significance["log10faps"] = c.significance["log10faps"]
        self.l1_nb_nonull = np.sum(c.smoothed_solution != 0)

        if len(c.omega_peaks) > 0:
            sort = np.argsort(c.significance["log10faps"])
            c.omega_peaks = c.omega_peaks[sort]
            c.peakvalues = c.peakvalues[sort]
            c.significance["log10_bayesf_laplace"] = c.significance[
                "log10_bayesf_laplace"
            ][sort]
            c.significance["log10faps"] = c.significance["log10faps"][sort]

            matrix = np.array(
                [
                    2 * np.pi / c.omega_peaks,
                    amp,
                    phi,
                    c.peakvalues,
                    c.significance["log10faps"],
                    c.significance["log10_bayesf_laplace"],
                ]
            ).T
            dataframe = pd.DataFrame(
                matrix,
                index=["peak%.0f" % (k) for k in range(1, 1 + signi)],
                columns=[
                    "period",
                    "amp",
                    "phi",
                    "peak_amp",
                    "log10faps",
                    "log10_bayesf_laplace",
                ],
            )
            dataframe = dataframe.sort_values(by="log10faps")
            if verbose:
                print("\nTABLE OF SIGNALS")
                print("----------------")
                print(
                    dataframe[
                        ["period", "peak_amp", "log10faps", "log10_bayesf_laplace"]
                    ]
                )
                print(
                    "\nTABLE OF SIGNALS SIGNIFICANT (FAP < %.1f%%)"
                    % (10 ** (2 + fap_min))
                )
                print("----------------------------")
                print(
                    dataframe.loc[
                        dataframe["log10faps"] < fap_min,
                        ["period", "peak_amp", "log10faps", "log10_bayesf_laplace"],
                    ]
                )
                print(
                    "\nPARAMETERS OF SIGNALS SIGNIFICANT (FAP < %.1f%%)"
                    % (10 ** (2 + fap_min))
                )
                print("----------------------------")
                print(
                    dataframe.loc[
                        dataframe["log10faps"] < fap_min,
                        ["period", "amp", "phi", "log10faps"],
                    ]
                )

            self.l1_table = dataframe.loc[
                dataframe["log10faps"] < fap_min,
                ["period", "peak_amp", "amp", "phi", "log10faps"],
            ]
            if Plot:
                c.plot_with_list(
                    sum(dataframe["log10faps"] < fap_min),
                    significance_values="log10faps",
                    save=True,
                    sort_val=sort_val,
                )
        else:
            self.l1_table = {"period": [None]}
            if Plot:
                c.plot_with_list(
                    1, significance_values="log10faps", save=True, sort_val=sort_val
                )

    def diff_phase_periodogram(self, vec2, detrend_deg=2, Plot=True):
        self.substract_polyfit(detrend_deg, replace=False, Draw=False)
        vec2.substract_polyfit(detrend_deg, replace=False, Draw=False)

        if Plot:
            plt.figure(figsize=(15, 10))
            plt.subplot(5, 1, 1)
            ax = plt.gca()
            self.detrend_poly.periodogram(
                nb_perm=1, all_outputs=True, color="k", Norm=True
            )
            vec2.detrend_poly.periodogram(
                nb_perm=1, all_outputs=True, color="r", Norm=True
            )

            plt.subplot(5, 1, 2, sharex=ax)
            plt.plot(
                1 / self.detrend_poly.freq,
                self.detrend_poly.phase * 180 / np.pi + 180,
                color="k",
            )
            plt.plot(
                1 / vec2.detrend_poly.freq,
                vec2.detrend_poly.phase * 180 / np.pi + 180,
                color="r",
            )

            dist = myf.dist_modulo(self.detrend_poly.phase, vec2.detrend_poly.phase)

            plt.subplot(5, 1, 3, sharex=ax)
            plt.plot(1 / self.detrend_poly.freq, dist, color="k")

            plt.subplot(5, 1, 4, sharex=ax)
            plt.scatter(
                1 / self.detrend_poly.freq, self.detrend_poly.power, c=dist, cmap="jet"
            )
            plt.subplot(5, 1, 5, sharex=ax)
            grad_dist = abs(np.gradient(dist))
            plt.scatter(
                1 / self.detrend_poly.freq,
                self.detrend_poly.power,
                c=grad_dist,
                vmin=np.percentile(grad_dist, 16),
                vmax=np.percentile(grad_dist, 84),
                cmap="jet",
            )

            plt.figure(figsize=(15, 10))
            plt.subplot(3, 1, 1)
            ax = plt.gca()
            self.detrend_poly.periodogram(
                nb_perm=1, all_outputs=True, color="k", Norm=True
            )
            plt.subplot(3, 1, 2, sharex=ax)
            prim = myf.transform_prim(1 / self.detrend_poly.freq, grad_dist)
            plt.plot(1 / self.detrend_poly.freq, prim[1], color="k")
            plt.subplot(3, 1, 3, sharex=ax)
            plt.plot(
                1 / self.detrend_poly.freq, prim[1] * self.detrend_poly.power, color="k"
            )

            self.phase_penality = prim[1]

        else:
            self.detrend_poly.periodogram(
                nb_perm=1, all_outputs=True, color="k", Norm=True, Plot=False
            )
            vec2.detrend_poly.periodogram(
                nb_perm=1, all_outputs=True, color="r", Norm=True, Plot=False
            )

            phase1 = self.detrend_poly.phase
            phase2 = vec2.detrend_poly.phase

            dist = myf.dist_modulo(phase1, phase2)
            grad_dist = abs(np.gradient(dist))
            prim = myf.transform_prim(1 / self.detrend_poly.freq, grad_dist)

            self.phase_penality = prim[1]

        return self.phase_penality

    def polar_periodogram(self, dtime=1):
        self.periodogram(nb_perm=1, Plot=False, Norm=False, all_outputs=True, ofac=30)
        # times = 1/self.freq
        power = self.power
        phi = self.phase
        norm = np.max(self.power)
        fap10 = self.fap10 / norm
        fap1 = self.fap1 / norm
        fap01 = self.fap01 / norm
        fap = self.fap / norm

        plt.figure(figsize=(20, 5))
        plt.axes([0.1, 0.1, 0.6, 0.8])
        plt.plot(1 / self.freq, self.power / norm, color="k")
        # plt.axhline(y=fap10,color='r',ls='-.',lw=2.5,label='FAP = 10%')
        plt.axhline(y=fap1, color="r", ls="-", lw=2.5, label="FAP = 1%")
        # plt.axhline(y=fap,color='r',ls='-',lw=2.5)
        # plt.axhline(y=fap01,color='r',ls=':',lw=2.5,label='FAP = 0.1%')
        plt.scatter(self.period_maxima, self.power_maxima / norm, color="b")
        plt.xlabel("Period [days]", fontsize=14)
        plt.ylabel("Power", fontsize=14)
        plt.xlim(np.min(1 / self.freq), np.max(1 / self.freq))
        plt.xscale("log")
        plt.legend()

        plt.axes([0.75, 0.1, 0.2, 0.8], projection="polar")
        plt.plot(phi, power / norm, color="k")
        # plt.plot(np.linspace(0,2*np.pi,100),fap10*np.ones(100),color='r',ls='-.',lw=2.5)
        plt.plot(
            np.linspace(0, 2 * np.pi, 100),
            fap1 * np.ones(100),
            color="r",
            ls="-",
            lw=2.5,
        )
        # plt.plot(np.linspace(0,2*np.pi,100),fap*np.ones(100),color='r',ls='-',lw=2.5)
        # plt.plot(np.linspace(0,2*np.pi,100),fap01*np.ones(100),color='r',ls=':',lw=2.5)
        # time_step = np.arange(np.min(times),np.max(times),dtime)
        # match = myf.match_nearest(times,time_step)
        plt.scatter(self.phase_maxima, self.power_maxima / norm, color="b")
        plt.grid(False)

    def corr(
        self,
        tablexy,
        new=False,
        detrend=0,
        color="k",
        cmap=None,
        capsize=3,
        fontsize=13,
        Draw=True,
    ):

        table = self.copy()
        table2 = tablexy.copy()
        if detrend:
            table.substract_polyfit(detrend, replace=True)
            table2.substract_polyfit(detrend, replace=True)

        n = tableXY(table2.y, table.y, table2.yerr, table.yerr)
        if Draw:
            n.plot(new=new, color=color, capsize=capsize)
            if cmap is not None:
                plt.scatter(n.x, n.y, c=self.x, cmap=cmap, zorder=10)
        n.fit_spearman()
        n.fit_line(info=True, Draw=Draw, fontsize=fontsize)

        return n

    def corr_elliptical(self, tablexy, window=5):

        table = self.copy()
        table2 = tablexy.copy()

        baseline = np.max(table.x) - np.min(table.x)
        period_ref = baseline / 10
        ref = table.copy()
        ref.y *= 0
        ref.y += np.sin(2 * np.pi / period_ref * ref.x)

        table.periodogram(Plot=False)
        table2.periodogram(Plot=False)
        ref.periodogram(Plot=False)

        power_spectrum = tableXY(table.freq, table.power)
        power_spectrum2 = tableXY(table2.freq, table2.power)
        power_spectrum_ref = tableXY(ref.freq, ref.power)

        power_spectrum_ref.find_min()
        width_peak_freq = abs(
            myf.find_nearest(power_spectrum_ref.min_extremum.y, 1 / period_ref)[2][0]
        )

        new_grid = np.arange(
            np.min(power_spectrum.x), np.max(power_spectrum.x), width_peak_freq / 20
        )
        power_spectrum.interpolate(new_grid=new_grid)
        power_spectrum2.interpolate(new_grid=new_grid)

    def hist_weighted(self):
        v = []
        for mean, sigma in zip(self.y, self.yerr):
            values = sigma * np.random.randn(100) + mean
            v.append(values)
        v = np.array(v)
        v = np.ravel(v)
        return v

    def periodogram(
        self,
        nb_perm=1,
        ofac=10,
        p_min=0,
        detrending=0,
        p_max=None,
        Plot=True,
        Norm=False,
        norm_val=False,
        supp=None,
        infos=False,
        level=None,
        color="k",
        ls="-",
        P_th=None,
        axis_y_var="p",
        axis_x_var="p",
        xlim=[None, None],
        all_outputs=False,
        compute_fap_levels=True,
        legend="",
        zorder=1,
        lw=1.5,
        alpha=1,
        warning=True,
        planet=[0, 365.25, 2 * np.pi],
    ):
        """nb_perm : number of permutation for the bootsrap and fap, ofac : oversampling for the plot, Plot : Draw the plot, Norm : normalise by the 1% fap value"""
        self.supress_nan()

        if supp != None:
            self.x_backup = self.x.copy()
            self.y_backup = self.y.copy()
            self.yerr_backup = self.yerr.copy()
            self.x = np.delete(self.x, supp)
            self.y = np.delete(self.y, supp)
            self.yerr = np.delete(self.yerr, supp)

        baseline = np.nanmax(self.x) - np.nanmin(self.x)

        if infos:
            all_outputs = True

        if axis_y_var != "p":
            all_outputs = True
            Norm = False
            norm_val = False
            compute_fap_levels = False
            p_min2 = 0
            if p_min != 0:
                p_min2 = p_min
            p_min = 0

        if all_outputs:
            all_outputs = "yes"
        else:
            all_outputs = "no"

        cond = len(self.x)
        if len(np.unique(self.x)) != len(self.x):
            cond = 0
            if warning:
                print("There are several x with the same values")

        if np.product(self.yerr != 0) == 0:
            if warning:
                print(
                    "At least one weight is null ! Reperform the analysis taking care of it."
                )
            self.yerr[self.yerr == 0] = np.min(self.yerr[self.yerr != 0]) / 10
            cond = 0

        if np.sum(self.yerr != 0) == 0:
            if warning:
                print(
                    "All the weights are null ! Warning : Kernal could died. Reperform the analysis taking care of it."
                )
            self.yerr[self.yerr == 0] = 1
            cond = 0

        if np.sum(abs(self.y)) == 0:
            if warning:
                print(
                    "All the y are null ! Warning : Kernal could died. Reperform the analysis taking care of it."
                )
            cond = 0

        self2 = self.copy()
        if detrending:
            self.substract_polyfit(detrending, replace=False)
            self2 = self.detrend_poly.copy()

        if planet[0]:
            self2.y += planet[0] * np.sin(2 * np.pi * self2.x / planet[1] + planet[2])
            P_th = planet[1]

        if cond > 5:
            pXav = f3.periodogram(
                self2.x,
                self2.y,
                sig_y=self2.yerr,
                ofac=ofac,
                all_outputs=all_outputs,
                min_P=p_min,
            )

            n = []
            for j in range(len(pXav)):
                if type(pXav[j]) == np.ndarray:
                    n.append(pXav[j])
            pXav = np.array(n).copy()

            if p_max is None:
                p_max = baseline

            last = int(myf.find_nearest(1 / pXav[0], p_max)[0])
            self.debug = pXav
            pXav = pXav[:, last:]

            fap = np.max(pXav[1])
            self.fap = fap
            self.fap01 = None
            self.fap1 = None
            self.fap10 = None

            if axis_x_var == "f":
                pXav[0] = 1 / pXav[0]
                try:
                    P_th = 1 / P_th
                except TypeError:
                    pass

            if all_outputs != "no":
                self.amplitude = np.sqrt(pXav[3] ** 2 + pXav[4] ** 2)
                self.phase = pXav[
                    2
                ]  # -2*np.pi*pXav[0]*np.min(self.x))%(2*np.pi) #np.sin(t+phi) sign positif convention

            self.freq, self.power = pXav[0], pXav[1]
            self.perio_curve = tableXY(1 / self.freq[::-1], self.power[::-1])

            high2 = argrelextrema(pXav[1], np.greater)[0]
            high2 = high2[self.power[high2].argsort()][::-1]
            self.period_maxima = 1 / pXav[0][high2[0:5]]
            self.power_maxima = pXav[1][high2[0:5]]

            self.perio_max = 1 / pXav[0][high2[0]]
            self.power_max = pXav[1][high2[0]]
            self.fap_max = 100 * myf.fap(
                self.power_max, self2.x, self2.yerr, np.max(self.freq), p0=1, Teff=None
            )

            if compute_fap_levels:
                if nb_perm > 1:
                    if level != None:
                        fap = f3.calculate_fap(
                            self2.x,
                            self2.y,
                            sig_y=self2.yerr,
                            level=level,
                            nb_perm=nb_perm,
                            ofac=ofac,
                        )
                        self.fap = fap
                    else:
                        fap01 = f3.calculate_fap(
                            self2.x,
                            self2.y,
                            sig_y=self2.yerr,
                            level=0.999,
                            nb_perm=nb_perm,
                            ofac=ofac,
                        )
                        fap = f3.calculate_fap(
                            self2.x,
                            self2.y,
                            sig_y=self2.yerr,
                            level=0.99,
                            nb_perm=nb_perm,
                            ofac=ofac,
                        )
                        fap10 = f3.calculate_fap(
                            self2.x,
                            self2.y,
                            sig_y=self2.yerr,
                            level=0.90,
                            nb_perm=nb_perm,
                            ofac=ofac,
                        )
                        self.fap01 = fap01
                        self.fap1 = fap
                        self.fap10 = fap10
                else:
                    dichotomic = np.array(
                        [
                            100
                            * myf.fap(
                                self.power_max * 0.01 * j,
                                self2.x,
                                self2.yerr,
                                np.max(self.freq),
                                p0=1,
                                Teff=None,
                            )
                            for j in range(1, 201, 10)
                        ]
                    )
                    dicho_power = np.array(
                        [self.power_max * 0.01 * j for j in range(1, 201, 10)]
                    )
                    dichotomic[dichotomic <= 0] = 100
                    dic = tableXY(np.log10(dichotomic), dicho_power)
                    mask = np.gradient(dic.x) < 0
                    if sum(mask) > 1:
                        dic.masked(mask)

                    if level == None:
                        fap_liste = np.array([-1, 0, 1])
                        dic.interpolate(new_grid=fap_liste, method="linear")
                        self.fap01 = dic.y[0]
                        self.fap1 = dic.y[1]
                        self.fap10 = dic.y[2]
                        fap = dic.y[1]
                        self.fap = fap
                    else:
                        fap_liste = np.array([np.log10(100 * (1 - level))])
                        dic.interpolate(new_grid=fap_liste, method="linear")
                        fap = dic.y[0]
                        self.fap = fap

            high = np.where(pXav[1] > fap)[0][
                np.in1d(
                    np.where(pXav[1] > fap)[0], argrelextrema(pXav[1], np.greater)[0]
                )
            ]
            self.period_significant = 1 / pXav[0][high[pXav[1][high].argsort()[::-1]]]
            self.power_significant = pXav[1][high[pXav[1][high].argsort()[::-1]]]
            if all_outputs != "no":
                self.amplitude_significant = self.amplitude[
                    high[pXav[1][high].argsort()[::-1]]
                ]
                self.damplitude_significant = 0.5 * abs(
                    self.amplitude[(high[pXav[1][high].argsort()[::-1]]) + 1]
                    - self.amplitude[(high[pXav[1][high].argsort()[::-1]]) - 1]
                )
                self.phase_significant = self.phase[high[pXav[1][high].argsort()[::-1]]]
                self.dphase_significant = 0.5 * abs(
                    self.phase[(high[pXav[1][high].argsort()[::-1]]) + 1]
                    - self.phase[(high[pXav[1][high].argsort()[::-1]]) - 1]
                )

            if all_outputs != "no":
                self.amplitude_maxima = self.amplitude[high2[0:5]]
                self.phase_maxima = self.phase[high2[0:5]]

            if self.fap01 is None:
                self.fap01 = self.power_max + 0.5 - int(self.fap_max <= 0.1)
            if self.fap1 is None:
                self.fap1 = self.power_max + 0.5 - int(self.fap_max <= 1)
            if self.fap10 is None:
                self.fap10 = self.power_max + 0.5 - int(self.fap_max <= 10)

            if self.fap01 < 0:
                self.fap01 = 0
            if self.fap1 < 0:
                self.fap1 = 0
            if self.fap10 < 0:
                self.fap10 = 0

            if axis_y_var != "p":
                self.fap01 = None
                self.fap1 = None
                self.fap10 = None
                color_power = pXav[1] / fap
                fap = 0
                pXav[1] = self.amplitude

            if Plot == True:
                if axis_x_var == "f":
                    fig = plt.figure()
                    ax1 = fig.add_subplot(111)
                    if Norm == True:
                        ax1.plot(1 / pXav[0], pXav[1] / fap, color=color, ls=ls)
                        ax1.axhline(y=1, linestyle="-.", color="k")
                        ax1.set_ylabel("Power normalised", fontsize=14)
                    else:
                        ax1.plot(1 / pXav[0], pXav[1], color=color, ls=ls)
                        ax1.axhline(y=fap, linestyle="-.", color="k")
                        ax1.set_ylabel(
                            ["Power", "K Amplitude"][int(axis_y_var != "p")],
                            fontsize=14,
                        )
                    if P_th != None:
                        ax1.axvline(x=P_th, linestyle=":", linewidth=1.5, color="k")
                    ax1.set_xlim([1 / pXav[0].max(), 1 / pXav[0].min()])
                    ax1.set_xlim(xlim)  # change the xlim if a value is precised
                    ax2 = ax1.twiny()
                    ax1.set_xlabel("Frequency [1/days]", fontsize=14)
                    ax2.set_xticks(1 / np.array([1, 3, 6, 10, 20, 30, 50, 140]))
                    ax2.set_xticklabels(np.array([1, 3, 6, 10, 20, 30, 50, 140]))
                    ax2.set_xlim(ax1.get_xlim())
                    ax2.set_xlabel("Period [days]", fontsize=14)
                else:
                    if Norm == True:
                        if norm_val:
                            plt.plot(
                                1 / pXav[0],
                                pXav[1] / fap,
                                color=color,
                                ls=ls,
                                label=legend,
                                zorder=zorder,
                                lw=lw,
                                alpha=alpha,
                            )
                            plt.axhline(
                                y=1, linestyle="-.", color="k", zorder=zorder + 1
                            )
                            plt.ylabel("Power normalised", fontsize=14)
                        else:
                            plt.plot(
                                1 / pXav[0],
                                pXav[1],
                                color=color,
                                ls=ls,
                                label=legend,
                                zorder=zorder,
                                lw=lw,
                                alpha=alpha,
                            )
                            plt.axhline(
                                y=fap, linestyle="-.", color="k", zorder=zorder + 1
                            )
                            plt.ylabel(
                                ["Power", "K Amplitude"][int(axis_y_var != "p")],
                                fontsize=14,
                            )

                    else:
                        plt.plot(
                            1 / pXav[0],
                            pXav[1],
                            color=color,
                            ls=ls,
                            label=legend,
                            zorder=zorder,
                            lw=lw,
                            alpha=alpha,
                        )
                        plt.axhline(y=fap, linestyle="-.", color="k")
                        if self.fap01 is not None:
                            plt.axhline(y=self.fap01, linestyle=":", color="k")
                        if self.fap1 is not None:
                            plt.axhline(
                                y=self.fap1, linestyle="-", color="k", label="e"
                            )
                        if self.fap10 is not None:
                            plt.axhline(y=self.fap10, linestyle="-.", color="k")
                        plt.ylabel(
                            ["Power", "K Amplitude"][int(axis_y_var != "p")],
                            fontsize=14,
                        )
                    if P_th != None:
                        plt.axvline(x=P_th, linestyle=":", linewidth=1.5, color="k")
                    plt.xscale("log")
                    plt.xlabel("Period [days]", fontsize=14)
                    plt.xlim(1 / np.max(self.freq), 1 / np.min(self.freq))
                    plt.ylim(0, None)

                if axis_y_var != "p":
                    plt.plot(1 / pXav[0], pXav[1], color=color)
                    # jet = plt.get_cmap('Reds')
                    # cNorm  = colors.Normalize(vmin=0, vmax=1)
                    # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
                    # plt.scatter(1/pXav[0][0:2], pXav[1][0:2], c=color_power[0:2],cmap='Reds',vmin=0,vmax=1)
                    # #plt.colorbar()
                    # for i in tqdm(range(len(pXav[0])-2)):
                    #     if 1/pXav[0][i]>p_min2:
                    #         colorVal = scalarMap.to_rgba(color_power)
                    #         plt.plot(1/pXav[0][i:i+2], pXav[1][i:i+2],c=colorVal[i])

            if all_outputs != "no":
                self.amplitude_max = self.amplitude[high2[0]]
                self.phase_max = self.phase[high2[0]]

            if Norm == False:
                fap = 1
            if infos:
                for k in range(int(infos)):
                    plt.annotate(
                        " P = %.2f \n K = %.2f +/- %.2f \n Phi = %.2f +/- %.2f"
                        % (
                            self.period_significant[k],
                            self.amplitude_significant[k],
                            self.damplitude_significant[k],
                            self.phase_significant[k],
                            self.dphase_significant[k],
                        ),
                        xy=(
                            self.period_significant[k],
                            self.power_significant[k] / fap,
                        ),
                        xytext=(
                            self.period_significant[k],
                            1.5 * (self.power_significant[k] / fap),
                        ),
                        ha="center",
                        va="bottom",
                        arrowprops=dict(facecolor="black", lw=1),
                    )
                    plt.ylim(0, 1.5 * (self.power_max / fap) + 0.2)
            if supp != None:
                self.x = self.x_backup.copy()
                self.y = self.y_backup.copy()
                self.yerr = self.yerr_backup.copy()
        else:
            if warning:
                print("Less than 6 points, periodogram not launched")

    def periodogram_keplerian_hierarchical(
        self,
        periods=[None],
        fap=0.1,
        photon_noise=0,
        nb_planet=8,
        Plot=False,
        deg=0,
        ms=1,
        rs=1,
        ofac=20,
        p_min=0,
        model=False,
        auto_long_trend=True,
        fit_ecc=True,
        mcmc=False,
        jitter=0.8,
        periodic=True,
        species=None,
        min_nb_cycle=1,
        nb_bins=9,
        capsize=0,
        eval_on_t=None,
        transits=None,
        sort_planets=True,
        known_planet=[],
    ):

        yerr = np.sqrt(self.yerr**2 + photon_noise**2)

        epoch_rjd = np.mean(self.x)
        y_backup = self.y.copy()
        x_backup = self.x.copy()

        timespan = np.max(self.x) - np.min(self.x)
        p_max = timespan / min_nb_cycle

        if periods[0] is not None:
            periods = list(np.array(periods)[np.array(periods) < p_max])
            if not len(periods):
                periods = [None]

        if deg > -1:
            self.substract_polyfit(deg)
            self.detrend_poly.periodogram(
                nb_perm=1, Norm=True, level=1 - fap / 100, Plot=False, p_min=p_min
            )
            long_power = int((self.detrend_poly.power[0] / self.detrend_poly.fap) > 1)
            if (long_power) & (auto_long_trend):
                deg += 1
                print(
                    "Power at long period found, power detrending increased to : %.0f"
                    % (deg)
                )

        if species is None:
            species = np.zeros(len(self.x)).astype("int")

        species_id = myf.label_int(species)

        if spleaf_version == "old":
            rv_model = rvmodel(
                x_backup - epoch_rjd,
                y_backup,
                yerr**2,
                inst_id=species_id,
                var_jitter_inst=np.array([float(jitter)] * len(np.unique(species))),
            )
        else:
            yerr_rv = term.Error(yerr)
            instjit = {}
            instruments = np.unique(species)
            for inst in instruments:
                instjit[f"inst_jit_{inst}"] = term.InstrumentJitter(
                    species == inst, jitter
                )
            rv_model = rvmodel(x_backup - epoch_rjd, y_backup, err=yerr_rv, **instjit)

        for kpow in range(deg + 1):
            if spleaf_version == "old":
                rv_model.addlin(rv_model.t ** (kpow), "drift_pow{}".format(kpow))
            else:
                rv_model.add_lin(rv_model.t ** (kpow), "drift_pow{}".format(kpow))

        # Add linear parameters
        if len(np.unique(species)) - 1:
            for num, kinst in enumerate(np.unique(species)):
                if spleaf_version == "old":
                    rv_model.addlin(
                        1.0 * (species == kinst), "offset_inst_{}".format(kinst)
                    )
                else:
                    rv_model.add_lin(
                        1.0 * (species == kinst), "offset_inst_{}".format(kinst)
                    )

        P_kep = []
        K_kep = []
        e_kep = []
        la0_kep = []
        w_kep = []
        a_kep = []
        mass_kep = []
        phi_kep = []

        if Plot:
            plt.figure(figsize=(14, 3 + 3 * (nb_planet)))
            plt.subplots_adjust(
                left=0.06, right=0.97, top=0.95, bottom=0.10, hspace=0, wspace=0.4
            )

        if periods[0] is not None:
            nb_planet = len(periods)
            button = 1
        else:
            periods = [None] * 99
            button = 0
        i = 0
        self.keplerian_converged = True
        for i in range(1, nb_planet + 1):
            try:
                rv_model.fit(method="L-BFGS-B")
            except:
                try:
                    rv_model.fit(method="tnc")
                except:
                    try:
                        rv_model.fit(method="slsqp")
                    except:
                        print("Fit did not converge")
                        # rv_model.addlin(rv_model.t**(deg+1), 'drift_pow{}'.format(kpow))
                        self.keplerian_converged = False

            if Plot:
                ax = plt.subplot(nb_planet + 1, 2, 2 * i)
            ts = tableXY(x_backup, rv_model.residuals(), yerr)
            ts.periodogram(
                nb_perm=1,
                Plot=Plot,
                level=1 - fap / 100,
                Norm=True,
                ofac=ofac,
                p_min=p_min,
                p_max=p_max,
            )
            for periods_planets in known_planet:
                plt.axvline(x=periods_planets, color="b", alpha=0.25)

            if Plot:
                plt.ylim(0.001, None)
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                plt.ylabel("Power", fontsize=14)
                plt.xlabel(None)
                plt.tick_params(
                    direction="in", top=True, labeltop=(i == 0), labelbottom=False
                )
                if periods[i - 1] is not None:
                    curve = tableXY(1 / ts.freq[::-1], ts.power[::-1])

                    curve.find_max()
                    close = myf.find_nearest(curve.x_max, periods[i - 1])[0]
                    per_max = curve.x_max[close]
                    pow_max = curve.y_max[close]
                    fap_max = [10, 1, 0.1][
                        int(curve.x_max[close] > ts.fap01)
                        + int(curve.x_max[close] > ts.fap1)
                    ]
                else:
                    per_max = ts.perio_max
                    pow_max = ts.power_max
                    fap_max = ts.fap_max
                plt.axvline(x=per_max, color="r", alpha=0.7)
                al = myf.calculate_alias(per_max)[0][0:-2]
                for l in al:
                    plt.axvline(x=l, color="r", alpha=0.1)
                plt.scatter(
                    per_max,
                    pow_max,
                    color="r",
                    marker="o",
                    zorder=100,
                    label=r"$\log_{10}$(FAP) = %.1f" % (np.log10(fap_max) - 2),
                )
                plt.legend(loc=1)

            if periods[0] is not None:
                perio_max = periods[i - 1]
            else:
                perio_max = ts.perio_max.copy()

            if ((ts.power_max >= ts.fap) & (not button)) | (
                (i != len(periods) + 1) & (button)
            ):
                if spleaf_version == "old":
                    rv_model.smartaddpla(perio_max)
                    rv_model.changeparpla(
                        rv_model.planame[i - 1],
                        params=["P", "la0", "K", "ecosw", "esinw"],
                    )
                    if not fit_ecc:
                        rv_model.set_params(np.zeros(2), rv_model.fitparams[-2:])
                        rv_model.fitparams = rv_model.fitparams[:-2]
                else:
                    rv_model.add_keplerian_from_period(perio_max)
                    rv_model.set_keplerian_param(
                        f"{rv_model.nkep-1}", param=["P", "la0", "K", "ecosw", "esinw"]
                    )
                    if not fit_ecc:
                        rv_model.set_param(np.zeros(2), rv_model.fit_param[-2:])
                        rv_model.fit_param = rv_model.fit_param[:-2]
            else:
                i -= 1
                break

        try:
            rv_model.fit(method="L-BFGS-B")
        except:
            try:
                rv_model.fit(method="tnc")
            except:
                try:
                    rv_model.fit(method="slsqp")
                except:
                    print("Fit did not converge")

        jitter_params = []
        for kinst in np.unique(species_id):
            if spleaf_version == "old":
                jitter_params.append("cov.var_jitter_inst.{}".format(kinst))
            else:
                jitter_params = [f"cov.{key}.sig" for key in instjit]

        if jitter:
            if spleaf_version == "old":
                rv_model.fitparams += jitter_params
            else:
                rv_model.fit_param += jitter_params

        try:
            rv_model.fit(method="L-BFGS-B")
        except:
            try:
                rv_model.fit(method="tnc")
            except:
                try:
                    rv_model.fit(method="slsqp")
                except:
                    print("Fit did not converge")

        if i:
            for j in range(1, i + 1):
                if spleaf_version == "old":
                    rv_model.changeparpla(
                        rv_model.planame[j - 1], params=["P", "la0", "K", "w", "e"]
                    )
                    P_planet = rv_model.get_params("pla." + str(j - 1) + ".P")
                    K_planet = rv_model.get_params("pla." + str(j - 1) + ".K")
                    e_planet = rv_model.get_params("pla." + str(j - 1) + ".e")
                    la0_planet = rv_model.get_params("pla." + str(j - 1) + ".la0")
                    w_planet = rv_model.get_params("pla." + str(j - 1) + ".w")
                    rv_model.changeparpla(
                        rv_model.planame[j - 1],
                        params=["P", "la0", "K", "ecosw", "esinw"],
                    )
                else:
                    rv_model.set_keplerian_param(
                        str(j - 1), param=["P", "la0", "K", "w", "e"]
                    )
                    P_planet = rv_model.get_param("kep." + str(j - 1) + ".P")
                    K_planet = rv_model.get_param("kep." + str(j - 1) + ".K")
                    e_planet = rv_model.get_param("kep." + str(j - 1) + ".e")
                    la0_planet = rv_model.get_param("kep." + str(j - 1) + ".la0")
                    w_planet = rv_model.get_param("kep." + str(j - 1) + ".w")
                    rv_model.set_keplerian_param(
                        str(j - 1), param=["P", "la0", "K", "ecosw", "esinw"]
                    )
                mass_planet = myf.AmpStar(
                    ms, 0, P_planet, abs(K_planet), i=90, e=e_planet, code="Sun-Earth"
                )
                a_planet = (ms * 7.496e-6 * P_planet**2) ** (1 / 3)
                phi_planet = (
                    la0_planet * 180 / np.pi - 360 * epoch_rjd / P_planet - 270
                ) % 360

                P_kep.append(P_planet)
                K_kep.append(K_planet)
                e_kep.append(e_planet)
                la0_kep.append(la0_planet * 180 / np.pi)
                w_kep.append(w_planet * 180 / np.pi)
                a_kep.append(a_planet)
                mass_kep.append(mass_planet)
                phi_kep.append(phi_planet)

                if Plot:
                    if spleaf_version == "old":
                        res = tableXY(
                            x_backup,
                            rv_model.residuals()
                            + rv_model.kep[j - 1].rv(x_backup - epoch_rjd),
                            yerr,
                        )
                    else:
                        res = tableXY(
                            x_backup,
                            rv_model.residuals()
                            + rv_model.keplerian[str(j - 1)].rv(x_backup - epoch_rjd),
                            yerr,
                        )
                    sup = np.nanpercentile(res.y, 75) + 3 * myf.IQ(res.y)
                    inf = np.nanpercentile(res.y, 25) - 3 * myf.IQ(res.y)
                    ax = plt.subplot(nb_planet + 1, 2, 2 * j - 1)
                    plt.tick_params(
                        direction="in",
                        top=True,
                        labeltop=((j - 1) == 0),
                        labelbottom=False,
                    )
                    if not j - 1:
                        ax.xaxis.set_label_position("top")
                        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
                        plt.xlabel("Phase", fontsize=14)

                    ax.annotate(
                        "P = %s days \nK = %.2f m/s \ne = %.2f \nMsin(i) = %.1f"
                        % (
                            myf.format_number(P_planet, digit=2),
                            K_planet,
                            e_planet,
                            mass_planet,
                        ),
                        xy=(1.02, 0.33),
                        xycoords="axes fraction",
                        horizontalalignment="left",
                        fontsize=13,
                    )

                    res.plot(
                        modulo=P_planet,
                        periodic=True,
                        modulo_norm=True,
                        alpha=0.5,
                        capsize=capsize,
                    )
                    res.modulo(P_planet, modulo_norm=True)
                    res.mod.binning(1 / nb_bins)
                    # res.mod.binned_scatter(nb_bins,color='r',alpha_p=0)
                    # res.mod.binned.plot(color='b',zorder=1001)
                    plt.errorbar(
                        res.mod.binned.x,
                        res.mod.binned.y,
                        yerr=res.mod.binned.yerr,
                        capsize=0,
                        fmt="ro",
                        markersize=8,
                        markeredgecolor="k",
                        markeredgewidth=2,
                        zorder=1001,
                    )
                    new_t = np.linspace(0, P_planet, 500)
                    if spleaf_version == "old":
                        curve = rv_model.kep[j - 1].rv(new_t - epoch_rjd)
                    else:
                        curve = rv_model.keplerian[str(j - 1)].rv(new_t - epoch_rjd)
                    sort = np.argsort(new_t % P_planet)
                    plt.plot(new_t / P_planet, curve, color="r", zorder=1000, lw=3)
                    plt.plot(new_t / P_planet - 1, curve, color="r", zorder=1000, lw=3)
                    plt.plot(new_t / P_planet + 1, curve, color="r", zorder=1000, lw=3)
                    if transits is not None:
                        if type(transits) == dict:
                            all_p = np.array(list(transits.keys()))
                            p_detect = all_p[
                                myf.find_nearest(all_p.astype("float"), P_planet)[0][0]
                            ]
                            if transits[p_detect][0] is not None:
                                jdb_transit = transits[p_detect]
                                for trans in jdb_transit:
                                    plt.axvline(
                                        x=(trans % P_planet) / P_planet,
                                        color="k",
                                        alpha=0.8,
                                        lw=2,
                                        zorder=1001,
                                    )
                        else:
                            for color_trans, trans in zip(
                                [
                                    "r",
                                    "g",
                                    "b",
                                    "purple",
                                    "olive",
                                    "cyan",
                                    "pink",
                                    "lime",
                                    "navy",
                                    "brown",
                                    "gold",
                                    "teal",
                                    "k",
                                ][0 : len(transits)],
                                transits,
                            ):
                                plt.axvline(
                                    x=(trans % P_planet) / P_planet,
                                    color=color_trans,
                                    alpha=0.8,
                                    lw=2,
                                    zorder=1001,
                                )
                        plt.axhline(y=0, color="k", zorder=1003)

                    plt.ylabel("RV [m/s]", fontsize=14)
                    plt.xlim(-0.1 - np.float(periodic), 1.1 + np.float(periodic))
                    if plt.gca().get_ylim()[0] < inf:
                        plt.ylim(inf, None)
                    if plt.gca().get_ylim()[1] > sup:
                        plt.ylim(None, sup)

        nb_planet = len(P_kep)

        self.rv_model = rv_model
        self.nb_planet_fitted = nb_planet
        self.model_periods = P_kep

        proba_transit = [np.round(myf.transit_proba(i, Rs=rs, Ms=ms), 2) for i in P_kep]
        dt_transit = [
            np.round(myf.transit_circular_dt(i, Rs=rs, Ms=ms), 2) for i in P_kep
        ]
        Tc = (
            np.array(epoch_rjd)
            - np.array(la0_kep) / 360 * np.array(P_kep)
            + np.array(P_kep) / 4
        )  # empirical formulae

        self.planet_fitted = pd.DataFrame(
            {
                "p": P_kep,
                "p_std": 0,
                "k": K_kep,
                "k_std": 0,
                "e": e_kep,
                "e_std": 0,
                "phi": phi_kep,
                "phi_std": 0,
                "peri": w_kep,
                "peri_std": 0,  # before update node
                "long": la0_kep,
                "long_std": 0,  # before update peri
                "a": a_kep,
                "a_std": 0,
                "mass": mass_kep,
                "mass_std": 0,
                "i": [0] * len(P_kep),
                "i_std": 0,
                "t0": epoch_rjd,
                "proba_t": proba_transit,
                "dt": dt_transit,
                "Tc": Tc,
            }
        )

        if sort_planets:
            self.planet_fitted = self.planet_fitted.sort_values(by="p")
        self.planet_fitted.index = [
            "planet %.0f" % (i) for i in range(1, 1 + nb_planet)
        ]

        if spleaf_version == "old":
            nb_par = np.sum(np.array(rv_model.get_params()) != 0)
        else:
            nb_par = np.sum(np.array(rv_model.get_param()) != 0)
        chi2 = self.rv_model.chi2() / (len(self.x) - 1 - nb_par)
        self.nb_par = nb_par
        self.chi2 = chi2

        self.model_drift = np.zeros(len(x_backup))
        self.model_keplerian = np.zeros(len(x_backup))
        self.model_keplerian_i = []
        self.model_deg = deg
        for kpow in range(deg + 1):
            if spleaf_version == "old":
                self.model_drift += rv_model.get_params(
                    "lin.drift_pow%.0f" % (kpow)
                ) * (x_backup - epoch_rjd) ** (kpow)
            else:
                self.model_drift += rv_model.get_param("lin.drift_pow%.0f" % (kpow)) * (
                    x_backup - epoch_rjd
                ) ** (kpow)
        for kpla in range(nb_planet):
            if spleaf_version == "old":
                self.model_keplerian += rv_model.kep[kpla].rv(x_backup - epoch_rjd)
                self.model_keplerian_i.append(
                    tableXY(x_backup, rv_model.kep[kpla].rv(x_backup - epoch_rjd))
                )
            else:
                self.model_keplerian += rv_model.keplerian[str(kpla)].rv(
                    x_backup - epoch_rjd
                )
                self.model_keplerian_i.append(
                    tableXY(
                        x_backup, rv_model.keplerian[str(kpla)].rv(x_backup - epoch_rjd)
                    )
                )

        if eval_on_t is not None:
            self.model_eval_t = np.zeros(len(eval_on_t))
            self.model_eval_t_i = []
            for kpla in range(nb_planet):
                if spleaf_version == "old":
                    self.model_eval_t += rv_model.kep[kpla].rv(eval_on_t - epoch_rjd)
                    self.model_eval_t_i.append(
                        tableXY(eval_on_t, rv_model.kep[kpla].rv(eval_on_t - epoch_rjd))
                    )
                else:
                    self.model_eval_t += rv_model.keplerian[str(kpla)].rv(
                        eval_on_t - epoch_rjd
                    )
                    self.model_eval_t_i.append(
                        tableXY(
                            eval_on_t,
                            rv_model.keplerian[str(kpla)].rv(eval_on_t - epoch_rjd),
                        )
                    )

        if Plot:
            if i:
                plt.subplot(nb_planet + 3, 2, 2 * (i + 2) + 2)
            else:
                plt.close()
                plt.figure(figsize=(14, 3))
                plt.subplot(1, 2, 2)
                plt.subplots_adjust(
                    left=0.06, right=0.97, top=0.90, bottom=0.20, hspace=0, wspace=0.4
                )

            res = tableXY(x_backup, rv_model.residuals(), yerr)

            res.periodogram(
                nb_perm=1,
                Plot=Plot,
                level=1 - fap / 100,
                Norm=True,
                ofac=ofac,
                p_min=p_min,
                p_max=p_max,
            )
            for periods_planets in known_planet:
                plt.axvline(x=periods_planets, color="b", alpha=0.25)

            axe = plt.gca()
            plt.plot(
                axe.get_xlim(),
                [res.fap, res.fap],
                ls="-.",
                label="FAP = %.1f%%" % (fap),
                color="k",
            )
            plt.legend(loc=1)
            plt.ylabel("Power", fontsize=14)
            plt.ylim(0.001, None)
            plt.tick_params(direction="in", top=True)
            if i:
                nax = plt.subplot(nb_planet + 3, 2, 2 * (i + 2) + 1)
            else:
                nax = plt.subplot(1, 2, 1)
            plt.xlabel("Time jdb", fontsize=14)
            res.rms_w()
            vec_input = self.copy()

            self.y -= self.model_drift
            self.rms_w()
            self.recenter(who="Y")

            self.plot(color="gray", label="Input", capsize=capsize, markersize=4)
            self.y += self.model_drift
            res.plot(
                label="Res.", color="k", species=species, capsize=capsize, markersize=4
            )

            vec_init = self.copy()
            vec_init.y -= self.model_drift

            vec_init.yerr = res.yerr
            vec_init.rms_w()

            if model:
                new_t = np.linspace(np.min(x_backup), np.max(x_backup), 10000)
                drift = np.zeros(len(new_t))
                keplerian = np.zeros(len(new_t))
                for kpow in range(deg + 1):
                    if spleaf_version == "old":
                        drift += rv_model.get_params("lin.drift_pow%.0f" % (kpow)) * (
                            new_t - epoch_rjd
                        ) ** (kpow)
                    else:
                        drift += rv_model.get_param("lin.drift_pow%.0f" % (kpow)) * (
                            new_t - epoch_rjd
                        ) ** (kpow)
                for kpla in range(nb_planet):
                    if spleaf_version == "old":
                        keplerian += rv_model.kep[kpla].rv(new_t - epoch_rjd)
                    else:
                        keplerian += rv_model.keplerian[str(kpla)].rv(new_t - epoch_rjd)

                model = tableXY(new_t, drift + keplerian)
                model.plot(color="r", ls="-")

            bic = nb_par * np.log(len(res.x)) + self.rv_model.chi2()
            self.bic = bic
            nax.annotate(
                r"$\chi^2_r$ = %.3f" % (chi2),
                xy=(1.02, 0.57),
                xycoords="axes fraction",
                horizontalalignment="left",
                fontsize=13,
            )
            nax.annotate(
                r"BIC = %.2f" % (bic),
                xy=(1.02, 0.33),
                xycoords="axes fraction",
                horizontalalignment="left",
                fontsize=13,
            )
            nax.annotate(
                r"LTD = %.0f" % (deg),
                xy=(1.02, 0.09),
                xycoords="axes fraction",
                horizontalalignment="left",
                fontsize=13,
            )

            plt.title(
                "rms_input = %.2f m/s         rms_residuals = %.2f m/s"
                % (vec_init.rms, res.rms),
                fontsize=13,
            )
            plt.legend()
            plt.ylabel("RV [m/s]", fontsize=14)
            self.rv_residues = res
            self.rv_residues_rms = res.rms
            self.model_info = {
                "rms_res": res.rms,
                "rms_input": vec_init.rms,
                "chi2": self.chi2,
                "bic": self.bic,
                "nb_obs": len(self.x),
                "timespan": np.max(self.x) - np.min(self.x),
            }

        # if mcmc:deg=§
        #     plt.figure(69)
        #     nsamples = 100000
        #     samples, diagnos = rv_model.sample(nsamples=nsamples, logprior=lambda fitparams, x:0)
        #     corner.corner(samples[nsamples//4:],labels=rv_model.fitparams, quantiles=[0.159, 0.5, 0.841], show_titles=True)

    def planetary_phase_test(
        self,
        periods=None,
        vecs=None,
        ref_vector=None,
        ref_time=0,
        split_vector=None,
        fit_ecc=False,
        deg_detrend=0,
        deg=0,
        photon_noise=0,
        nb_planet=1,
        nb_cut=2,
        bbox=(-0.25, -0.35),
        markers=["o", "x", "^", "s", "v", "<", ">"],
        colors=["blue", "red", "green", "orange", "purple"],
        Plot=True,
        rmax=None,
        ymax=None,
        xmin=None,
        xmax=None,
    ):

        vec_lbl = self.copy()

        if deg_detrend:
            vec_lbl.substract_polyfit(deg_detrend, replace=True)

        if periods is None:
            vec_lbl.periodogram_keplerian_hierarchical(
                Plot=False,
                nb_planet=nb_planet,
                fit_ecc=fit_ecc,
                photon_noise=photon_noise,
                fap=1,
            )
            tab = vec_lbl.planet_fitted.sort_values(by="k", ascending=False)["p"]
            periods = list(tab)

        if ref_vector is None:
            ref_vector = tableXY(vec_lbl.x, np.zeros(len(vec_lbl.x)))

        if len(periods):
            if nb_cut > 5:
                nb_cut = 5
                print("Nb cut maximum equal to 5")

            if vecs is None:
                if split_vector is None:
                    limits = np.array(
                        [
                            np.percentile(vec_lbl.x, 100 / nb_cut * i)
                            for i in range(0, nb_cut)
                        ]
                    )
                    if np.min(np.diff(limits)) < 1.5 * np.max(periods):
                        nb_cut -= 1
                        print(
                            "Cut number reduced of 1 otherwise baseline to short for the planetary fit"
                        )
                    vecs = [
                        vec_lbl.masked(
                            (vec_lbl.x >= np.percentile(vec_lbl.x, 100 / nb_cut * i))
                            & (
                                vec_lbl.x
                                <= np.percentile(vec_lbl.x, 100 / nb_cut * (i + 1))
                            ),
                            replace=False,
                        )
                        for i in range(0, nb_cut)
                    ]
                    rhks = [
                        ref_vector.masked(
                            (vec_lbl.x >= np.percentile(vec_lbl.x, 100 / nb_cut * i))
                            & (
                                vec_lbl.x
                                <= np.percentile(vec_lbl.x, 100 / nb_cut * (i + 1))
                            ),
                            replace=False,
                        )
                        for i in range(0, nb_cut)
                    ]
                else:
                    if len(split_vector) == len(vec_lbl.x):
                        vecs = [
                            vec_lbl.masked(
                                (
                                    split_vector
                                    >= np.percentile(split_vector, 100 / nb_cut * i)
                                )
                                & (
                                    split_vector
                                    <= np.percentile(
                                        split_vector, 100 / nb_cut * (i + 1)
                                    )
                                ),
                                replace=False,
                            )
                            for i in range(0, nb_cut)
                        ]
                        rhks = [
                            ref_vector.masked(
                                (
                                    split_vector
                                    >= np.percentile(split_vector, 100 / nb_cut * i)
                                )
                                & (
                                    split_vector
                                    <= np.percentile(
                                        split_vector, 100 / nb_cut * (i + 1)
                                    )
                                ),
                                replace=False,
                            )
                            for i in range(0, nb_cut)
                        ]
                    else:
                        print("RHK length is not matching the vector length")
            else:
                rhks = []

            if nb_cut < 2:
                nb_cut = 2
                print("Nb cut minimum equal to 2")

            vecs = [vec_lbl] + vecs
            rhks = [ref_vector] + rhks

            for j in range(len(vecs)):
                vecs[j].yerr = np.sqrt(vecs[j].yerr ** 2 + photon_noise**2)

            nb_cut = len(vecs) - 1

            colors = np.sort(colors[0:nb_cut])

            k = []
            f = []
            dk = []
            df = []

            for b in range(len(vecs)):
                v = vecs[b]
                base = [(v.x - np.median(v.x)) ** d for d in range(deg + 1)]
                for p in periods:
                    base.append(np.cos(2 * np.pi * v.x / p))
                    base.append(np.sin(2 * np.pi * v.x / p))
                base = np.array(base)

                v.fit_base(base)
                v.y = v.y - np.sum(
                    v.coeff_fitted[0 : 1 + deg] * base[0 : 1 + deg].T, axis=1
                )

                k.append(
                    np.sqrt(
                        v.coeff_fitted[1 + deg :: 2] ** 2
                        + v.coeff_fitted[2 + deg :: 2] ** 2
                    )
                )
                f.append(
                    (
                        np.arctan2(
                            v.coeff_fitted[1 + deg :: 2], v.coeff_fitted[2 + deg :: 2]
                        )
                        * 180
                        / np.pi
                    )
                    % 360
                )
                dk.append(
                    np.sqrt(
                        (
                            v.coeff_fitted[1 + deg :: 2] ** 2
                            * v.coeff_fitted_std[1 + deg :: 2] ** 2
                            + v.coeff_fitted[2 + deg :: 2] ** 2
                            * v.coeff_fitted_std[2 + deg :: 2] ** 2
                        )
                        / k[-1] ** 2
                    )
                )
                # df_b = f*(abs(vec_lbl.coeff_fitted_std[1::2]/vec_lbl.coeff_fitted[1::2])+abs(vec_lbl.coeff_fitted_std[2::2]/vec_lbl.coeff_fitted[2::2]))
                df.append(
                    180
                    / np.pi
                    * np.arctan(
                        np.sqrt(
                            (
                                v.coeff_fitted_std[2 + deg :: 2]
                                / v.coeff_fitted[1 + deg :: 2]
                            )
                            ** 2
                            + (
                                v.coeff_fitted_std[1 + deg :: 2]
                                * v.coeff_fitted[2 + deg :: 2]
                                / v.coeff_fitted[1 + deg :: 2] ** 2
                            )
                            ** 2
                        )
                    )
                )

            f_ref = (2 * np.pi * ref_time / np.array(periods) * 180 / np.pi) % (360)

            k = np.array(k)
            f = np.array(f) + f_ref
            dk = np.array(dk)
            df = np.array(df)
            f = f % 360

            if Plot:

                if np.sum(ref_vector.y):
                    nb_raw = 3
                    ysize = 0.25
                else:
                    nb_raw = 2
                    ysize = 0.4

                fig = plt.figure(figsize=(18, 5))

                plt.axes([0.05, 0.11, 0.325, ysize])
                vec_lbl.periodogram(Norm=True, zorder=100)
                plt.xlim(xmin, xmax)
                for p in periods:
                    plt.axvline(x=p, color="magenta", alpha=1, ls=":")
                plt.axes([0.05, 0.11 + ysize + 0.05, 0.325, ysize])
                vec_lbl.plot()
                plt.ylabel("RV [m/s]", fontsize=14)

                plt.axes([0.425, 0.11, 0.325, ysize])
                save_x = []
                save_x2 = []
                for b in range(1, len(vecs)):
                    v = vecs[b]
                    v.periodogram(color=colors[b - 1], Norm=True, norm_val=True)
                    save_x.append(1 / v.freq[0])
                    save_x2.append(1 / v.freq[-1])

                plt.ylabel("Power", fontsize=14)
                ax = plt.gca()
                save_y = ax.get_ylim()[1]
                save_x = np.min(save_x)
                save_x2 = np.max(save_x2)

                plt.ylim(None, ymax)
                plt.xlim(xmin, xmax)

                plt.axes([0.425, 0.11 + ysize + 0.05, 0.325, ysize])
                for b in range(1, len(vecs)):
                    v = vecs[b]
                    v.plot(color=colors[b - 1])
                plt.ylabel("RV [m/s]", fontsize=14)

                if nb_raw == 3:
                    plt.axes([0.05, 0.11 + 2 * ysize + 0.10, 0.325, ysize])
                    ref_vector.plot()
                    plt.ylabel("Ref vector", fontsize=14)

                    plt.axes([0.425, 0.11 + 2 * ysize + 0.10, 0.325, ysize])
                    for b in range(1, len(rhks)):
                        v = rhks[b]
                        v.plot(color=colors[b - 1])
                    plt.ylabel("Ref vector", fontsize=14)

                ax = fig.add_subplot(133, projection="polar")
                plt.title("%.0f-Keplerian model" % (len(periods)))
                for j in range(len(periods)):
                    m = markers[j]
                    for b in range(1, len(vecs)):
                        t1 = myf.polar_err(
                            f[b][j],
                            k[b][j],
                            df[b][j],
                            dk[b][j],
                            color=colors[b - 1],
                            marker=m,
                        )
                    t = myf.polar_err(
                        f[0][j], k[0][j], df[0][j], dk[0][j], color="k", marker=m
                    )
                    plt.polar(
                        f[0][j] * np.pi / 180,
                        k[0][j],
                        "k" + m,
                        label="P=%.2f" % (list(periods)[j]),
                    )
                    plt.legend(loc=3, bbox_to_anchor=bbox, ncol=3)

                dist = [
                    sum(myf.dist_modulo(d * np.pi / 4, np.pi / 180 * np.ravel(f)))
                    for d in [1, 2, 3, 5, 6, 7]
                ]

                # val = np.linspace(0,360,36)[np.where(dist==np.max(dist))[0][0]]
                # print(val)
                plt.ylim(0, rmax)
                ax.set_rlabel_position([45, 90, 135, -135, -90, -45][np.argmax(dist)])
                plt.subplots_adjust(left=0.4, right=0.97, bottom=0.25)

            return {
                "phi": f,
                "k": k,
                "phi_std": df,
                "k_std": dk,
                "axe_x": save_x,
                "axe_x2": save_x2,
                "axe_y": save_y,
                "p": periods,
            }

    def periodogram_auto_detrend(
        self,
        kmin=7,
        emax=0.85,
        photon_noise=0.3,
        fap=0.1,
        p_min=1.1,
        jitter=0.7,
        fit_ecc=True,
        species=None,
        power_max=3,
        degree_max=2,
        x_export=None,
    ):

        save = []
        for power in np.arange(power_max + 1):
            vec = self.copy()
            for deg in np.arange(degree_max + 1):
                vec.substract_polyfit(power, replace=False)
                vec.detrend_poly.periodogram_keplerian_hierarchical(
                    photon_noise=photon_noise,
                    jitter=jitter,
                    fap=fap,
                    deg=deg,
                    p_min=p_min,
                    fit_ecc=fit_ecc,
                    species=species,
                    periodic=0.25,
                    Plot=False,
                    auto_long_trend=False,
                )
                vec.detrend_poly.periodogram_keplerian_hierarchical(
                    photon_noise=photon_noise,
                    jitter=jitter,
                    fap=fap,
                    deg=deg,
                    p_min=p_min,
                    fit_ecc=fit_ecc,
                    species=species,
                    periodic=0.25,
                    Plot=True,
                    nb_planet=vec.detrend_poly.nb_planet_fitted,
                    auto_long_trend=False,
                )

                bic = (vec.detrend_poly.nb_par + power) * np.log(
                    len(vec.detrend_poly.x)
                ) + vec.detrend_poly.rv_model.chi2()  # np.sum(vec.detrend_poly.rv_residues.y**2)/np.sum(vec.detrend_poly.yerr**2)

                save.append(
                    [
                        power,
                        deg,
                        vec.detrend_poly.keplerian_converged,
                        vec.detrend_poly.bic,
                        bic,
                        vec.detrend_poly.rv_residues_rms,
                        vec.detrend_poly.nb_planet_fitted,
                    ]
                )
                plt.close("all")
        tab = pd.DataFrame(
            save,
            columns=[
                "deg_detrend",
                "deg_fit_kep",
                "conv",
                "bic",
                "bic2",
                "rms",
                "nb_planet",
            ],
        )
        tab = tab.dropna()
        tab = tab.loc[tab["conv"] == True]
        if len(tab):
            tab = tab.sort_values(by="bic2")
            print(tab)
            tab = tab.reset_index(drop=True).loc[0]
            power = tab["deg_detrend"]
            deg = tab["deg_fit_kep"]
            nb_planet = tab["nb_planet"]
            vec = self.copy()
            vec.substract_polyfit(power, replace=False)
            vec.detrend_poly.periodogram_keplerian_hierarchical(
                sort_planets=False,
                photon_noise=photon_noise,
                jitter=jitter,
                fap=fap,
                deg=deg,
                p_min=p_min,
                fit_ecc=fit_ecc,
                species=species,
                periodic=0.25,
                Plot=True,
                nb_planet=nb_planet,
                auto_long_trend=False,
            )
            print(
                "\n----------------------------\n[INFO] Keplerian model :\n----------------------------\n"
            )
            print(
                vec.detrend_poly.planet_fitted[["p", "k", "e", "mass", "peri", "long"]]
            )

            self.auto_deg = np.max([power, deg])

            def produce_model(x):
                model_drift = np.zeros(len(x))
                model_keplerian = np.zeros(len(x))
                model_power = np.zeros(len(x))

                for kpow in range(deg + 1):
                    if spleaf_version == "old":
                        model_drift += vec.detrend_poly.rv_model.get_params(
                            "lin.drift_pow%.0f" % (kpow)
                        ) * (x - np.mean(vec.x)) ** (kpow)
                    else:
                        model_drift += vec.detrend_poly.rv_model.get_param(
                            "lin.drift_pow%.0f" % (kpow)
                        ) * (x - np.mean(vec.x)) ** (kpow)

                t = vec.detrend_poly.planet_fitted.loc[
                    (vec.detrend_poly.planet_fitted["k"] > kmin)
                    & (vec.detrend_poly.planet_fitted["e"] < emax)
                ]
                pp = np.where(
                    (vec.detrend_poly.planet_fitted["k"] > kmin)
                    & (vec.detrend_poly.planet_fitted["e"] < emax)
                )[0]
                self.auto_nb_planet = len(pp)
                for kpla in pp:
                    if spleaf_version == "old":
                        model_keplerian += vec.detrend_poly.rv_model.kep[kpla].rv(
                            x - np.mean(vec.x)
                        )
                    else:
                        model_keplerian += vec.detrend_poly.rv_model.keplerian[
                            str(kpla)
                        ].rv(x - np.mean(vec.x))

                model_power = np.polyval(vec.poly_coefficient, x)

                return model_drift + model_keplerian + model_power, t

            model = produce_model(vec.x)[0]
            model_smooth = tableXY(
                myf.vec_oversamp(vec.x), produce_model(myf.vec_oversamp(vec.x))[0]
            )
            if x_export is None:
                x_export = vec.x
            model_export, t = produce_model(x_export)
            print(
                "\n--------------------------------------------------------\n[INFO] Keplerian model with k>%.2f and e<%.2f :\n--------------------------------------------------------\n"
                % (kmin, emax)
            )
            print("\n", t[["p", "k", "e", "mass", "peri", "long"]])

        else:
            model = np.zeros(len(self.x))
            model_smooth = tableXY(vec.x, model)
            model_export = np.zeros(len(self.x))

        offset = np.median(model)
        model -= offset
        model_export -= offset
        model_smooth.y -= offset

        return model, model_smooth, model_export

    def evaluate_keplerian(self, planet_fitted, epoch_rjd=None):

        params = ["p", "long", "k", "e", "w"]
        params2 = ["P", "la0", "K", "ecosw", "esinw", "t0"]  # don't change this list

        table_param = planet_fitted.copy()

        table_param["ecosw"] = table_param["e"] * np.cos(
            table_param["peri"] * np.pi / 180
        )
        table_param["esinw"] = table_param["e"] * np.sin(
            table_param["peri"] * np.pi / 180
        )
        table_param["P"] = table_param["p"]
        table_param["K"] = table_param["k"]
        table_param["la0"] = table_param["long"] * np.pi / 180

        table_keplerian = table_param[params2].copy()

        jitter = 0.7
        species = np.zeros(len(self.x)).astype("int")
        species_id = myf.label_int(species)

        if spleaf_version == "old":
            rv_model = rvmodel(
                self.x,
                self.y,
                self.yerr**2,
                inst_id=species_id,
                var_jitter_inst=np.array([float(jitter)] * len(np.unique(species))),
            )
        else:
            yerr_rv = term.Error(self.yerr)
            instjit = {}
            instruments = np.unique(species)
            for inst in instruments:
                instjit[f"inst_jit_{inst}"] = term.InstrumentJitter(
                    species == inst, jitter
                )
            rv_model = rvmodel(self.x, self.y, err=yerr_rv, **instjit)

        model_keplerian = np.zeros(len(self.x))
        nkep = 0
        for kpla in table_keplerian.index:
            nkep += 1
            period = table_keplerian.loc[kpla, "P"]
            # print(table_keplerian.loc[kpla])
            if epoch_rjd is None:
                try:
                    t0 = table_keplerian.loc[kpla, "t0"]
                except:
                    t0 = np.mean(self.x)
            else:
                t0 = epoch_rjd

            if spleaf_version == "old":
                rv_model.smartaddpla(period)
                rv_model.changeparpla(rv_model.planame[nkep - 1], params=params2[:-1])
                rv_model.set_params(
                    np.array(table_keplerian.loc[kpla][params2[:-1]]),
                    rv_model.fitparams,
                )
                model_keplerian += rv_model.kep[nkep - 1].rv(self.x - t0)
            else:
                rv_model.add_keplerian_from_period(period)
                rv_model.set_keplerian_param(f"{rv_model.nkep-1}", param=params2[:-1])
                rv_model.set_param(
                    np.array(table_keplerian.loc[kpla][params2[:-1]]),
                    rv_model.fit_param,
                )
                model_keplerian += rv_model.keplerian["0"].rv(self.x - t0)
                # print(rv_model.keplerian['0'].get_value())
                # plt.subplot(len(table_keplerian.index),1,nkep)
                # plt.plot(self.x,rv_model.keplerian['0'].rv(self.x - t0),marker='.')

        self.model_keplerian = model_keplerian
        self.keplerian = tableXY(self.x, model_keplerian)
        self.keplerian.null()

    def periodogram_rolling(
        self, prot_estimated=25, windows=100, min_nb_points=25, ofac=10, inv=False
    ):
        # plt.title(proxy_name[num],fontsize=13)
        self.periodogram(Plot=False)
        proxy = self
        jdb = proxy.x
        # proxy.rm_outliers(m=2,kind='inter')
        jdb_int = np.array(jdb).astype("int")
        times = []
        # m2 = []
        # m = np.zeros(len(np.arange(int(jdb_int[0]),int(jdb_int[-1]))))
        time_idx = []
        all_power = []
        jdb_old_min = -1
        jdb_old_max = -1
        count = 0
        liste = np.arange(int(jdb_int[0]), int(jdb_int[-1]))
        pmax = int(prot_estimated * 3)
        pmin = 1 / np.max(proxy.freq)

        for i in tqdm(range(len(liste))):
            j = liste[i]
            mask = (jdb >= j) & (jdb <= j + windows)
            proxy1 = proxy.copy()
            proxy1.x = proxy1.x[mask]
            proxy1.y = proxy1.y[mask]
            proxy1.yerr = proxy1.yerr[mask]
            proxy1.xerr = proxy1.xerr[mask]

            # m[i] = 0
            if len(proxy1.x) > min_nb_points:
                # m[i] = 1
                if (jdb_old_min != proxy1.x[0]) | (jdb_old_max != proxy1.x[-1]):
                    # m[i] = 2
                    if (np.max(proxy1.x) - np.min(proxy1.x)) > 1.5 * prot_estimated:
                        # m[i] = 3
                        jdb_old_min = proxy1.x[0].copy()
                        jdb_old_max = proxy1.x[-1].copy()
                        count += 1
                        times.append(np.mean(proxy1.x))
                        time_idx.append(count)
                        proxy1.substract_polyfit(1, replace=True)
                        proxy1.rm_outliers(m=2, kind="inter")
                        proxy1.periodogram(nb_perm=1, ofac=ofac, level=None, Plot=False)
                        power = tableXY(
                            proxy1.freq, proxy1.power / proxy1.fap01, 0 * proxy1.freq
                        )
                        power.interpolate(
                            1 / np.linspace(pmin, pmax, 1000),
                            method="linear",
                            interpolate_x=False,
                        )
                        all_power.append(power.y)

        # m2 = np.array(m2)
        all_power = np.array(all_power)
        times = np.array(times)
        time_idx = np.array(time_idx)

        if len(all_power) > 5:
            if not inv:
                myf.my_colormesh(
                    np.linspace(pmin, pmax, 1000), time_idx, all_power, vmin=0, vmax=2
                )
                plt.title("Rotational period : %.1f" % (prot_estimated), fontsize=14)
                plt.xlabel("Period [days]", fontsize=14)
                plt.ylabel("Window (%.0f days) center idx" % (windows), fontsize=14)
            else:
                myf.my_colormesh(
                    time_idx, np.linspace(pmin, pmax, 1000), all_power.T, vmin=0, vmax=2
                )
                plt.title("Rotational period : %.1f" % (prot_estimated), fontsize=14)
                plt.ylabel("Period [days]", fontsize=14)
                plt.xlabel("Window (%.0f days) center idx" % (windows), fontsize=14)
        return times, all_power

    def periodogram_cumu(
        self,
        min_nbpoints=20,
        nb_perm=1,
        ofac=5,
        grid_period=np.arange(0.5, 400, 0.1),
        Plot=True,
        zoom=5,
        time_direction=1,
        all_plot=False,
    ):
        all_power = []
        all_fap = []
        all_amplitude = []
        all_phase = []

        if np.sign(time_direction) < 0:
            self.x = self.x[::-1]
            self.y = self.y[::-1]
            self.yerr = self.yerr[::-1]

        for j in tqdm(np.arange(min_nbpoints, len(self.x) - 1)[::-1]):
            new = tableXY(self.x[0:j], self.y[0:j], self.yerr[0:j])
            new.periodogram(nb_perm=nb_perm, ofac=ofac, Plot=False, all_outputs=True)
            all_fap.append(new.fap1)
            for obs, save in zip(
                [new.power, new.amplitude, new.phase],
                [all_power, all_amplitude, all_phase],
            ):
                test = tableXY(1 / new.freq, obs)
                test.order()
                test.interpolate(new_grid=grid_period, replace=False, method="linear")
                test.y_interp[
                    (grid_period < np.min(test.x)) | (grid_period > np.max(test.x))
                ] = 0
                save.append(test.y_interp)

        self.sequence = np.arange(min_nbpoints, len(self.x) - 1)[::-1]
        self.all_fap = np.array(all_fap)
        self.all_amplitude = np.array(all_amplitude)
        self.all_phase = np.array(all_phase)
        self.all_power = np.array(all_power)
        self.all_power[self.all_power < 0] = 0
        self.all_period = grid_period

        if np.sign(time_direction) < 0:
            self.x = self.x[::-1]
            self.y = self.y[::-1]
            self.yerr = self.yerr[::-1]

        if Plot:
            self.periodogram_cumu_plot(zoom=zoom, all_plot=all_plot)

    def periodogram_cumu_plot(
        self, zoom=5, minp=2.5, maxp=50, vmin=0.5, vmax=1.5, logz=False, all_plot=False
    ):
        mask_period = (self.all_period > minp) & (self.all_period < maxp)
        z = self.all_power[:, mask_period] / self.all_fap[:, np.newaxis]
        if logz:
            z = np.log10(z + 1e-3)
        if all_plot:
            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1)
            myf.my_colormesh(
                self.all_period[mask_period],
                self.sequence,
                z,
                vmin=vmin,
                vmax=vmax,
                zoom=zoom,
            )
            ax = plt.gca()
            plt.colorbar()
            plt.subplot(1, 3, 2, sharex=ax, sharey=ax)
            myf.my_colormesh(
                self.all_period[mask_period],
                self.sequence,
                self.all_amplitude[:, mask_period],
                zoom=zoom,
                vmin=np.nanpercentile(self.all_amplitude, 16),
                vmax=np.nanpercentile(self.all_amplitude, 84),
                cmap="jet",
            )
            plt.colorbar()
            plt.subplot(1, 3, 3, sharex=ax, sharey=ax)
            myf.my_colormesh(
                self.all_period[mask_period],
                self.sequence,
                self.all_phase[:, mask_period],
                zoom=zoom,
                vmin=np.nanpercentile(self.all_phase, 16),
                vmax=np.nanpercentile(self.all_phase, 84),
                cmap="twilight",
            )
            plt.colorbar()
        else:
            myf.my_colormesh(
                self.all_period[mask_period],
                self.sequence,
                z,
                vmin=vmin,
                vmax=vmax,
                zoom=zoom,
            )
        plt.xlim(minp, maxp)
        plt.xlabel("Period [days]", fontsize=13)
        plt.ylabel("Number of points", fontsize=13)
        plt.subplots_adjust(left=0.06, right=0.93, top=0.93, bottom=0.05)

    def xav_bis(self):
        (
            model,
            offset,
            contrast,
            rv,
            fwhm,
            sig_offset,
            sig_contrast,
            sig_v0,
            sig_fwhm,
            iterations,
            span,
            ff,
            ee,
            len_depth,
        ) = fit3.gauss_bis(self.x, self.y, self.yerr, np.ones(10000, "d"))
        contrast = abs(contrast)
        depth = 1 - ee[:len_depth] * contrast
        bis = ff[:len_depth]

        print(bis)
        print(offset)
        print(contrast)
        print(span)
        print(rv)

        return 0

    def my_bisector(
        self,
        kind="cubic",
        oversampling=1,
        weighted=True,
        num_outpoints="none",
        between_max=False,
        vic=10,
    ):
        """Compute the bisector of a line if the table(XY) is a table(wavelength/flux) ---"""
        self.order()
        maxi, flux = myf.local_max(self.y, vicinity=vic)
        maxi_left = maxi[maxi < len(self.x) / 2]
        flux_left = flux[maxi < len(self.x) / 2]
        maxi_right = maxi[maxi > len(self.x) / 2]
        flux_right = flux[maxi > len(self.x) / 2]

        maxi1 = 0
        maxi2 = len(self.x) - 1

        if between_max:
            if len(flux_left) > 0:
                maxi1 = int(maxi_left[flux_left.argmax()])

            if len(flux_right) > 0:
                maxi2 = int(maxi_right[flux_right.argmax()])

        selfx = self.x[maxi1 : maxi2 + 1]
        selfy = self.y[maxi1 : maxi2 + 1]

        if between_max:
            self.xerr = self.xerr[maxi1 : maxi2 + 1]
            self.yerr = self.yerr[maxi1 : maxi2 + 1]
            self.x = self.x[maxi1 : maxi2 + 1]
            self.y = self.y[maxi1 : maxi2 + 1]
        normalisation = abs(selfx[selfy.argmin() - 1] - selfx[selfy.argmin() + 1])
        Interpol = interp1d(
            selfx, selfy, kind=kind, bounds_error=False, fill_value="extrapolate"
        )
        new_x = np.linspace(
            selfx.min(), selfx.max(), oversampling * (len(selfx) - 1) + 1
        )
        new_y = Interpol(new_x)

        min_idx = new_y.argmin()
        liste1 = (new_y[0 : min_idx + 1])[::-1]
        liste2 = new_y[min_idx:]
        left = (new_x[0 : min_idx + 1])[::-1]
        right = new_x[min_idx:]
        save = myf.match_nearest(liste1, liste2)
        bisector_x = []
        bisector_y = []
        bisector_xerr = []
        for num in np.arange(len(save[1:-1, 0])) + 1:
            j = save[num, 0].astype("int")
            k = save[num, 1].astype("int")
            bisector_y.append(np.mean([liste1[j], liste2[k]]))
            bisector_x.append(np.mean([left[j], right[k]]))
            if weighted:
                diff_left = (liste1[j - 1] - liste1[j + 1]) / (
                    left[j - 1] - left[j + 1]
                )
                diff_right = (liste2[k - 1] - liste2[k + 1]) / (
                    right[k - 1] - right[k + 1]
                )
                diff = np.max([abs(diff_left), abs(diff_right)])
                bisector_xerr.append(1 / diff)

        if weighted:
            bisector_xerr = np.array(bisector_xerr)
            bisector_xerr = bisector_xerr * normalisation / bisector_xerr[-1] / 2
        else:
            bisector_xerr = np.zeros(len(bisector_y))
        bisector = np.vstack([bisector_x, bisector_y, bisector_xerr]).T
        bisector = np.insert(
            bisector, 0, [new_x[min_idx], new_y[min_idx], np.max(bisector_xerr)], axis=0
        )
        Interpol = interp1d(
            bisector[:, 1],
            bisector[:, 0],
            kind=kind,
            bounds_error=False,
            fill_value="extrapolate",
        )
        if type(num_outpoints) == str:
            num_outpoints = len(bisector[:, 1])
        new_x = np.linspace(bisector[:, 1].min(), bisector[:, 1].max(), num_outpoints)
        new_y = Interpol(new_x)
        Interpol = interp1d(
            bisector[:, 1],
            bisector[:, 2],
            kind=kind,
            bounds_error=False,
            fill_value="extrapolate",
        )
        new_yerr = Interpol(new_x)
        bisector = np.vstack([new_y, new_x, new_yerr]).T
        self.bisector = bisector.copy()
        self.bis = tableXY(bisector[:, 1], bisector[:, 0], bisector[:, 2])

    def night_stack(self, db=0, bin_length=1, replace=False):

        jdb = self.x
        vrad = self.y
        vrad_std = self.yerr.copy()

        if not np.sum(vrad_std):  # to avoid null vector
            vrad_std += 1

        vrad_std[vrad_std == 0] = np.nanmax(vrad_std[vrad_std != 0] * 10)

        weights = 1 / (vrad_std) ** 2

        if bin_length:
            groups = ((jdb - db) // bin_length).astype("int")
            groups -= groups[0]
            group = np.unique(groups)
        else:
            group = np.arange(len(jdb))
            groups = np.arange(len(jdb))

        mean_jdb = []
        mean_vrad = []
        mean_svrad = []

        for j in group:
            g = np.where(groups == j)[0]
            mean_jdb.append(np.sum(jdb[g] * weights[g]) / np.sum(weights[g]))
            mean_svrad.append(1 / np.sqrt(np.sum(weights[g])))
            mean_vrad.append(np.sum(vrad[g] * weights[g]) / np.sum(weights[g]))

        mean_jdb = np.array(mean_jdb)
        mean_vrad = np.array(mean_vrad)
        mean_svrad = np.array(mean_svrad)

        if replace:
            self.x, self.y, self.xerr, self.yerr = (
                mean_jdb,
                mean_vrad,
                0 * mean_svrad,
                mean_svrad,
            )
        else:
            self.stacked = tableXY(mean_jdb, mean_vrad, mean_svrad)

    def match_x(self, table_xy, replace=False):
        match = myf.match_nearest(self.x, table_xy.x)[:, 0:2].astype("int")
        if replace:
            self.x, self.y, self.xerr, self.yerr = (
                self.x[match[:, 0]],
                self.y[match[:, 0]],
                self.xerr[match[:, 0]],
                self.yerr[match[:, 0]],
            )
            table_xy.x, table_xy.y, table_xy.xerr, table_xy.yerr = (
                table_xy.x[match[:, 1]],
                table_xy.y[match[:, 1]],
                table_xy.xerr[match[:, 1]],
                table_xy.yerr[match[:, 1]],
            )
        else:
            v1 = tableXY(
                self.x[match[:, 0]],
                self.y[match[:, 0]],
                self.xerr[match[:, 0]],
                self.yerr[match[:, 0]],
            )
            v2 = tableXY(
                table_xy.x[match[:, 1]],
                table_xy.y[match[:, 1]],
                table_xy.xerr[match[:, 1]],
                table_xy.yerr[match[:, 1]],
            )
            return v1, v2

    def merge_diff1d(self, table_xy):
        match = myf.match_nearest(self.x, table_xy.x)[:, 0:2].astype("int")

        mask = ~np.in1d(table_xy.x, self.x)
        new = table_xy.masked(mask, replace=False)

        self.merge(new)

    def mirror(self):
        mirror_left = self.y[0 : int((len(self.y) - 1) / 2) + 1]
        mirror_left = np.hstack([mirror_left[0:-1], mirror_left[::-1]])
        self.mirror_left = tableXY(self.x, mirror_left)
        self.mirror_left.null()

        self.y = self.y[::-1]

        mirror_right = self.y[0 : int((len(self.y) - 1) / 2) + 1]
        mirror_right = np.hstack([mirror_right[0:-1], mirror_right[::-1]])
        self.mirror_right = tableXY(self.x, mirror_right[::-1])
        self.mirror_right.null()

        self.y = self.y[::-1]

    def fit_gaussian(self, guess=None, Plot=True, color="r", free_offset=True):
        """guess = [amp,cen,width,offset]"""
        if guess is None:
            guess = [-0.5, 0, 3, 1]

        gmodel = Model(myf.gaussian)
        fit_params = Parameters()
        fit_params.add("amp", value=guess[0], min=-1, max=0)
        fit_params.add("cen", value=guess[1])
        fit_params.add("wid", value=guess[2], min=1)
        fit_params.add("offset", value=guess[3], min=0, max=2)
        if not free_offset:
            fit_params["offset"].vary = False
        result1 = gmodel.fit(self.y, fit_params, 1 / self.yerr**2, x=self.x)
        self.lmfit = result1
        self.params = result1.params
        if Plot:
            newx = np.linspace(np.min(self.x), np.max(self.x), 10 * len(self.x))
            plt.plot(newx, gmodel.eval(result1.params, x=newx), color=color)

    def fit_kernel0(
        self, guess=None, Plot=True, color="r", free_offset=True, free_center=True
    ):
        """guess = [a,b,amp,cen,width,offset]"""
        if guess is None:
            guess = [2, 2, 1, 0, 1, 0]

        gmodel = Model(myf.kernel)
        fit_params = Parameters()
        fit_params.add("a", value=guess[0], min=1, max=4)
        fit_params.add("b", value=guess[1], min=1, max=4)
        fit_params.add("amp", value=guess[2], min=0, max=5)
        fit_params.add("cen", value=guess[3])
        fit_params.add("wid", value=guess[4], min=0.01)
        fit_params.add("offset", value=guess[5], min=0, max=2)
        if not free_offset:
            fit_params["offset"].vary = False
        if not free_center:
            fit_params["cen"].vary = False

        result1 = gmodel.fit(self.y, fit_params, 1 / self.yerr**2, x=self.x)
        self.lmfit = result1
        self.params = result1.params
        if Plot:
            newx = np.linspace(np.min(self.x), np.max(self.x), 10 * len(self.x))
            plt.plot(newx, gmodel.eval(result1.params, x=newx), color=color)

    def fit_gauss_vincent(self, guess=None, Plot=True, color="r", free_offset=True):

        if guess is None:
            guess = [-0.5, 0.2, 0, 0, 3, 4, 1]

        gmodel = Model(myf.gauss_vincent)
        fit_params = Parameters()
        fit_params.add("amp", value=guess[0], min=-1, max=0)
        fit_params.add("amp2", value=guess[1], min=0, max=1)
        fit_params.add("cen", value=guess[2])
        fit_params.add("cen2", value=guess[3])
        fit_params.add("wid", value=guess[4], min=1)
        fit_params.add("wid2", value=guess[5], min=0)
        fit_params.add("offset", value=guess[6], min=0, max=2)
        if not free_offset:
            fit_params["offset"].vary = False
        result1 = gmodel.fit(self.y, fit_params, 1 / self.yerr**2, x=self.x)
        self.lmfit = result1
        self.params = result1.params
        if Plot:
            newx = np.linspace(np.min(self.x), np.max(self.x), 10 * len(self.x))
            plt.plot(newx, gmodel.eval(result1.params, x=newx), color=color)

    def fit_voigt(self, guess=None, Plot=True, color="r"):

        if guess is None:
            guess = [-0.5, 0, 0.02, 0.02]

        gmodel = Model(myf.voigt)
        fit_params = Parameters()
        fit_params.add("amp", value=guess[0], min=-1, max=0)
        fit_params.add("cen", value=guess[1])
        fit_params.add("wid", value=guess[2], min=0.01)
        fit_params.add("wid2", value=guess[3], min=0)
        result1 = gmodel.fit(self.y, fit_params, 1 / self.yerr**2, x=self.x)
        self.lmfit = result1
        self.params = result1.params
        if Plot:
            newx = np.linspace(np.min(self.x), np.max(self.x), 100)
            plt.plot(newx, gmodel.eval(result1.params, x=newx), color=color)

    def fit_multigaussian(self, guess=[], model="gauss", vrot=1.5, lim="pouet"):
        if model == "gauss":

            def model(x, m, a, s1, s2):
                return a * np.exp(-((x - m) ** 2) / (2 * s1**2))

        if model == "voigt":

            def model(x, m, a, s1, s2):
                val = Voigt1D(x_0=m, amplitude_L=1, fwhm_L=s1, fwhm_G=s2)(x)
                return a * (val) / np.max(val)

        def wrapper(selfx, *args):
            N = int((len(args) - 2) / 3)
            center = list(args[0:N])
            amplitude = list(args[N : 2 * N])
            sig = list(args[2 * N : 3 * N])
            sig2 = args[-2]
            offset = args[-1]
            return fit_func(selfx, center, amplitude, sig, sig2, offset)

        def fit_func(selfx, center, amplitude, sig, sig2, offset):
            fit = np.zeros(len(selfx))
            for m, a, s in zip(center, amplitude, sig):
                fit += model(selfx, m, a, s, sig2)
            fit[fit < -1] = -1
            return (fit + 1) * offset

        def gen_data(x, center, amplitude, sig, sig2, offset, noise=0.1):
            y = np.zeros(len(x))
            for m, a, s in zip(center, amplitude, sig):
                y += model(x, m, a, s, sig2)
            if noise:
                y += np.random.normal(0, noise, size=len(x))
            return (y + 1) * offset

        selfx = self.x
        selfy = self.y

        lengh = int(len(guess) / 3)
        guess = guess + [2 * (myf.doppler_c(guess[0], vrot * 1000)[0] - guess[0])] + [1]
        vrot_min = 2 * (myf.doppler_c(guess[0], 0.7 * vrot * 1000)[0] - guess[0])
        vrot_max = 2 * (myf.doppler_c(guess[0], 1.3 * vrot * 1000)[0] - guess[0])
        dop_max = myf.doppler_c(guess[0], 300)[0] - guess[0]
        p0 = guess.copy()

        bounds = (
            list(
                np.hstack(
                    [
                        np.array(guess[0:lengh]) - dop_max,
                        -1 * np.ones(lengh),
                        0 * np.ones(lengh),
                        vrot_min,
                        0.8,
                    ]
                )
            ),
            list(
                np.hstack(
                    [
                        np.array(guess[0:lengh]) + dop_max,
                        0 * np.ones(lengh),
                        1 * np.ones(lengh),
                        vrot_max,
                        1,
                    ]
                )
            ),
        )  # 100 m/s doppler max

        if lim == "none":
            popt, pcov = opt.curve_fit(
                lambda selfx, *p0: wrapper(selfx, *p0), selfx, selfy, p0=p0
            )  # call with lambda function
        else:
            popt, pcov = opt.curve_fit(
                lambda selfx, *p0: wrapper(selfx, *p0),
                selfx,
                selfy,
                p0=p0,
                method="trf",
                bounds=bounds,
            )  # call with lambda function
        yfit = gen_data(
            selfx,
            popt[0:lengh],
            popt[lengh : 2 * lengh],
            popt[2 * lengh : 3 * lengh],
            popt[-2],
            popt[-1],
            noise=0,
        )
        plt.plot(selfx, yfit)
        for j in range(lengh):
            if lengh != 1:
                plt.plot(
                    selfx,
                    gen_data(
                        selfx,
                        [popt[0:lengh][j]],
                        [popt[lengh : 2 * lengh][j]],
                        [popt[2 * lengh : 3 * lengh][j]],
                        [popt[-2]],
                        [popt[-1]],
                        noise=0,
                    ),
                )
        self.gauss_par = popt
        self.gauss_cov = pcov
        print("chi2 : %.3f" % (sum((selfy - yfit) ** 2)))

    def underpolate(self, factor, replace=True):
        x = self.x[::factor]
        y = self.y[::factor]
        xerr = self.xerr[::factor]
        yerr = self.yerr[::factor]

        if replace:
            self.x, self.y, self.xerr, self.yerr = x, y, xerr, yerr
        else:
            self.underpolated = tableXY(x, y, xerr, yerr)

    def interpolate(
        self,
        new_grid="auto",
        method="cubic",
        replace=True,
        interpolate_x=True,
        fill_value="extrapolate",
        scale="lin",
    ):

        if scale != "lin":
            self.inv()

        if type(new_grid) == str:
            new_grid = np.linspace(self.x.min(), self.x.max(), 10 * len(self.x))
        if type(new_grid) == int:
            new_grid = np.linspace(self.x.min(), self.x.max(), new_grid * len(self.x))

        if np.sum(new_grid != self.x) != 0:
            if replace:
                self.x_backup = self.x.copy()
                self.y_backup = self.y.copy()
                self.xerr_backup = self.xerr.copy()
                self.yerr_backup = self.yerr.copy()
                self.y = interp1d(
                    self.x,
                    self.y,
                    kind=method,
                    bounds_error=False,
                    fill_value=fill_value,
                )(new_grid)
                if np.sum(abs(self.yerr)):
                    self.yerr = interp1d(
                        self.x,
                        self.yerr,
                        kind=method,
                        bounds_error=False,
                        fill_value=fill_value,
                    )(new_grid)
                else:
                    self.yerr = np.zeros(len(new_grid))
                if (interpolate_x) & (bool(np.sum(abs(self.xerr)))):
                    self.xerr = interp1d(
                        self.x,
                        self.xerr,
                        kind=method,
                        bounds_error=False,
                        fill_value=fill_value,
                    )(new_grid)
                else:
                    self.xerr = np.zeros(len(new_grid))
                self.x = new_grid

                if scale != "lin":
                    self.inv()

            else:
                self.y_interp = interp1d(
                    self.x,
                    self.y,
                    kind=method,
                    bounds_error=False,
                    fill_value=fill_value,
                )(new_grid)
                if np.sum(abs(self.yerr)):
                    self.yerr_interp = interp1d(
                        self.x,
                        self.yerr,
                        kind=method,
                        bounds_error=False,
                        fill_value=fill_value,
                    )(new_grid)
                else:
                    self.yerr_interp = np.zeros(len(new_grid))
                if (interpolate_x) & (bool(np.sum(abs(self.xerr)))):
                    self.xerr_interp = interp1d(
                        self.x,
                        self.xerr,
                        kind=method,
                        bounds_error=False,
                        fill_value=fill_value,
                    )(new_grid)
                else:
                    self.xerr_interp = np.zeros(len(new_grid))
                self.x_interp = new_grid
                self.interpolated = tableXY(
                    self.x_interp, self.y_interp, self.xerr_interp, self.yerr_interp
                )

                if scale != "lin":
                    self.interpolated.inv()
                    self.inv()

    def interpolate2d(
        self,
        z,
        new_grid="auto",
        method="linear",
        replace=True,
        fill_value="extrapolate",
    ):
        if type(new_grid) == str:
            X, Y = np.meshgrid(
                np.linspace(self.x.min(), self.x.max(), 50),
                np.linspace(self.y.min(), self.y.max(), 50),
            )

        if type(new_grid) == int:
            X, Y = np.meshgrid(
                np.linspace(self.x.min(), self.x.max(), new_grid),
                np.linspace(self.y.min(), self.y.max(), new_grid),
            )

        if replace:
            self.x_backup = self.x.copy()
            self.y_backup = self.y.copy()
            self.z_backup = z.copy()

            self.z = np.ravel(griddata((self.x, self.y), z, (X, Y), method=method))
            self.x = np.ravel(X)
            self.y = np.ravel(Y)

        else:
            self.z_interp = griddata((self.x, self.y), z, (X, Y), method=method)
            self.x_interp = X
            self.y_interp = Y

            self.interpolated = tableXY(
                self.x_interp, self.y_interp, self.z_interp, self.z_interp
            )

    def export_to_dace(self, file_name, convert_to_kms=1, convert_to_jdb=0):
        if len(file_name.split(".")) > 1:
            if file_name.split(".")[-1] != "rdb":
                file_name = file_name.split(".")[0] + ".rdb"
        else:
            file_name += ".rdb"

        matrice = np.array(
            [
                self.x + convert_to_jdb,
                convert_to_kms * self.y,
                convert_to_kms * self.yerr,
            ]
        ).T
        try:
            np.savetxt(
                file_name,
                matrice,
                delimiter="\t",
                header="jdb\tvrad\tsvrad\n---\t----\t-----",
            )

            f = open(file_name, "r")
            lines = f.readlines()
            lines[0] = lines[0][2:]
            lines[1] = lines[1][2:]
            lines[-1] = lines[-1][:-1]
            f.close()
            f = open(file_name, "w")
            f.writelines(lines)
            f.close()
        except:
            return matrice

    def slippery_filter(self, box=20, sigma=False):
        """Slippery median to filter a signal (flattening). slippery(x, y, box=size, sigma=val)."""
        grille = range(box, len(self.y) - box)
        slippery = np.array([np.median(self.y[j - box : j + box]) for j in grille])
        self.filterx = self.x[grille]
        self.filtery = slippery
        if sigma != False:
            self.env = np.array(
                [np.std(self.y[j - box : j + box]) * sigma for j in grille]
            )

    def gp_derivative(self, kernel=["M52", 1, 3.0], Plot=True):
        if cwd.split("/")[2] == "cretignier":
            self.gp_interp(kernel=kernel, Plot=Plot)
            new_vec = np.gradient(self.gp_interpolated.y) / np.gradient(
                self.gp_interpolated.x
            )
            new_vec = tableXY(
                self.gp_interpolated.x,
                new_vec,
                np.sqrt(
                    np.roll(self.gp_interpolated.yerr, 1) ** 2
                    + np.roll(self.gp_interpolated.yerr, -1) ** 2
                ),
            )

            new_vec.interpolate(new_grid=self.x, replace=False)

            new_vec.yerr /= myf.mad(new_vec.interpolated.y)
            new_vec.yerr *= myf.mad(self.y)
            new_vec.y -= np.median(new_vec.y)
            new_vec.y /= myf.mad(new_vec.interpolated.y)
            new_vec.y *= myf.mad(self.y)
            new_vec.y += np.median(self.y)

            plt.plot(new_vec.x, new_vec.y, color="b")
            plt.fill_between(
                new_vec.x,
                new_vec.y + new_vec.yerr,
                new_vec.y - new_vec.yerr,
                color="b",
                alpha=0.3,
            )
            self.gp_gradient = new_vec.interpolated
        else:
            self.gp_gradient = self.copy()

    def gp_interp(self, kernel=["M52", 1, 3.0], Plot=True):
        if cwd.split("/")[2] == "cretignier":
            import george

            dt = np.nanmedian(np.diff(self.x))
            over = int(10 * (np.nanmax(self.x) - np.nanmin(self.x)) / dt)
            new_vec = self.baseline_oversampled(oversampling=over)

            if kernel[0] == "M52":
                k = kernel[1] * george.kernels.Matern52Kernel(kernel[2])

            gp = george.GP(k)
            p0 = gp.get_parameter_vector()

            def neg_ln_like(p):
                gp.set_parameter_vector(p)
                return -gp.log_likelihood(zobs)

            def grad_neg_ln_like(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(zobs)

            gp.compute(self.x, yerr=self.yerr)
            zobs = self.y
            result = minimize(neg_ln_like, p0, jac=grad_neg_ln_like)
            gp.set_parameter_vector(result.x)
            new_vec.y, var = gp.predict(
                zobs, new_vec.x, return_var=True, return_cov=False
            )
            s = np.sqrt(var)
            new_vec.yerr = s
            self.gp_interpolated = new_vec
            if Plot:
                # self.plot()
                new_vec.plot(ls="-", color="r")
                plt.fill_between(
                    new_vec.x, new_vec.y + s, new_vec.y - s, color="r", alpha=0.3, lw=0
                )

    def gp_interp2d(self, zobs, zerr=None, coupled=True, grid=1000, Plot=False):
        if cwd.split("/")[2] == "cretignier":
            import george

            if zerr is None:
                zerr = np.ones(len(zobs)) * myf.mad(zobs) / 10

            X2D, Y2D = np.meshgrid(
                np.linspace(np.nanmin(self.x), np.nanmax(self.x), grid),
                np.linspace(np.nanmin(self.y), np.nanmax(self.y), grid),
            )

            if coupled:
                k = (
                    1.0
                    * george.kernels.ExpSquaredKernel(1.0, ndim=2, axes=0)
                    * george.kernels.ExpSquaredKernel(1.0, ndim=2, axes=1)
                )
            else:
                k1 = 1.0 * george.kernels.ExpSquaredKernel(1.0, ndim=2, axes=0)
                k2 = 1.0 * george.kernels.ExpSquaredKernel(1.0, ndim=2, axes=1)
                k = k1 + k2

            gp = george.GP(k)
            Xobs = np.array([self.x, self.y]).T
            gp.compute(Xobs, yerr=zerr)

            def neg_ln_like(p):
                gp.set_parameter_vector(p)
                return -gp.log_likelihood(zobs)

            def grad_neg_ln_like(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(zobs)

            print(gp.get_parameter_vector())
            res = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
            print(res.x)
            gp.set_parameter_vector(res.x)
            Xpred = np.array([X2D.flatten(), Y2D.flatten()]).T
            zpred = gp.predict(zobs, Xpred, return_var=False, return_cov=False)
            Z2D = zpred.reshape(X2D.shape)
            if Plot:
                plt.scatter(self.x, self.y, c=zobs, cmap="brg", ec="k")
                plt.contour(X2D, Y2D, Z2D, cmap="brg")

    def gp_prot(self, Prot=None, evol=None, gamma=1.0, Plot=True):
        if cwd.split("/")[2] == "cretignier":
            import george

            dt = np.nanmedian(np.diff(self.x))
            over = int(10 * (np.nanmax(self.x) - np.nanmin(self.x)) / dt)
            new_vec = self.baseline_oversampled(oversampling=over)

            if Prot is None:
                self.periodogram(Plot=False)
                print("[INFO] Highest period = %.1f days" % (self.perio_max))
                Prot = self.perio_max

            if evol is None:
                evol = 5 * Prot

            k = (
                np.var(self.y)
                * george.kernels.ExpSine2Kernel(log_period=np.log(Prot), gamma=gamma)
                * george.kernels.ExpSquaredKernel(2 * evol**2)
            )

            gp = george.GP(k, mean=np.nanmedian(self.y), fit_mean=True)

            gp.compute(self.x, yerr=self.yerr)
            p0 = gp.get_parameter_vector()

            def nll_qp(p):
                if abs(np.exp(p[-2]) - Prot) * 100 / Prot > 10:  # 10%prior on Prot
                    return 1e25
                else:
                    gp.set_parameter_vector(p)
                    try:
                        return -gp.log_likelihood(self.y)
                    except:
                        return 1e25

            def grad_nll_qp(p):
                if abs(np.exp(p[-2]) - Prot) * 100 / Prot > 10:
                    return np.array([1e25] * len(p))
                else:
                    gp.set_parameter_vector(p)
                    try:
                        return -gp.grad_log_likelihood(self.y)
                    except:
                        return np.array([1e25] * len(p))

            res = minimize(nll_qp, p0, jac=grad_nll_qp)
            print("P={}, l_e={}".format(np.exp(res.x[-2]), np.exp(res.x[-1]) / 2))
            self.gp_param_prot = np.exp(res.x[-2])
            self.gp_param_evol = np.exp(res.x[-1] / 2)

            gp.set_parameter_vector(res.x)
            gp.compute(self.x, yerr=self.yerr)
            mu, var = gp.predict(self.y, self.x, return_var=True)
            s = np.sqrt(var)

            self.plot()

            mu, var = gp.predict(self.y, new_vec.x, return_var=True)
            s = np.sqrt(var)
            new_vec.yerr = s
            new_vec.y = mu

            plt.plot(new_vec.x, mu, "C0-")
            plt.fill_between(new_vec.x, mu + s, mu - s, color="C0", alpha=0.3, lw=0)

    def light_curve_plot(self, table_keplerian):

        all_periods = np.array(table_keplerian["p"])
        all_transit_time = np.array(table_keplerian["Tc"])
        all_transit_duration = np.array(table_keplerian["dt"])

        lc = self.copy()
        lc.supress_nan()

        plt.figure(figsize=(18, 6))
        for count in range(1, 1 + len(all_periods)):
            dura = all_transit_duration[count - 1] / 2
            p_fold = all_periods[count - 1]
            dura_norm = (dura / 24) / p_fold  # demi transit

            tc = all_transit_time[count - 1]

            plt.subplot(1, len(all_periods), count)
            new_t = ((lc.x - tc % p_fold + p_fold / 2) % p_fold) / p_fold
            new_t = (new_t - 0.5) / dura_norm

            out_t = abs(new_t) > 2
            conti = np.mean(lc.y[out_t])
            plt.scatter(new_t, lc.y - conti, c=lc.x)
            plt.xlim(-2.5, 2.5)
            plt.axhline(y=0, color="gray", alpha=0.5, zorder=101)

    def light_curve_periodogram(
        self,
        table_keplerian,
        std_p=3,
        nb_period=1000,
        std_phase=1 / 18,
        pts_transit=15,
        exclude_out=True,
    ):

        all_periods = np.array(table_keplerian["p"])
        all_transit_time = np.array(table_keplerian["Tc"])
        all_transit_proba = np.array(table_keplerian["proba_t"])
        all_transit_duration = np.array(table_keplerian["dt"])

        table_keplerian2 = table_keplerian.copy()

        lc = self.copy()
        lc.supress_nan()

        model_i = []
        best_p_i = []
        best_tc_i = []
        best_d_i = []
        plt.figure(figsize=(18, 10))
        for count in range(1, 1 + len(all_periods)):
            print(" [INFO] Light_curve_periodogram planet : %.0f" % (count))
            p, proba, tc, dura = (
                all_periods[count - 1],
                all_transit_proba[count - 1],
                all_transit_time[count - 1],
                all_transit_duration[count - 1],
            )
            freq = 1 / p
            binning_transit = (dura / 24) / p / pts_transit
            all_p_scanned = 1 / np.linspace(
                freq + std_p / 100 * freq, freq - std_p / 100 * freq, nb_period
            )
            plt.subplot(2, len(all_periods), count)
            for j in range(2):
                m1 = []
                m2 = []
                tc_corr = []
                for p_fold in all_p_scanned:
                    lc.modulo(
                        p_fold, phase_mod=tc % p_fold + p_fold / 2, modulo_norm=True
                    )
                    lc.mod.x -= 0.5
                    lc.mod.clip(min=[-std_phase, None], max=[std_phase, None])
                    if len(lc.mod.x) > 10:
                        lc.mod.binning(binning_transit)
                        tc_corr.append(lc.mod.binned.x[np.argmin(lc.mod.binned.y)])
                        lc.mod.binned.interpolate(new_grid=lc.mod.x, method="linear")
                        m1.append(np.std(lc.mod.binned.y))
                        m2.append(np.std(lc.mod.y) / np.std(lc.mod.y - lc.mod.binned.y))
                    else:
                        m1.append(0)
                        m2.append(1)
                        tc_corr.append(np.nan)

                plt.plot(all_p_scanned, m2)
                p2 = all_p_scanned[np.argmax(m2)]
                t2 = tc_corr[np.argmax(m2)]
                freq = 1 / p2
                std_p2 = 0.5
                all_p_scanned = 1 / np.linspace(
                    freq + std_p2 / 100 * freq, freq - std_p2 / 100 * freq, nb_period
                )

            plt.xlabel("Period [days]")
            plt.title("Best period = %.6f days" % (p2))
            p_fold = p2
            best_p_i.append(p2)
            best_tc_i.append(tc + t2 * p2)
            lc.modulo(p_fold, phase_mod=tc % p_fold + p_fold / 2, modulo_norm=True)
            lc.mod.binning(binning_transit)
            lc.mod.binned.interpolate(new_grid=lc.mod.x, method="linear")

            model = np.zeros(len(lc.x))
            model[lc.mod.old_index_modulo] = lc.mod.binned.y

            if exclude_out:
                dura_norm = (dura / 24) / p_fold  # demi transit
                new_t = ((lc.x - best_tc_i[-1] % p_fold + p_fold / 2) % p_fold) / p_fold
                new_t = (new_t - 0.5) / dura_norm
                out_t = abs(new_t) > 2
                model[out_t] = np.median(model)

            model_i.append(model)
            lc.y -= model

        table_keplerian2["d_transit"] = 0
        for count in range(1, 1 + len(all_periods)):
            dura = all_transit_duration[count - 1] / 2
            p_fold = best_p_i[count - 1]
            proba = all_transit_proba[count - 1]
            dura_norm = (dura / 24) / p_fold  # demi transit

            tc = best_tc_i[count - 1]

            plt.subplot(2, len(all_periods), count + len(all_periods))
            lc.y += model_i[count - 1]
            new_t = ((lc.x - tc % p_fold + p_fold / 2) % p_fold) / p_fold
            new_t = (new_t - 0.5) / dura_norm

            in_t = abs(new_t) < 0.5
            out_t = abs(new_t) > 2
            conti = np.mean(model_i[count - 1][out_t])
            depth = np.mean(model_i[count - 1][in_t] - conti)
            depth_std = np.std(model_i[count - 1][in_t])
            best_d_i.append(depth)

            plt.title(
                "Depth = %.0f $\pm$ %.0f \n Transit proba = %.1f %%"
                % (abs(depth), depth_std, proba)
            )
            plt.scatter(new_t, lc.y - conti, c=lc.x)
            plt.scatter(new_t, model_i[count - 1] - conti, zorder=100, color="k")
            lc.y -= model_i[count - 1]
            plt.xlim(-2.5, 2.5)
            plt.axhline(y=0, color="gray", alpha=0.5, zorder=101)

            if -depth / depth_std > 5:
                table_keplerian2.loc["planet %.0f" % (count), "p"] = p_fold
                table_keplerian2.loc["planet %.0f" % (count), "Tc"] = tc
                table_keplerian2.loc["planet %.0f" % (count), "d_transit"] = abs(depth)

        plt.subplots_adjust(hspace=0.5, top=0.96, left=0.07, right=0.95, bottom=0.11)

        self.planet_fitted2 = table_keplerian2
        self.transit_model = model_i
