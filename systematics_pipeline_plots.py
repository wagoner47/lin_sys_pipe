import pathlib
import re
import copy
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
from matplotlib.ticker import MaxNLocator, ScalarFormatter, LogLocator
from matplotlib.gridspec import GridSpec
import numpy as np
import healpy as hp
from scipy import stats, linalg, special
import twopoint
from astropy.table import Table
from chainconsumer import ChainConsumer
from chainconsumer.plotter import Plotter
from linear_systematics_pipeline_lsssys import read_chain, map_importance


class LazyFStr(object):
    def __init__(self, func):
        self.func = func
    def __str__(self):
        return self.func()


class MyPlotter(Plotter):
    def _plot_walk(self, ax, parameter, data, truth=None, extents=None,     
                   convolve=None, color=None, log_scale=False, x_min=0, 
                   nburnin=0):
        if extents is not None:
            ax.set_ylim(extents)
        assert convolve is None or isinstance(convolve, int), \
            "Convolve must be an integer pixel window width"
        x = np.arange(x_min, data.size + x_min)
        ax.set_xlim([x[0], x[-1]])
        ax.set_ylabel(parameter)
        if ax.is_last_row():
            ax.set_xlabel(r"Step")
        if color is None:
            color = "k"
        ax.scatter(
            x, data, c=color, s=2, marker=".", edgecolors="none", alpha=0.5)
        max_ticks = self.parent.config["max_ticks"]
        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(numticks=max_ticks))
        else:
            if ax.is_first_row():
                ax.yaxis.set_major_locator(
                    MaxNLocator(max_ticks, prune="lower"))
            elif ax.is_last_row():
                ax.yaxis.set_major_locator(
                    MaxNLocator(max_ticks, prune="upper"))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(max_ticks, prune="both"))
        if convolve is not None:
            color2 = self.parent.color_finder.scale_colour(color, 0.5)
            filt = np.ones(convolve) / convolve
            filtered = np.convolve(data, filt, mode="same")
            ax.plot(x[:-1], filtered[:-1], ls=":", color=color2, alpha=1)
        if nburnin > 0:
            ax.axvline(nburnin, c="b")
        return
    
    def plot_walks(self, parameters=None, truth=None, extents=None, 
                   display=False, filename=None, chains=None, convolve=None, 
                   figsize=None, plot_weights=True, plot_posterior=True, 
                   log_weight=None, log_scales=None, num_cols=1, nburn=0, 
                   post_burn=False):
        """
        Plots the chain walk; the parameter values as a function of step index.
        This plot is more for a sanity or consistency check than for use with 
        final results. Plotting this before plotting with :func:`plot` allows 
        you to quickly see if the chains are well behaved, or if certain 
        parameters are suspect or require a greater burn in period.
        The desired outcome is to see an unchanging distribution along the 
        x-axis of the plot. If there are obvious tails or features in the 
        parameters, you probably want to investigate.
        
        :param parameters: Specify a subset of parameters to plot. If not set, 
            all parameters are plotted. If an integer is given, only the first 
            so many parameters are plotted.
        :type parameters: list[str]|int, optional
        :param truth: A list of truth values corresponding to parameters, or a 
            dictionary of truth values keyed by the parameter.
        :type truth: list[float]|dict[str], optional
        :param extents: A list of two-tuples for plot extents per parameter, or 
            a dictionary of extents keyed by the parameter.
        :type extents: list[tuple]|dict[str], optional
        :param display: If True, shows the plot using 
            :func:`matplotlib.pyplot.show`
        :type display: bool, optional
        :param filename: If set, saves the figure to the filename
        :type filename: str, optional
        :param chains: Used to specify which chain to show if more than one 
            chain is loaded in. Can be an integer, specifying the chain index, 
            or a str, specifying the chain name.
        :type chains: int|str, list[str|int], optional
        :param convolve: If set, overplots a smoothed version of the steps using 
            ``convolve`` as the width of the smoothing filter.
        :type convolve: int, optional
        :param figsize: If set, sets the created figure size. Default is 
            `(0.75 + (1.5 * num_cols), 0.75 + nrows)` inches
        :type figsize: tuple, optional
        :param plot_weights: If True (default), plots the weight if they are 
            available
        :type plot_weights: bool, optional
        :param plot_posterior: If True (default), plots the log posterior if 
            they are available
        :type plot_posterior: bool, optional
        :param log_weight: Whether to display weights in log space or not. If 
            None, the value is inferred by the mean weights of the plotted 
            chains.
        :type log_weight: bool, optional
        :param log_scales: Whether or not to use a log scale on any given axis. 
            Can be a list of True/False, a list of param names to set to true, a 
            dictionary of param names with true/false or just a bool (just 
            `True` would set everything to log scales).
        :type log_scales: bool, list[bool] or dict[bool], optional
        :param num_cols: Used to specify the number of columns in which to 
            arrange the subplots. Will be set to 1 if a negative value or 0 is 
            given. Default 1
        :type num_cols: int, optional
        :param nburn: Used to show the length of the burn-in, if any. Behavior 
            depends on ``post_burn``, where ``post_burn = True`` results in the 
            step numbering being shifted and ``post_burn = False`` results in a 
            vertical line being drawn in each subplot. Default 0
        :type nburn: int, optional
        :param post_burn: If ``nburn`` is positive, plot a vertical line in each 
            subplot at ``nburn`` if `False` (default) or shift the step numbers 
            if `True`
        :type post_burn: bool, optional
        :return: The matplotlib figure created
        :rtype: :class:`matplotlib.figure.Figure`
        """
        if nburn < 0:
            nburn = 0
        x_min = 0
        if post_burn:
            x_min = nburn
            nburn = 0
        chains, parameters, truth, extents, _, log_scales = self._sanitise(
            chains, parameters, truth, extents, log_scales=log_scales)
        chains = [c for c in chains if c.mcmc_chain]
        n = len(parameters)
        if plot_weights:
            plot_weights = plot_weights and np.any(
                [np.any(c.weights != 1.0) for c in chains])
        plot_posterior = plot_posterior and np.any(
            [c.posterior is not None for c in chains])
        extra = plot_weights + plot_posterior
        num_cols = min(max(1, int(num_cols)), n + extra)
        num_rows = int(np.ceil((n + extra) / num_cols))
        num_full_cols = (n + extra) - (num_cols * (num_rows - 1))
        if num_full_cols < num_cols:
            skipped_indices = np.ravel_multi_index(
                ([0] * (num_cols - num_full_cols), np.arange(
                    num_ful_cols, num_cols)), (num_rows, num_cols))
        else:
            skipped_indices = np.array([])
        if figsize is None:
            figsize = (0.75 + (1.5 * num_cols), 0.75 + num_rows)
        fig, axes = plt.subplots(
            num_rows, num_cols, squeeze=False, sharex="all", figsize=figsize, 
            gridspec_kw={
                "left": 0.165, "top": 0.95, "right": 0.95, "bottom": 0.1})
        for idx, ax in np.ndenumerate(axes):
            if idx in skipped_indices:
                ax.set_frame_on(False)
                ax.tick_params(
                    axis="both", which="both", bottom=False, left=False, 
                    top=False, right=False, labelbottom=False, labelleft=False)
                for ax in axes[:, idx[1]]:
                    ax.rowNum += 1
            else:
                flat_idx = np.ravel_multi_index(idx, axes.shape)
                if flat_idx >= extra:
                    p = parameters[flat_idx - extra]
                    for chain in chains:
                        if p in chain.parameters:
                            chain_row = chain.get_data(p)
                            is_log = log_scales.get(p, False)
                            self._plot_walk(
                                ax, p, chain_row, extents=extents.get(p), 
                                convolve=convolve, color=chain.config["color"], 
                                log_scale=is_log, x_min=x_min, nburnin=nburn)
                    if truth.get(p) is not None:
                        self._plot_walk_truth(ax, truth.get(p))
                else:
                    if flat_idx == 0 and plot_posterior:
                        for chain in chains:
                            if chain.posterior is not None:
                                self._plot_walk(
                                    ax, r"$\Delta \log(\mathcal{L})$", 
                                    chain.posterior - chain.posterior.max(), 
                                    convolve=convolve, 
                                    color=chain.config["color"], x_min=x_min, 
                                    nburnin=nburn)
                    else:
                        if log_weight is None:
                            log_weight = np.any([
                                chain.weights.mean() < 0.1 for chain in chains])
                        if log_weight:
                            for chain in chains:
                                self._plot_walk(
                                    ax, r"$\log_{10}(w)$", 
                                    np.log10(chain.weights), convolve=convolve, 
                                    color=chain.config["color"], x_min=x_min, 
                                    nburnin=nburn)
                        else:
                            for chain in chains:
                                self._plot_walk(
                                    ax, r"$w$", chain.weights, 
                                    convolve=convolve, 
                                    color=chain.config["color"], x_min=x_min, 
                                    nburnin=nburn)
        if filename is not None:
            fig.savefig(filename)
        if display:
            plt.show()


class MyChainConsumer(ChainConsumer):
    def __init__(self):
        super(MyChainConsumer, self).__init__()
        self.plotter = MyPlotter(self)
        
        
def _unravel_axes_index(index, outer_shape, with_resid=False, top=True):
    row, col = np.unravel_index(index, outer_shape)
    if with_resid:
        row *= 2
        if not top:
            row += 1
    return row, col


def is_first_row(ax, inner_only=False, outer_only=False):
    gs_inner = ax.get_gridspec()
    try:
        gs_outer = gs_inner.get_topmost_subplotspec().get_gridspec()
    except AttributeError:
        return ax.rowNum == 0
    if gs_inner is gs_outer:
        return ax.rowNum == 0
    nrows_inner = gs_inner.get_geometry()[0]
    nrows_outer = gs_outer.get_geometry()[0]
    if inner_only:
        return (ax.rowNum % nrows_inner) == 0
    if outer_only:
        return (ax.rowNum // nrows_inner) == 0
    return ax.rowNum == 0


def is_first_col(ax, inner_only=False, outer_only=False):
    gs_inner = ax.get_gridspec()
    try:
        gs_outer = gs_inner.get_topmost_subplotspec().get_gridspec()
    except AttributeError:
        return ax.colNum == 0
    if gs_inner is gs_outer:
        return ax.colNum == 0
    ncols_inner = gs_inner.get_geometry()[1]
    ncols_outer = gs_outer.get_geometry()[1]
    if inner_only:
        return (ax.colNum % ncols_inner) == 0
    if outer_only:
        return (ax.colNum // ncols_inner) == 0
    return ax.colNum == 0


def is_last_row(ax, inner_only=False, outer_only=False):
    gs_inner = ax.get_gridspec()
    nrows_inner = gs_inner.get_geometry()[0]
    try:
        gs_outer = gs_inner.get_topmost_subplotspec().get_gridspec()
    except AttributeError:
        return ax.rowNum == (nrows_inner - 1)
    if gs_inner is gs_outer:
        return ax.rowNum == (nrows_inner - 1)
    nrows_outer = gs_outer.get_geometry()[0]
    if inner_only:
        return (ax.rowNum % nrows_inner) == (nrows_inner - 1)
    if outer_only:
        return (ax.rowNum // nrows_inner) == (nrows_outer - 1)
    return ax.rowNum == (nrows_inner * nrows_outer) - 1


def is_last_col(ax, inner_only=False, outer_only=False):
    gs_inner = ax.get_gridspec()
    ncols_inner = gs_inner.get_geometry()[1]
    try:
        gs_outer = gs_inner.get_topmost_subplotspec().get_gridspec()
    except AttributeError:
        return ax.colNum == (ncols_inner - 1)
    if gs_inner is gs_outer:
        return ax.colNum == (ncols_inner - 1)
    ncols_outer = gs_outer.get_geometry()[1]
    if inner_only:
        return (ax.colNum % ncols_inner) == (ncols_inner - 1)
    if outer_only:
        return (ax.colNum // ncols_inner) == (ncols_outer - 1)
    return ax.colNum == (ncols_inner * ncols_outer) - 1


def _init_axes(n_ax, n_rows=2, with_resid=False, sharex="none", sharey="none", 
               gridspec_kw=None, subplot_kw=None, **kwargs):
    n_ax = max(int(n_ax), 1)
    n_rows = min(max(int(n_rows), 1), n_ax)
    n_cols = int(np.ceil(n_ax / n_rows))
    if isinstance(sharex, bool):
        sharex = "all" if sharex else "none"
    if isinstance(sharey, bool):
        sharey = "all" if sharey else "none"
    if subplot_kw is None:
        subplot_kw = {}
    if gridspec_kw is None:
        gridspec_kw = {}
    subplot_kw_ = copy.deepcopy(subplot_kw)
    gridspec_kw_ = copy.deepcopy(gridspec_kw)
    fig = plt.figure(**kwargs)
    gs = fig.add_gridspec(n_rows, n_cols, **gridspec_kw)
    if with_resid:
        axes = np.empty((n_rows * 2, n_cols), dtype=object)
    else:
        axes = np.empty((n_rows, n_cols), dtype=object)
    for i in range(n_ax):
        row, col = _unravel_axes_index(i, (n_rows, n_cols), with_resid)
        shared_with = {
            "none": None, "all": axes[0, 0], "row": axes[row, 0], 
            "col": axes[0, col]}
        subplot_kw_["sharex"] = shared_with[sharex]
        subplot_kw_["sharey"] = shared_with[sharey]
        if with_resid:
            sub_gs = gs[np.unravel_index(i, (n_rows, n_cols))].subgridspec(
                2, 1, hspace=0.0, height_ratios=[3, 1])
            axes[row, col] = fig.add_subplot(sub_gs[0], **subplot_kw_)
            axes[row, col].rowNum = row
            axes[row, col].colNum = col
            if subplot_kw_["sharex"] is None:
                subplot_kw_["sharex"] = axes[row, col]
            axes[row + 1, col] = fig.add_subplot(sub_gs[1], **subplot_kw_)
            axes[row + 1, col].rowNum = row + 1
            axes[row + 1, col].colNum = col
        else:
            axes[row, col] = fig.add_subplot(gs[row, col], **subplot_kw_)
    for i in range(n_ax, n_rows * n_cols):
        row, col = _unravel_axes_index(i, (n_rows, n_cols), with_resid)
        orow, ocol = np.unravel_index(i, (n_rows, n_cols))
        for ax in axes[:row, col]:
            if ax.rowNum < n_rows:
                if with_resid:
                    ax.rowNum += 2
                else:
                    ax.rowNum += 1
            else:
                break
        ax = fig.add_subplot(gs[orow, ocol])
        if axes[row - 1, col].rowNum < n_rows:
            if with_resid:
                ax.rowNum = 2 * n_rows
            else:
                ax.rowNum = n_rows
        else:
            ax.rowNum = axes[row - 1, col].rowNum + 1
        ax.set_frame_on(False)
        ax.tick_params(
            axis="both", which="both", bottom=False, left=False, top=False, 
            right=False, labelbottom=False, labelleft=False)
        axes[row, col] = ax
        if with_resid:
            axes[row + 1, col] = ax
    return fig, axes


def setup_axes_noresid(n_ax, x_axis_label, y_axis_labels, n_rows=2, 
                       label_all_x=False, label_all_y=False, x_scale="log", 
                       y_scale="linear", x_scale_kwargs=None, 
                       y_scale_kwargs=None, sharex="none", sharey="none", 
                       gridspec_kw=None, subplot_kw=None, **kwargs):
    n_ax = max(int(n_ax), 1)
    if isinstance(sharex, bool):
        sharex = "all" if sharex else "none"
    if isinstance(sharey, bool):
        sharey = "all" if sharey else "none"
    fig, axes = _init_axes(
        n_ax, n_rows, False, sharex, sharey, gridspec_kw, subplot_kw, **kwargs)
    if not isinstance(y_axis_labels, str):
        label_all_y = True
    else:
        if label_all_y:
            y_axis_labels = [y_axis_labels] * n_ax
    label_all_y_ticks = label_all_y or (sharey == "none")
    if gridspec_kw is not None and "wspace" in gridspec_kw:
        max_yticklabel_width = max([
            fig.transFigure.inverted().transform_bbox(
                t.get_window_extent()).width for t in axes[
                    0, 0].get_yticklabels()])
        label_all_y_ticks = label_all_y_ticks and not (
            gridspec_kw["wspace"] < max_yticklabel_width)
    label_all_x_ticks = label_all_x or (sharex == "none")
    if gridspec_kw is not None and "hspace" in gridspec_kw:
        max_xticklabel_height = max([
            fig.transFigure.inverted().transform_bbox(
                t.get_window_extent()).height for t in axes[
                    0, 0].get_xticklabels()])
        label_all_x_ticks = label_all_x_ticks and not (
            gridspec_kw["hspace"] < max_xticklabel_height)
    if x_scale_kwargs is None:
        x_scale_kwargs = {}
    if x_scale == "log" and "nonposx" not in x_scale_kwargs:
        x_scale_kwargs["nonposx"] = "mask"
    if y_scale_kwargs is None:
        y_scale_kwargs = {}
    if y_scale == "log" and "nonposy" not in y_scale_kwargs:
        y_scale_kwargs["nonposy"] = "mask"
    for i in range(n_ax):
        ax = axes[_unravel_axes_index(i, axes.shape)]
        ax.set_xscale(x_scale, **x_scale_kwargs)
        ax.set_yscale(y_scale, **y_scale_kwargs)
        ax.tick_params(
            axis="both", which="both", bottom=True, left=True, top=True, 
            right=True, labeltop=False, labelright=False, direction="inout")
        if label_all_y:
            ax.tick_params(axis="y", which="both", labelleft=True)
            ax.set_ylabel(y_axis_labels[i])
        else:
            if is_first_col(ax):
                ax.set_ylabel(y_axis_labels)
            if is_first_col(ax) or label_all_y_ticks:
                ax.tick_params(axis="y", which="both", labelleft=True)
            else:
                ax.tick_params(axis="y", which="both", labelleft=False)
        if label_all_x:
            ax.tick_params(axis="x", which="both", labelbottom=True)
            ax.set_xlabel(x_axis_label)
        else:
            if is_last_row(ax):
                ax.set_xlabel(x_axis_label)
            if is_last_row(ax) or label_all_x_ticks:
                ax.tick_params(axis="x", which="both", labelbottom=True)
            else:
                ax.tick_params(axis="x", which="both", labelbottom=False)
    return fig, axes


def setup_axes_resid(n_ax, x_axis_label, y_taxis_labels, y_baxis_labels, 
                     n_rows=2, label_all_x=False, label_all_y=False, 
                     x_scale="log", y_scale="linear", x_scale_kwargs=None, 
                     y_scale_kwargs=None, sharex="none", sharey="none", 
                     gridspec_kw=None, subplot_kw=None, **kwargs):
    n_ax = max(int(n_ax), 1)
    if isinstance(sharex, bool):
        sharex = "all" if sharex else "none"
    if isinstance(sharey, bool):
        sharey = "all" if sharey else "none"
    fig, axes = _init_axes(
        n_ax, n_rows, True, sharex, sharey, gridspec_kw, subplot_kw, **kwargs)
    if not isinstance(y_taxis_labels, str):
        label_all_y = True
    else:
        if label_all_y:
            y_taxis_labels = [y_taxis_labels] * n_ax
            y_baxis_labels = [y_baxis_labels] * n_ax
    label_all_y_ticks = label_all_y or (sharey == "none")
    if gridspec_kw is not None and "wspace" in gridspec_kw:
        max_yticklabel_width = max([
            fig.transFigure.inverted().transform_bbox(
                t.get_window_extent()).width for t in axes[
                    0, 0].get_yticklabels()])
        label_all_y_ticks = label_all_y_ticks and not (
            (gridspec_kw["wspace"] < max_yticklabel_width) or 
            np.isclose(gridspec_kw["wspace"], 0.0))
    label_all_x_ticks = label_all_x or (sharex == "none")
    if gridspec_kw is not None and "hspace" in gridspec_kw:
        max_xticklabel_height = max([
            fig.transFigure.inverted().transform_bbox(
                t.get_window_extent()).height for t in axes[
                    0, 0].get_xticklabels()])
        label_all_x_ticks = label_all_x_ticks and not (
            (gridspec_kw["hspace"] < max_xticklabel_height) or 
            np.isclose(gridspec_kw["hspace"], 0.0))
    if x_scale_kwargs is None:
        x_scale_kwargs = {}
    if x_scale == "log" and "nonposx" not in x_scale_kwargs:
        x_scale_kwargs["nonposx"] = "mask"
    if y_scale_kwargs is None:
        y_scale_kwargs = {}
    if y_scale == "log" and "nonposy" not in y_scale_kwargs:
        y_scale_kwargs["nonposy"] = "mask"
    for i in range(n_ax):
        tax = axes[_unravel_axes_index(i, axes.shape, True, True)]
        tax.set_xscale(x_scale, **x_scale_kwargs)
        tax.set_yscale(y_scale, **y_scale_kwargs)
        tax.tick_params(
            axis="both", which="both", bottom=True, left=True, top=True, 
            right=True, labelbottom=False, labeltop=False, labelright=False, 
            direction="inout")
        bax = axes[_unravel_axes_index(i, axes.shape, True, False)]
        bax.set_xscale(x_scale, **x_scale_kwargs)
        bax.tick_params(
            axis="both", which="both", bottom=True, left=True, top=True, 
            right=True, labeltop=False, labelright=False, direction="inout")
        if label_all_y:
            tax.set_ylabel(y_taxis_labels[i])
            bax.set_ylabel(y_baxis_labels[i])
        elif is_first_col(tax, outer_only=True):
            tax.set_ylabel(y_taxis_labels)
            bax.set_ylabel(y_baxis_labels)
        if is_first_col(tax, outer_only=True) or label_all_y_ticks:
            tax.tick_params(axis="y", which="both", labelleft=True)
            bax.tick_params(axis="y", which="both", labelleft=True)
        else:
            tax.tick_params(axis="y", which="both", labelleft=False)
            bax.tick_params(axis="y", which="both", labelleft=False)
        if label_all_x:
            bax.set_xlabel(x_axis_label)
        else:
            if is_last_row(bax, outer_only=True):
                bax.set_xlabel(x_axis_label)
        if is_last_row(bax, outer_only=True) or label_all_x_ticks:
            bax.tick_params(axis="x", which="both", labelbottom=True)
        else:
            bax.tick_params(axis="x", which="both", labelbottom=False)
    return fig, axes
    
    
def _calc_chain_chi2(truth, chain_file, nburnin=0):
    if isinstance(chain_file, str) or not hasattr(chain_file, "__len__"):
        chain = read_chain(chain_file, nburnin=nburnin, flat=True)[0]
        mean = chain.mean(axis=0)
        cov = np.cov(chain, rowvar=False)
        return np.dot(mean - truth, np.linalg.solve(cov, mean - truth))
    return np.array([_calc_chain_chi2(truth, cf, nburnin) for cf in chain_file])


def calc_chain_chi2(truth, chain_file, nburnin=0):
    return np.vectorize(
        _calc_chain_chi2, otypes=[float], excluded=[2, "nburnin"], 
        signature="(k),(n)->(n)")(truth, chain_file, nburnin)
    
    
def _get_all_wtheta(wtheta_dir, search_str):
    dir = pathlib.Path(wtheta_dir).expanduser().resolve()
    dir_files = np.array(
        sorted(
            dir.glob(search_str), 
                key=lambda p: int(re.findall("mock(\d+)", p.name)[0])))
    return np.array([Table.read(p)["xi"].data for p in dir_files])

    
def _get_delta_wtheta(wtheta_dir1, wtheta_dir2, search_str1, search_str2):
    dir1 = pathlib.Path(wtheta_dir1).expanduser().resolve()
    dir2 = pathlib.Path(wtheta_dir2).expanduser().resolve()
    dir1_files = np.array(
        sorted(
            dir1.glob(search_str1), 
            key=lambda p: int(re.findall("mock(\d+)", p.name)[0])))
    dir2_files = np.array(
        sorted(
            dir2.glob(search_str2), 
            key=lambda p: int(re.findall("mock(\d+)", p.name)[0])))
    if dir1_files.size != dir2_files.size:
        dir1_mocks = np.array([
            int(re.findall("mock(\d+)", p.name)[0]) for p in dir1_files])
        dir2_mocks = np.array([
            int(re.findall("mock(\d+)", p.name)[0]) for p in dir2_files])
        _, dir1_idx, dir2_idx = np.intersect1d(
            dir1_mocks, dir2_mocks, return_indices=True)
        dir1_files = dir1_files[dir1_idx]
        dir2_files = dir2_files[dir2_idx]
        del dir1_mocks, dir2_mocks, dir1_idx, dir2_idx
    return np.array([
        Table.read(p1)["xi"].data - Table.read(p2)["xi"].data for p1, p2 in 
        zip(dir1_files, dir2_files)])


def get_delta_wtheta_mean(wtheta_dir1, wtheta_dir2, search_str1, search_str2):
    return _get_delta_wtheta(
        wtheta_dir1, wtheta_dir2, search_str1, search_str2).mean(axis=0)


def get_delta_wtheta_err(wtheta_dir1, wtheta_dir2, search_str1, search_str2, 
                         eom):
    delta = _get_delta_wtheta(
        wtheta_dir1, wtheta_dir2, search_str1, search_str2)
    if eom:
        return delta.std(axis=0, ddof=1) / np.sqrt(delta.shape[0])
    return delta.std(axis=0, ddof=1)
    
    
def _get_bias(n0_delta_wtheta, nall_delta_wtheta, correction=True):
    delta0 = np.atleast_2d(n0_delta_wtheta)
    deltan = np.atleast_2d(nall_delta_wtheta)
    assert delta0.shape == deltan.shape
    if correction:
        return 0.5 * np.mean(deltan + delta0, axis=-2)
    else:
        return 0.5 * np.mean(deltan - delta0, axis=-2)


def get_bias(n0_wtheta_dir, nall_wtheta_dir, corr0_search_str, true_search_str):
    return _get_bias(
        _get_delta_wtheta(
            n0_wtheta_dir, n0_wtheta_dir, corr0_search_str, true_search_str), 
        _get_delta_wtheta(
            nall_wtheta_dir, n0_wtheta_dir, corr0_search_str, true_search_str))


def _c_sys(n0_delta_wtheta, nall_delta_wtheta):
    delta0 = np.atleast_2d(n0_delta_wtheta)
    deltan = np.atleast_2d(nall_delta_wtheta)
    assert delta0.shape == deltan.shape
    bias = _get_bias(delta0, deltan)
    diff = _get_bias(delta0, deltan, False)
    return 0.25 * (
        np.einsum("...i,...j->...ij", bias, bias) + 
        np.einsum("...i,...j->...ij", diff, diff))


def get_c_sys(results_root_dir, mock_run, chain_version, nside_fit, nside, 
              nmaps):
    rdir = pathlib.Path(results_root_dir).expanduser().resolve()
    mdir = rdir / mock_run
    n0_dir = mdir / "contamination_0"
    nall_dir = mdir / f"contamination_{nmaps}"
    true_search_str = "wtheta_unweighted_mock*.fits"
    corr0_search_str = (f"wtheta_mean_const_cov_mock*_fit{nside_fit}_"
                        f"nside{nside}_v{chain_version}.fits")
    sys_cov_fname = (f"wtheta_sys_cov_fit{nside_fit}_nside{nside}_nmocks{{}}_"
                     f"v{chain_version}.pkl")
    delta0 = [_get_delta_wtheta(
        n0_dir / f"zbin{zbin}", n0_dir / f"zbin{zbin}", corr0_search_str, 
        true_search_str) for zbin in range(1, 6)]
    deltan = [_get_delta_wtheta(
        nall_dir / f"zbin{zbin}", n0_dir / f"zbin{zbin}", corr0_search_str, 
        true_search_str) for zbin in range(1, 6)]
    nmocks = np.array([[
        deltai.shape[0] for deltai in deltan] for deltan in [delta0, deltan]])
    assert np.unique(nmocks).size == 1
    nmocks = nmocks[0, 0]
    sys_cov_fname = (f"wtheta_sys_cov_fit{nside_fit}_nside{nside}_"
                     f"nmocks{nmocks}_v{chain_version}.pkl")
    c_sys = []
    for zbin, (this_delta0, this_deltan) in enumerate(zip(delta0, deltan), 1):
        try:
            sys_cov = np.load(
                rdir / f"zbin{zbin}" / sys_cov_fname, allow_pickle=True)
        except FileNotFoundError:
            sys_cov = _c_sys(this_delta0, this_deltan)
            rdir.joinpath(f"zbin{zbin}", sys_cov_fname).write_bytes(
                sys_cov.dumps())
        c_sys.append(sys_cov.copy())
        del sys_cov
    return np.array(c_sys), nmocks


def get_c_sys_var(results_root_dir, mock_run, chain_version, nside_fit, nside, 
                  nmaps):
    rdir = pathlib.Path(results_root_dir).expanduser().resolve()
    mdir = rdir / mock_run
    n0_dir = mdir / "contamination_0"
    nall_dir = mdir / f"contamination_{nmaps}"
    true_search_str = "wtheta_unweighted_mock*.fits"
    corr0_search_str = (f"wtheta_mean_const_cov_mock*_fit{nside_fit}_"
                        f"nside{nside}_v{chain_version}.fits")
    sys_cov_fname = (f"wtheta_sys_cov_fit{nside_fit}_nside{nside}_nmocks{{}}_"
                     f"v{chain_version}.pkl")
    delta0 = [_get_delta_wtheta(
        n0_dir / f"zbin{zbin}", n0_dir / f"zbin{zbin}", corr0_search_str, 
        true_search_str) for zbin in range(1, 6)]
    deltan = [_get_delta_wtheta(
        nall_dir / f"zbin{zbin}", n0_dir / f"zbin{zbin}", corr0_search_str, 
        true_search_str) for zbin in range(1, 6)]
    nmocks = np.array([[
        deltai.shape[0] for deltai in deltan] for deltan in [delta0, deltan]])
    assert np.unique(nmocks).size == 1
    nmocks = nmocks[0, 0]
    delta0_loo = np.array([np.delete(delta0, i, axis=1) for i in range(nmocks)])
    deltan_loo = np.array([
        np.delete(deltan, i, axis=1) for i in range(nmocks)])
    c_sys_loo = _c_sys(delta0_loo, deltan_loo)
    return (nmocks - 1) * c_sys_loo.var(axis=0)


def get_c_stat(results_root_dir, chain_version, nside_fit, nside, nsteps, 
               nmocks, nsteps_mocks=700, nburnin_mocks=300, mock_run=None, 
               chain_root_dir=None):
    rdir = pathlib.Path(results_root_dir).expanduser().resolve()
    stat_cov_fname = (f"wtheta_stat_cov_fit{nside_fit}_nside{nside}_"
                      f"nsteps{nsteps}_v{chain_version}.pkl")
    stat_weighted_cov_fname = (f"wtheta_stat_cov_fit{nside_fit}_nside{nside}_"
                               f"nsteps{nsteps}_weighted_nmocks{nmocks}_"
                               f"v{chain_version}.pkl")
    wtheta_real_fname = (f"wtheta_real_fit{nside_fit}_nside{nside}_"
                         f"nsteps{nsteps}_v{chain_version}.npy")
    lam = 1.0
    if not all([rdir.joinpath(
            f"zbin{zbin}", stat_weighted_cov_fname).exists() 
            for zbin in range(1, 6)]):
        if chain_root_dir is None:
            raise ValueError(
                "Need chain_root_dir to get weighted covariance matrix")
        if mock_run is None:
            raise ValueError("Need mock_run to get weighted covariance matrix")
        cdir = pathlib.Path(chain_root_dir).expanduser().resolve() / mock_run
        chain_str_const_part = ("_linear_eigenbasis_const_cov_"
                                f"nside{nside_fit}_"
                                f"nsteps{nsteps_mocks+nburnin_mocks}"
                                f"_v{chain_version}.fits")
        chain_fname = LazyFStr(
            lambda: f"zbin{zbin}_mock{mock}" + chain_str_const_part)
        truths_fname = f"mean_parameters_nside{nside_fit}.pkl"
        nmaps = np.load(rdir / "zbin1" / truths_fname, allow_pickle=True).size
        truths = np.stack([
            np.zeros((5, nmaps)), np.array([
                np.load(rdir / f"zbin{zbin}" / truths_fname, allow_pickle=True) 
                for zbin in range(1, 6)])])
        chain_files = np.empty((2, 5, nmocks), dtype=object)
        for i, n in enumerate([0, nmaps]):
            for j, zbin in enumerate(range(1, 6)):
                for mock in range(nmocks):
                    chain_files[i, j, mock] = cdir.joinpath(
                        f"contamination_{n}", str(chain_fname))
        lam = calc_chain_chi2(
            truths, chain_files, nburnin=nburnin_mocks).mean() / nmaps
        del chain_files, truths
    c_stat = []
    for zbin in range(1, 6):
        try:
            stat_cov = np.load(
                rdir / f"zbin{zbin}" / stat_weighted_cov_fname, 
                allow_pickle=True)
        except FileNotFoundError:
            try:
                stat_cov = np.load(
                    rdir / f"zbin{zbin}" / stat_cov_fname, allow_pickle=True)
            except FileNotFoundError:
                w_real = np.load(rdir / f"zbin{zbin}" / wtheta_real_fname)
                stat_cov = np.cov(w_real.copy(), rowvar=False, bias=True)
                del w_real
                rdir.joinpath(f"zbin{zbin}", stat_cov_fname).write_bytes(
                    stat_cov.dumps())
            stat_cov *= lam
            rdir.joinpath(f"zbin{zbin}", stat_weighted_cov_fname).write_bytes(
                stat_cov.dumps())
        c_stat.append(stat_cov.copy())
        del stat_cov
    return np.array(c_stat)


def get_c_stat_var(results_root_dir, chain_version, nside_fit, nside, nsteps,
                   nmocks, nsteps_mocks, nburnin_mocks, mock_run, 
                   chain_root_dir):
    rdir = pathlib.Path(results_root_dir).expanduser().resolve()
    wtheta_real_fname = (f"wtheta_real_fit{nside_fit}_nside{nside}_"
                         f"nsteps{nsteps}_v{chain_version}.npy")
    wtheta_real = np.array([
        np.load(
            rdir / f"zbin{zbin}" / wtheta_real_fname) 
        for zbin in range(1, 6)]).swapaxes(1, 2)
    cdir = pathlib.Path(chain_root_dir).expanduser().resolve() / mock_run
    chain_str_const_part = ("_linear_eigenbasis_const_cov_"
                            f"nside{nside_fit}_"
                            f"nsteps{nsteps_mocks+nburnin_mocks}"
                            f"_v{chain_version}.fits")
    chain_fname = LazyFStr(
        lambda: f"zbin{zbin}_mock{mock}" + chain_str_const_part)
    truths_fname = f"mean_parameters_nside{nside_fit}.pkl"
    nmaps = np.load(rdir / "zbin1" / truths_fname, allow_pickle=True).size
    truths = np.stack([
        np.zeros((5, nmaps)), np.array([
            np.load(rdir / f"zbin{zbin}" / truths_fname, allow_pickle=True) 
            for zbin in range(1, 6)])])
    chain_files = np.empty((2, 5, nmocks), dtype=object)
    for i, n in enumerate([0, nmaps]):
        for j, zbin in enumerate(range(1, 6)):
            for mock in range(nmocks):
                chain_files[i, j, mock] = cdir.joinpath(
                    f"contamination_{n}", str(chain_fname))
    lam = calc_chain_chi2(
        truths, chain_files, nburnin=nburnin_mocks).mean() / nmaps
    del chain_files, truths
    return lam**2 * np.array(
        [[stats.kstatvar(w_theta) for w_theta in w_z] for w_z in wtheta_real])

    
    
def n_of_z_table(area_y1, area_now, ngals_y1, ngals_now, bin_edges):
    z_range_str = r"${edges[0]} < z < {edges[1]}$"
    table_body = " \\\\\n".join([
        fr"{z_range_str.format(edges=bin_edges[i-1:i+1])} & {ny} & "
        fr"{np.around(ny / area_y1, 4)} & {nn} & {np.around(nn / area_now, 4)}" 
        for i, (ny, nn) in enumerate(zip(ngals_y1, ngals_now), 1)])
    col_names = " & ".join([
        r"$z$ range", r"Y1 $N_{\rm gal}$", r"Y1 $n_{\rm gal}$", 
        r"$N_{\rm gal}$", r"$n_{\rm gal}$"])
    col_units = " & ".join([
        "", "", r"(\si{\per\square\amin})", "", r"(\si{\per\square\amin})"])
    tabular_str = (f"\\begin{{tabular}}{{lcccc}}\n\\toprule\n{col_names} "
                   f"\\\\\n{col_units} \\\\ \\midrule\n{table_body} \\\\ "
                   f"\\bottomrule\n\\end{{tabular}}")
    return tabular_str


def plot_n_of_z(redmagic_dir, chain_version, bin_edges, output_dir, 
                delta_max=0.2, nside=4096):
    rm_dir = pathlib.Path(redmagic_dir).expanduser().resolve()
    odir = pathlib.Path(output_dir).expanduser().resolve()
    fgood = hp.read_map(
        rm_dir / f"DES_Y1A1_3x2pt_redMaGiC_MASK_HPIX{nside}RING.fits",     
        verbose=False, dtype=float)
    wmask = np.load(rm_dir.joinpath(
        f"systematics_weights_mask_max{delta_max}_nside{nside}_"
        f"v{chain_version}.npy"))
    pix_area = hp.nside2pixarea(nside, degrees=True) * 60**2
    area_y1 = pix_area * fgood[fgood >= 0.8].sum()
    area_now = pix_area * fgood[wmask].sum()
    cat = Table.read(rm_dir / "DES_Y1A1_3x2pt_redMaGiC_zerr_CATALOG.fits")
    cat["HPIX"] = hp.ang2pix(nside, cat["RA"], cat["DEC"], lonlat=True)
    bins_y1 = cat.group_by(np.digitize(cat["ZREDMAGIC"], bin_edges))
    n_y1 = bins_y1["RA"].groups.aggregate(len).data
    cat.remove_rows(~wmask[cat["HPIX"]])
    bins_now = cat.group_by(np.digitize(cat["ZREDMAGIC"], bin_edges))
    n_now = bins_now["RA"].groups.aggregate(len).data
    odir.joinpath("redmagic_bin_table.tex").write_text(
        n_of_z_table(area_y1, area_now, n_y1, n_now, bin_edges))
    del n_y1, n_now, area_y1, area_now, pix_area
    z_edges, z_width = np.linspace(0.0, 2.0, num=201, retstep=True)
    z_table = 0.5 * (z_edges[:-1] + z_edges[1:])[None, :]
    n_y1 = np.empty((len(bin_edges) - 1, z_table.size))
    n_now = np.empty_like(n_y1)
    for i, (y1_group, now_group) in enumerate(
            zip(bins_y1.groups, bins_now.groups)):
        n_y1[i] = np.sum(
            stats.norm.pdf(
                (y1_group["ZREDMAGIC"][:, None] - z_table) / 
                    y1_group["ZREDMAGIC_E"][:, None]) / 
                y1_group["ZREDMAGIC_E"][:, None], axis=0)
        n_now[i] = np.sum(
            stats.norm.pdf(
                (now_group["ZREDMAGIC"][:, None] - z_table) / 
                    now_group["ZREDMAGIC_E"][:, None]) / 
                now_group["ZREDMAGIC_E"][:, None], axis=0)
    n_y1 /= z_width
    n_now /= z_width
    n_y1 *= 1e-8
    n_now *= 1e-8
    z_table = z_table[0]
    del cat, bins_y1, bins_now
    line_styles = [
        (0, ()), (0, (3.7, 1.6)), (0, (6.4, 1.6, 1.0, 1.6)), 
        (0, (5.2, 1.4, 1.0, 1.4, 1.0, 1.4)), (0, (1.0, 1.65))]
    with plt.style.context(["paper", "colorblind", "onecol-wide"]):
        plt.figure()
        plt.xlabel(r"$z$")
        plt.ylabel(r"$\displaystyle\frac{\mathrm{d} N}{\mathrm{d} z} "
                   r"\times 10^{-8}$")
        l_y1, = plt.plot(z_table, n_y1[0], c="k", ls=":", alpha=0.5)
        l_now, = plt.plot(z_table, n_now[0], c="k", ls="--", alpha=0.5)
        l_bins = []
        for i, (yi, ni) in enumerate(zip(n_y1, n_now)):
            l_bins.append(plt.plot(z_table, yi, c=f"C{i}", ls=":")[0])
            plt.plot(z_table, ni, ls="--", c=f"C{i}")
        plt.xlim([0.0, 1.0])
        ymax = plt.ylim()[1]
        plt.ylim([0.0, ymax])
        leg_mask = plt.legend(
            [l_y1, l_now], [r"DES Y1", r"This work"], fontsize=7, 
            loc="upper left")
        l_y1.remove()
        l_now.remove()
        leg_bins = plt.legend(
            l_bins, [fr"${bmin} \leq z < {bmax}$" for bmin, bmax in 
                zip(bin_edges[:-1], bin_edges[1:])], handlelength=0, 
            handletextpad=0, loc="upper right", fontsize=7)
        for l, t in zip(l_bins, leg_bins.get_texts()):
            t.set_color(l.get_c())
        plt.gca().add_artist(leg_mask)
        plt.savefig(odir / "des_y1_redmagic_n_of_z.pgf")
        plt.savefig(odir / "des_y1_redmagic_n_of_z.pdf")
        plt.savefig(odir / "des_y1_redmagic_n_of_z.png")
        plt.close()
    return
    
    
def map_importance_band(sys_root_dir, chain_root_dir, chain_version, mock_run, 
                        nside, nsteps, nburnin, contamination):
    cdir = pathlib.Path(chain_root_dir).expanduser().resolve()
    cdir = cdir / mock_run / f"contamination_{contamination}"
    nsteps_tot = nsteps + nburnin
    sdir = pathlib.Path(sys_root_dir).expanduser().resolve()
    sys_maps = np.load(
        sdir / f"standard_systematics_eigenbasis_fit{nside}_nside{nside}.pkl", 
        allow_pickle=True)
    cfile_search_strs = [(f"zbin{zbin}_mock*_linear_eigenbasis_const_cov_"
                          f"nside{nside}_nsteps{nsteps_tot}_v{chain_version}"
                          ".fits") for zbin in range(1, 6)]
    tau_percentiles = np.empty((5, 3, sys_maps.shape[0]))
    for i, cstr in enumerate(cfile_search_strs):
        tau_real = np.array([
            np.sort(map_importance(
                read_chain(cfile, nburnin, True)[0], sys_maps)[0])[::-1] 
            for cfile in cdir.glob(cstr)])
        tau_percentiles[i] = np.percentile(tau_real, [16, 50, 84], axis=0)
    return tau_percentiles
    


def map_importance_plot(results_root_dir, chain_version, nside, nmaps, 
                        output_dir, sys_root_dir, chain_root_dir, mock_run, 
                        nsteps_mocks, nburnin_mocks):
    rdir = pathlib.Path(results_root_dir).expanduser().resolve()
    odir = pathlib.Path(output_dir).expanduser().resolve()
    order_alpha = np.array([
        np.load(
            rdir.joinpath(
                f"zbin{zbin}", 
                f"map_importance_order_const_cov_fit{nside}_v{chain_version}"
                ".npy")) 
        for zbin in range(1, 6)])
    tau_alpha = np.array([
        np.load(
            rdir.joinpath(
                f"zbin{zbin}", 
                f"map_importance_const_cov_fit{nside}_unsorted_v{chain_version}"
                ".npy"))
        for zbin in range(1, 6)])
    tau_err_alpha = np.array([
        np.load(
            rdir.joinpath(
                f"zbin{zbin}", 
                f"map_importance_err_const_cov_fit{nside}_unsorted_"
                f"v{chain_version}.npy")) * np.sqrt(1000)
        for zbin in range(1, 6)])
    tau_bands0 = map_importance_band(
        sys_root_dir, chain_root_dir, chain_version, mock_run, nside, 
        nsteps_mocks, nburnin_mocks, 0)
    tau_bandsn = map_importance_band(
        sys_root_dir, chain_root_dir, chain_version, mock_run, nside, 
        nsteps_mocks, nburnin_mocks, nmaps)
    with plt.style.context(["paper", "colorblind", "twocol-wide"]):
        fig, axes = setup_axes_noresid(
            5, r"Importance Rank", r"$\tau_\alpha$", x_scale="linear", 
            sharey="all", sharex="all", gridspec_kw={"hspace": 0.0})
        for i in range(5):
            ax = axes[_unravel_axes_index(i, axes.shape)]
            ax.axhline(0, c="k", ls="-", lw=plt.rcParams["axes.linewidth"])
            ax.fill_between(
                np.arange(nmaps), tau_bands0[i, 0], tau_bands0[i, 2], fc="C1", 
                alpha=0.3, ec="C1", hatch="\\", label=r"Uncontaminated")
            ax.fill_between(
                np.arange(nmaps), tau_bandsn[i, 0], tau_bandsn[i, 2], fc="C2", 
                alpha=0.3, ec="C2", hatch="/", label=r"Contaminated")
            ax.errorbar(
                np.arange(nmaps), 
                tau_alpha[i][order_alpha[i]], 
                yerr=tau_err_alpha[i][order_alpha[i]], fmt="C0o", label=r"Data")
            ax.legend(
                [ax.lines[0]], [fr"${i + 1}, {i + 1}$"], frameon=False, 
                handlelength=0.0, handletextpad=0.0)
        axes[-1, -1].legend(
            *axes[0, 0].get_legend_handles_labels(), loc="center", 
            bbox_to_anchor=(0.45, 0.5))
        plt.savefig(
            odir / f"map_importance_all-z_const_cov_v{chain_version}.pgf")
        plt.savefig(
            odir / f"map_importance_all-z_const_cov_v{chain_version}.pdf")
        plt.savefig(
            odir / f"map_importance_all-z_const_cov_v{chain_version}.png")
        plt.close()
    return


def plot_mock_corners(results_root_dir, chain_root_dir, mock_run, chain_version, 
                      nside_fit, nsteps, nburnin, output_dir, 
                      param_choice=None):
    rdir = pathlib.Path(results_root_dir).expanduser().resolve()
    cdir = pathlib.Path(chain_root_dir).expanduser().resolve() / mock_run
    odir = pathlib.Path(output_dir)
    chain_str_const_part = (f"_linear_eigenbasis_const_cov_nside{nside_fit}_"
                            f"nsteps{nsteps + nburnin}_v{chain_version}.fits")
    chain_search_str = LazyFStr(
        lambda: f"zbin{zbin}_mock*" + chain_str_const_part)
    chain_fname = LazyFStr(
        lambda: f"zbin{zbin}_mock{mock}" + chain_str_const_part)
    truths_fname = f"mean_parameters_nside{nside_fit}.pkl"
    nmaps = np.load(rdir / "zbin1" / truths_fname, allow_pickle=True).size
    params = [fr"$a_{{{i}}}$" for i in range(nmaps)]
    if param_choice is None:
        plot_params = [fr"$a_{{{i}}}$" for i in [0, 2, 6, 11, 17]]
    else:
        plot_params = [
            p if isinstance(p, str) else params[p] for p in param_choice]
    plot_params = sorted(
        plot_params, key=lambda p: int(re.findall("(\d+)", p)[0]))
    plot_fname = LazyFStr(
        lambda: f"corner_zbin{zbin}_contamination_{n}" + str(
            f"_const_cov_nside{nside_fit}_nburn{nburnin}_nsteps{nsteps}"
            f"_v{chain_version}"))
    truths = dict([
        (0, dict.fromkeys(range(1, 6), dict.fromkeys(params, 0.0))), 
        (nmaps, dict([
            (zbin, dict(zip(params, np.load(
                rdir / f"zbin{zbin}" / truths_fname, allow_pickle=True)))) 
            for zbin in range(1, 6)]))])
    for zbin in range(1, 6):
        for n in [0, nmaps]:
            mocks = np.sort(
                np.random.choice(
                    [int(re.findall("mock(\d+)", p.name)[0]) for p in 
                     cdir.joinpath(
                        f"contamination_{n}").glob(str(chain_search_str))], 
                    5, replace=False))
            cc = MyChainConsumer()
            for mock in mocks:
                cc.add_chain(
                    read_chain(
                        cdir / f"contamination_{n}" / str(chain_fname), 
                        nburnin=nburnin, flat=True)[0], 
                    parameters=params, name=fr"Mock {mock}")
            cc.configure()
            cc.configure_truth(ls="-")
            with plt.rc_context(rc={"figure.autolayout": False}):
                fig = cc.plotter.plot(
                    parameters=plot_params, truth=truths[n][zbin], 
                    figsize="COLUMN", display=False)
            fig.savefig(odir.joinpath(str(plot_fname) + ".pgf"))
            fig.savefig(odir.joinpath(str(plot_fname) + ".pdf"))
            fig.savefig(odir.joinpath(str(plot_fname) + ".png"))
            plt.close()
    return


def chi2_histogram(results_root_dir, chain_root_dir, mock_run, chain_version, 
                   nside_fit, nsteps, nburnin, output_dir):
    rdir = pathlib.Path(results_root_dir).expanduser().resolve()
    cdir = pathlib.Path(chain_root_dir).expanduser().resolve() / mock_run
    odir = pathlib.Path(output_dir)
    chain_str_const_part = (f"_linear_eigenbasis_const_cov_nside{nside_fit}_"
                            f"nsteps{nsteps + nburnin}_v{chain_version}.fits")
    chain_search_str = LazyFStr(
        lambda: f"zbin{zbin}_mock*" + chain_str_const_part)
    truths_fname = f"mean_parameters_nside{nside_fit}.pkl"
    nmaps = np.load(rdir / "zbin1" / truths_fname, allow_pickle=True).size
    truths = np.stack([
        np.zeros((5, nmaps)), np.array([
            np.load(rdir / f"zbin{zbin}" / truths_fname, allow_pickle=True) 
            for zbin in range(1, 6)])])
    chain_files = [[], []]
    for i, n in enumerate([0, nmaps]):
        for zbin in range(1, 6):
            chain_files[i].append(
                list(
                    cdir.joinpath(
                        f"contamination_{n}").glob(str(chain_search_str))))
    chi2 = calc_chain_chi2(truths, chain_files, nburnin=nburnin)
    chi2_means = chi2.mean(axis=-1)
    worst_idx = np.unravel_index(chi2_means.argmax(), chi2_means.shape)
    best_idx = np.unravel_index(chi2_means.argmin(), chi2_means.shape)
    ave_chi2 = chi2.mean()
    print(ave_chi2)
    with plt.style.context(["paper", "colorblind", "onecol-wide"]):
        fig = plt.figure()
        plt.xlabel(fr"$\chi^2_{{{nmaps}}}$")
        _, wbins, _ = plt.hist(
            chi2[worst_idx[0], worst_idx[1]], bins="doane", density=True, 
            histtype="step", color="C0", fill=True, hatch="/",
            fc=mpl.colors.to_rgba("C0", alpha=0.3), 
            lw=plt.rcParams["hatch.linewidth"], 
            label=(fr"$\overline{{\chi^2_{{{nmaps}}}}} = "
                   fr"{np.around(chi2_means[worst_idx], 2)}$"))
        _, bbins, _ = plt.hist(
            chi2[best_idx[0], best_idx[1]], bins="doane", density=True, 
            histtype="step", color="C1", fill=True, hatch="\\", 
            fc=mpl.colors.to_rgba("C1", alpha=0.3), 
            lw=plt.rcParams["hatch.linewidth"],
            label=(fr"$\overline{{\chi^2_{{{nmaps}}}}} = "
                   fr"{np.around(chi2_means[best_idx], 2)}$"))
        x = np.arange(
            min(wbins[0], bbins[0]), max(wbins[-1], bbins[-1]) + 0.1, 0.1)
        plt.plot(
            x, stats.chi2.pdf(x, nmaps), "C2-", 
            label=fr"$P_{{{nmaps}}}(\chi^2)$")
        plt.legend()
        fig.savefig(odir.joinpath(
            f"chi2_histogram_worst_best_const_cov_nside{nside_fit}"
            f"_nburnin{nburnin}_nsteps{nsteps}_v{chain_version}.pgf"))
        fig.savefig(odir.joinpath(
            f"chi2_histogram_worst_best_const_cov_nside{nside_fit}"
            f"_nburnin{nburnin}_nsteps{nsteps}_v{chain_version}.pdf"))
        fig.savefig(odir.joinpath(
            f"chi2_histogram_worst_best_const_cov_nside{nside_fit}"
            f"_nburnin{nburnin}_nsteps{nsteps}_v{chain_version}.png"))
        plt.close()
        fig = plt.figure()
        plt.xlabel(fr"$\chi^2_{{{nmaps}}} / \lambda$")
        _, obins, _ = plt.hist(
            chi2.flatten(), bins="doane", density=True, histtype="step", 
            color="C0", fill=True, fc=mpl.colors.to_rgba("C0", alpha=0.3), 
            hatch="/", label=r"$\lambda = 1$", 
            lw=plt.rcParams["hatch.linewidth"])
        _, nbins, _ = plt.hist(
            18 * chi2.flatten() / ave_chi2, bins="doane", density=True, 
            histtype="step", color="C1", fill=True, hatch="\\",
            lw=plt.rcParams["hatch.linewidth"], 
            fc=mpl.colors.to_rgba("C1", alpha=0.3), 
            label=(fr"$\lambda = {np.around(ave_chi2 / nmaps, 2)}$"))
        x = np.arange(
            min(obins[0], nbins[0]), max(obins[-1], nbins[-1]) + 0.1, 0.1)
        plt.plot(
            x, stats.chi2.pdf(x, nmaps), "C2-", 
            label=fr"$P_{{{nmaps}}}(\chi^2)$")
        plt.legend()
        fig.savefig(odir.joinpath(
            f"chi2_histogram_flat_const_cov_nside{nside_fit}"
            f"_nburnin{nburnin}_nsteps{nsteps}_v{chain_version}.pgf"))
        fig.savefig(odir.joinpath(
            f"chi2_histogram_flat_const_cov_nside{nside_fit}"
            f"_nburnin{nburnin}_nsteps{nsteps}_v{chain_version}.pdf"))
        fig.savefig(odir.joinpath(
            f"chi2_histogram_flat_const_cov_nside{nside_fit}"
            f"_nburnin{nburnin}_nsteps{nsteps}_v{chain_version}.png"))
        plt.close()
    return
    
    

def corr0_bias_plot(results_root_dir, mock_run, chain_version, nside_fit, nside, 
                    nmaps, output_dir, mask, theta, theta_min=0.0, ylim=None):
    rdir = pathlib.Path(results_root_dir).expanduser().resolve() / mock_run
    odir = pathlib.Path(output_dir).expanduser().resolve()
    true_search_str = "wtheta_unweighted_mock*.fits"
    corr0_search_str = (f"wtheta_mean_const_cov_mock*_fit{nside_fit}_"
                        f"nside{nside}_v{chain_version}.fits")
    theta_mask = theta >= theta_min
    theta_ = np.copy(theta)[theta_mask]
    sigma_true = np.std([
        _get_all_wtheta(
            rdir / "contamination_0" / f"zbin{zbin}", true_search_str) 
        for zbin in range(1, 6)], axis=1, ddof=1)[:, theta_mask]
    delta0_mean = np.array([
        get_delta_wtheta_mean(
            rdir / "contamination_0" / f"zbin{zbin}", 
            rdir / "contamination_0" / f"zbin{zbin}", 
            corr0_search_str, true_search_str) 
        for zbin in range(1, 6)])[:, theta_mask]
    delta0_err = np.array([
        get_delta_wtheta_err(
            rdir / "contamination_0" / f"zbin{zbin}", 
            rdir / "contamination_0" / f"zbin{zbin}", 
            corr0_search_str, true_search_str, True) 
        for zbin in range(1, 6)])[:, theta_mask]
    deltan_mean = np.array([
        get_delta_wtheta_mean(
            rdir / f"contamination_{nmaps}" / f"zbin{zbin}", 
            rdir / "contamination_0" / f"zbin{zbin}", 
            corr0_search_str, true_search_str) 
        for zbin in range(1, 6)])[:, theta_mask]
    deltan_err = np.array([
        get_delta_wtheta_err(
            rdir / f"contamination_{nmaps}" / f"zbin{zbin}", 
            rdir / "contamination_0" / f"zbin{zbin}", 
            corr0_search_str, true_search_str, True) 
        for zbin in range(1, 6)])[:, theta_mask]
    with plt.style.context(["paper", "colorblind", "twocol-wide"]):
        fig, axes = setup_axes_noresid(
            5, r"$\theta \left[\mathrm{arcmin}\right]$", 
            r"$\displaystyle\frac{w_{\rm corr0}(\theta) - "
            r"w_{\rm true}(\theta)}{\sigma_{w_{\rm true}}}$", sharex="all", 
            sharey="all", gridspec_kw={"hspace": 0.0, "wspace": 0.0})
        for i in range(5):
            ax = axes[_unravel_axes_index(i, axes.shape)]
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.axhline(
                0, c="k", ls="-", lw=plt.rcParams["axes.linewidth"], zorder=0)
            l0, = ax.plot(
                theta_, (delta0_mean / sigma_true)[i], "C1--", zorder=1)
            e0 = ax.fill_between(
                theta_, ((delta0_mean - delta0_err) / sigma_true)[i], 
                ((delta0_mean + delta0_err) / sigma_true)[i], fc="C1", 
                alpha=0.3, ec="C1", hatch="\\", zorder=1)
            ln, = ax.plot(
                theta_, (deltan_mean / sigma_true)[i], "C0-", zorder=2)
            en = ax.fill_between(
                theta_, ((deltan_mean - deltan_err) / sigma_true)[i], 
                ((deltan_mean + deltan_err) / sigma_true)[i], fc="C0", 
                alpha=0.3, ec="C0", hatch="/", zorder=2)
            xlim = ax.get_xlim()
            ax.axvspan(0, mask[i], fc="k", alpha=0.1, zorder=0)
            ax.set_xlim(xlim)
            ax.legend(
                [ax.lines[0]], [fr"${i + 1}, {i + 1}$"], frameon=False, 
                handlelength=0.0, handletextpad=0.0)
        axes[-1, -1].legend(
            [(ln, en), (l0, e0)], [r"Contaminated", r"Uncontaminated"], 
            loc="center", bbox_to_anchor=(0.5, 0.4))
        plt.savefig(odir.joinpath(
            f"w_corr0_bias_truth_mean+eom_const_cov_v{chain_version}.pdf"))
        plt.savefig(odir.joinpath(
            f"w_corr0_bias_truth_mean+eom_const_cov_v{chain_version}.png"))
        plt.close()
    return


def contaminated_bias_plot(results_root_dir, mock_run, chain_version, nside_fit, 
                           nside, nmaps, output_dir, mask, theta, theta_min=0.0, 
                           ylim=None):
    rdir = pathlib.Path(results_root_dir).expanduser().resolve() / mock_run
    odir = pathlib.Path(output_dir).expanduser().resolve()
    true_search_str = "wtheta_unweighted_mock*.fits"
    corr0_search_str = (f"wtheta_mean_const_cov_mock*_fit{nside_fit}_"
                        f"nside{nside}_v{chain_version}.fits")
    theta_mask = theta >= theta_min
    theta_ = np.copy(theta)[theta_mask]
    sigma_true = np.std([
        _get_all_wtheta(
            rdir / "contamination_0" / f"zbin{zbin}", true_search_str) 
        for zbin in range(1, 6)], axis=1, ddof=1)[:, theta_mask]
    bias = np.array([
        get_bias(
            rdir / "contamination_0" / f"zbin{zbin}", 
            rdir / f"contamination_{nmaps}" / f"zbin{zbin}", corr0_search_str, 
            true_search_str)
        for zbin in range(1, 6)])[:, theta_mask]
    delta_cont_mean = np.array([
        get_delta_wtheta_mean(
            rdir / f"contamination_{nmaps}" / f"zbin{zbin}", 
            rdir / "contamination_0" / f"zbin{zbin}", 
            true_search_str, true_search_str) 
        for zbin in range(1, 6)])[:, theta_mask]
    delta_cont_err = np.array([
        get_delta_wtheta_err(
            rdir / f"contamination_{nmaps}" / f"zbin{zbin}", 
            rdir / "contamination_0" / f"zbin{zbin}", 
            true_search_str, true_search_str, True) 
        for zbin in range(1, 6)])[:, theta_mask]
    delta_corr0_mean = np.array([
        get_delta_wtheta_mean(
            rdir / f"contamination_{nmaps}" / f"zbin{zbin}", 
            rdir / "contamination_0" / f"zbin{zbin}", 
            corr0_search_str, true_search_str) 
        for zbin in range(1, 6)])[:, theta_mask]
    delta_corr0_err = np.array([
        get_delta_wtheta_err(
            rdir / f"contamination_{nmaps}" / f"zbin{zbin}", 
            rdir / "contamination_0" / f"zbin{zbin}", 
            corr0_search_str, true_search_str, True) 
        for zbin in range(1, 6)])[:, theta_mask]
    delta_corr1_mean = delta_corr0_mean - bias
    delta_corr1_err = np.sqrt(
        delta_corr0_err**2 + np.diagonal(
            get_c_sys(
                rdir.parent, mock_run, chain_version, nside_fit, nside, 
                nmaps)[0], 
            axis1=1, axis2=2)[:, theta_mask])
    with plt.style.context(["paper", "colorblind", "twocol-wide"]):
        fig, axes = setup_axes_noresid(
            5, r"$\theta \left[\mathrm{arcmin}\right]$", 
            r"$\displaystyle\frac{w(\theta) - "
            r"w_{\rm true}(\theta)}{\sigma_{w_{\rm true}}}$", sharex="all", 
            sharey="row", gridspec_kw={"hspace": 0.0, "wspace": 0.0})
        for i in range(5):
            ax = axes[_unravel_axes_index(i, axes.shape)]
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.axhline(
                0, c="k", ls="-", lw=plt.rcParams["axes.linewidth"], zorder=0)
            lc, = ax.plot(
                theta_, (delta_cont_mean / sigma_true)[i], "C1--", zorder=1)
            ec = ax.fill_between(
                theta_, ((delta_cont_mean - delta_cont_err) / sigma_true)[i], 
                ((delta_cont_mean + delta_cont_err) / sigma_true)[i], fc="C1", 
                alpha=0.3, ec="C1", hatch="\\", zorder=1)
            l0, = ax.plot(
                theta_, (delta_corr0_mean / sigma_true)[i], "C0-", zorder=2)
            e0 = ax.fill_between(
                theta_, ((delta_corr0_mean - delta_corr0_err) / sigma_true)[i], 
                ((delta_corr0_mean + delta_corr0_err) / sigma_true)[i], fc="C0", 
                alpha=0.3, ec="C0", hatch="/", zorder=2)
            l1, = ax.plot(
                theta_, (delta_corr1_mean / sigma_true)[i], "C2-.", zorder=3)
            e1 = ax.fill_between(
                theta_, ((delta_corr1_mean - delta_corr1_err) / sigma_true)[i], 
                ((delta_corr1_mean + delta_corr1_err) / sigma_true)[i], fc="C2", 
                alpha=0.3, ec="C2", hatch="|", zorder=3)
            xlim = ax.get_xlim()
            ax.axvspan(0, mask[i], fc="k", alpha=0.1, zorder=0)
            ax.set_xlim(xlim)
            ax.legend(
                [ax.lines[0]], [fr"${i + 1}, {i + 1}$"], frameon=False, 
                handlelength=0.0, handletextpad=0.0)
        axes[1, 0].yaxis.set_major_locator(
            MaxNLocator(nbins=6, prune="upper"))
        axes[-1, -1].legend(
            [(lc, ec), (l0, e0), (l1, e1)], [
                r"Uncorrected", r"Linear correction", 
                "Linear + Bias correction"], 
            loc="center", bbox_to_anchor=(0.54, 0.48))
        plt.savefig(odir.joinpath(
            f"w_cont_corr0_corr1_bias_truth_mean+eom_const_cov_v{chain_version}"
            ".pdf"))
        plt.savefig(odir.joinpath(
            f"w_cont_corr0_corr1_bias_truth_mean+eom_const_cov_v{chain_version}"
            ".png"))
        plt.close()
    return


def correction_plot(results_root_dir, mock_run, chain_version, nside_fit, nside, 
                    nmaps, output_dir, des_dir, mask, theta, theta_min=0.0, 
                    ylim=None, des_2pt_name="2pt_NG_mcal_1110.fits"):
    rdir = pathlib.Path(results_root_dir).expanduser().resolve()
    odir = pathlib.Path(output_dir).expanduser().resolve()
    cont_search_str = "wtheta_unweighted_mock*.fits"
    corr0_search_str = (f"wtheta_mean_const_cov_mock*_fit{nside_fit}_"
                        f"nside{nside}_v{chain_version}.fits")
    des_spec = twopoint.TwoPointFile.from_fits(
        pathlib.Path(des_dir) / des_2pt_name).get_spectrum("wtheta")
    theta_mask = theta >= theta_min
    theta_ = np.copy(theta)[theta_mask]
    delta0_mean = np.array([
        get_delta_wtheta_mean(
            rdir / mock_run / "contamination_0" / f"zbin{zbin}", 
            rdir / mock_run / "contamination_0" / f"zbin{zbin}", 
            corr0_search_str, cont_search_str) 
        for zbin in range(1, 6)])[:, theta_mask]
    delta0_err = np.array([
        get_delta_wtheta_err(
            rdir / mock_run / "contamination_0" / f"zbin{zbin}", 
            rdir / mock_run / "contamination_0" / f"zbin{zbin}", 
            corr0_search_str, cont_search_str, False) 
        for zbin in range(1, 6)])[:, theta_mask]
    deltan_mean = np.array([
        get_delta_wtheta_mean(
            rdir / mock_run / f"contamination_{nmaps}" / f"zbin{zbin}", 
            rdir / mock_run / f"contamination_{nmaps}" / f"zbin{zbin}", 
            corr0_search_str, cont_search_str) 
        for zbin in range(1, 6)])[:, theta_mask]
    deltan_err = np.array([
        get_delta_wtheta_err(
            rdir / mock_run / f"contamination_{nmaps}" / f"zbin{zbin}", 
            rdir / mock_run / f"contamination_{nmaps}" / f"zbin{zbin}", 
            corr0_search_str, cont_search_str, False) 
        for zbin in range(1, 6)])[:, theta_mask]
    delta_data = np.array([
        Table.read(
            rdir.joinpath(
                f"zbin{zbin}", 
                f"wtheta_mean_const_cov_fit{nside_fit}_nside{nside}_"
                f"v{chain_version}.fits"))["xi"].data - Table.read(
                    rdir.joinpath(
                        f"zbin{zbin}", "wtheta_unweighted.fits"))["xi"].data 
        for zbin in range(1, 6)])[:, theta_mask]
    with plt.style.context(["paper", "colorblind", "twocol-wide"]):
        fig, axes = setup_axes_noresid(
            5, r"$\theta \left[\mathrm{arcmin}\right]$", 
            r"$\displaystyle\frac{w_{\rm corr0}(\theta) - "
            r"w_{\rm cont}(\theta)}{\sigma_{Y1}}$", sharex="all", 
            gridspec_kw={"hspace": 0.0})
        for i in range(5):
            ax = axes[_unravel_axes_index(i, axes.shape)]
            if ylim is not None:
                ax.set_ylim(ylim)
            erri = des_spec.get_error(i + 1, i + 1)[theta >= theta_min]
            ax.axhline(
                0, c="k", ls="-", lw=plt.rcParams["axes.linewidth"], zorder=0)
            l0, = ax.plot(theta_, delta0_mean[i] / erri, "C2-.", zorder=1)
            e0 = ax.fill_between(
                theta_, (delta0_mean - delta0_err)[i] / erri, 
                (delta0_mean + delta0_err)[i] / erri, fc="C2", 
                alpha=0.3, ec="C2", hatch="\\", zorder=1)
            ln, = ax.plot(theta_, deltan_mean[i] / erri, "C1--", zorder=2)
            en = ax.fill_between(
                theta_, (deltan_mean - deltan_err)[i] / erri, 
                (deltan_mean + deltan_err)[i] / erri, fc="C1", 
                alpha=0.3, ec="C1", hatch="/", zorder=2)
            ld, = ax.plot(theta_, delta_data[i] / erri, "C0-", zorder=3)
            xlim = ax.get_xlim()
            ax.axvspan(0, mask[i], fc="k", alpha=0.1, zorder=0)
            ax.set_xlim(xlim)
            ax.legend(
                [ax.lines[0]], [fr"${i + 1}, {i + 1}$"], frameon=False, 
                handlelength=0.0, handletextpad=0.0)
        axes[-1, -1].legend(
            [ld, (ln, en), (l0, e0)], 
            [r"Data", r"Contaminated", r"Uncontaminated"], 
            loc="center", bbox_to_anchor=(0.5, 0.4))
        plt.savefig(odir.joinpath(
            f"w_corr0_bias_cont_data_mean+std_const_cov_v{chain_version}.pdf"))
        plt.savefig(odir.joinpath(
            f"w_corr0_bias_cont_data_mean+std_const_cov_v{chain_version}.png"))
        plt.close()
    return


def _generate_spectrum_measurement(des_spec, wtheta, err=None, npairs=None):
    if npairs is None:
        npairs_ = des_spec.npairs.copy()
    else:
        npairs_ = np.copy(npairs)
    if npairs_.ndim == 2:
        npairs_ = npairs_.flatten()
    wtheta_ = np.copy(wtheta)
    if wtheta_.ndim == 2:
        wtheta_ = wtheta_.flatten()
    if err is not None:
        err_ = np.copy(err)
        if err_.ndim == 2:
            err_ = err_.flatten()
        if des_spec.varxi is not None:
            varxi_ = err_**2
        else:
            varxi_ = None
    else:
        err_ = None
        varxi_ = None
    return twopoint.SpectrumMeasurement(
        des_spec.name, [des_spec.bin1, des_spec.bin2], 
        [des_spec.type1, des_spec.type2], [des_spec.kernel1, des_spec.kernel2], 
        des_spec.windows, des_spec.angular_bin, wtheta_, angle=des_spec.angle, 
        error=err_, angle_unit=des_spec.angle_unit, metadata=des_spec.metadata, 
        npairs=npairs_, varxi=varxi_, extra_cols=None, 
        angle_min=des_spec.angle_min, angle_max=des_spec.angle_max)


def save_dvec(des_dir, results_root_dir, mock_run, chain_version, nside_fit, 
              nside, nsteps, nmaps, nsteps_mocks=700, nburnin_mocks=300, 
              chain_root_dir=None, des_2pt_name="2pt_NG_mcal_1110.fits"):
    odir = pathlib.Path(des_dir).expanduser().resolve()
    rdir = pathlib.Path(results_root_dir).expanduser().resolve()
    true_search_str = "wtheta_unweighted_mock*.fits"
    corr0_search_str = (f"wtheta_mean_const_cov_mock*_fit{nside_fit}_"
                        f"nside{nside}_v{chain_version}.fits")
    des_dvec = twopoint.TwoPointFile.from_fits(odir / des_2pt_name)
    des_spec = des_dvec.get_spectrum("wtheta")
    wtheta_pos = np.where(
        [n == "wtheta" for n in des_dvec.covmat_info.names])[0][0]
    cov_start = des_dvec.covmat_info.starts[wtheta_pos]
    cov_end = cov_start + des_dvec.covmat_info.lengths[wtheta_pos]
    c_sys, nmocks = get_c_sys(
        results_root_dir, mock_run, chain_version, nside_fit, nside, nmaps)
    c_sys = linalg.block_diag(*c_sys)
    c_stat = linalg.block_diag(
        *get_c_stat(
            results_root_dir, chain_version, nside_fit, nside, nsteps, 
            nmocks, nsteps_mocks, nburnin_mocks, mock_run, chain_root_dir))
    cov = des_dvec.covmat.copy()
    cov[cov_start:cov_end, cov_start:cov_end] += c_stat
    cinfo_stat = twopoint.CovarianceMatrixInfo(
        des_dvec.covmat_info.name, des_dvec.covmat_info.names, 
        des_dvec.covmat_info.lengths, cov)
    data_corr0 = np.array([
        Table.read(
            rdir.joinpath(
                f"zbin{zbin}", 
                f"wtheta_mean_const_cov_fit{nside_fit}_nside{nside}_"
                f"v{chain_version}.fits"))["xi"].data 
        for zbin in range(1, 6)])
    data_npairs = np.array([
        Table.read(
            rdir.joinpath(
                f"zbin{zbin}", 
                f"wtheta_mean_const_cov_fit{nside_fit}_nside{nside}_"
                f"v{chain_version}.fits"))["DD"].data 
        for zbin in range(1, 6)])
    lin_dvec = twopoint.TwoPointFile(
        [s if s.name != des_spec.name else _generate_spectrum_measurement(
            s, data_corr0, npairs=data_npairs) for s in des_dvec.spectra], 
        des_dvec.kernels, des_dvec.windows, cinfo_stat)
    lin_dvec.to_fits(
        odir.joinpath(
            f"2pt_NG_mcal_1110_linear_weights_const_cov_fit{nside_fit}_"
            f"nside{nside}_nreal{nsteps}_v{chain_version}.fits"), 
        overwrite=True)
    cov[cov_start:cov_end, cov_start:cov_end] += c_sys
    cinfo_sys = twopoint.CovarianceMatrixInfo(
        des_dvec.covmat_info.name, des_dvec.covmat_info.names, 
        des_dvec.covmat_info.lengths, cov)
    data_corr1 = data_corr0 - np.array([
        get_bias(
            rdir / mock_run / "contamination_0" / f"zbin{zbin}", 
            rdir / mock_run / f"contamination_{nmaps}" / f"zbin{zbin}",
            corr0_search_str, true_search_str)
        for zbin in range(1, 6)])
    db_dvec = twopoint.TwoPointFile(
        [s if s.name != des_spec.name else _generate_spectrum_measurement(
            s, data_corr1, npairs=data_npairs) for s in des_dvec.spectra], 
        des_dvec.kernels, des_dvec.windows, cinfo_stat)
    db_dvec.to_fits(
        odir.joinpath(
            f"2pt_NG_mcal_1110_linear_weights_debias_const_cov_fit{nside_fit}_"
            f"nside{nside}_nreal{nsteps}_nmocks{nmocks}_v{chain_version}.fits"), 
        overwrite=True)
    return


def wtheta_plot(des_dir, output_dir, mask, chain_version, nside_fit, nside, 
                nsteps, nmocks, theta_min=0.0, 
                des_2pt_name="2pt_NG_mcal_1110.fits"):
    """
    Make plots of the correlation functions
    
    :param des_dir: The directory containing the DES catalogs, redMaGiC mask,
        systematics mask, and fiducial data vector (in 2point 
        format). The data vector file should be in this directory, and 
        everything else is assumed to be in a subdirectory named 'redmagic'
    :type des_dir: `str` or :class:`os.PathLike`
    :param output_dir: The location in which to save the resulting plots and 
        tables. This should already exist
    :type output_dir: `str` or :class:`os.PathLike`
    :param mask: An array of the angular scales to be masked in each redshift 
        bin, in arcminutes
    :type mask: array-like `float`
    :param chain_version: The version of the chains to use
    :type chain_version: `int` or `str`
    :param nside_fit: The Nside parameter of the resolution at which the fits 
        were run
    :type nside_fit: `int`
    :param nside: The Nside parameter of the resolution at which the weights and 
        systematics mask were applied
    :type nside: `int`
    :param nsteps: The number of steps from the data chains to use for 
        estimating the statistical covariance due to the weights
    :type nsteps: `int`
    :param nmocks: The number of mock catalogs that were fit
    :type nmocks: `int`
    :param theta_min: The minimum angular separation to include in the plots, in 
        arcminutes. Default 0.0
    :type theta_min: `float`, optional
    :param des_2pt_name: The name of the 2point file containing the fiducial DES 
        data vector. Default '2pt_NG_mcal_1110.fits'
    :type des_2pt_name: `str`, optional
    """
    ddir = pathlib.Path(des_dir).expanduser().resolve()
    odir = pathlib.Path(output_dir).expanduser().resolve()
    des_spec = twopoint.TwoPointFile.from_fits(
        ddir / des_2pt_name).get_spectrum("wtheta")
    theta = des_spec.get_pair(1, 1)[0]
    theta_mask = theta >= theta_min
    theta_ = theta[theta_mask]
    db_spec = twopoint.TwoPointFile.from_fits(
        ddir.joinpath(
            f"2pt_NG_mcal_1110_linear_weights_debias_const_cov_fit{nside_fit}_"
            f"nside{nside}_nreal{nsteps}_nmocks{nmocks}_v{chain_version}"
            ".fits")).get_spectrum("wtheta")
    uw_spec = twopoint.TwoPointFile.from_fits(
        ddir / "2pt_NG_mcal_1110_uncorrected.fits").get_spectrum("wtheta")
    with plt.style.context(["paper", "colorblind", "twocol-wide"]):
        fig, axes = setup_axes_resid(
            5, r"$\theta \left[\mathrm{arcmin}\right]$", 
            r"$\theta\, w\!(\theta)$", r"Diff.", 
            sharex="all", gridspec_kw={"hspace": 0.0, "wspace": 0.0})
        for i in range(5):
            zbin = i + 1
            tax = axes[_unravel_axes_index(i, (2, 3), True, True)]
            bax = axes[_unravel_axes_index(i, (2, 3), True, False)]
            tax.set_ylim([-0.5, 2.5])
            bax.set_ylim([-0.5, 0.5])
            wy1 = des_spec.get_pair(zbin, zbin)[1][theta_mask]
            wuw = uw_spec.get_pair(zbin, zbin)[1][theta_mask]
            lu, = tax.plot(
                theta_, theta_ * wuw, "k--", alpha=0.3, zorder=1, 
                label=r"Uncorrected")
            bax.plot(theta_, theta_ * (wuw - wy1), "k--", alpha=0.3, zorder=1)
            del wuw
            ly, = tax.plot(
                theta_, theta_ * wy1, "C1-", zorder=2, label=r"Y1 Correction")
            bax.axhline(0, c="C1", ls="-", zorder=2)
            wn = db_spec.get_pair(zbin, zbin)[1][theta_mask]
            wn_err = db_spec.get_error(zbin, zbin)[theta_mask]
            ln = tax.errorbar(
                theta_, theta_ * wn, yerr=(theta_ * wn_err), fmt="C0o", 
                zorder=3, label=r"Linear+bias Correction", ms=2)
            bax.errorbar(
                theta_, theta_ * (wn - wy1), yerr=(theta_ * wn_err), fmt="C0o", 
                zorder=3, ms=2)
            del wn, wy1
            if tax.get_ylim()[0] < 0.0:
                tax.axhline(0, c="k", alpha=0.2, zorder=0)
            xlim = bax.get_xlim()
            tax.axvspan(0, mask[i], fc="k", alpha=0.1, zorder=0)
            bax.axvspan(0, mask[i], fc="k", alpha=0.1, zorder=0)
            bax.set_xlim(xlim)
            tax.legend(
                [tax.lines[0]], [fr"${zbin}, {zbin}$"], frameon=False, 
                handlelength=0.0, handletextpad=0.0)
        handles = [lu, ly, ln]
        labels = [h.get_label() for h in handles]
        axes[-1, -1].legend(
            handles, labels, loc="center", bbox_to_anchor=(0.5, 0.4))
        plt.savefig(
            odir.joinpath(
                f"wtheta_const_cov_fit{nside_fit}_nside{nside}_nreal{nsteps}"
                f"_nmocks{nmocks}_v{chain_version}.pgf"))
        plt.savefig(
            odir.joinpath(
                f"wtheta_const_cov_fit{nside_fit}_nside{nside}_nreal{nsteps}"
                f"_nmocks{nmocks}_v{chain_version}.pdf"))
        plt.savefig(
            odir.joinpath(
                f"wtheta_const_cov_fit{nside_fit}_nside{nside}_nreal{nsteps}"
                f"_nmocks{nmocks}_v{chain_version}.png"))
        plt.close()
    return


def covariance_plot(des_dir, results_root_dir, mock_run, output_dir, 
                    chain_version, nside_fit, nside, nsteps, mask, 
                    nsteps_mocks, nburnin_mocks, chain_root_dir, nmaps, 
                    theta_min=0.0, des_2pt_name="2pt_NG_mcal_1110.fits"):
    """
    Plot the diagonal elements of the ratio of the new components of the 
    covariance matrix to the fiducial Y1 covariance matrix
    
    :param des_dir: The directory containing the DES catalogs, redMaGiC mask,
        systematics mask, and fiducial data vector (in 2point 
        format). The data vector file should be in this directory, and 
        everything else is assumed to be in a subdirectory named 'redmagic'
    :type des_dir: `str` or :class:`os.PathLike`
    :param results_root_dir: The root directory containing the final results for
        both the data and the mocks
    :type results_root_dir: `str` or :class:`os.PathLike`
    :param mock_run: The name for the mock generation run, which is also the 
        name of the subdirectories containing the mock results and chains
    :type mock_run: `str`
    :param output_dir: The location in which to save the resulting plots and 
        tables. This should already exist
    :type output_dir: `str` or :class:`os.PathLike`
    :param chain_version: The version of the chains to use
    :type chain_version: `int` or `str`
    :param nside_fit: The Nside parameter of the resolution at which the fits 
        were run
    :type nside_fit: `int`
    :param nside: The Nside parameter of the resolution at which the weights and 
        systematics mask were applied
    :type nside: `int`
    :param nsteps: The number of steps from the data chains to use for 
        estimating the statistical covariance due to the weights
    :type nsteps: `int`
    :param mask: An array of the angular scales to be masked in each redshift 
        bin, in arcminutes
    :type mask: array-like `float`
    :param nsteps_mocks: The number of steps (after burn-in) that were run for 
        the mock chains
    :type nsteps_mocks: `int`
    :param nburnin_mocks: The number of steps removed as the burn-in phase for 
        the mock chains
    :type nburnin_mocks: `int`
    :param chain_root_dir: The root directory containing the chains for both the
        data and the mocks
    :type chain_root_dir: `str` or :class:`os.PathLike`
    :param nmaps: The number of systematics maps that were considered (and thus 
        the number of coefficients that were fit)
    :type nmaps: `int`
    :param theta_min: The minimum angular separation to include in the plots, in 
        arcminutes. Default 0.0
    :type theta_min: `float`, optional
    :param des_2pt_name: The name of the 2point file containing the fiducial DES 
        data vector. Default '2pt_NG_mcal_1110.fits'
    :type des_2pt_name: `str`, optional
    """
    ddir = pathlib.Path(des_dir).expanduser().resolve()
    odir = pathlib.Path(output_dir).expanduser().resolve()
    rdir = pathlib.Path(results_root_dir).expanduser().resolve()
    des_dvec = twopoint.TwoPointFile.from_fits(ddir / des_2pt_name)
    theta = des_dvec.get_spectrum("wtheta").get_pair(1, 1)[0]
    theta_mask = theta >= theta_min
    theta_ = theta[theta_mask]
    wtheta_pos = np.where(
        [n == "wtheta" for n in des_dvec.covmat_info.names])[0][0]
    cov_start = des_dvec.covmat_info.starts[wtheta_pos]
    cov_end = cov_start + des_dvec.covmat_info.lengths[wtheta_pos]
    var_des = np.diag(
        des_dvec.covmat[cov_start:cov_end, cov_start:cov_end]).reshape(
            (5, -1))[:, theta_mask]
    var_sys, nmocks = get_c_sys(
        rdir, mock_run, chain_version, nside_fit, nside, nmaps)
    var_sys = np.diagonal(var_sys, axis1=1, axis2=2)[:, theta_mask]
    var_sys_err = np.diagonal(
        get_c_sys_var(rdir, mock_run, chain_version, nside_fit, nside, nmaps), 
        axis1=1, axis2=2)[:, theta_mask]
    var_stat = np.diagonal(
        get_c_stat(
            rdir, chain_version, nside_fit, nside, nsteps, nmocks, nsteps_mocks, 
            nburnin_mocks, mock_run, chain_root_dir), 
        axis1=1, axis2=2)[:, theta_mask]
    var_stat_err = get_c_stat_var(
        rdir, chain_version, nside_fit, nside, nsteps, nmocks, nsteps_mocks, 
        nburnin_mocks, mock_run, chain_root_dir)[:, theta_mask]
    var_both_err = np.sqrt(var_sys_err + var_stat_err)
    var_both = var_sys + var_stat
    var_sys_err = np.sqrt(var_sys_err)
    var_stat_err = np.sqrt(var_stat_err)
    with plt.style.context(["paper", "colorblind", "twocol-wide"]):
        fig, axes = setup_axes_noresid(
            5, r"$\theta \left[\mathrm{arcmin}\right]$", 
            r"$\displaystyle\frac{\bm{C}_{ii}}{\bm{C}^{Y1}_{ii}}$", 
            sharex="all", sharey="all", 
            gridspec_kw={"hspace": 0.0, "wspace": 0.0})
        for i in range(5):
            ax = axes[_unravel_axes_index(i, axes.shape)]
            ft, = ax.plot(theta_, (var_stat / var_des)[i], "C1--", zorder=1)
            et = ax.fill_between(
                theta_, ((var_stat - var_stat_err) / var_des)[i], 
                ((var_stat + var_stat_err) / var_des)[i], fc="C1", alpha=0.3, 
                ec="C1", hatch="\\", zorder=1)
            fy, = ax.plot(theta_, (var_sys / var_des)[i], "C2-.", zorder=2)
            ey = ax.fill_between(
                theta_, ((var_sys - var_sys_err) / var_des)[i], 
                ((var_sys + var_sys_err) / var_des)[i], fc="C2", alpha=0.3, 
                ec="C2", hatch="/", zorder=2)
            fb, = ax.plot(theta_, (var_both / var_des)[i], "C0-", zorder=3)
            eb = ax.fill_between(
                theta_, ((var_both - var_both_err) / var_des)[i], 
                ((var_both + var_both_err) / var_des)[i], fc="C0", alpha=0.3, 
                ec="C0", hatch="|", zorder=3)
            if ax.get_ylim()[0] < 0.0:
                ax.axhline(
                    0, c="k", ls="-", lw=plt.rcParams["axes.linewidth"], 
                    alpha=0.2, zorder=0)
            xlim = ax.get_xlim()
            ax.axvspan(0, mask[i], fc="k", alpha=0.1, zorder=0)
            ax.set_xlim(xlim)
            ax.legend(
                [ax.lines[0]], [fr"${i + 1}, {i + 1}$"], frameon=False, 
                handlelength=0.0, handletextpad=0.0)
        axes[-1, -1].legend(
            [(ft, et), (fy, ey), (fb, eb)], [r"Stat", r"Sys", "Stat + Sys"], 
            loc="center", bbox_to_anchor=(0.5, 0.4))
        fig.savefig(odir.joinpath(
            f"covariance_diagonals_sys+stat_y1_mean_const_cov_fit{nside_fit}"
            f"_nside{nside}_nsteps{nsteps}_nmocks{nmocks}_v{chain_version}"
            ".pgf"))
        fig.savefig(odir.joinpath(
            f"covariance_diagonals_sys+stat_y1_mean_const_cov_fit{nside_fit}"
            f"_nside{nside}_nsteps{nsteps}_nmocks{nmocks}_v{chain_version}"
            ".pdf"))
        fig.savefig(odir.joinpath(
            f"covariance_diagonals_sys+stat_y1_mean_const_cov_fit{nside_fit}"
            f"_nside{nside}_nsteps{nsteps}_nmocks{nmocks}_v{chain_version}"
            ".png"))
        plt.close()
    return


def chi2_des_us(des_dir, chain_version, nside_fit, nside, nsteps, nmocks, 
                mask, y3=False, delta_max=0.2, new_cov_only=False, 
                des_2pt_name="2pt_NG_mcal_1110.fits"):
    """
    Get the :math:`\chi^2` between the fiducial correlation function and the one 
    with our weights in each redshift bin
    
    :param des_dir: The directory containing the DES catalogs, redMaGiC mask,
        systematics mask, and fiducial data vector (in 2point 
        format). The data vector file should be in this directory, and 
        everything else is assumed to be in a subdirectory named 'redmagic'
    :type des_dir: `str` or :class:`os.PathLike`
    :param chain_version: The version of the chains to use
    :type chain_version: `int` or `str`
    :param nside_fit: The Nside parameter of the resolution at which the fits 
        were run
    :type nside_fit: `int`
    :param nside: The Nside parameter of the resolution at which the weights and 
        systematics mask were applied
    :type nside: `int`
    :param nsteps: The number of steps from the data chains to use for 
        estimating the statistical covariance due to the weights
    :type nsteps: `int`
    :param nmocks: The number of mock catalogs that were fit
    :type nmocks: `int`
    :param mask: An array of the angular scales to be masked in each redshift 
        bin, in arcminutes
    :type mask: array-like `float`
    :param y3: If `True`, multiply the fiducial Y1 covariance matrix to forecast 
        the :math:`\chi^2` for the Y3 area rather than the Y1 area. Default 
        `False`
    :type y3: `bool`, optional
    :param delta_max: The maximum systematic overdensity that was allowed in the 
        systematics mask. Default 0.2
    :type delta_max: `float`, optional
    :param new_cov_only: If `True`, do not include the fiducial Y1 covariance 
        matrix in the calculation. Default `False`
    :type new_cov_only: `bool`, optional
    :param des_2pt_name: The name of the 2point file containing the fiducial DES 
        data vector. Default '2pt_NG_mcal_1110.fits'
    :type des_2pt_name: `str`, optional
    :return: An array of the :math:`\chi^2` in each redshift bin and an array of 
        the number of angular bins in each redshift bin
    :rtype: `tuple` of array-like `float` and array-like `int`
    """
    ddir = pathlib.Path(des_dir).expanduser().resolve()
    des_dvec = twopoint.TwoPointFile.from_fits(ddir / des_2pt_name)
    new_dvec = twopoint.TwoPointFile.from_fits(
        ddir.joinpath(
            f"2pt_NG_mcal_1110_linear_weights_debias_const_cov_fit{nside_fit}_"
            f"nside{nside}_nreal{nsteps}_nmocks{nmocks}_v{chain_version}"
            ".fits"))
    masks = dict([(
        ("wtheta", zbin, zbin), [m, np.inf]) for zbin, m in enumerate(mask, 1)])
    des_dvec.mask_scales(masks)
    new_dvec.mask_scales(masks)
    des_spec = des_dvec.get_spectrum("wtheta")
    new_spec = new_dvec.get_spectrum("wtheta")
    num_bins = np.array([des_spec.get_pair(i, i)[0].size for i in range(1, 6)])
    bin_pos = np.insert(np.cumsum(num_bins), 0, 0)
    wtheta_pos = np.where(
        [n == "wtheta" for n in des_dvec.covmat_info.names])[0][0]
    cov_start = des_dvec.covmat_info.starts[wtheta_pos]
    cov_end = cov_start + des_dvec.covmat_info.lengths[wtheta_pos]
    cov = new_dvec.covmat[cov_start:cov_end, cov_start:cov_end]
    if y3:
        fgood = hp.read_map(
            ddir.joinpath(
                "redmagic", 
                f"DES_Y1A1_3x2pt_redMaGiC_MASK_HPIX{nside}RING.fits"),     
            verbose=False, dtype=float)
        wmask = np.load(
            ddir.joinpath(
                "redmagic", 
                f"systematics_weights_mask_max{delta_max}_nside{nside}_"
                f"v{chain_version}.npy"))
        pix_area = hp.nside2pixarea(nside, degrees=True)
        area_diff = (5000. / (pix_area * fgood[wmask].sum())) - 1.
    else:
        if new_cov_only:
            area_diff = -1.
        else:
            area_diff = 0.
    cov += area_diff * des_dvec.covmat[cov_start:cov_end, cov_start:cov_end]
    chi2 = np.array([
        np.dot(
            des_spec.get_pair(i, i)[1] - new_spec.get_pair(i, i)[1], 
            np.linalg.solve(
                cov[bin_pos[i - 1]:bin_pos[i], bin_pos[i - 1]:bin_pos[i]], 
                des_spec.get_pair(i, i)[1] - new_spec.get_pair(i, i)[1])) 
        for i in range(1, 6)])
    return chi2, num_bins
    
    
def _round_to_sig_figs(num, n_sig_figs):
    if not hasattr(num, "__len__"):
        num_f, num_i = np.modf(num)
        ndigits_i = len(str(np.abs(num_i)))
        return np.around(num, n_sig_figs - ndigits_i)
    return np.array([
        _round_to_sig_figs(this_num, n_sig_figs) for this_num in num])

    
    
def chi2_table(bin_edges, des_dir, chain_version, nside_fit, nside, nsteps,     
               nmocks, mask, delta_max=0.2, 
               des_2pt_name="2pt_NG_mcal_1110.fits"):
    """
    Generate a table of :math:`\chi^2` values for the fiducial correlation 
    function and the one with our weights. This will include both the 
    :math:`\chi^2` with the full covariance matrix and the :math:`\chi^2` with 
    only the statistical and systematic covariances from our weights
    
    :param bin_edges: The redshift bin edges
    :type bin_edges: array-like `float`
    :param des_dir: The directory containing the DES catalogs, redMaGiC mask,
        systematics mask, and fiducial data vector (in 2point 
        format). The data vector file should be in this directory, and 
        everything else is assumed to be in a subdirectory named 'redmagic'
    :type des_dir: `str` or :class:`os.PathLike`
    :param chain_version: The version of the chains to use
    :type chain_version: `int` or `str`
    :param nside_fit: The Nside parameter of the resolution at which the fits 
        were run
    :type nside_fit: `int`
    :param nside: The Nside parameter of the resolution at which the weights and 
        systematics mask were applied
    :type nside: `int`
    :param nsteps: The number of steps from the data chains to use for 
        estimating the statistical covariance due to the weights
    :type nsteps: `int`
    :param nmocks: The number of mock catalogs that were fit
    :type nmocks: `int`
    :param mask: An array of the angular scales to be masked in each redshift 
        bin, in arcminutes
    :type mask: array-like `float`
    :param delta_max: The maximum systematic overdensity that was allowed in the 
        systematics mask. Default 0.2
    :type delta_max: `float`, optional
    :param des_2pt_name: The name of the 2point file containing the fiducial DES 
        data vector. Default '2pt_NG_mcal_1110.fits'
    :type des_2pt_name: `str`, optional
    :return: A LaTeX string that can be used to generate the tablular data (only 
        includes the tabular environment)
    :rtype: `str`
    """
    z_range_str = r"${edges[0]} < z < {edges[1]}$"
    chi2_ss, nbins = chi2_des_us(
        des_dir, chain_version, nside_fit, nside, nsteps, nmocks, mask, False, 
        delta_max, True, des_2pt_name)
    print(
        "\chi^2 / dof = ", ", ".join(f"{c} / {n}" for c, n in zip(
            chi2_ss, nbins)))
    chi2_tot = chi2_des_us(
        des_dir, chain_version, nside_fit, nside, nsteps, nmocks, mask, False, 
        delta_max, False, des_2pt_name)[0]
    print(
        "Y1: \chi^2 / dof = ", ", ".join(f"{c} / {n}" for c, n in zip(
            chi2_ss, nbins)))
    table_body = " \\\\\n".join([
        fr"{z_range_str.format(edges=bin_edges[i-1:i+1])} & {ct} & "
        fr" {cs} & {n}" for i, (ct, cs, n) in enumerate(
            zip(
                _round_to_sig_figs(chi2_tot, 4), 
                _round_to_sig_figs(chi2_ss, 4), nbins), 1)])
    col_names = " & ".join([
        r"$z$ range", r"$\chi^2_{\rm stat+sys}$", r"$\chi^2_{\rm tot}$", 
        r"Angular bins"])
    tabular_str = (f"\\begin{{tabular}}{{lccc}}\n\\toprule\n{col_names} "
                   f"\\\\ \\midrule\n{table_body} \\\\ "
                   f"\\bottomrule\n\\end{{tabular}}")
    return tabular_str
    

def main(des_dir, results_root_dir, chain_root_dir, sys_root_dir, output_dir, 
         chain_version, nside_fit, nside, nsteps, nmocks, nmaps, theta_min=0.0, 
         theta_min_wtheta=0.0, delta_max=0.2, nsteps_mocks=700, 
         nburnin_mocks=300, mock_run=None, 
         des_2pt_name="2pt_NG_mcal_1110.fits"):
    """
    Run all plotting routines
    
    :param des_dir: The directory containing the DES catalogs, redMaGiC mask,
        systematics mask, and fiducial data vector (in 2point 
        format). The data vector file should be in this directory, and 
        everything else is assumed to be in a subdirectory named 'redmagic'
    :type des_dir: `str` or :class:`os.PathLike`
    :param results_root_dir: The root directory containing the final results for
        both the data and the mocks
    :type results_root_dir: `str` or :class:`os.PathLike`
    :param chain_root_dir: The root directory containing the chains for both the
        data and the mocks
    :type chain_root_dir: `str` or :class:`os.PathLike`
    :param sys_root_dir: The directory containing the systematics maps. It is 
        assumed that this directory contains files with the masked, standardized 
        (and possibly rotated) systematics as an (Nmaps, Npix) array
    :type sys_root_dir: `str` or :class:`os.PathLike`
    :param output_dir: The location in which to save the resulting plots and 
        tables. This should already exist
    :type output_dir: `str` or :class:`os.PathLike`
    :param chain_version: The version of the chains to use
    :type chain_version: `int` or `str`
    :param nside_fit: The Nside parameter of the resolution at which the fits 
        were run
    :type nside_fit: `int`
    :param nside: The Nside parameter of the resolution at which the weights and 
        systematics mask were applied
    :type nside: `int`
    :param nsteps: The number of steps from the data chains to use for 
        estimating the statistical covariance due to the weights
    :type nsteps: `int`
    :param nmocks: The number of mock catalogs that were fit
    :type nmocks: `int`
    :param nmaps: The number of systematics maps that were considered (and thus 
        the number of coefficients that were fit)
    :type nmaps: `int`
    :param theta_min: The minimum angular separation to include in the bias 
        plots, in arcminutes. Default 0.0
    :type theta_min: `float`, optional
    :param theta_min_wtheta: The minimum angular separation to include in the 
        plot of the correlation function and covariance matrix terms, in 
        arcminutes. Default 0.0
    :type theta_min_wtheta: `float`, optional
    :param delta_max: The maximum systematic overdensity that was allowed in the 
        systematics mask. Default 0.2
    :type delta_max: `float`, optional
    :param nsteps_mocks: The number of steps (after burn-in) that were run for 
        the mock chains. Default 700
    :type nsteps_mocks: `int`, optional
    :param nburnin_mocks: The number of steps removed as the burn-in phase for 
        the mock chains. Default 300
    :type nburnin_mocks: `int`, optional
    :param mock_run: The name for the mock generation run, which is also the 
        name of the subdirectories containing the mock results and chains. If 
        `None` (default), this is set to 'full_mocks_v{chain_version}'
    :type mock_run: `str` or `NoneType`, optional
    :param des_2pt_name: The name of the 2point file containing the fiducial DES 
        data vector. Default '2pt_NG_mcal_1110.fits'
    :type des_2pt_name: `str`, optional
    """
    mask = [43, 27, 20, 16, 14]
    bin_edges = np.around(np.arange(15, 91, 15) / 100, 2)
    if mock_run is None:
        mock_run = f"full_mocks_v{chain_version}"
    ddir = pathlib.Path(des_dir).expanduser().resolve()
    rdir = pathlib.Path(results_root_dir).expanduser().resolve()
    odir = pathlib.Path(output_dir).expanduser().resolve()
    cdir = pathlib.Path(chain_root_dir).expanduser().resolve()
    sdir = pathlib.Path(sys_root_dir).expanduser().resolve()
    theta = twopoint.TwoPointFile.from_fits(
        ddir / des_2pt_name).get_spectrum("wtheta").get_pair(1, 1)[0]
    plot_n_of_z(
        ddir / "redmagic", chain_version, bin_edges, odir, delta_max, nside)
    map_importance_plot(
        rdir, chain_version, nside_fit, nmaps, odir, sdir, cdir, mock_run, 
        nsteps_mocks, nburnin_mocks)
    chi2_histogram(
        rdir, cdir, mock_run, chain_version, nside_fit, nsteps_mocks, 
        nburnin_mocks, odir)
    if mpl.get_backend() == "pgf":
        is_pgf = True
        mpl.use("Agg")
    else:
        is_pgf = False
    corr0_bias_plot(
        rdir, mock_run, chain_version, nside_fit, nside, nmaps, odir, mask, 
        theta, theta_min, (-0.5, 0.5))
    contaminated_bias_plot(
        rdir, mock_run, chain_version, nside_fit, nside, nmaps, odir, mask, 
        theta, theta_min, (-0.5, 2.0))
    correction_plot(
        rdir, mock_run, chain_version, nside_fit, nside, nmaps, odir, ddir, 
        mask, theta, theta_min, None, des_2pt_name)
    if is_pgf:
        mpl.use("pgf")
    save_dvec(
        ddir, rdir, mock_run, chain_version, nside_fit, nside, nsteps, nmaps, 
        nsteps_mocks, nburnin_mocks, cdir, des_2pt_name)
    wtheta_plot(
        ddir, odir, mask, chain_version, nside_fit, nside, nsteps, nmocks, 
        theta_min_wtheta, des_2pt_name)
    covariance_plot(
        ddir, rdir, mock_run, odir, chain_version, nside_fit, nside, nsteps, 
        mask, nsteps_mocks, nburnin_mocks, cdir, nmaps, theta_min_wtheta,
        des_2pt_name)
    chi2_tab = chi2_table(
        bin_edges, ddir, chain_version, nside_fit, nside, nsteps, nmocks, mask, 
        delta_max, des_2pt_name)
    odir.joinpath("wtheta_chi2_table.tex").write_text(chi2_tab)
    return


if __name__ == "__main__":
    mpl.use("pgf")
    chain_version = 6
    ddir = pathlib.Path("/spiff/wagoner47/des/y1")
    sdir = ddir / "systematics"
    rdir = pathlib.Path("/spiff/wagoner47/finalized_systematics_results_y1")
    # rdir = rdir / f"v{chain_version}_results"
    pdir = rdir.joinpath(
        "finalized_systematics_plots_y1", f"v{chain_version}_results")
    cdir = rdir.joinpath(
        "finalized_systematics_chains_y1", f"v{chain_version}_results")
    odir = rdir / f"paper_plots_v{chain_version}"
    des_2pt_name = "2pt_NG_mcal_1110.fits"
    odir.mkdir(parents=True, exist_ok=True)
    main(
        ddir, pdir, cdir, sdir, odir, chain_version, 128, 4096, 250, 100, 18, 
        8.0, 0.0, 0.2, 700, 300, des_2pt_name)