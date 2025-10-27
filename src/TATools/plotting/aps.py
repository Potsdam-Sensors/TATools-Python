import TATools.processing.aps as aps
from TATools.plotting.util import Axes, GraphReturn, FigSize, find_time_splits_period
from TATools.plotting import title_append_norm_smooth
from TATools.smoothing import Smoothing, Normalization

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import pandas as pd
from typing import Optional, List
import numpy as np



VAR_NAME_MAP = {"dN/dlogDp": r"$dN / dlogD_p$"}
VAR_UNIT_MAP = {"dN/dlogDp": r"$\#/cc$"}
def plot_aps(df: pd.DataFrame, channels: List[str] = aps.APS_BIN_HEADERS, bounds = aps.aps_bin_boundaries(),
             normalization: Optional[Normalization] = None, smoothing: Optional[Smoothing] = None,
             do_cbar: bool = True, cbar_label: Optional[str] = "auto", cnorm: Optional[Normalize] = LogNorm(),
             logscale_y: bool = True, title: Optional[str] = "APS %s", do_title_addons: bool = True,
             ylabel: str = "$D_p$ (%s)", use_nm: bool = True,
             ax: Optional[Axes] = None, figsize: FigSize = (13, 5)) -> GraphReturn:
    fig = None
    new_fig = ax is None
    if not new_fig:
        fig = ax.get_figure()
        assert fig != None
    else:
        fig, ax = plt.subplots(1,1, figsize=figsize)

    ts = find_time_splits_period(df.index, df['Sample Period'])
    split_data = np.split(df, ts) if len(ts) else [df]

    vmin = None
    vmax = None
    for i in range(len(split_data)):
        if normalization:
            split_data[i][channels] = normalization.apply(split_data[i][channels])
        if smoothing:
            split_data[i][channels] = smoothing.smooth(split_data[i][channels])
        
        dmin = split_data[i][channels].replace(0, np.NAN).min(axis=None)
        dmax = split_data[i][channels].max(axis=None)
        if vmin is None: vmin = dmin
        else: vmin = min([vmin, dmin])
        if vmax is None: vmax = dmax
        else: vmax = max([vmax, dmax])
    assert isinstance(vmax, float) and isinstance(vmin, float)
    if cnorm and isinstance(cnorm, LogNorm) and (not cnorm.vmin and not cnorm.vmax): cnorm = LogNorm(vmin=vmin, vmax=vmax)

    if use_nm:
        bounds = np.array(bounds)*1000
    
    m = None
    for d in split_data:
        x = pd.DatetimeIndex(np.concatenate([d.index, [d.index[-1]+d['Sample Period'].iloc[-1]]]))
        c = d[channels]
        if normalization:
            c = normalization.apply(c)
        if smoothing:
            c = smoothing.smooth(c)
        m = ax.pcolormesh(
            x,
            bounds,
            c.transpose(),
            norm=cnorm
        )
    
    if do_cbar and m:
        auto_label = (lambda st: VAR_UNIT_MAP.get(st) or st)(df['Aerodynamic Diameter'].iloc[0])
        if normalization:
            auto_label = normalization.units(auto_label.replace("$", ""))
            print(auto_label)
        fig.colorbar(m, ax=ax, label=cbar_label if cbar_label != "auto" else auto_label)
    if logscale_y:
        ax.set_yscale("log")
    
    if title and "%s" in title:
        title = title%(lambda st: VAR_NAME_MAP.get(st) or st)(df['Aerodynamic Diameter'].iloc[0])
    ax.set_title(title or "")
    title_append_norm_smooth(ax, smoothing, normalization, do_title_addons, do_title_addons)

    if ylabel:
        if "%s" in ylabel:
            ylabel = ylabel%("nm" if use_nm else "$\\mu m$")
        ax.set_ylabel(ylabel)

    if new_fig: fig.tight_layout()
    return fig, ax