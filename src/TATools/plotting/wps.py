import TATools.processing.aps as aps
from TATools.plotting.util import Axes, GraphReturn, FigSize, find_time_splits_period, find_time_splits
from TATools.plotting import title_append_norm_smooth
from TATools.smoothing import Smoothing, Normalization

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import pandas as pd
from typing import Optional, List, Union, Literal
import numpy as np
find_channel_changes = lambda df: df.index[np.argwhere(df['Lower diameter in nm'].unstack(level="Channel#").astype(float).diff().any(axis=1) | df['Upper diameter in nm'].unstack(level="Channel#").astype(float).diff().any(axis=1))]

VAR_NAME_MAP = {
    'dN in #/cc': r"dN",
    'dN/dlogDp': r"dN / dLogDp"
    }
VAR_UNIT_MAP = {
    'dN in #/cc': r"#/cc",
    'dN/dlogDp': r"#/cc"
    }
def plot_wps(df: pd.DataFrame, var: Union[Literal['dN in #/cc', 'dN/dlogDp'], str] = 'dN in #/cc', split_timedelta: Optional[str] = "4min", channels: List[float] = None,
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


    indices = np.unique(np.stack(df.index)[:, 0])
    ts = np.argwhere(np.diff(indices) > pd.Timedelta(split_timedelta))[0]
    split_data = [df.loc[sidx] for sidx in np.split(indices, ts)]

    vmin = None
    vmax = None
    m = None
    for i in range(len(split_data)):
        channels = split_data[i]['Lower diameter in nm'].unstack(level="Channel#").columns.astype(float)
        bounds = np.concatenate([split_data[i]['Lower diameter in nm'].unstack("Channel#").iloc[0].values, [split_data[i]['Upper diameter in nm'].unstack("Channel#").iloc[0].values[-1]]])

        to_plot = split_data[i][var].unstack(level="Channel#")
        if normalization:
            to_plot = normalization.apply(to_plot)
        if smoothing:
            to_plot = smoothing.smooth(to_plot)
        
        dmin = to_plot.replace(0, np.NAN).min(axis=None)
        dmax = to_plot.max(axis=None)
        if vmin is None: vmin = dmin
        else: vmin = min([vmin, dmin])
        if vmax is None: vmax = dmax
        else: vmax = max([vmax, dmax])

        split_data[i] = to_plot, channels, bounds

    assert isinstance(vmax, float) and isinstance(vmin, float)
    if cnorm and isinstance(cnorm, LogNorm) and (not cnorm.vmin and not cnorm.vmax): cnorm = LogNorm(vmin=vmin, vmax=vmax)

    for to_plot, channels, bounds in split_data:
        if not use_nm: # WPS is in nm already, so switch to um if needed
            bounds = np.array(bounds)/1000
    
        period = to_plot.index.diff(1).median()
        x = pd.DatetimeIndex(np.concatenate([to_plot.index, [to_plot.index[-1]+period]]))
        c = to_plot[channels]
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
        auto_label = (lambda st: VAR_UNIT_MAP.get(st) or st)(var)
        if normalization:
            auto_label = normalization.units(auto_label.replace("$", ""))
            print(auto_label)
        fig.colorbar(m, ax=ax, label=cbar_label if cbar_label != "auto" else auto_label)
    if logscale_y:
        ax.set_yscale("log")
    
    if title and "%s" in title:
        title = title%(lambda st: VAR_NAME_MAP.get(st) or st)(var)
    ax.set_title(title or "")
    title_append_norm_smooth(ax, smoothing, normalization, do_title_addons, do_title_addons)

    if ylabel:
        if "%s" in ylabel:
            ylabel = ylabel%("nm" if use_nm else "$\\mu m$")
        ax.set_ylabel(ylabel)

    if new_fig: fig.tight_layout()
    return fig, ax