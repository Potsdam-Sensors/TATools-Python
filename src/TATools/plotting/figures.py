from .util import GraphReturn, Axes, Figure, FigSize, CyclicList
from typing import Optional, Literal, List, Union, Tuple, Any
import matplotlib.pyplot as plt

def set_yaxis_color(ax: Axes, color: str, spine: Literal['left','right'] = 'right') -> None:
    ax.tick_params(axis='y', colors=color)
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color(color)
    ax.yaxis.label.set_color(color)

def add_yaxis(ax: Axes, color: str = "black", label: Optional[str] = None, third_axis_offset_mult: float = 60) -> Axes:
    """
    Add another y-axis to the given plot. 
    If there are already two y-axes, the new axis created will have an offset, `third_axis_offset_mult`, from the right.
    """
    fig = ax.get_figure()
    if not fig:
        raise ValueError("figure for `ax` returned None")
    n_existing_axes = len(fig.get_axes())

    
    new_ax = ax.twinx()
    new_ax.spines['left'].set_visible(False)
    set_yaxis_color(new_ax, color, 'right')
    if n_existing_axes > 1:
        new_ax.spines["right"].set_position(("outward", third_axis_offset_mult*(n_existing_axes-1)))
    
    if label: new_ax.set_ylabel(label)
        
    fig.tight_layout()
    return new_ax

def multi_yaxis_figure(n_yaxes: int, figsize: FigSize = (13, 6),
                       colors: Union[str,List[str]] = CyclicList(["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]),
                       ylabels: Optional[List[str]] = None, to_plot: Optional[Any] = None) -> GraphReturn:
    fig, ax = plt.subplots(1,1, figsize=figsize)

    axes: 'list[Axes]' = [ax] + [add_yaxis(ax, colors[i]) for i in range(1,n_yaxes)]
    set_yaxis_color(ax, colors[0], 'left')

    if to_plot:
        for ax, c, data in zip(axes, CyclicList(colors), to_plot):
            ax.plot(data, color=c)
    if ylabels:
        for axis, label in zip(axes, ylabels):
            axis.set_ylabel(label)

    fig.tight_layout()

    return fig, axes

from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from typing import Dict
import pandas as pd
def plot_timespans(ax: Axes, timespans_df: pd.DataFrame, col: str, col_start: str = "Start", col_stop: str = "Stop", colors: Optional[Dict[str, str]] = None, alpha: float = .1) -> List[Rectangle]:
    if not colors:
        colors = dict([(l, c['color']) for l, c in zip(timespans_df[col].unique(), plt.rcParams['axes.prop_cycle'])])
    
    rects = []
    for _, row in timespans_df.iterrows():
        category: str = row[col]
        start, stop = row[col_start], row[col_stop]
        color = colors.get(category)
        if not color: continue

        rects.append(
            ax.axvspan(start, stop, color=color, label=category, alpha=alpha, linestyle='')
        )
    return rects

from matplotlib.artist import Artist

get_label_handle_unique = lambda artists: dict([(a.get_label(), a) for a in artists])
get_handles_unique = lambda artists: list(get_label_handle_unique(artists).values())

from typing import Iterable, Sized, List, Optional, Dict
from matplotlib.axes import Axes
from matplotlib.pyplot import Line2D
_empty_line_w_label = lambda label: Line2D([],[],linestyle="none",label=label)
def ax_legend_segmented(ax: Axes, handles: List, titles: List, make_handles_unqiue: bool = True, bold_titles: bool = True, loc: Optional[str] = None, legend_kwargs: Dict = {}): 
    assert len(handles) == len(titles), "'handles' and 'titles' must have same length"
    if len(handles) == 0: return

    if make_handles_unqiue:
        for i in range(len(handles)):
            handles[i] = get_handles_unique(handles[i])
    
    handles_ = [_empty_line_w_label(titles[0])] + handles[0]
    if len(handles) > 1:
        for t, h in zip(titles[1:],handles[1:]):
            handles_ += [_empty_line_w_label(""), _empty_line_w_label(t)] + h

    legend = ax.legend(handles=handles_, loc=loc, **legend_kwargs)
    
    if bold_titles:
        # Title rows occur exactly where we inserted `_empty_line_w_label(t)`
        # i.e., they are the rows whose label is nonempty *and* whose associated handle is empty.
        legend_handles = legend.legend_handles  # artists
        legend_texts = legend.get_texts()

        for handle, text in zip(legend_handles, legend_texts):
            if isinstance(handle, Line2D):
                # Detect section titles: these have no marker, no data, and no linestyle
                    label = text.get_text()
                    if label in titles:
                        # Bold the title
                        text.set_weight("bold")
    
    return legend

import matplotlib.pyplot as plt

def move_legend_outside_auto(
    ax,
    legend,
    *,
    pad=0.01,
    fig_pad=0.02,
    shrink_all_axes=False,
):
    """
    Move legend just outside the axes (or the whole subplot area) on the right,
    automatically shrinking the axes so the legend fits perfectly inside the figure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes containing the plot.
    legend : matplotlib.legend.Legend
        Legend to move.
    pad : float
        Padding between the right edge of the axes block and the *left* edge
        of the legend (in figure coordinates).
    fig_pad : float
        Padding between the *right* edge of the legend and the right edge
        of the figure (in figure coordinates).
    shrink_all_axes : bool
        If False (default), only shrink the given `ax` to make room for the legend.
        If True, shrink all visible axes in the figure together so that the legend
        sits to the right of the entire subplot area.

    Returns
    -------
    legend : matplotlib.legend.Legend
        The repositioned legend.
    """
    fig = ax.figure

    # Need a renderer to measure the legend.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Legend bbox in display coords -> figure coords
    leg_bbox_disp = legend.get_window_extent(renderer=renderer)
    leg_bbox_fig = leg_bbox_disp.transformed(fig.transFigure.inverted())
    leg_width = leg_bbox_fig.width

    # Where the legend's LEFT edge should go (fixed by fig_pad)
    x_leg_left = 1.0 - fig_pad - leg_width

    if shrink_all_axes:
        # Use all visible axes to define a "subplot block"
        axes = [a for a in fig.axes if a.get_visible()]

        if not axes:
            return legend

        bboxes = [a.get_position() for a in axes]

        left = min(b.x0 for b in bboxes)
        right = max(b.x0 + b.width for b in bboxes)
        total_width = right - left

        # New right edge of the axes block
        axes_right = x_leg_left - pad
        new_total_width = axes_right - left

        if new_total_width <= 0:
            # Legend too wide; bail out gracefully
            return legend

        scale = new_total_width / total_width

        # Rescale all axes horizontally, preserving relative positions
        new_bboxes = []
        for a, b in zip(axes, bboxes):
            new_x0 = left + (b.x0 - left) * scale
            new_width = b.width * scale
            a.set_position([new_x0, b.y0, new_width, b.height])
            new_bboxes.append(a.get_position())

        # Vertical center of the whole subplot block
        bottom = min(b.y0 for b in new_bboxes)
        top = max(b.y0 + b.height for b in new_bboxes)
        y_leg_center = bottom + 0.5 * (top - bottom)

    else:
        # Original behavior: only adjust this one axes
        box = ax.get_position()

        axes_right = x_leg_left - pad
        new_width = axes_right - box.x0

        if new_width <= 0:
            return legend

        ax.set_position([box.x0, box.y0, new_width, box.height])

        # Center legend next to this axes only
        y_leg_center = box.y0 + box.height * 0.5

    # Place legend centered vertically next to the chosen block
    legend.set_bbox_to_anchor(
        (x_leg_left, y_leg_center),
        transform=fig.transFigure,
    )
    legend.set_loc("center left")

    return legend
