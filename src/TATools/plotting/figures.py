from .util import GraphReturn, Axes, Figure, FigSize
from typing import Optional, Literal, List, Union, Tuple
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

def multi_yaxis_figure(n_yaxes: int, colors: Union[str,List[str]], figsize: FigSize = (13, 6)) -> GraphReturn:
    fig, ax = plt.subplots(1,1, figsize=figsize)

    axes: 'list[Axes]' = [ax] + [add_yaxis(ax, colors[i]) for i in range(1,n_yaxes)]
    set_yaxis_color(ax, colors[0], 'left')

    return fig, axes