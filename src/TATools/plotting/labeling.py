from typing import Union, Optional, Iterable, Any
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
from tzlocal import get_localzone
from .util import decide_timezone, timezone_now

from TATools.smoothing import Normalization, Smoothing

def title_append_norm_smooth(o: Union[Axes, Figure], smoothing: Optional[Smoothing] = None, normalization: Optional[Normalization] = None,
                             label_smoothing: bool = True, label_normalization: bool = True, reverse_order: bool = False,
                             initial_sep: str = "\n", sep: str = "\n") -> None:
    """
    Add to an existing Axis or Figure title a label string with info regarding normalization and smoothing, if any.
    
    See kwargs for customization.
    """
    smoothing_str = None
    norm_str = None
    if smoothing is not None and label_smoothing:
        smoothing_str = smoothing.label()
    if normalization is not None and label_normalization:
        norm_str = normalization.label()

    strs = [norm_str, smoothing_str]
    if reverse_order:
        strs.reverse()
    
    norm_smooth_str = ""
    if strs[0] is not None or strs[1] is not None:
        norm_smooth_str += initial_sep
    if strs[0] is not None:
        norm_smooth_str += strs[0]
        if strs[1] is not None:
            norm_smooth_str += sep
    if strs[1] is not None:
        norm_smooth_str += strs[1]

    if isinstance(o, Figure):
        o.suptitle(o.get_suptitle() + norm_smooth_str)
    elif isinstance(o, Axes):
        o.set_title(o.get_title() + norm_smooth_str)
    else:
        raise ValueError("arg `o` must be Figure or Axes")    

        
def set_datetime_xaxis_format(ax: Axes, date_format: str = "%m-%d %H:%M",
                              rotation: Optional[int] = None, label_tz: Optional[Union[bool, str]] = None,
                              data_index: Optional[Iterable[Any]] = None, verbose: bool = False, try_tight_layout: bool = True) -> None:
    """
    Set the x-axis of the given Axes to a datetime format.
    Optionally rotate the x-axis labels by `rotation` degrees.
    Optionally label the x-axis with the local timezone if `label_tz` is True, or with the given string if `label_tz` is a string.

    `date_format` uses the same format codes as `datetime.strftime`. See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes for details.


    For example, "%Y-%m-%d %H:%M:%S" would format dates like "2023-03-15 14:30:00", but if you don't want the year and seconds, you could use "%m-%d %H:%M" to get "03-15 14:30".
    """
    if data_index is not None and label_tz is not True:
        raise ValueError("If `data_index` is provided, `label_tz` must be True to decide timezone from data.")
    
    ax.xaxis.set_major_formatter(DateFormatter(date_format))
    if rotation is not None:
        for label in ax.get_xticklabels():
            label.set_rotation(rotation)

    xlabel = None
    if label_tz is not None:
        if label_tz is True:
            if data_index is not None:
                xlabel = decide_timezone(data_index)
            else:
                xlabel = timezone_now()
        elif isinstance(label_tz, str):
            xlabel = label_tz
    if xlabel:
        if verbose:
            print(f"Setting x-axis label to: \"{xlabel}\"")
        ax.set_xlabel(xlabel)

    fig = ax.get_figure()
    if fig and (try_tight_layout or fig.get_tight_layout()):
        fig.tight_layout() # Call this again just to make sure things fit