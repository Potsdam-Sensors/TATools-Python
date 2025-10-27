from typing import Union, Optional
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .normalization import Normalization
from .smoothing import Smoothing

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

        
            
        

