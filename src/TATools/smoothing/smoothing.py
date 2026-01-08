from typing import Literal, Union, Optional, List, Tuple, Callable
import numpy as np
import pandas as pd

AGG_OPTIONS = Literal["mean", "median", "mode", "sum", "std", "min", "max"]
PandasData = Union[pd.DataFrame, pd.Series]
Data = Union[PandasData, np.ndarray]

# --- helpers ---------------------------------------------------------------

def _sliding_window_view(a: np.ndarray, window: int, axis: int) -> np.ndarray:
    """Like pandas rolling(window) along axis; returns view of shape
       [..., L-window+1, window, ...]."""
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(a, window_shape=window, axis=axis)

def _pad_leading_nans(out: np.ndarray, pad: int, axis: int) -> np.ndarray:
    if pad <= 0:
        return out
    pad_shape = list(out.shape)
    pad_shape[axis] = pad
    pad_block = np.full(pad_shape, np.nan, dtype=out.dtype)
    return np.concatenate([pad_block, out], axis=axis)

def _agg_dispatch(name: AGG_OPTIONS, ddof: int = 1) -> Callable:
    if name == "mean":   return lambda x, axis: np.nanmean(x, axis=axis)
    if name == "median": return lambda x, axis: np.nanmedian(x, axis=axis)
    if name == "sum":    return lambda x, axis: np.nansum(x, axis=axis)
    if name == "std":    return lambda x, axis: np.nanstd(x, axis=axis, ddof=ddof)
    if name == "min":    return lambda x, axis: np.nanmin(x, axis=axis)
    if name == "max":    return lambda x, axis: np.nanmax(x, axis=axis)
    if name == "mode":
        # Simple mode: works best for integer/boolean data. For float data,
        # you’ll likely want to bin first. Break ties by first occurrence.
        def _mode(x, axis):
            # reshape to 2D: [N, W] then compute per-row mode
            x2 = np.moveaxis(x, axis, -1)  # shape [..., W]
            flat = x2.reshape(-1, x2.shape[-1])
            res = np.empty(flat.shape[0], dtype=flat.dtype)
            for i, row in enumerate(flat):
                # ignore NaNs for mode
                row = row[~np.isnan(row)] if row.dtype.kind in "fc" else row
                if row.size == 0:
                    res[i] = np.nan
                    continue
                if np.issubdtype(row.dtype, np.integer) or row.dtype == bool:
                    vals, counts = np.unique(row, return_counts=True)
                    res[i] = vals[np.argmax(counts)]
                else:
                    # Fallback: treat as continuous -> median as proxy
                    res[i] = np.nanmedian(row) if row.size else np.nan
            out = res.reshape(x2.shape[:-1])
            return out
        return _mode
    raise ValueError(f"Unknown agg: {name}")

def _ensure_numpy(d: Data) -> Tuple[np.ndarray, Optional[pd.Index], Optional[pd.Index], bool]:
    if isinstance(d, pd.Series):
        return d.to_numpy(dtype=float, copy=False), d.index, None, True
    if isinstance(d, pd.DataFrame):
        return d.to_numpy(dtype=float, copy=False), d.index, d.columns, False
    arr = np.asarray(d, dtype=float)
    return arr, None, None, False

def _restore_pandas(arr: np.ndarray, idx, cols, was_series: bool) -> Data:
    if idx is None and cols is None:
        return arr
    if was_series:
        # match pandas rolling: index preserved; leading NaNs allowed
        return pd.Series(arr, index=idx)
    return pd.DataFrame(arr, index=idx, columns=cols)

# --- your classes ----------------------------------------------------------

class SmoothingOp(object):
    def __init__(self, window: Union[str, int], agg: AGG_OPTIONS, *, axis: int = -1, ddof_std: int = 1):
        self.window = window
        self.agg = agg
        self.axis = axis
        self.ddof_std = ddof_std

    def _label(self) -> str:
        raise NotImplementedError()
    def label(self) -> str:
        return self._label()
    
    def _filename_label(self) -> str:
        raise NotImplementedError()
    def filename_label(self) -> str:
        return self._filename_label()

    # pandas paths
    def _grouping_pd(self, d: PandasData):
        raise NotImplementedError()
    def _agging_pd(self, grouped) -> PandasData:
        raise NotImplementedError()

    # numpy paths
    def _grouping_np(self, arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    def _agging_np(self, grouped: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def smooth_pandas(self, d: PandasData) -> PandasData:
        return self._agging_pd(self._grouping_pd(d))

    def smooth_numpy(self, d: np.ndarray) -> np.ndarray:
        return self._agging_np(self._grouping_np(d))

    def smooth(self, d: Data) -> Data:
        if isinstance(d, (pd.Series, pd.DataFrame)):
            return self.smooth_pandas(d)
        elif isinstance(d, np.ndarray):
            return self.smooth_numpy(d)
        else:
            raise TypeError("Must be pandas or numpy")

class RollingSmooth(SmoothingOp):
    def _label(self) -> str:
        return f"Rolling {self.window} {self.agg.title()}"
    def _filename_label(self) -> str:
        return f"rolling{self.agg}_{self.window}"
    
    # pandas
    def _grouping_pd(self, d: PandasData):
        # center/closed params can be added to match your needs
        if not isinstance(self.window, int) and not isinstance(self.window, str):
            raise TypeError("rolling window must be int (periods) or offset string")
        return d.rolling(self.window)
    def _agging_pd(self, grouped) -> PandasData:
        return grouped.aggregate(self.agg)

    # numpy
    def _grouping_np(self, arr: np.ndarray) -> np.ndarray:
        if not isinstance(self.window, int):
            raise TypeError("NumPy rolling requires integer window size")
        if self.window <= 0:
            raise ValueError("window must be positive")
        return _sliding_window_view(arr, self.window, axis=self.axis)

    def _agging_np(self, windows: np.ndarray) -> np.ndarray:
        # windows shape: original with one extra "window" dim at `axis+1`
        agg = _agg_dispatch(self.agg, ddof=self.ddof_std)
        # aggregate over the new trailing window dimension
        out = agg(windows, axis=self.axis + 1 if self.axis >= 0 else windows.ndim - 1)
        # pad to match pandas' leading NaNs
        return _pad_leading_nans(out, pad=self.window - 1, axis=self.axis)

class ResampleSmooth(SmoothingOp):
    def _label(self) -> str:
        return f"Aggregated {self.window} {self.agg.title()}s"
    def _filename_label(self) -> str:
        return f"resample{self.agg}_{self.window}"
    
    # pandas (needs DatetimeIndex/TimedeltaIndex/PeriodIndex)
    def _grouping_pd(self, d: PandasData):
        if not isinstance(self.window, str):
            raise TypeError("pandas resample expects a frequency string (e.g., '1S', '100ms')")
        return d.resample(self.window)

    def _agging_pd(self, grouped) -> PandasData:
        return grouped.aggregate(self.agg)

    # numpy: block grouping by a fixed integer length along axis
    def _grouping_np(self, arr: np.ndarray) -> Tuple[np.ndarray, int]:
        if not isinstance(self.window, int):
            raise TypeError("NumPy resample here groups by integer block length")
        if self.window <= 0:
            raise ValueError("window must be positive")
        axis = self.axis if self.axis >= 0 else arr.ndim + self.axis
        L = arr.shape[axis]
        n_blocks = L // self.window  # drop remainder like pandas 'label=left, closed=left' default
        if n_blocks == 0:
            # nothing to aggregate; return empty slice
            slicer = [slice(None)] * arr.ndim
            slicer[axis] = slice(0, 0)
            return arr[tuple(slicer)], 0
        # trim to multiple of window
        trim_len = n_blocks * self.window
        slicer = [slice(None)] * arr.ndim
        slicer[axis] = slice(0, trim_len)
        trimmed = arr[tuple(slicer)]
        # reshape to stack blocks
        new_shape = list(trimmed.shape)
        new_shape[axis:axis+1] = [n_blocks, self.window]
        # move axis into two dims
        trimmed = np.reshape(trimmed, new_shape)
        return trimmed, n_blocks

    def _agging_np(self, grouped_and_n: Tuple[np.ndarray, int]) -> np.ndarray:
        grouped, _ = grouped_and_n  # shape has block and window dims at `axis`..`axis+1`
        axis = self.axis if self.axis >= 0 else grouped.ndim + self.axis
        agg = _agg_dispatch(self.agg, ddof=self.ddof_std)
        # aggregate over the window dim (axis+1)
        out = agg(grouped, axis=axis + 1)
        # out shape has the block count in place of the original axis
        return out

        
class Smoothing(object):
    def __init__(self, smooths: List[SmoothingOp]):
        self.smooths = smooths
    def smooth(self, d: Data) -> Data:
        dc = d.copy()
        for sm in self.smooths:
            dc = sm.smooth(dc)
        return dc
    def label(self) -> str:
        label = self.smooths[0]._label()
        if len(self.smooths) > 1:
            for sm in self.smooths[1:]:
                label += ", " + sm._label()
        return label
    def filename_label(self) -> str:
        label = self.smooths[0].filename_label()
        if len(self.smooths) > 1:
            for sm in self.smooths[1:]:
                label += "_" + sm.filename_label()
        return label
