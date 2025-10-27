from typing import List, Optional, Union, Tuple
import numpy as np
import pandas as pd

PandasData = Union[pd.DataFrame, pd.Series]
Data = Union[PandasData, np.ndarray]

def _as_numpy(d: Data) -> Tuple[np.ndarray, Optional[pd.Index], Optional[pd.Index], bool]:
    if isinstance(d, pd.Series):
        arr = d.to_numpy(dtype=float, copy=False)
        return arr, d.index, None, True
    if isinstance(d, pd.DataFrame):
        arr = d.to_numpy(dtype=float, copy=False)
        return arr, d.index, d.columns, False
    # ndarray
    return np.asarray(d, dtype=float), None, None, False

def _from_numpy(arr: np.ndarray, idx: Optional[pd.Index], cols: Optional[pd.Index], was_series: bool) -> Data:
    if idx is None and cols is None:
        return arr
    if was_series:
        return pd.Series(arr, index=idx)
    return pd.DataFrame(arr, index=idx, columns=cols)

class Normalization:
    def _func(self, d: np.ndarray, axis: Optional[int]) -> np.ndarray:
        raise NotImplementedError()

    def _label_latex(self) -> str:
        raise NotImplementedError()

    def _label_no_latex(self) -> str:
        raise NotImplementedError()
    
    def _units(self, *args) -> str:
        raise NotImplementedError()

    def apply(self, d: Data, axis: Optional[int] = None) -> Data:
        arr, idx, cols, was_series = _as_numpy(d)
        out = self._func(arr, axis)
        return _from_numpy(out, idx, cols, was_series)

    def label(self, latex: bool = True) -> str:
        return self._label_latex() if latex else self._label_no_latex()
    
    def units(self, *args) -> str:
        return self._units(*args)

class L1Normalization(Normalization):
    def __init__(self, eps: Optional[float] = None):
        self.eps = eps

    def _func(self, d: np.ndarray, axis: Optional[int]) -> np.ndarray:
        denom = np.sum(np.abs(d), axis=axis, keepdims=True)
        if not self.eps:
            if np.any(denom == 0): raise ValueError("zero norm along axis")
        else:
            denom = np.maximum(denom, self.eps)
        return d / denom

    def _label_latex(self) -> str:
        return r"$L_1$ Normalization"

    def _label_no_latex(self) -> str:
        return "L1 Normalization"
    
    def _units(self, *args) -> str:
        return ""

class MinMaxNormalization(Normalization):
    def __init__(self, eps: Optional[float] = None, nan_safe: bool = False):
        self.eps = eps
        self.nan_safe = nan_safe

    def _reduce(self, f, d, axis, keepdims):
        return (np.nanmin if self.nan_safe else np.min)(d, axis=axis, keepdims=keepdims)

    def _func(self, d: np.ndarray, axis: Optional[int]) -> np.ndarray:
        mn = (np.nanmin if self.nan_safe else np.min)(d, axis=axis, keepdims=True)
        mx = (np.nanmax if self.nan_safe else np.max)(d, axis=axis, keepdims=True)
        rng = mx - mn
        if not self.eps:
            if np.any(rng == 0): raise ValueError("zero norm along axis")
        else:
            rng = np.maximum(rng, self.eps)
        return (d - mn) / rng

    def _label_latex(self) -> str:
        return "Min–Max Normalization"

    def _label_no_latex(self) -> str:
        return "Min–Max Normalization"

    def _units(self, *args) -> str:
        return "% of range"

class PDFNormalization(Normalization):
    """
    Normalize so that sum(f_i * w_i) = 1 along `axis`, i.e. f_i = d_i / (sum(d) * w_i).
    `bin_widths` must align with the dimension being normalized.
    """
    def __init__(self, bin_widths: Union[np.ndarray, List[float]], eps: Optional[float] = None):
        self.w = np.asarray(bin_widths, dtype=float)
        self.eps = eps
        if np.any(self.w <= 0):
            raise ValueError("bin widths must be positive")

    @classmethod
    def from_edges(cls, edges: Union[np.ndarray, List[float]], **kwargs):
        e = np.asarray(edges, dtype=float)
        if e.ndim != 1 or e.size < 2:
            raise ValueError("edges must be 1D with length >= 2")
        widths = np.diff(e)
        return cls(widths, **kwargs)

    def _func(self, d: np.ndarray, axis: Optional[int]) -> np.ndarray:
        # Sum along axis
        s = np.sum(d, axis=axis, keepdims=True)
        if not self.eps:
            if np.any(s == 0): raise ValueError("zero norm along axis")
        else:
            s = np.maximum(s, self.eps)

        # Prepare widths for broadcasting: put a singleton in every axis except the normalized one
        if axis is None:
            # Flatten case: require w to match d.size
            if self.w.shape != d.shape:
                raise ValueError("For axis=None, bin_widths must match data shape")
            w_b = self.w
        else:
            # Build shape like (1,1,...,W,...,1) with W in the 'axis' slot
            if d.shape[axis] != self.w.shape[0]:
                raise ValueError(f"bin_widths length ({self.w.shape[0]}) must match d.shape[axis] ({d.shape[axis]})")
            shape = [1] * d.ndim
            shape[axis] = self.w.shape[0]
            w_b = self.w.reshape(shape)

        return d / (s * w_b)

    def _infer_axis_for_pandas(self, d: PandasData, axis: Optional[int]) -> Optional[int]:
        if axis is not None:
            return axis
        if isinstance(d, pd.Series):
            if len(d) == len(self.w):
                return 0
            raise ValueError(
                f"Cannot infer axis: Series length {len(d)} != bin_widths length {len(self.w)}"
            )
        if isinstance(d, pd.DataFrame):
            candidates = [i for i, n in enumerate(d.shape) if n == len(self.w)]
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) == 0:
                raise ValueError(
                    f"Cannot infer axis: neither DataFrame axis matches bin_widths length {len(self.w)} "
                    f"(df.shape={d.shape})"
                )
            # len==2: both match; ambiguous
            raise ValueError(
                f"Ambiguous axis: both DataFrame axes match bin_widths length {len(self.w)} "
                f"(df.shape={d.shape}). Please specify axis explicitly (0 for rows, 1 for columns)."
            )
        return None  # numpy: keep existing behavior

    def apply(self, d: Data, axis: Optional[int] = None) -> Data:
        # If pandas and axis not given, try to infer it
        inferred_axis = self._infer_axis_for_pandas(d, axis) if isinstance(d, (pd.Series, pd.DataFrame)) else axis
        arr, idx, cols, was_series = _as_numpy(d)
        out = self._func(arr, inferred_axis)
        return _from_numpy(out, idx, cols, was_series)

    def _label_latex(self) -> str:
        return "PDF Normalization"

    def _label_no_latex(self) -> str:
        return "PDF Normalization"

    def units(self, input_units: str = "unit", *args, latex: bool = True) -> str:
        if latex:
            return f"$({input_units})^{{-1}}$"
        else:
            return f"1/({input_units})"

