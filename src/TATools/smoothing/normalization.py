from typing import List, Optional, Union, Tuple
import numpy as np
import pandas as pd

PandasData = Union[pd.DataFrame, pd.Series]
Data = Union[PandasData, np.ndarray]
_label_float = lambda f: f"{f}".replace(".","p")

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
    
    def _filename_label(self) -> str:
        raise NotImplementedError()
    
    def _units(self, *args) -> str:
        raise NotImplementedError()

    def apply(self, d: Data, axis: Optional[int] = None) -> Data:
        arr, idx, cols, was_series = _as_numpy(d)
        out = self._func(arr, axis)
        return _from_numpy(out, idx, cols, was_series)

    def label(self, latex: bool = True) -> str:
        return self._label_latex() if latex else self._label_no_latex()

    def filename_label(self) -> str:
        return self._filename_label()
    
    def units(self, *args) -> str:
        return self._units(*args)

class L1Normalization(Normalization):
    def __init__(self, eps: Optional[float] = None, zero_norm_nan: bool = True):
        self.eps = eps
        self.zero_norm_nan = zero_norm_nan

    # def _func(self, d: np.ndarray, axis: Optional[int]) -> np.ndarray:
    #     denom = np.sum(np.abs(d), axis=axis, keepdims=True)
    #     if not self.eps:
    #         if np.any(denom == 0): raise ValueError("zero norm along axis")
    #     else:
    #         denom = np.maximum(denom, self.eps)
    #     return d / denom
    def _func(self, d: np.ndarray, axis: Optional[int]) -> np.ndarray:
        denom = np.sum(np.abs(d), axis=axis, keepdims=True)

        if not self.eps:
            zero_mask = (denom == 0)
            if np.any(zero_mask):
                if not getattr(self, "zero_norm_nan", False):
                    raise ValueError("zero norm along axis")

                # Replace zero denominators with NaN to localize failure
                denom = np.where(zero_mask, np.nan, denom)
        else:
            denom = np.maximum(denom, self.eps)

        return d / denom

    
    def _filename_label(self) -> str:
        return "L1Normalized" + (f"_eps{_label_float(self.eps)}" if self.eps else "")

    def _label_latex(self) -> str:
        return r"$L_1$ Normalization"

    def _label_no_latex(self) -> str:
        return "L1 Normalization"
    
    def _units(self, *args) -> str:
        return ""

class MinMaxNormalization(Normalization):
    def __init__(self, eps: Optional[float] = None, reference_level = 1.0, nan_safe: bool = False, zero_norm_nan: bool = True):
        self.eps = eps
        self.reference_level = reference_level
        self.nan_safe = nan_safe
        self.zero_norm_nan = zero_norm_nan

    def _reduce(self, f, d, axis, keepdims):
        return (np.nanmin if self.nan_safe else np.min)(d, axis=axis, keepdims=keepdims)

    # def _func(self, d: np.ndarray, axis: Optional[int]) -> np.ndarray:
    #     mn = (np.nanmin if self.nan_safe else np.min)(d, axis=axis, keepdims=True)
    #     mx = (np.nanmax if self.nan_safe else np.max)(d, axis=axis, keepdims=True)
    #     if self.reference_level != 1.0:
    #         mx = mx * self.reference_level
    #     rng = mx - mn
    #     if not self.eps:
    #         if np.any(rng == 0): raise ValueError("zero norm along axis")
    #     else:
    #         rng = np.maximum(rng, self.eps)
    #     return (d - mn) / rng

    def _func(self, d: np.ndarray, axis: Optional[int]) -> np.ndarray:
        mn = (np.nanmin if self.nan_safe else np.min)(d, axis=axis, keepdims=True)
        mx = (np.nanmax if self.nan_safe else np.max)(d, axis=axis, keepdims=True)

        if self.reference_level != 1.0:
            mx = mx * self.reference_level

        rng = mx - mn

        # Handle zero-range slices
        if not self.eps:
            zero_mask = (rng == 0)
            if np.any(zero_mask):
                if not getattr(self, "zero_norm_nan", False):
                    raise ValueError("zero norm along axis")

                # Avoid divide-by-zero warnings
                rng = np.where(zero_mask, np.nan, rng)
        else:
            rng = np.maximum(rng, self.eps)

        out = (d - mn) / rng

        return out


    
    def _filename_label(self) -> str:
        return "MinMaxNormalized" \
            + (f"_reflvl{_label_float(self.reference_level)}" if self.reference_level else "") \
            + (f"_eps{_label_float(self.eps)}" if self.eps else "")
    
    def _label_latex(self) -> str:
        return self._label_no_latex()

    def _label_no_latex(self) -> str:
        st = "Min-Max Normalization"
        if self.reference_level != 1.0:
            st += f" [ref. level = {self.reference_level}x peak]"
        return st

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

    def _filename_label(self) -> str:
        return "PDFNormalized" \
            + (f"_eps{_label_float(self.eps)}" if self.eps else "")

    def _label_latex(self) -> str:
        return "PDF Normalization"

    def _label_no_latex(self) -> str:
        return "PDF Normalization"

    def units(self, input_units: str = "unit", *args, latex: bool = True) -> str:
        if latex:
            return f"$({input_units})^{{-1}}$"
        else:
            return f"1/({input_units})"


class QuantileMinMaxNormalization(Normalization):
    """
    Robust Min-Max normalization using quantiles instead of the true min/max.

    For each slice along `axis`:
        lo = quantile(d, q_low)
        hi = quantile(d, q_high)
        y  = (d - lo) / (hi - lo)

    Notes:
    - If `clip=True`, outputs are clipped to [0, 1]. Otherwise, values can be
      <0 or >1 when data fall outside [lo, hi].
    - If `nan_safe=True`, NaNs are ignored when computing quantiles.
    """

    def __init__(
        self,
        q_high: float = 0.99,
        q_low: float = 0.00,
        eps: Optional[float] = None,
        clip: bool = False,
        nan_safe: bool = False,
        zero_norm_nan: bool = True,
    ):
        if not (0.0 <= q_low < q_high <= 1.0):
            raise ValueError(f"require 0 <= q_low < q_high <= 1; got {q_low=}, {q_high=}")
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.eps = eps
        self.clip = clip
        self.nan_safe = nan_safe
        self.zero_norm_nan = zero_norm_nan
    
    # def _func(self, d: np.ndarray, axis: Optional[int]) -> np.ndarray:
    #     if axis is None:
    #         # Flatten semantics: compute quantiles over all elements
    #         flat = d.reshape(-1)
    #         qfunc = np.nanquantile if self.nan_safe else np.quantile
    #         lo = qfunc(flat, self.q_low)
    #         hi = qfunc(flat, self.q_high)
    #         rng = hi - lo
    #         if not self.eps:
    #             if rng == 0:
    #                 raise ValueError("zero norm (hi - lo == 0)")
    #         else:
    #             rng = max(rng, self.eps)
    #         out = (d - lo) / rng
    #     else:
    #         qfunc = np.nanquantile if self.nan_safe else np.quantile
    #         lo = qfunc(d, self.q_low, axis=axis, keepdims=True)
    #         hi = qfunc(d, self.q_high, axis=axis, keepdims=True)
    #         rng = hi - lo
    #         if not self.eps:
    #             if np.any(rng == 0):
    #                 raise ValueError("zero norm along axis (hi - lo == 0)")
    #         else:
    #             rng = np.maximum(rng, self.eps)
    #         out = (d - lo) / rng

    #     if self.clip:
    #         out = np.clip(out, 0.0, 1.0)
    #     return out

    def _func(self, d: np.ndarray, axis: Optional[int]) -> np.ndarray:
        qfunc = np.nanquantile if self.nan_safe else np.quantile

        if axis is None:
            # Flatten semantics: compute quantiles over all elements
            flat = d.reshape(-1)
            lo = qfunc(flat, self.q_low)
            hi = qfunc(flat, self.q_high)
            rng = hi - lo

            if not self.eps:
                if rng == 0:
                    if not getattr(self, "zero_norm_nan", False):
                        raise ValueError("zero norm (hi - lo == 0)")
                    rng = np.nan  # makes whole output NaN (since it's a single global range)
            else:
                rng = max(rng, self.eps)

            out = (d - lo) / rng

        else:
            lo = qfunc(d, self.q_low, axis=axis, keepdims=True)
            hi = qfunc(d, self.q_high, axis=axis, keepdims=True)
            rng = hi - lo

            if not self.eps:
                zero_mask = (rng == 0)
                if np.any(zero_mask):
                    if not getattr(self, "zero_norm_nan", False):
                        raise ValueError("zero norm along axis (hi - lo == 0)")
                    # Localize failure: only affected slices become NaN
                    rng = np.where(zero_mask, np.nan, rng)
            else:
                rng = np.maximum(rng, self.eps)

            out = (d - lo) / rng

        if self.clip:
            out = np.clip(out, 0.0, 1.0)
        return out


    def _filename_label(self) -> str:
        return "QuantileMinMaxNormalized" \
            + (f"_ql{_label_float(self.q_low)}" if self.q_low else "") \
            + (f"_qh{_label_float(self.q_high)}" if self.q_high else "") \
            + (f"_eps{_label_float(self.eps)}" if self.eps else "")
    
    def _label_latex(self) -> str:
        # Keep it simple; you can make this fancier if you want.
        return self._label_no_latex()

    def _label_no_latex(self) -> str:
        st = f"Quantile Min-Max Normalization [q=({self.q_low:g}, {self.q_high:g})]"
        if self.clip:
            st += " [clipped]"
        if self.nan_safe:
            st += " [nan-safe]"
        return st

    def _units(self, *args) -> str:
        return "% of robust range"
