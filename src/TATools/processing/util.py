import os
import pathlib
from typing import Union, Callable, List, Dict
import pandas as pd
import re
FilePath = Union[pathlib.Path, str]
ReadFunc = Callable[[FilePath], pd.DataFrame]
MatchFunc = Callable[[FilePath], bool]

listdir = lambda d: [pathlib.Path(d).joinpath(x) for x in os.listdir(d)]
def matchdir(f: MatchFunc) -> Callable[[FilePath], List[FilePath]]:
    return lambda d: [pathlib.Path(d).joinpath(fp) for fp in os.listdir(d) if f(fp)]

def read_multiple_(filepaths: List[FilePath], f: ReadFunc, concat_kwargs: Dict = {}) -> pd.DataFrame:
    return pd.concat(
        [f(fp) for fp in filepaths],
        **concat_kwargs
    )
def read_multiple(f: ReadFunc, concat_kwargs: Dict = {}) -> Callable[[List[FilePath]], pd.DataFrame]:
    return lambda filepaths: read_multiple_(filepaths, f, concat_kwargs)

def read_match(f: ReadFunc, m: MatchFunc, concat_kwargs: Dict = {}) -> Callable[[FilePath], pd.DataFrame]:
    return lambda dir: read_multiple(f, concat_kwargs)(matchdir(m)(dir)).sort_index()

def re_match(regexp: Union[re.Pattern, str]) -> MatchFunc:
    if isinstance(regexp, str):
        regexp = re.compile(regexp)
    return lambda st: bool(regexp.match(st))

match_extension = lambda ext: lambda fp: pathlib.Path(fp).suffix == ext

# Timestamps
from typing import Iterable, Union, Optional, Dict, Callable, Any
from numbers import Number
from pandas import Timestamp
from zoneinfo import ZoneInfo
import datetime
import numpy as np
import numpy.typing as npt
from tzlocal import get_localzone
TimestampLike = Union[str, Number, Timestamp, datetime.datetime]
TimezoneLike = Union[str, ZoneInfo]
_ts_type_convmap: Dict[Any, Callable[[Union[TimestampLike, Iterable[TimestampLike]]], Union[Timestamp, npt.NDArray[np.datetime64]]]] = {
    Number: lambda ts: pd.to_datetime(ts, unit='s', utc=True),
    str: pd.to_datetime,
    Timestamp: None,
}
def to_timestamp(ts: Union[TimestampLike, Iterable[TimestampLike]], tz: Optional[ZoneInfo] = get_localzone()) -> Union[Timestamp, npt.NDArray[np.datetime64]]:
    # Check for iterable, deal with that case and convert to numpy
    is_iterable = isinstance(ts, Iterable)
    if is_iterable and len(ts) == 0: return ts
    if is_iterable and not isinstance(ts, np.ndarray):
        ts = np.array(ts)

    # Decide what conversion function we need, if any
    sample_elem = ts if not is_iterable else ts[0]
    conv_func: Callable[[Union[TimestampLike, Iterable[TimestampLike]]], Union[Timestamp, npt.NDArray[np.datetime64]]] = pd.to_datetime
    for t, f in _ts_type_convmap.items():
        if isinstance(sample_elem, t):
            conv_func = f
            break
    
    # Convert ts to Timestamp, the conv_func options should all be vectorized
    if conv_func:
        ts: npt.NDArray[np.datetime64] = conv_func(ts)
    
    # Convert timestamp timezone
    sample_elem: Timestamp = ts if not is_iterable else ts[0]
    tz_aware = bool(sample_elem.tz)
    if tz:
        if tz_aware:
            ts = ts.tz_convert(tz)
        else:
            ts = ts.tz_localize(tz)

    return ts

# Applying timerange data
from typing import Iterable, Optional, Any

def apply_timerange_column(target_df: pd.DataFrame, 
                           time_df: pd.DataFrame, 
                           col_start, 
                           col_end, 
                           col_apply,
                           use_column_index: Optional[Any] = None):
    """
    Notes:
    * `target_df` must be indexed on datetime or provide `use_column_index`.
    * Columns in `time_df` given by `col_start` and `col_end` must be a datetime type and not be index columns.
    """

    # Treat strings as scalars, not iterables
    is_multi = isinstance(col_apply, Iterable) and not isinstance(col_apply, (str, bytes))

    # Fix index, re-setting temporarily if needed
    index_names = target_df.index.names
    if use_column_index is not None:
        target_df.reset_index(inplace=True)
        target_df.set_index(use_column_index, inplace=True)
    elif len(index_names) > 1:
        raise ValueError("When arg 'target_df' is indexed on multiple columns, optional arg `use_column_index` must be provided.")
    target_df.sort_index(inplace=True)

    for _, row in time_df.iterrows():
        start, end = row[col_start], row[col_end]

        if is_multi:
            # Set same value across multiple columns
            for c in col_apply:
                target_df.loc[start:end, c] = row[c]
        else:
            # Normal single-column case
            target_df.loc[start:end, col_apply] = row[col_apply]

    # Put back original index if needed
    if use_column_index is not None:
        target_df.reset_index(inplace=True)
        target_df.set_index(index_names, inplace=True)
