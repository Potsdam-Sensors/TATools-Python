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

def read_match(f: ReadFunc, m: MatchFunc, concat_kwargs: Dict = {}) -> Callable[[List[FilePath]], pd.DataFrame]:
    return lambda dir: read_multiple(f, concat_kwargs)(matchdir(m)(dir))

def re_match(regexp: Union[re.Pattern, str]) -> MatchFunc:
    if isinstance(regexp, str):
        regexp = re.compile(regexp)
    return lambda st: bool(regexp.match(st))

match_extension = lambda ext: lambda fp: pathlib.Path(fp).suffix == ext