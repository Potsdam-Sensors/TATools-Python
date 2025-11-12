# Processing Functions
from pandas import DataFrame, to_numeric
import numpy as np
from typing import List

def coerce_or_drop(df: DataFrame, target_columns: List[str], data_type, verbose=True) -> DataFrame:
    for col in target_columns:
        len_before = len(df)

        # Coerce column values to the desired type and set invalid ones to NaN
        if data_type in [int, float]:
            df[col] = to_numeric(df[col], errors='coerce')  # Coerce to numeric, set invalid values to NaN
        else:
            try:
                df[col] = df[col].astype(data_type, errors='ignore')  # Attempt coercion
                df[col] = df[col].map(lambda v: v if isinstance(v, data_type) else np.NaN)  # Handle non-matching values
            except Exception as e:
                raise ValueError(f"Failed to coerce column '{col}' to {data_type}: {e}")

        # Drop rows with NaN in this column
        df = df[df[col].notna()]
        len_after = len(df)
        num_dropped = len_before - len_after

        # Verbose output
        if verbose and num_dropped > 0:
            print(f"Dropped {num_dropped} rows after coercing column \"{col}\" to {data_type}.")
            if len_after == 0:
                print("DataFrame now has 0 rows.")
    return df

remove_rows_with_nan = lambda df, col_name: df.loc[df[col_name].notna()]
us_per_point = lambda df: (df['ms_read'] * 1000) / (df['num_buff']*3500) 


from TATools.processing import FilePath

from typing import Union, Optional, Iterable, Any
from pandas import DataFrame, read_csv, to_datetime
from tzlocal import get_localzone
from zoneinfo import ZoneInfo

def treat_opera_data(df: DataFrame, target_tz: Optional[Union[str, ZoneInfo]] = get_localzone(),
                     required_not_na_colname: Optional[Union[Iterable[Any], Any]] = None,
                     unix_column: str = "unix", verbose: bool = False) -> DataFrame:
    df = df.copy()

    # Trim where no unix
    len_before = len(df)
    df = df[df[unix_column].notna()]
    len_after = len(df)
    trimmed = len_before - len_after
    if verbose and trimmed: print(f"Trimmed {trimmed} rows due to NaN unix col, \"{unix_column}\".")

    # Trim where invalid unix
    df = coerce_or_drop(df, [unix_column], int, verbose=verbose) # Type validation, prints trimmed already

    len_before = len(df)
    df = df[df[unix_column] < 1920836990] # Just a backup in case data really wrong
    len_after = len_after - len(df)
    if verbose and trimmed: print(f"Trimmed {trimmed} rows due to too-high unix value in col \"{unix_column}\".")

    # Convert to datetime
    df[unix_column] = to_datetime(df[unix_column], unit='s', utc=True).dt.tz_convert(target_tz)

    # Trim rows where required not NA columns are NAN
    if required_not_na_colname and required_not_na_colname in df.columns:
        if not isinstance(required_not_na_colname, Iterable): required_not_na_colname = [required_not_na_colname]

        for c in required_not_na_colname:
            len_before = len(df)
            df = remove_rows_with_nan(df, c)
            len_after = len(df)
            trimmed = len_before-len_after
            if verbose and trimmed: print(f"Removed {trimmed} NA rows using column, \"{required_not_na_colname}\".")

    return df.set_index(unix_column).sort_index()

# Matching Functions
import re
filename_format = 'OPERA_(\\d|\\w+)_%s_(\\d{8})(?:.raw)?.csv'
secondary_re = re.compile(filename_format%"Secondary")
raw_re = re.compile(filename_format%"Raw")
housekeeping_re = re.compile(filename_format%"Housekeeping")
primary_raw_re = re.compile(filename_format%'PrimaryRaw')
secondary_raw_re = re.compile(filename_format%'SecondaryRaw')
output_re = re.compile(filename_format%"Output")

# Reading Functions
def read_opera_primaryraw(fp: FilePath) -> DataFrame:
    df = read_csv(fp)
    df = treat_opera_data(df)
    df['us_per_point'] = us_per_point(df)
    return df

def read_opera_secondaryraw(fp: FilePath) -> DataFrame:
    df = read_csv(fp)
    df = treat_opera_data(df)

    for col in ["sps30_pm1","sps30_pm2p5","sps30_pm4","sps30_pm10","sps30_pn0p5","sps30_pn1","sps30_pn2p5","sps30_pn4","sps30_pn10"]:
        df.loc[df[col] < 0, col] = np.nan
    for col in ["sps30_pm1","sps30_pm2p5","sps30_pm4","sps30_pm10"]:
        df.loc[df[col] > 1e20, col] = np.nan
    for col in ["sps30_pn0p5","sps30_pn1","sps30_pn2p5","sps30_pn4","sps30_pn10"]:
        df.loc[df[col] > 1e20, col] = np.nan

    return df

def read_opera_output(fp: FilePath) -> DataFrame:
    df = read_csv(fp)
    df = treat_opera_data(df)
    return df

from .util import read_match, re_match
read_folder_opera_primaryraw = read_match(read_opera_primaryraw, re_match(primary_raw_re))
read_folder_opera_secondaryraw = read_match(read_opera_secondaryraw, re_match(secondary_raw_re))
read_folder_opera_output = read_match(read_opera_output, re_match(output_re))

