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
pulses_re = re.compile(filename_format%"Pulses")

# Reading Functions
def read_opera_primaryraw(fp: FilePath) -> DataFrame:
    df = read_csv(fp)
    df = treat_opera_data(df)
    df['us_per_point'] = us_per_point(df)

    df.set_index(["laser", "pd0", "pd1"], inplace=True, append=True)
    return df.reorder_levels(["laser", "pd0", "pd1", "unix"])

def read_opera_secondaryraw(fp: FilePath) -> DataFrame:
    df = read_csv(fp)
    df = treat_opera_data(df)

    for col in ["sps30_pm1","sps30_pm2p5","sps30_pm4","sps30_pm10","sps30_pn0p5","sps30_pn1","sps30_pn2p5","sps30_pn4","sps30_pn10"]:
        df.loc[df[col] < 0, col] = np.nan
    for col in ["sps30_pm1","sps30_pm2p5","sps30_pm4","sps30_pm10"]:
        df.loc[df[col] > 1e5, col] = np.nan
    for col in ["sps30_pn0p5","sps30_pn1","sps30_pn2p5","sps30_pn4","sps30_pn10"]:
        df.loc[df[col] > 1e5, col] = np.nan

    return df

def read_opera_output(fp: FilePath) -> DataFrame:
    df = read_csv(fp)
    df = treat_opera_data(df)
    return df

height_scalars_integral = np.mean((lambda arr: [arr[1:], arr[:-1]])(np.array([.25, .5, .75, 1, .75, .5, .25])), axis=0)
pulse_integral = lambda pulse: np.sum((pulse['height']*height_scalars_integral)*np.diff(pulse['indices'][1:-1]))
from json import loads
def read_opera_pulses(fp: FilePath) -> DataFrame:
    df = read_csv(fp)
    df = treat_opera_data(df)
    df['raw_indices'] = df['indices']
    df['indices'] = df['indices'].map(loads).map(np.array).map(lambda arr: arr * np.array([-1,-1,-1,-1, 1, 1, 1, 1])).map(lambda arr: np.concatenate([arr[:4], [0], arr[4:]]))
    df['indices'] = df['indices'].map(lambda arr: arr + (-1*arr.min()) if arr.min() >= 0 else arr - arr.min())
    df['quarter_height'] = df['height']/4
    df['values'] = (df['raw_upper_th0'].map(lambda v: [v]) + (df['baseline0'] + df['quarter_height'].map(lambda v: (v*np.array([1, 2, 3, 4, 3, 2, 1])))).map(lambda arr: arr.round(2).tolist()) + df['raw_upper_th0'].map(lambda v: [v])).map(np.array)
    df.reset_index(inplace=True)

    edge_case = df['raw_upper_th0'] > ((df['height'] / 4) + df['baseline0'])
    df.loc[edge_case, 'indices'] = df[edge_case]['indices'].map(lambda arr: arr[[1,0,2,3,4,5,6,8,7]])
    df.loc[edge_case, 'values'] = df[edge_case]['values'].map(lambda arr: arr[[1,0,2,3,4,5,6,8,7]])
    df.set_index("unix", inplace=True)
    df['integral'] = df.apply(pulse_integral, axis=1)
    return df

from .util import read_match, re_match
read_folder_opera_primaryraw = read_match(read_opera_primaryraw, re_match(primary_raw_re))
read_folder_opera_secondaryraw = read_match(read_opera_secondaryraw, re_match(secondary_raw_re))
read_folder_opera_output = read_match(read_opera_output, re_match(output_re))
read_folder_opera_pulses = read_match(read_opera_pulses, re_match(pulses_re))


# opera_pulses_column_to_list = lambda s: json.loads(s.replace("(", "[").replace(")", "]")) if (isinstance(s, str) and s.endswith("]")) else s

# def _extract_pulses_process_row(row, columns_of_interest, pulse_columns, pulses_colname):
#     pulses = row[pulses_colname]
#     row_data = {col: row[col] for col in columns_of_interest}
    
#     # Create a DataFrame for each set of pulses in the row
#     pulse_df = pd.DataFrame(pulses, columns=pulse_columns)
#     pulse_df = pulse_df.assign(**row_data, index=row.name)
#     return pulse_df

# def extract_pulses_column(df: pd.DataFrame, pulses_colname: str = 'pulses',
#                           columns_of_interest: 'list[str]' = ['portenta', 'laser', 'pd0', 'pd1', 'hv_enabled', 'baseline0', 'baseline1', 'raw_upper_th0', 'us_per_point'],
#                           pulse_columns: 'list[str]' = ['raw_peak', 'raw_side_peak', 'indices']) -> pd.DataFrame:
#     # Apply the conversion in one go for efficiency
#     df[pulses_colname] = df[pulses_colname].map(opera_pulses_column_to_list)

#     # print(df[pulses_colname])
    
#     # Filter rows with valid pulses
#     valid_pulses_df = df[df[pulses_colname].map(lambda x: isinstance(x, list))]

#     # print(valid_df)

#     # Use joblib to parallelize the processing of each row
#     expanded_rows = Parallel(n_jobs=-1)(delayed(_extract_pulses_process_row)(row, columns_of_interest, pulse_columns, pulses_colname) 
#                                         for _, row in valid_pulses_df.iterrows())

#     if len(expanded_rows) == 0:
#         return None
#     # Concatenate all expanded rows into a single DataFrame
#     df = pd.concat(expanded_rows).reset_index().set_index("index").drop("level_0", axis=1)

#     df['height'] = df['raw_peak'] - df['baseline0']
#     df['side'] = df['raw_side_peak'] - df['baseline1']
#     df['raw_indices'] = df['indices']
#     df['indices'] = df['indices'].map(np.array).map(lambda arr: arr * np.array([-1,-1,-1,-1, 1, 1, 1, 1])).map(lambda arr: np.concatenate([arr[:4], [0], arr[4:]]))
#     df['indices'] = df['indices'].map(lambda arr: arr + (-1*arr.min()) if arr.min() >= 0 else arr - arr.min())
#     df['raw_75pc_width'] = df['indices'].map(lambda arr: arr[5] - arr[3])
#     df['raw_50pc_width'] = df['indices'].map(lambda arr: arr[6] - arr[2])
#     df['raw_25pc_width'] = df['indices'].map(lambda arr: arr[7] - arr[1])
#     df['quarter_height'] = df['height']/4
#     df['values'] = (df['raw_upper_th0'].map(lambda v: [v]) + (df['baseline0'] + df['quarter_height'].map(lambda v: (v*np.array([1, 2, 3, 4, 3, 2, 1])))).map(lambda arr: arr.round(2).tolist()) + df['raw_upper_th0'].map(lambda v: [v])).map(np.array)
#     df.reset_index(inplace=True)

#     edge_case = df['raw_upper_th0'] > ((df['height'] / 4) + df['baseline0'])
#     df.loc[edge_case, 'indices'] = df[edge_case]['indices'].map(lambda arr: arr[[1,0,2,3,4,5,6,8,7]])
#     df.loc[edge_case, 'values'] = df[edge_case]['values'].map(lambda arr: arr[[1,0,2,3,4,5,6,8,7]])
#     df['half_max_width'] = df['raw_50pc_width'] * df['us_per_point']
#     df['quarter_max_width'] = df['raw_25pc_width'] * df['us_per_point']
#     df.set_index("index", inplace=True)

#     return df