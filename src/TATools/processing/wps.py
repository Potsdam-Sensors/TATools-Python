import pandas as pd
import re
from typing import Union
import pathlib
import numpy as np
from TATools.processing.util import read_match, match_extension
FilePath = Union[str, pathlib.Path]

wps_sample_cycle_time_re = re.compile(r"SampleCycleTime=(\d+(?:.\d+)?)\s")
wps_sample_cycle_wait_time_re = re.compile(r"SampleCycleWaitTime=(\d+(?:.\d+)?)\s")
wps_timestamp_re = re.compile(r"TimeStamp=(\w*, \w* \d\d, \d\d\d\d \d{1,2}:\d{1,2}:\d{1,2}\ [A|P]M)")
wps_data_re = re.compile(r"DataFormat:(Channel#,Lower diameter in nm,Upper diameter in nm, dN in #/cc,Corrected Counts,lower voltage,upper voltage,Raw Counts\n(?:(?:\s*\d+(?:.\d+){0,1}\s*,){7}(?:\s*\d+(?:.\d+){0,1}\s*)\s*){48})")
def read_wps(filepath: FilePath):
    with open(filepath, "r") as f:
        text = f.read()
   
    wps_timestamps = pd.to_datetime(wps_timestamp_re.findall(text), format="mixed")
    # wps_sample_cycle_time = wps_sample_cycle_time_re.findall(text)[0]
    # wps_sample_cycle_wait_time = wps_sample_cycle_wait_time_re.findall(text)[0]
    _wps_dfs = []
    for match in re.findall(wps_data_re, text):
        wps_tokens = [[y.strip() for y in x.split(",")] for x in match.splitlines()]
        _wps_dfs.append(pd.DataFrame(wps_tokens[1:], columns=wps_tokens[0]))
    
    for d, t in zip(_wps_dfs, wps_timestamps):
        d['DateTime'] = t
    wps_df = pd.concat(_wps_dfs)
    for col, ty in [("Channel#", int), ('Lower diameter in nm', float), ('Upper diameter in nm', float), ('dN in #/cc', float)]:
        wps_df[col] = wps_df[col].astype(ty)

    # wps_df['SampleCycleTime'] = wps_sample_cycle_time
    # wps_df['SampleCycleWaitTime'] = wps_sample_cycle_wait_time
    wps_df['dlogDp'] = np.log10(wps_df['Upper diameter in nm'] / wps_df['Lower diameter in nm'])
    wps_df['dN/dlogDp'] = wps_df['dN in #/cc'] / wps_df['dlogDp']
    return wps_df.set_index(["DateTime", "Channel#"])

read_folder_wps = read_match(read_wps, match_extension(".SMS"))