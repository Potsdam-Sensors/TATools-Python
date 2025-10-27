import pandas as pd
import re
from typing import Union
import pathlib
from .util import read_match, match_extension
FilePath = Union[str, pathlib.Path]

wps_timestamp_re = re.compile(r"TimeStamp=(\w*, \w* \d\d, \d\d\d\d \d{1,2}:\d{1,2}:\d{1,2}\ [A|P]M)")
wps_data_re = re.compile(r"DataFormat:(Channel#,Lower diameter in nm,Upper diameter in nm, dN in #/cc,Corrected Counts,lower voltage,upper voltage,Raw Counts\n(?:(?:\s*\d+(?:.\d+){0,1}\s*,){7}(?:\s*\d+(?:.\d+){0,1}\s*)\s*){48})")
def read_wps(filepath: FilePath):
    with open(filepath, "r") as f:
        text = f.read()
   
    wps_timestamps = pd.to_datetime(wps_timestamp_re.findall(text), format="mixed")
    _wps_dfs = []
    for match in re.findall(wps_data_re, text):
        wps_tokens = [[y.strip() for y in x.split(",")] for x in match.splitlines()]
        _wps_dfs.append(pd.DataFrame(wps_tokens[1:], columns=wps_tokens[0]))
   
    wps_df = pd.DataFrame([x['dN in #/cc'] for x in _wps_dfs]).reset_index(drop=True).set_index(wps_timestamps).astype(float)
    wps_df.columns = _wps_dfs[0]['Upper diameter in nm'].astype(float)
    return wps_df

read_folder_wps = read_match(read_wps, match_extension(".SMS"))