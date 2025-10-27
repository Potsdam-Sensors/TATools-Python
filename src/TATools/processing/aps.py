import pandas as pd
from typing import Union, Tuple, Dict
import numpy as np
import pathlib

FilePath = Union[str, pathlib.Path]
APS_BIN_HEADERS = ['<0.523', '0.542', '0.583', '0.626', '0.673', '0.723', '0.777', '0.835',
       '0.898', '0.965', '1.037', '1.114', '1.197', '1.286', '1.382', '1.486',
       '1.596', '1.715', '1.843', '1.981', '2.129', '2.288', '2.458', '2.642',
       '2.839', '3.051', '3.278', '3.523', '3.786', '4.068', '4.371', '4.698',
       '5.048', '5.425', '5.829', '6.264', '6.732', '7.234', '7.774', '8.354',
       '8.977', '9.647', '10.37', '11.14', '11.97', '12.86', '13.82', '14.86',
       '15.96', '17.15', '18.43', '19.81']

_APS_STAT_HEADERS = ['Median(um)', 'Mode(um)', 'Mean(um)', 'Geo. Std. Dev.', 'Geo. Mean(um)']
def read_aps(filepath: FilePath) -> Tuple[pd.DataFrame, Dict]:
    with open(filepath, "rb") as f:
        d = f.read().replace(b'\xef\xbf\xbdm', b'um'
                             ).replace(bytes([0xb5]),b"u"
                                       ).replace(bytes([0xb3]), b"3"
                                                 ).replace(b'cm\xef\xbf\xbd', b'cm3'
                                                           ).decode()

    lines = d.splitlines()
    lens = [len(l.split("\t")) for l in lines]
    lens_u = np.unique(lens)
    lens_u.sort()
    assert len(lens_u) == 2, "unexpected format, expected only two row widths"

    header = [line for line, l in zip(lines, lens) if l == lens_u[0]]
    header = dict([h.split("\t") for h in header])
    header['Lower Channel Bound'] = float(header['Lower Channel Bound'])
    header['Upper Channel Bound'] = float(header['Upper Channel Bound'])

    data = [line.split("\t") for line, l in zip(lines, lens) if l == lens_u[1]]

    df = pd.DataFrame(data[1:], columns=data[0])
    df['DateTime'] = pd.to_datetime(df['Date'] + " " + df['Start Time'], format="%m/%d/%y %H:%M:%S")
    df['Total Conc.'] = df['Total Conc.'].map(lambda s: float(s.split("(")[0]))
    df[APS_BIN_HEADERS] = df[APS_BIN_HEADERS].astype(float)
    df[_APS_STAT_HEADERS] = df[_APS_STAT_HEADERS].astype(float)
    df.drop(columns=["Date", "Start Time"], inplace=True)
    if len(df['Aerodynamic Diameter'].unique()) > 1:
        df = df.set_index(['Aerodynamic Diameter', 'DateTime']).unstack('Aerodynamic Diameter').swaplevel(0, axis=1)
    else:
        df.set_index("DateTime", inplace=True)
        
    return df, header