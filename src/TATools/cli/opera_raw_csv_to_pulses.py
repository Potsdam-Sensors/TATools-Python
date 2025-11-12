import sys
import pandas as pd
import os
import json

pulse_csv_headers = "unix_timestamp,portenta_serial,pd0,pd1,laser,raw_scalar0,raw_scalar1,diff_scalar0,diff_scalar1,baseline0,baseline1,max_laser_on,height,side_height,raw_width_75pc,raw_width_50pc,raw_width_25pc\n"

def __main__():
    # Get cmd line args, if no args, print usage, if 3 args use 2nd as input file and 3rd as output file
    if len(sys.argv) < 2:
        print("Usage: python opera_raw_csv_to_pulses.py <input_file.raw> [output_file.csv]")
        return
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None

    raw_df = pd.read_csv(input_file)
    output_filename = output_file or os.path.splitext(input_file)[0] + "_pulses.csv"

    if not os.path.exists(output_filename):
        with open(output_filename, "w") as f:
            f.write(pulse_csv_headers)

    for _, row in raw_df.iterrows():
        with open(output_filename, "a+") as f:
            unix_timestamp = row['unix']
            portenta_serial = row['portenta']
            pd0 = row['pd0']; pd1 = row['pd1']; laser = row['laser']
            raw_scalar0 = row['raw_scalar0']; raw_scalar1 = row['raw_scalar1']
            diff_scalar0 = row['diff_scalar0']; diff_scalar1 = row['diff_scalar1']
            baseline0 = row['baseline0']
            baseline1 = row['baseline1']
            max_laser_on = row['max_laser_on']
            pulses = row['pulses']

            if not pulses or pd.isna(pulses): continue
            for raw_height, raw_side, indices in json.loads(pulses.replace("(", "[").replace(")", "]")):
                peak = raw_height - baseline0
                side = raw_side - baseline1
                raw_25pc_width = indices[1] + indices[-2]
                raw_50pc_width = indices[2] + indices[-3]
                raw_75pc_width = indices[3] + indices[-4]
                
                f.write(f"{unix_timestamp},{portenta_serial},{pd0},{pd1},{laser},{raw_scalar0},{raw_scalar1},{diff_scalar0},{diff_scalar1},{baseline0},{baseline1},{max_laser_on},{peak},{side},{raw_75pc_width},{raw_50pc_width},{raw_25pc_width}\n")

if __name__ == "__main__":
    __main__()

def main():
    __main__()