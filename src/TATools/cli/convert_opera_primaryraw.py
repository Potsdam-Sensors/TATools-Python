import argparse
import pathlib
import os

from .file_util import *
from .structs import OPERA_V1, OPERA_V1_OldStructs

def read_file(opera, filepath: pathlib.Path, output_filename: pathlib.Path):
    if not os.path.exists(output_filename):
        with open(output_filename, "w") as f:
            f.write(opera.Pulse.pulse_csv_headers)

    with open(filepath, "rb") as f:
        while seek_until_separator(f):
            res = opera._read_primary_raw_chunk(f)
            if isinstance(res, Exception):
                print(f"[ERROR] Failed to read primary raw chunk: {res}")
                # try to resync instead of returning
                continue
            chunk = res
            opera.write_chunk_csv(chunk, output_filename)


from typing import Union, List, Optional
def convert(no_buffnum: bool, input_files: List[pathlib.Path], output_files: List[pathlib.Path]):
    opera = None
    if no_buffnum:
        opera = OPERA_V1_OldStructs()
    else:
        opera = OPERA_V1()
    if not opera: # Future proofing
        raise ValueError()
    
    for ipf, opf in zip(input_files, output_files):
        read_file(opera, ipf, opf)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--output-name", "-o", help="Output filename.")
    parser.add_argument("--output-suffix", "-s", help="Suffix to replace original suffix for output.")
    parser.add_argument("--ignore-ext", "-i", help="Ignore extensions requirements (must end w/ '.raw').", const=True, nargs="?")
    parser.add_argument("--no-buffnum", "-n", const=True, nargs="?")
    args = parser.parse_args()

    input_file: pathlib.Path = pathlib.Path(args.input_file).absolute()
    is_dir = input_file.is_dir()
    no_buffnum = args.no_buffnum
    output_suffix = args.output_suffix
    output_filename = args.output_name
    ignore_extension = args.ignore_ext

    if output_filename and output_suffix:
        raise ValueError("Args '--output-name' and '--output-suffix' cannot both be provided.")

    inputs = []
    if is_dir:
        inputs = [input_file.joinpath(f) for f in os.listdir(input_file)]
    else:
        inputs = [input_file]
    if not ignore_extension:
        inputs = [f for f in inputs if f.suffix == ".raw"]
    
    if not inputs:
        raise ValueError("No valid input files were found at given path. Try the '-h' option for help.")
    print("Input files:")
    for f in inputs:
        print(f"\t{f}")

    outputs = []
    if not output_suffix and not output_filename:
        is_expected_format = all(["PrimaryRaw" in f.name for f in inputs])
        if is_expected_format:
            outputs = [f.with_name(f.name.replace("PrimaryRaw", "Pulses")).with_suffix(".csv") for f in inputs]
        else:
            output_suffix = "_Pulses"

    if output_suffix:
        outputs = [f.parent.joinpath(f"{f.stem}{output_suffix}").with_suffix(".csv") for f in inputs]
    elif output_filename:
        outputs = [output_filename]*len(inputs)

    print("Output files:")
    for f in outputs:
        print(f"\t{f}")

    convert(no_buffnum, inputs, outputs)


__main__ = main

if __name__=="__main__":
    main()