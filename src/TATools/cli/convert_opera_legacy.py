import argparse
import struct
import io
import time
import abc
import pathlib
import os
import sys
import shutil
from typing import Tuple, Optional, Literal, Any, Union, List

DEBUG_SEEK = False
DEBUG_SLOW = False

# File Utils
format_csv = lambda l: (",".join([str(v) for v in l])+"\n").encode()

def read_n(f: io.BufferedReader, n: int) -> bytearray:
    buf = bytearray()
    while len(buf) != n:
        new_byte = f.read(1)
        if new_byte == b'':
            raise EOFError()
        buf.append(new_byte[0])
    return buf

def seek_until_separator(
    file_handle: io.BufferedReader,
    separator: bytes = bytes([0xff, 0xff, 0x00, 0x00, 0xff, 0xff]),
) -> bool:
    n = len(separator)
    try:
        buffer = read_n(file_handle, n)
    except EOFError:
        return False

    seeked_bytes = 0

    while True:
        if buffer == separator:
            # We have just read the separator; file pointer is already after it
            if DEBUG_SEEK:
                print(f"[DEBUG] Seeked {seeked_bytes} bytes to find the separator at pos {file_handle.tell()}")
            return True

        b = file_handle.read(1)
        if b == b'':
            return False  # EOF

        buffer = buffer[1:] + b
        seeked_bytes += 1

        if DEBUG_SLOW:
            print(f"[DEBUG] Current buffer({len(buffer)}): {buffer}")
            time.sleep(1)

def read_binary(r: io.BufferedReader, struct_text: str) -> Any:
    st = struct.Struct(OPERA_ENDIANNESS+struct_text)
    return st.unpack(r.read(st.size))

def read_variable_length_str(r: io.BufferedReader) -> str:
    str_len, = read_binary(r, "L")
    st = r.read(str_len).decode()
    return st

# Structs
OPERA_ENDIANNESS = "<"

FRAME_MARKER_SECONDARY = b'S'
FRAME_MARKER_OUTPUT = b'O'
FRAME_MARKER_PRIMARY = b'P'

class OperaFrame(abc.ABC):
    @abc.abstractmethod
    def populate(self, f: io.BufferedReader):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def output(self, ifp: pathlib.Path):
        raise NotImplementedError()
    
class SPS30Frame(object):
    PM1: float
    PM2p5: float
    PM4: float
    PM10: float
    PN0p5: float
    PN1: float
    PN2p5: float
    PN4: float
    PN10: float
    TypicalParticleSize: float

    def populate(self, f: io.BufferedReader):
        self.PM1, self.PM2p5, self.PM4, self.PM10 = read_binary(f, "ffff")
        self.PN0p5, self.PN1, self.PN2p5, self.PN4, self.PN10 = read_binary(f, "fffff")
        self.TypicalParticleSize, = read_binary(f, "f")
    
    def __str__(self) -> str:
        return f"PM 1: {self.PM1:.1f}, 2.5: {self.PM2p5:.1f}, 4: {self.PM4:.1f}, 10: {self.PM10:.1f} | PN 0.5: {self.PN0p5:.1f}, 1: {self.PN1:.1f}, 2.5: {self.PN2p5:.1f}, 4: {self.PN4:.1f}, 10: {self.PN10:.1f} | Typical: {self.TypicalParticleSize:.1f}"

class SecondaryFrame(OperaFrame):
    unix: int
    portenta_serial: str
    sps30: SPS30Frame
    pressure: float
    co2: int
    voc_index: int
    flow_temp: float
    flow_hum: float
    flow_rate: float
    imx8_temp: float
    teensy_mcu_temp: float
    optical_temperatures: List[float]
    temp_htu: float
    hum_htu: float
    temp_scd: float
    hum_scd: float
    mon_5v: float
    mon_5v_mean: float


    def populate(self, f: io.BufferedReader):
        self.unix, = read_binary(f, "L")
        self.portenta_serial = read_variable_length_str(f)
        self.sps30 = SPS30Frame()
        self.sps30.populate(f)
        self.pressure, self.co2, self.voc_index = read_binary(f, "fLl")
        self.flow_temp, self.flow_hum, self.flow_rate = read_binary(f, "fff")
        self.imx8_temp, self.teensy_mcu_temp = read_binary(f, "ff")
        self.optical_temperatures = list(read_binary(f, "fff"))
        self.temp_htu, self.hum_htu, self.temp_scd, self.hum_scd = read_binary(f, "ffff")
        self.mon_5v, self.mon_5v_mean = read_binary(f, "ff")

    def values(self) -> List:
        return [self.portenta_serial, self.unix, \
                round(self.sps30.PM1,3), round(self.sps30.PM2p5,3), round(self.sps30.PM4,3), round(self.sps30.PM10,3),
                round(self.sps30.PN0p5,3), round(self.sps30.PN1,3), round(self.sps30.PN2p5,3), round(self.sps30.PN4,3),
                round(self.sps30.PN10,3), round(self.sps30.TypicalParticleSize,2), round(self.pressure,1), self.co2, 
                self.voc_index, round(self.flow_temp,1), round(self.flow_hum,1), round(self.flow_rate,4), round(self.imx8_temp,1),
                round(self.teensy_mcu_temp,3), round(self.optical_temperatures[0],1),round(self.optical_temperatures[1],1),
                round(self.optical_temperatures[2],1), round(self.temp_htu,1), round(self.hum_htu,1), round(self.temp_scd,1),
                round(self.hum_scd,1), round(self.mon_5v,1), round(self.mon_5v_mean,1)]
    
    def headers(self) -> List:
        return ["portenta_serial", "unix", "sps30_pm1", "sps30_pm2p5", "sps30_pm4", "sps30_pm10",
                "sps30_pn0p5", "sps30_pn1", "sps30_pn2p5", "sps30_pn4", "sps30_pn10", "sps30_typical_size",
                "pressure", "co2", "voc_index", "flow_temp", "flow_hum", "flow_rate", "imx8_temp", "teensy_mcu_temp",
                "optical_temp0", "optical_temp1", "optical_temp2", "omb_temp_htu", "omb_hum_htu",
                "omb_temp_scd", "omb_hum_scd", "mon_5v", "mon_5v_mean"]

    def __str__(self) -> str:
        return f"{self.__repr__()[:-1]} | SPS30 {str(self.sps30)} | {self.pressure:.1f}kPa, {self.co2}ppm,\
 {self.flow_temp:.1f}C, {self.flow_hum:.1f}%, {self.flow_rate:.1f}m/s | MCU I.MX8: {self.imx8_temp:.1f}, Teensy: {self.teensy_mcu_temp:.1f} |\
 Optical: {self.optical_temperatures[0]:.1f}, {self.optical_temperatures[1]:.1f}, {self.optical_temperatures[2]:.1f}F\
 | HTU: {self.temp_htu:.1f}C, {self.hum_htu:.1f}% | SCD: {self.temp_scd:.1f}C, {self.hum_scd:.1f}% | {self.mon_5v:.1f}V, {self.mon_5v_mean:.1f}V >"
    def __repr__(self) -> str:
        return f"<SecondaryFrame@{self.unix}, OPERA {self.portenta_serial}>"
    
    def output(self, ifp: pathlib.Path):
        ofp = ifp.with_suffix(".csv")
        new_file = not os.path.exists(ofp)
        with open(ofp, "a+b") as f:
            if new_file:
                f.write(format_csv(self.headers()))
            f.write(format_csv(self.values()))


class ConcentrationFrame(object):
    pm: List[float]
    pn: List[float]

    pm_diameters = [.3, 1, 2.5, 10]
    pn_diameters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1, 2.5, 5, 10]

    def populate(self, f: io.BufferedReader):
        self.pm = list(read_binary(f, "4f"))
        self.pn = list(read_binary(f, "12f"))

    def __init__(self):
        self.pm = [0 for _ in range(4)]
        self.pn = [0 for _ in range(12)]

    def values(self) -> List[float]:
        return [round(c,3) for c in self.pm + self.pn]

    def headers(self) -> List[str]:
        return [f"PM{d}" for d in self.pm_diameters] + [f"PN{d}" for d in self.pn_diameters]

class OutputFrame(OperaFrame):
    unix: int
    portenta_serial: str
    concentration_frame: ConcentrationFrame
    class_label: str
    n_classes: int
    classes: List[str]
    class_probabilities: List[float]
    temp: float
    hum: float
    sps30_pm2p5: float
    pressure: float
    co2: int
    voc_index: int

    def populate(self, f: io.BufferedReader):
        self.unix, = read_binary(f, "L")
        self.portenta_serial = read_variable_length_str(f)
        self.concentration_frame = ConcentrationFrame()
        self.concentration_frame.populate(f)
        self.class_label = read_variable_length_str(f)
        self.n_classes, = read_binary(f, "L")
        self.classes = [read_variable_length_str(f) for _ in range(self.n_classes)]
        self.class_probabilities = list(read_binary(f, f"{self.n_classes}f"))
        self.temp, self.hum, self.sps30_pm2p5, self.pressure = read_binary(f, "ffff")
        self.co2, self.voc_index = read_binary(f, "Ll")
    
    def values(self) -> List:
        return [self.portenta_serial, self.unix, *self.concentration_frame.values(),
                f"\"{self.class_label}\"", f"\"{self.classes}\"", f"\"{[round(c,2) for c in self.class_probabilities]}\"",
                round(self.temp,1), round(self.hum,1), round(self.sps30_pm2p5, 3), round(self.pressure,1),
                self.co2, self.voc_index]
    
    def headers(self) -> List:
        return ["portenta_serial", "unix", *self.concentration_frame.headers(), "class", "classes", "probabilities",
                "temp", "hum", "sps30_pm2p5", "pressure", "co2", "voc_index"]
    
    def output(self, ifp: pathlib.Path):
        ofp = ifp.with_suffix(".csv")
        new_file = not os.path.exists(ofp)
        with open(ofp, "a+b") as f:
            if new_file:
                f.write(format_csv(self.headers()))
            f.write(format_csv(self.values()))
        


class Pulse(object):
    raw_height: int
    raw_side_height: int
    indices: List[int]

    height: float
    side_height: float
    raw_width_75pc: float
    raw_width_50pc: float
    raw_width_25pc: float
    
    def values(self) -> List:
        return [self.raw_height, self.raw_side_height,
                f"\"[{self.indices[0]},{self.indices[1]},{self.indices[2]},{self.indices[3]},{self.indices[4]},{self.indices[5]},{self.indices[6]},{self.indices[7]}]\"", 
                round(self.height,2), round(self.side_height,2), round(self.raw_width_25pc,2),round(self.raw_width_50pc,2),
                round(self.raw_width_75pc,2)]
    
    def headers(self) -> List:
        return ["raw_height", "raw_side_height", "indices", "height", "side_height", "width_25pc", "width_50pc", "width_75pc"]
    
    def populate(self, f: io.BufferedReader):
        self.raw_height, self.raw_side_height = read_binary(f, "HH")
        self.indices = list(read_binary(f, "4x8H"))
        

class Counts(object):
    pd0: int
    pd1: int
    laser: int

    raw_scalar_pd0: float
    raw_scalar_pd1: float
    diff_scalar_pd0: float
    diff_scalar_pd1: float

    baseline0: float
    baseline1: float

    raw_th0: float
    raw_th1: float
    diff_th0: float
    diff_th1: float

    ms_read: int
    buffers_read: int
    num_pulse: int
    max_laser_on: int
    pulses_per_second: float

    pulses: List[Pulse]

    def values(self) -> List:
        return [self.pd0, self.pd1, self.laser, round(self.raw_scalar_pd0,2), round(self.raw_scalar_pd1,2),
                round(self.diff_scalar_pd0,2), round(self.diff_scalar_pd1,2), round(self.baseline0,2), round(self.baseline1,2),
                round(self.raw_th0,2), round(self.raw_th1,2), round(self.diff_th0,2), round(self.diff_th1,2),
                self.ms_read, self.buffers_read, self.num_pulse, self.max_laser_on, round(self.pulses_per_second,2)]
    
    def headers(self) -> List:
        return ["pd0", "pd1", "laser", "raw_scalar_pd0", "raw_scalar_pd1", "diff_scalar_pd0", "diff_scalar_pd1",
                "baseline0", "baseline1", "raw_th0", "raw_th1", "diff_th0", "diff_th1", "ms_read", "num_buff",
                "num_pulse", "max_laser_on", "pulses_per_second"]

    def populate(self, f: io.BufferedReader):
        self.pd0, self.pd1, self.laser = read_binary(f, "BBB")
        self.raw_scalar_pd0, self.raw_scalar_pd1 = read_binary(f, "ff")
        self.diff_scalar_pd0, self.diff_scalar_pd1 = read_binary(f, "ff")
        self.baseline0, self.baseline1 = read_binary(f, "ff")
        self.raw_th0, self.raw_th1, self.diff_th0, self.diff_th1 = read_binary(f, "ffff")
        self.ms_read, self.buffers_read, self.num_pulse, self.max_laser_on, self.pulses_per_second = read_binary(f, "3IHf")

        n_pulse, = read_binary(f, "I")
        self.pulses = [Pulse() for _ in range(n_pulse)]
        for p in self.pulses:
            p.populate(f)
        self.recalculate_pulses()
    
    def recalculate_pulses(self):
        for pulse in self.pulses:
            pulse.height = pulse.raw_height - self.baseline0
            pulse.side_height = pulse.raw_side_height - self.baseline1
            indices = [-1*i for i in pulse.indices[0:4]] + pulse.indices[4:8]  # Invert first four indices, keep last four as is
            pulse.raw_width_75pc = indices[4] - indices[3]
            pulse.raw_width_50pc = indices[5] - indices[2]
            pulse.raw_width_25pc = indices[6] - indices[1]

class RawFrame(OperaFrame):
    unix: int
    portenta_serial: str

    teensy_ms: int = 0
    teensy_mcu_temp: float = 0.0
    flow_temp: float = 0.0
    flow_hum: float = 0.0
    flow_rate: float = 0.0
    hv_enabled: bool = False
    hv_set: int = 0
    hv_mon: int = 0
    n_counts: int
    counts: List[Counts]

    def values(self, c: Counts) -> List:
        return [self.portenta_serial, self.unix, *c.values(), self.teensy_ms, round(self.teensy_mcu_temp,2), round(self.flow_temp,2),
                round(self.flow_hum,2), round(self.flow_rate,2), self.hv_enabled, self.hv_set, self.hv_mon]
    
    def headers(self, c: Counts) -> List:
        return ["portenta_serial", "unix", *c.headers(), "teensy_ms", "teensy_mcu_temp", "flow_temp",
                "flow_hum", "flow_rate", "hv_enabled", "hv_set", "hv_mon"]
    
    def values_pulses(self, c: Counts) -> List:
        return [self.portenta_serial, self.unix, c.pd0, c.pd1, c.laser, round(c.baseline0,2), round(c.baseline1,2)]
    
    def headers_pulses(self) -> List:
        return ["portenta_serial", "unix", "pd0", "pd1", "laser", "baseline0", "baseline1"]

    def populate(self, f: io.BufferedReader):
        self.unix, = read_binary(f, "L")
        self.portenta_serial = read_variable_length_str(f)
        self.teensy_ms, self.teensy_mcu_temp, self.flow_temp, self.flow_hum, self.flow_rate = read_binary(f, "Lffff")
        self.hv_enabled, self.hv_set, self.hv_mon = read_binary(f, "?BH")
        self.n_counts, = read_binary(f, "I")
        self.counts = [Counts() for _ in range(self.n_counts)]
        for c in self.counts:
            c.populate(f)

    def output(self, ifp: pathlib.Path):
        ofp_counts = ifp.with_suffix(".csv")
        ofp_pulses = ifp.stem+"_Pulses.csv"

        new_file = not os.path.exists(ofp_counts)
        with open(ofp_counts, "a+b") as f:
            if new_file:
                f.write(format_csv(self.headers(Counts())))
            for c in self.counts:
                f.write(format_csv(self.values(c)))
        
        new_file = not os.path.exists(ofp_pulses)
        with open(ofp_pulses, "a+b") as f:
            if new_file:
                f.write(format_csv(self.headers_pulses() + Pulse().headers()))
            for c in self.counts:
                for p in c.pulses:
                    f.write(format_csv(self.values_pulses(c) + p.values()))



FRAME_MAP = {
    FRAME_MARKER_PRIMARY: RawFrame,
    FRAME_MARKER_OUTPUT: OutputFrame,
    FRAME_MARKER_SECONDARY: SecondaryFrame,
}

# Control Flow
def get_args() -> List[pathlib.Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")

    args = parser.parse_args()
    input_file = pathlib.Path(args.input_file)

    if input_file.is_dir():
        input_file = [f for f in [pathlib.Path(f) for f in os.listdir(input_file)] if f.suffix == ".raw"]
    else:
        input_file = [input_file]

    return input_file

def seek_next_frame(f: io.BufferedReader) -> Union[bytes, None]:
    if not seek_until_separator(f): return None
    return f.read(1)

def parse_next_frame(f: io.BufferedReader) -> Union[OperaFrame, None]:
    frame_marker = seek_next_frame(f)
    
    if not frame_marker: return

    frame_type = FRAME_MAP.get(frame_marker)
    if not frame_type: return

    frame = frame_type()
    frame.populate(f)
    return frame


def convert_file(ifp: pathlib.Path):
    filesize = os.path.getsize(ifp)
    term_width = shutil.get_terminal_size().columns
    prefix = f"Converting {ifp}: |"
    bar_width = int(0.9 * term_width) - len(prefix) - 1  # -1 for the closing '|'
    segment_size = max(1, filesize // bar_width)

    # Hide cursor (ANSI escape)
    sys.stdout.write("\x1b[?25l")
    sys.stdout.flush()

    try:

        with open(ifp, "rb") as f:
            frame = parse_next_frame(f)
            while(frame is not None):
                frame.output(ifp)
                frame = parse_next_frame(f)

                current_pos = f.tell()
                # How many segments to draw
                n_segs = min(bar_width, current_pos // segment_size)
                bar = "=" * n_segs
                spaces = " " * (bar_width - n_segs)

                # Rewrite the entire line
                sys.stdout.write("\r" + prefix + bar + spaces + "|")
                sys.stdout.flush()

        # Clear entire line, return carriage
        sys.stdout.write("\r\x1b[2K")
        # Write your final message
        sys.stdout.write(f"Converted {ifp}.\n")
        sys.stdout.flush()
        # Show cursor again and move to next line
    finally:
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()

def main():
    ifps = get_args()

    print()
    for ifp in ifps:
        convert_file(ifp)

        

if __name__ == "__main__":
    main()