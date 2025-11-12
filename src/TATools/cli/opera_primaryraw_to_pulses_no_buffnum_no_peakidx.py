import struct
import io
import time
import os
import sys

DEBUG_SLOW = False
DEBUG_SEEK = False
DEBUG_READ = False

CHUNK_TYPE_PRIMARY_RAW = b'P'

def _seek_until_separator(file_handle: io.BufferedReader, separator: bytes = bytes([0xff, 0xff, 0x00, 0x00, 0xff, 0xff])) -> bool:
    n = len(separator)
    buffer = file_handle.peek(n)[:n]
    seeked_bytes = 0
    while buffer != separator:
        if file_handle.read(1) == b'':
            return False  # EOF
        buffer = file_handle.peek(n)
        if DEBUG_SLOW:
            print(f"[DEBUG] Current buffer: {buffer}")
            time.sleep(1)
        seeked_bytes += 1
    if DEBUG_SEEK:
        print(f"[DEBUG] Seeked {seeked_bytes} bytes to find the separator.")
    file_handle.read(n)
    return True

pulse_struct = struct.Struct("<2H4x8H")
class Pulse(object):
    def __init__(self, raw_height: float, raw_side_height: float, indices: 'list[int]'):
        self.raw_height = raw_height
        self.raw_side_height = raw_side_height
        self.indices = indices

        self.height: float = 0.0
        self.side_height: float = 0.0
        self.raw_width_75pc: float = 0.0
        self.raw_width_50pc: float = 0.0
        self.raw_width_25pc: float = 0.0
    
    def __str__(self) -> str:
        return f"(Pulse: raw_height={self.raw_height}, raw_side_height={self.raw_side_height}, indices={self.indices})"

teensy_counts_id_struct = struct.Struct("<3B")
scalars_struct = struct.Struct("<4f")
baselines_struct = struct.Struct("<2f")
thresholds_struct = struct.Struct("<4f")
misc_struct = struct.Struct("<3IHf")
class TeensyCounts(object):
    def __init__(self, pd0: int, pd1: int, laser: int):
        self.pd0 = pd0
        self.pd1 = pd1
        self.laser = laser

        self.raw_scalar_pd0: float = 0
        self.raw_scalar_pd1: float = 0
        self.diff_scalar_pd0: float = 0
        self.diff_scalar_pd1: float = 0

        self.baseline0: float = 0
        self.baseline1: float = 0

        self.raw_th0: float = 0
        self.raw_th1: float = 0
        self.diff_th0: float = 0
        self.diff_th1: float = 0

        self.ms_read: int = 0
        self.buffers_read: int = 0
        self.num_pulse: int = 0
        self.max_laser_on: int = 0
        self.pulses_per_second: float = 0

        self.pulses: 'list[Pulse]' = []

    def recalculate_pulses(self):
        for pulse in self.pulses:
            pulse.height = pulse.raw_height - self.baseline0
            pulse.side_height = pulse.raw_side_height - self.baseline1
            pulse.indices = [-1*i for i in pulse.indices[0:4]] + pulse.indices[4:8]  # Invert first four indices, keep last four as is
            pulse.raw_width_75pc = pulse.indices[4] - pulse.indices[3]
            pulse.raw_width_50pc = pulse.indices[5] - pulse.indices[2]
            pulse.raw_width_25pc = pulse.indices[6] - pulse.indices[1]


    def __str__(self) -> str:
        s = f"[Counts, Pins {self.pd0},{self.pd1},{self.laser}]"
        if DEBUG_READ:
            s += f"\n\t[Scal: {self.raw_scalar_pd0:.1f},{self.raw_scalar_pd1:.1f},{self.diff_scalar_pd0:.1f},{self.diff_scalar_pd1:.1f}]"
            s += f"\n\t[Base: {self.baseline0:.1f},{self.baseline1:.1f}]"
            s += f"\n\t[Thre: {self.raw_th0:.1f},{self.raw_th1:.1f},{self.diff_th0:.1f},{self.diff_th1:.1f}]"
            s += f"\n\t[Misc: ms_read={self.ms_read}, buffers_read={self.buffers_read}, num_pulse={self.num_pulse}, max_laser_on={self.max_laser_on}, pulses_per_second={self.pulses_per_second:.2f}]"
        return s
    
    def populate_scalars(self, raw0: float, raw1: float, diff0: float, diff1: float):
        self.raw_scalar_pd0 = raw0
        self.raw_scalar_pd1 = raw1
        self.diff_scalar_pd0 = diff0
        self.diff_scalar_pd1 = diff1
    
    def populate_baselines(self, b0: float, b1: float):
        self.baseline0 = b0
        self.baseline1 = b1

    def populate_thresholds(self, raw0: float, raw1: float, diff0: float, diff1: float):
        self.raw_th0 = raw0
        self.raw_th1 = raw1
        self.diff_th0 = diff0
        self.diff_th1 = diff1

    def populate_misc(self, ms_read: int, buffers_read: int, num_pulse: int, max_laser_on: int, pulses_per_second: float):
        self.ms_read = ms_read
        self.buffers_read = buffers_read
        self.num_pulse = num_pulse
        self.max_laser_on = max_laser_on
        self.pulses_per_second = pulses_per_second

def _read_counts(file_handle: io.BufferedReader) -> TeensyCounts:
    counts_id = teensy_counts_id_struct.unpack(file_handle.read(teensy_counts_id_struct.size))
    counts = TeensyCounts(*counts_id)

    scalars = scalars_struct.unpack(file_handle.read(scalars_struct.size))
    counts.populate_scalars(*scalars)

    baselines = baselines_struct.unpack(file_handle.read(baselines_struct.size))
    counts.populate_baselines(*baselines)

    thresholds = thresholds_struct.unpack(file_handle.read(thresholds_struct.size))
    counts.populate_thresholds(*thresholds)

    misc = misc_struct.unpack(file_handle.read(misc_struct.size))
    counts.populate_misc(*misc)

    # pulses, for some reason separate from num_pulse in misc
    num_pulse = struct.unpack('<I', file_handle.read(4))[0]
    if DEBUG_READ:
        print(f"[DEBUG] Number of Pulses: {num_pulse}")

    for i in range(num_pulse):
        pulse_data = pulse_struct.unpack(file_handle.read(pulse_struct.size))
        pulse = Pulse(pulse_data[0], pulse_data[1], list(pulse_data[2:]))
        counts.pulses.append(pulse)

    counts.recalculate_pulses()

    return counts

class PrimaryDataChunk(object):
    def __init__(self, unix_timestamp: int, portenta_serial: str):
        self.unix_timestamp = unix_timestamp
        self.portenta_serial = portenta_serial

        #Teensy Metadata
        self.teensy_ms: int = 0
        self.teensy_mcu_temp: float = 0.0
        self.flow_temp: float = 0.0
        self.flow_hum: float = 0.0
        self.flow_rate: float = 0.0
        self.hv_enabled: bool = False
        self.hv_set: int = 0
        self.hv_mon: int = 0

        self.counts: 'list[TeensyCounts]' = []

    def populate_teensy_metadata(self, ms: int, mcu_temp: float, flow_temp: float, flow_hum: float, flow_rate: float, hv_enabled: bool, hv_set: int, hv_mon: int):
        self.teensy_ms = ms
        self.teensy_mcu_temp = mcu_temp
        self.flow_temp = flow_temp
        self.flow_hum = flow_hum
        self.flow_rate = flow_rate
        self.hv_enabled = hv_enabled
        self.hv_set = hv_set
        self.hv_mon = hv_mon

teensy_metadata_struct = struct.Struct("<L4f?BH")
def _read_primary_raw_chunk(file_handle: io.BufferedReader) -> Exception | PrimaryDataChunk:
    try:
        # First we make sure the chunk we are pointed at is a primary raw chunk
        chunk_type = file_handle.read(1)
        assert chunk_type == CHUNK_TYPE_PRIMARY_RAW, f"Expected chunk type {CHUNK_TYPE_PRIMARY_RAW}, but got {chunk_type}"

        # Unix Timestamp (4 byte uint32)
        unix_timestamp = struct.unpack('<L', file_handle.read(4))[0]
        if DEBUG_READ:
            print(f"[DEBUG] Unix Timestamp: {unix_timestamp} ({time.ctime(unix_timestamp)})")

        # Next is Portenta Serial Number (n-byte string, first 4 bytes are length as uint32)
        portenta_serial_length = struct.unpack('<L', file_handle.read(4))[0]
        portenta_serial = file_handle.read(portenta_serial_length).decode('utf-8')
        if DEBUG_READ:
            print(f"[DEBUG] Portenta Serial Number: {portenta_serial}")

        chunk = PrimaryDataChunk(unix_timestamp, portenta_serial)

        # Read Teensy Metadata
        teensy_ms, teensy_mcu_temp, flow_temp, flow_hum, flow_rate, hv_enabled, hv_set, hv_mon, = teensy_metadata_struct.unpack(file_handle.read(teensy_metadata_struct.size))
        if DEBUG_READ:
            print(f"[DEBUG] Teensy Metadata: ms={teensy_ms}, mcu_temp={teensy_mcu_temp:.02f}, flow_temp={flow_temp:.02f}, flow_hum={flow_hum:.02f}, flow_rate={flow_rate:.02f}, hv_enabled={hv_enabled}, hv_set={hv_set}, hv_mon={hv_mon}")
        chunk.populate_teensy_metadata(teensy_ms, teensy_mcu_temp, flow_temp, flow_hum, flow_rate, hv_enabled, hv_set, hv_mon)

        # Number of counts
        num_counts = struct.unpack('<I', file_handle.read(4))[0]
        if DEBUG_READ:
            print(f"[DEBUG] Number of Counts: {num_counts}")

        for i in range(num_counts):
            counts = _read_counts(file_handle)
            if DEBUG_READ:
                print(f"[DEBUG] Counts: {str(counts)}")
            chunk.counts.append(counts)

        return chunk

        
    except AssertionError as e:
        return e
    

pulse_csv_headers = "unix_timestamp,portenta_serial,pd0,pd1,laser,baseline0,baseline1,max_laser_on,height,side_height,raw_width_75pc,raw_width_50pc,raw_width_25pc\n"

def write_chunk_csv(chunk: PrimaryDataChunk, filepath: str):
    with open(filepath, "a+") as f:
        for counts in chunk.counts:
            unix_timestamp = chunk.unix_timestamp
            portenta_serial = chunk.portenta_serial
            pd0, pd1, laser = counts.pd0, counts.pd1, counts.laser
            baseline0, baseline1 = counts.baseline0, counts.baseline1
            max_laser_on = counts.max_laser_on

            for pulse in counts.pulses:
                height = pulse.height
                side_height = pulse.side_height
                raw_width_75pc = pulse.raw_width_75pc
                raw_width_50pc = pulse.raw_width_50pc
                raw_width_25pc = pulse.raw_width_25pc
                f.write(f"{unix_timestamp},{portenta_serial},{pd0},{pd1},{laser},{baseline0:.2f},{baseline1:.2f},{max_laser_on},{height:.2f},{side_height:.2f},{raw_width_75pc},{raw_width_50pc},{raw_width_25pc}\n")

    

def read_file(filepath: str, output_filename: 'str | None' = None):
    # remove .raw extension and replace with "_pulses.csv"
    output_filename = output_filename or os.path.splitext(filepath)[0] + "_pulses.csv"
    print(f"Output filename will be: {output_filename}")

    # if the output file does not exist, create it and write the headers
    if not os.path.exists(output_filename):
        with open(output_filename, "w") as f:
            f.write(pulse_csv_headers)
    with open(filepath, "rb") as f:
        while _seek_until_separator(f):
            res = _read_primary_raw_chunk(f)
            if isinstance(res, Exception):
                print(f"[ERROR] Failed to read primary raw chunk: {res}")
                return
            chunk: PrimaryDataChunk = res
            write_chunk_csv(chunk, output_filename)


def __main__():
    # Get cmd line args, if no args, print usage, if 3 args use 2nd as input file and 3rd as output file
    if len(sys.argv) < 2:
        print("Usage: python opera_primaryraw_to_pulses.py <input_file.raw> [output_file.csv]")
        return
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    read_file(input_file, output_file)

if __name__ == "__main__":
    __main__()

def main():
    __main__()