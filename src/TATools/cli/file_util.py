import io
import time

DEBUG_SLOW = True
DEBUG_SEEK = False
DEBUG_READ = False

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



"""

def seek_until_separator(file_handle: io.BufferedReader,
                         separator: bytes = bytes([0xff, 0xff, 0x00, 0x00, 0xff, 0xff])
                         ) -> bool:
    n = len(separator)
    buffer = file_handle.peek(n)[:n]
    seeked_bytes = 0
    while buffer != separator:
        if file_handle.read(1) == b'':
            print("[DEBUG] EOF reached while seeking separator at pos", file_handle.tell())
            return False  # EOF
        buffer = file_handle.peek(n)[:n]
        seeked_bytes += 1

    sep_pos = file_handle.tell()
    file_handle.read(n)
    if DEBUG_SEEK:
        print(f"[DEBUG] Found separator at {sep_pos}, seeked {seeked_bytes} bytes")
    return True

"""