import mmap
import struct
import time

POLL_INTERVAL_TIMING = 0.001
HEADER_SIZE = 4 + 4 + 4
TIMING_SIZE = 8 + 8 + 8 + 8 + 8 + 8
SHMBUF_TIMING_SIZE = HEADER_SIZE + TIMING_SIZE
SHMBUF_TIMING_PATH = "/dev/shm/minecraft_timing"

def init():
    print('====init()====')
    with open(SHMBUF_TIMING_PATH, "w+b") as f:
        f.truncate(SHMBUF_TIMING_SIZE)
        shmbuf_timing = mmap.mmap(f.fileno(), SHMBUF_TIMING_SIZE, access=mmap.ACCESS_READ)
    print('====init() DONE====')
    return shmbuf_timing

    
def process(shmbuf_prompt):
    print('====process()====')

    timingcounter = 0
    while True:
        time.sleep(POLL_INTERVAL_TIMING)
        
        shmbuf_timing.seek(0)
        header0 = struct.unpack(">I", shmbuf_timing.read(4))[0]

        if header0 == timingcounter:
            continue
        elif header0 > timingcounter + 1:
            print('header0:', header0, 'timingcounter:', timingcounter)

        timingcounter = header0
        
        shmbuf_timing.seek(4)
        header4 = struct.unpack(">I", shmbuf_timing.read(4))[0]
        framecounter = header4

        shmbuf_timing.seek(8)
        header8 = struct.unpack(">I", shmbuf_timing.read(4))[0]
        frametype = header8

        shmbuf_timing.seek(HEADER_SIZE)
        tick_logic = struct.unpack(">q", shmbuf_timing.read(8))[0]
        shmbuf_timing.seek(HEADER_SIZE + 8)
        tick_render = struct.unpack(">q", shmbuf_timing.read(8))[0]
        shmbuf_timing.seek(HEADER_SIZE + 16)
        tick_playscreen_1 = struct.unpack(">q", shmbuf_timing.read(8))[0]
        shmbuf_timing.seek(HEADER_SIZE + 24)
        tick_playscreen_2 = struct.unpack(">q", shmbuf_timing.read(8))[0]
        shmbuf_timing.seek(HEADER_SIZE + 32)
        tick_menuscreen_1 = struct.unpack(">q", shmbuf_timing.read(8))[0]
        shmbuf_timing.seek(HEADER_SIZE + 40)
        tick_menuscreen_2 = struct.unpack(">q", shmbuf_timing.read(8))[0]

        d1 = tick_logic - tick_render
        d2 = tick_logic - tick_playscreen_1
        d3 = tick_logic - tick_playscreen_2
        d4 = tick_logic - tick_menuscreen_1
        d5 = tick_logic - tick_menuscreen_2
        print(f"{timingcounter:06d} {framecounter:06d} {frametype:01d} [{d1:04d} {d2:04d} {d3:04d} {d4:04d} {d5:04d}]")
        

shmbuf_timing = init()
process(shmbuf_timing)
