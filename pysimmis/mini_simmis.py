import audio_input
import audio_output
from time import sleep

FILENAME = '/tmp/simmis.wav'
SECONDS = 3

ctx_in = audio_input.open()
ctx_out = audio_output.open()

while True:
    #input('Enter to start recording:')
    print(f"{SECONDS} sec recording started")
    audio_input.record(FILENAME, SECONDS, *ctx_in)
    print("playback started")
    audio_output.play(FILENAME, *ctx_out)
    sleep(1)

audio_input.close(*ctx_in)
audio_output.close(*ctx_out)
