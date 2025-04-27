import pyaudio
import wave

# Settings
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000
FRAMES_PER_BUFFER = 1024
RECORD_SECONDS = 3
FILENAME = "test.wav"

def audio_output_open():
    p = pyaudio.PyAudio()
    #wf = open(FILENAME, 'wb')
    #stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
    #                channels=wf.getnchannels(),
    #                rate=wf.getframerate(),
    #                output=True)
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True)
    return p, stream

def audio_output_play(p, stream):
    wf = wave.open(FILENAME, 'rb')
    # Read and play in chunks
    chunk = 1024
    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)

def audio_output_close():
    stream.stop_stream()
    stream.close()
    p.terminate()

p, stream = audio_output_open()
audio_output_play(p, stream)
audio_output_close(p, stream)
