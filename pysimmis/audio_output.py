import pyaudio
import wave

# Settings
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000
FRAMES_PER_BUFFER = 1024

FILENAME = "test.wav"

def open():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True)
    return p, stream

def play(filename, p, stream):
    print(f"audio_output.play(): {filename}")
    wf = wave.open(filename, 'rb')
    # Read and play in chunks
    chunk = 1024
    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)

def close(p, stream):
    stream.stop_stream()
    stream.close()
    p.terminate()

def main():
    ctx = open()
    play(FILENAME, *ctx)
    close(*ctx)

if __name__ == '__main__':
    main()