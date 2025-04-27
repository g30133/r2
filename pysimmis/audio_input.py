import pyaudio
import wave
import time

# Settings
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000
FRAMES_PER_BUFFER = 1024

FILENAME = "/tmp/audio_input.wav"
SECONDS = 3

# Store recorded frames here
frames = []

# Callback function
def callback(in_data, frame_count, time_info, status):
    frames.append(in_data)
    return (None, pyaudio.paContinue)

def open():
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open input stream in non-blocking mode
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=FRAMES_PER_BUFFER,
                    stream_callback=callback)
    stream.start_stream()

    return p, stream

def record(filename, duration, p, stream):
    frames.clear()

    # Wait for specified duration while stream runs in background
    time.sleep(duration)
    print("Recording complete.")

    # Save recorded data to WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Saved to {filename}")
    return 0

def close(p, stream):
    stream.stop_stream()
    stream.close()
    p.terminate()

def main():
    ctx = open()
    input('Enter to start recording:')
    record(FILENAME, SECONDS, *ctx)
    print('recording finished')
    close(*ctx)

if __name__ == '__main__':
    main()
