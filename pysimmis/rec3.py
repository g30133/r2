import pyaudio
import wave

# Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
FRAMES_PER_BUFFER = 1024
RECORD_SECONDS = 3
FILENAME = "/tmp/audio_input.wav"

# Store recorded frames here
frames = []

# Callback function
def callback(in_data, frame_count, time_info, status):
    frames.append(in_data)
    return (None, pyaudio.paContinue)

def audio_input_open():
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open input stream in non-blocking mode
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=FRAMES_PER_BUFFER,
                    stream_callback=callback)

    # Start stream
    stream.start_stream()

    return p, stream

def audio_input_record(p, stream):
    frames.clear()

    # Wait for specified duration while stream runs in background
    import time
    time.sleep(RECORD_SECONDS)
    print("Recording complete.")

    # Save recorded data to WAV file
    wf = wave.open(FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Saved to {FILENAME}")

def audio_input_close(p, stream):
    # Stop stream
    stream.stop_stream()
    stream.close()
    p.terminate()

p, stream = audio_input_open()
input('Enter to start recording:')
audio_input_record(p, stream)
print('recording finished')
audio_input_close(p, stream)



