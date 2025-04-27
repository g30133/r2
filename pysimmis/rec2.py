import pyaudio
import wave

# Recording settings
filename = "/tmp/output_pyaudio.wav"
duration = 3  # seconds
channels = 1
rate = 48000  # sample rate (Hz)
chunk = 1024  # frame size
format = pyaudio.paInt16  # 16-bit audio

def record_audio():
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"{i}: {info['name']}")

    devix = int(input('Choose device index:'))

    # Open input stream
    stream = audio.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        input_device_index=devix,
                        frames_per_buffer=chunk)

    input("Enter to start recording:")
    stream.start_stream()

    print("Recording...")

    frames = []
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Recording finished.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save to WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    print(f"Saved as {filename}")

record_audio()

