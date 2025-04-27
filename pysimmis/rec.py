import sounddevice as sd
import wave

duration = 1 # seconds
samplerate = 48000
channels = 2
filename = "/tmp/output.wav"

devices = sd.query_devices()
print('devices:', devices)

devix = int(input('Choose device index:'))

print("Recording...")
audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate,
                    channels=channels, dtype='int16', device=devix)
sd.wait()
print("Done recording.")

# Save as a WAV file
with wave.open(filename, 'wb') as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(2)  # 16-bit audio
    wf.setframerate(samplerate)
    wf.writeframes(audio_data.tobytes())

print(f"Saved to {filename}")

