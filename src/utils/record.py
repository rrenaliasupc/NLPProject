import sys

import tempfile
import threading

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd

recording = []
samplerate = 16000  # Whisper prefers 16000Hz
is_recording = True

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Recording error: {status}", file=sys.stderr)
    recording.append(indata.copy())

def record_until_enter():
    global is_recording
    print("Recording... Press Enter to stop.", end="")
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate):
        input()  # Wait for Enter
        is_recording = False

def record_temp_file():
    thread = threading.Thread(target=record_until_enter)
    thread.start()
    thread.join()

    # Concatenate all recorded chunks
    audio = np.concatenate(recording, axis=0)

    # Save
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        wav.write(tmpfile.name, samplerate, audio)
        print(f"Saved recording to {tmpfile.name}")

    return tmpfile.name
