import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment

# Function to convert MP3 to WAV
def convert_mp3_to_wav(mp3_filename, wav_filename):
    audio = AudioSegment.from_mp3(mp3_filename)
    audio.export(wav_filename, format="wav")

# Function to analyze audio
def analyze_audio(filename):
    # Convert MP3 to WAV first
    wav_filename = filename.replace(".mp3", ".wav")
    convert_mp3_to_wav(filename, wav_filename)

    # Load the WAV file
    y, sr = librosa.load(wav_filename, sr=None)
    time = np.linspace(0, len(y) / sr, num=len(y))

    # Compute features
    amplitude = np.abs(y)
    pitch, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    rms_energy = librosa.feature.rms(y=y)[0]

    # Spectrogram
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(time, amplitude, label="Amplitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Amplitude of {filename}")

    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram of {filename}")

    plt.tight_layout()
    plt.show()

    print(f"Analysis of {filename}:")
    print(f" - RMS Energy: {np.mean(rms_energy):.4f}")
    print(f" - Pitch (Mean): {np.nanmean(pitch):.2f} Hz")
    print("\n")

# Analyze the audio
analyze_audio(f"dataset/jamie.mp3")


# Some Analysis / Observations
# All the dialogues are part of Game of Thrones TV series.
#  I have tried to enact their dialogues in the same way as the original characters.
# Dialogues with excitement, emotion tend to have higher amplitude, and higher RMS energy.
# Dialogues spoken by females have higher pitch compared to male dialogues