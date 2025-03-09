# This script.py file is for the second question of the minor project.
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pydub import AudioSegment
import os
from scipy.signal import medfilt

class SpeechAnalyzer:
    def __init__(self, sr=22050):
        self.sr = sr
        self.frame_length = 2048
        self.hop_length = 512
    
    def convert_mp3(self, mp3_path):
        wav_path = mp3_path.rsplit('.', 1)[0] + '.wav'
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format='wav')
        return wav_path
    
    def clean_audio(self, audio):
        cleaned = medfilt(audio, kernel_size=3)
        normalized = librosa.util.normalize(cleaned)
        
        chunks = librosa.effects.split(
            normalized,
            top_db=20,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        
        filtered = np.concatenate(
            [normalized[start:end] for start, end in chunks]
        )
        
        return filtered
    
    def load_file(self, file_path):
        try:
            if file_path.endswith('.mp3'):
                wav_path = self.convert_mp3(file_path)
                audio, _ = librosa.load(wav_path, sr=self.sr)
                os.remove(wav_path)
            else:
                audio, _ = librosa.load(file_path, sr=self.sr)
                
            return self.clean_audio(audio)
            
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            return None
    
    def get_zcr(self, audio):
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]
        
        clean_zcr = zcr[zcr < np.mean(zcr) + 2*np.std(zcr)]
        return clean_zcr
    
    def get_energy(self, audio):
        energy = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]
        
        clean_energy = energy[energy < np.mean(energy) + 2*np.std(energy)]
        return clean_energy
    
    def get_mfcc(self, audio):
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=13,
            n_mels=40,
            dct_type=2,
            norm='ortho'
        )
        return mfcc
    
    def analyze(self, file_path):
        print(f"\nAnalyzing speech file: {file_path}")
        
        audio = self.load_file(file_path)
        if audio is None:
            return
        
        features = {
            'zcr': self.get_zcr(audio),
            'energy': self.get_energy(audio),
            'mfcc': self.get_mfcc(audio)
        }
        
        self.make_plots(features, audio, os.path.basename(file_path))
        self.show_stats(features)
        
        return features
    
    def make_plots(self, features, audio, title):
        plt.figure(figsize=(15, 12))
        
        plt.subplot(5, 1, 1)
        plt.title(f'Speech Analysis: {title}')
        plt.plot(audio)
        plt.ylabel('Amplitude')
        
        plt.subplot(5, 1, 2)
        plt.title('Zero Crossing Rate')
        plt.plot(features['zcr'])
        plt.ylabel('ZCR')
        
        plt.subplot(5, 1, 3)
        plt.title('Short-Time Energy')
        plt.plot(features['energy'])
        plt.ylabel('Energy')
        
        plt.subplot(5, 1, 4)
        plt.title('MFCC Features')
        librosa.display.specshow(
            features['mfcc'],
            x_axis='time',
            sr=self.sr
        )
        plt.colorbar(format='%+2.0f dB')
        
        plt.subplot(5, 1, 5)
        plt.title('Spectrogram')
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio)),
            ref=np.max
        )
        librosa.display.specshow(
            D,
            y_axis='log',
            x_axis='time',
            sr=self.sr
        )
        plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(f'analysis_{title}.png')
        plt.close()
    
    def show_stats(self, features):
        print("\nFeature Statistics:")
        print("-" * 40)
        
        print("\nZero Crossing Rate:")
        print(f"Mean: {np.mean(features['zcr']):.4f}")
        print(f"Std: {np.std(features['zcr']):.4f}")
        print(f"Max: {np.max(features['zcr']):.4f}")
        print(f"Min: {np.min(features['zcr']):.4f}")
        
        print("\nEnergy:")
        print(f"Mean: {np.mean(features['energy']):.4f}")
        print(f"Std: {np.std(features['energy']):.4f}")
        print(f"Max: {np.max(features['energy']):.4f}")
        print(f"Min: {np.min(features['energy']):.4f}")
        
        print("\nMFCC Values:")
        for i in range(13):
            print(f"\nCoef {i+1}:")
            print(f"Mean: {np.mean(features['mfcc'][i]):.4f}")
            print(f"Std: {np.std(features['mfcc'][i]):.4f}")
            print(f"Max: {np.max(features['mfcc'][i]):.4f}")
            print(f"Min: {np.min(features['mfcc'][i]):.4f}")

if __name__ == "__main__":
    analyzer = SpeechAnalyzer()
    speech_file = "indira_gandhi.mp3"
    analyzer.analyze(speech_file)