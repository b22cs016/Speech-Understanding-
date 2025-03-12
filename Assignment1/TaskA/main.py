import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class UrbanSoundDataset(Dataset):
    def __init__(self, audio_paths, labels, window_type='hann'):
        self.audio_paths = audio_paths
        self.labels = labels
        self.window_type = window_type
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)

    def __len__(self):
        return len(self.audio_paths)

    def apply_window(self, signal):
        def hann_window(length):
            return torch.hann_window(length)
        
        def hamming_window(length):
            return torch.hamming_window(length)
        
        def rectangular_window(length):
            return torch.ones(length)

        window_funcs = {
            'hann': hann_window,
            'hamming': hamming_window,
            'rectangular': rectangular_window
        }

        window = window_funcs.get(self.window_type, hann_window)
        window_tensor = window(signal.shape[-1])
        return signal * window_tensor

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.audio_paths[idx])
        
        # Convert to mono if not already
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        windowed_signal = self.apply_window(waveform)
        
        spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=512,  # Reduced n_fft
            hop_length=256
        )
        spectrogram = spectrogram_transform(windowed_signal)
        
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000, 
            n_mfcc=10,  # Reduced number of MFCCs
            melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 23}
        )
        mfcc = mfcc_transform(windowed_signal)
        mfcc = mfcc.mean(dim=2).squeeze().numpy()  # Average over time
        
        return spectrogram, mfcc, self.encoded_labels[idx]

def visualize_spectrograms(dataset, window_types):
    plt.figure(figsize=(15, 5))
    
    for i, window_type in enumerate(window_types, 1):
        spectrogram, _, _ = dataset[0]
        
        plt.subplot(1, 3, i)
        plt.title(f'{window_type.capitalize()} Window Spectrogram')
        plt.imshow(spectrogram.log2()[0, :, :].numpy(), aspect='auto', origin='lower')
        plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def train_and_evaluate(train_loader, val_loader):
    # Prepare data for SVM
    X_train, y_train = [], []
    for _, features, label in train_loader:
        X_train.append(features)
        y_train.append(label)
    
    X_train = np.array(X_train).squeeze()
    y_train = np.array(y_train).squeeze()

    X_val, y_val = [], []
    for _, features, label in val_loader:
        X_val.append(features)
        y_val.append(label)
    
    X_val = np.array(X_val).squeeze()
    y_val = np.array(y_val).squeeze()

    # Train SVM
    svm = SVC(kernel='linear')  # Linear kernel for faster training
    svm.fit(X_train, y_train)

    # Evaluate
    y_pred = svm.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f'Validation Accuracy: {accuracy:.2f}%')

def main():
    dataset_path = 'UrbanSound8K/audio'
    
    audio_paths = []
    labels = []
    
    for fold in range(1, 11):
        fold_path = os.path.join(dataset_path, f'fold{fold}')
        for file in os.listdir(fold_path):
            if file.endswith('.wav'):
                audio_paths.append(os.path.join(fold_path, file))
                labels.append(file.split('-')[1])
    
    # Use a subset of the data for faster training
    audio_paths = audio_paths[:1000]
    labels = labels[:1000]
    
    window_types = ['hann', 'hamming', 'rectangular']
    
    for window_type in window_types:
        dataset = UrbanSoundDataset(audio_paths, labels, window_type)
        
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            audio_paths, labels, test_size=0.2, random_state=42
        )
        
        train_dataset = UrbanSoundDataset(train_paths, train_labels, window_type)
        val_dataset = UrbanSoundDataset(val_paths, val_labels, window_type)
        
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1)
        
        visualize_spectrograms(train_dataset, window_types)
        
        print(f"Training with {window_type} window")
        train_and_evaluate(train_loader, val_loader)

if __name__ == '__main__':
    main()