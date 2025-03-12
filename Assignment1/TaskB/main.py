import os
import torch
import torchaudio
import matplotlib.pyplot as plt

def generate_spectrograms(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    window_types = ['hann', 'hamming', 'rectangular']
    
    # Create figure for subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Spectrograms for {os.path.basename(audio_path)}', fontsize=14)
    
    for i, window_type in enumerate(window_types):
        # Apply window function
        if window_type == 'hann':
            window = torch.hann_window(waveform.shape[-1])
        elif window_type == 'hamming':
            window = torch.hamming_window(waveform.shape[-1])
        elif window_type == 'rectangular':
            window = torch.ones(waveform.shape[-1])
        else:
            raise ValueError("Unsupported window type")

        windowed_signal = waveform * window

        # Generate spectrogram
        spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512)
        spectrogram = spectrogram_transform(windowed_signal)

        # Plot spectrogram
        axes[i].imshow(spectrogram.log2()[0, :, :].numpy(), aspect='auto', origin='lower')
        axes[i].set_title(f'{window_type.capitalize()} Window')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
    
    # Save plot in the same directory as the audio file
    plot_path = os.path.splitext(audio_path)[0] + '.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved spectrogram plot as {plot_path}")

def main():
    # List of audio files to process
    song_paths = [
        '128-Dooriyan - Love Aaj Kal 128 Kbps.mp3',
        '128-Wada Karo Nahin Chodogi - Aa Gale Lag Jaa 128 Kbps.mp3',
        'soul-of-songs_shakira-waka-waka-this-time-for-africa.mp3',
        'trinity-(titoli)-(annibale-e-i-cantori-moderni)-made-with-Voicemod.mp3'
    ]
    
    for song_path in song_paths:
        if os.path.exists(song_path):
            print(f"Processing {song_path}")
            generate_spectrograms(song_path)
        else:
            print(f"File not found: {song_path}")

if __name__ == '__main__':
    main()
