# This code is for question 3 of the minor project
# The code is used to classify vowels from audio files
# Importing all the tools we need for audio processing and machine learning
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import pandas as pd
import os

class VowelAnalyzer:
    def __init__(self):
        # Setting up some basic parameters for chopping up our audio
        self.frame_size = 2048    # How big each chunk of audio should be
        self.hop_size = 512       # How far we jump ahead each time
        self.sample_rate = 16000  # Good enough for voice, no need for CD quality
        
        # Using KNN because it's simple and works well for vowel classification
        # 5 neighbors because we have 5 vowels - makes sense, right?
        self.model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    
    def read_audio(self, path):
        # Audio files can be tricky, better use try-except
        try:
            audio, _ = librosa.load(path, sr=self.sample_rate)
            return audio
        except:
            print(f"Uh oh, couldn't read this file: {path}")
            return None
    
    def split_frames(self, audio):
        # Breaking down our audio into smaller chunks
        # Makes it easier to analyze bit by bit
        frames = librosa.util.frame(audio, 
                                  frame_length=self.frame_size,
                                  hop_length=self.hop_size)
        return frames.T
    
    def add_window(self, frames):
        # Smoothing out the edges of our audio chunks
        # Helps avoid weird effects at the boundaries
        window = np.hamming(self.frame_size)
        return frames * window
    
    def get_lpc(self, frame):
        # This is where the magic happens - Linear Predictive Coding
        # Helps us find patterns in the speech
        order = 13  # This number works well for speech - kind of a magic number
        
        # Getting the signal to compare with itself
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        
        # Some linear algebra to find the patterns
        r = corr[:order+1]
        R = np.zeros((order, order))
        
        for i in range(order):
            for j in range(order):
                R[i,j] = r[abs(i-j)]
        
        # Solving the equation to get our coefficients
        a = np.linalg.solve(R, -r[1:order+1])
        return np.concatenate(([1], a))
    
    def find_formants(self, coeffs):
        # Finding the important frequencies that make up each vowel
        roots = np.roots(coeffs)
        roots = roots[np.imag(roots) >= 0]  # Only care about positive frequencies
        angles = np.arctan2(np.imag(roots), np.real(roots))
        
        # Converting to actual frequencies we can understand
        freqs = angles * (self.sample_rate / (2 * np.pi))
        return sorted(freqs)[:3]  # First three are the most important
    
    def get_pitch(self, frame):
        # Finding the basic pitch of the voice
        # Most human speech is between 50-500 Hz
        f0 = librosa.yin(frame, 
                        fmin=50, 
                        fmax=500, 
                        sr=self.sample_rate)
        return np.mean(f0)
    
    def get_features(self, file):
        # Main function to extract all the good stuff from each audio file
        audio = self.read_audio(file)
        if audio is None:
            return None
            
        # Process the audio in chunks
        frames = self.split_frames(audio)
        windowed = self.add_window(frames)
        
        # Get features from each chunk
        features = []
        for frame in windowed:
            coeffs = self.get_lpc(frame)
            formants = self.find_formants(coeffs)
            pitch = self.get_pitch(frame)
            
            # Only keep if we got all three formants
            if len(formants) >= 3:
                features.append([pitch] + list(formants[:3]))
        
        # Average everything to get one set of features
        return np.mean(features, axis=0) if features else np.zeros(4)
    
    def process_files(self, base_path):
        # Time to go through all our files
        features = []
        labels = []
        vowels = ['a', 'e', 'i', 'o', 'u']
        genders = ['Female', 'Male']
        
        # Look through both male and female folders
        for gender in genders:
            gender_path = os.path.join(base_path, 'Dataset', gender)
            
            if not os.path.exists(gender_path):
                print(f"Can't find this folder: {gender_path}")
                continue
            
            # Go through each vowel folder
            for v in vowels:
                vowel_path = os.path.join(gender_path, v)
                if not os.path.exists(vowel_path):
                    print(f"Missing vowel folder: {vowel_path}")
                    continue
                
                # Process each wav file
                for file in os.listdir(vowel_path):
                    if file.endswith('.wav'):
                        file_path = os.path.join(vowel_path, file)
                        feat = self.get_features(file_path)
                        if feat is not None:
                            features.append(feat)
                            labels.append(vowels.index(v))
                            print(f"Just processed: {gender}/{v}/{file}")
        
        return np.array(features), np.array(labels)
    
    def train_model(self, X_train, y_train):
        # Teaching our model what each vowel looks like
        self.model.fit(X_train, y_train)
    
    def test_model(self, X_test):
        # Let's see how well it learned
        return self.model.predict(X_test)
    
    def plot_space(self, features, labels):
        # Making a nice plot of our vowel space
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features[:, 1], features[:, 2], c=labels, cmap='Set1')
        plt.xlabel('First Formant (F1) Hz')
        plt.ylabel('Second Formant (F2) Hz')
        plt.title('How Our Vowels Are Distributed')
        
        # Add labels for each vowel
        vowels = ['a', 'e', 'i', 'o', 'u']
        legend1 = plt.legend(scatter.legend_elements()[0], 
                           vowels,
                           title="Vowels",
                           loc="upper right")
        plt.gca().add_artist(legend1)
        
        plt.savefig('vowel_space.png')
        plt.close()
    
    def plot_results(self, true, pred):
        # Making a confusion matrix to see where we got things wrong
        vowels = ['a', 'e', 'i', 'o', 'u']
        cm = confusion_matrix(true, pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=vowels,
                   yticklabels=vowels)
        plt.xlabel('What we predicted')
        plt.ylabel('What it actually was')
        plt.title('Our Prediction Results')
        plt.savefig('confusion_matrix.png')
        plt.close()

def main():
    # Let's get this party started
    analyzer = VowelAnalyzer()
    
    # Figure out where we are
    base_path = os.path.dirname(os.path.abspath(__file__))
    print("Starting to read all those files...")
    
    features, labels = analyzer.process_files(base_path)
    
    if len(features) == 0:
        print("Oops, couldn't process any files!")
        return
    
    print(f"Great! Processed {len(features)} files")
    
    # Split our data for training and testing
    print("Splitting up our data...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=0.2,  # Keep 20% for testing
        random_state=45,  # For reproducibility
        stratify=labels  # Make sure we have all vowels represented
    )
    
    print("Training our classifier...")
    analyzer.train_model(X_train, y_train)
    
    print("Testing how well it works...")
    predictions = analyzer.test_model(X_test)
    
    print("Making some nice plots...")
    analyzer.plot_space(features, labels)
    analyzer.plot_results(y_test, predictions)
    
    # Let's see how well we did
    acc = accuracy_score(y_test, predictions)
    print("\nHere's how we did:")
    print("-" * 20)
    print(f"Overall accuracy: {acc:.2f}")
    
    # Check each vowel separately
    vowels = ['a', 'e', 'i', 'o', 'u']
    cm = confusion_matrix(y_test, predictions)
    for i, vowel in enumerate(vowels):
        vowel_acc = cm[i,i] / cm[i,:].sum()
        print(f"Vowel '{vowel}' accuracy: {vowel_acc:.2f}")

if __name__ == "__main__":
    main()