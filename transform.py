import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pywt
import soundfile as sf
from tqdm import tqdm

def save_spectrogram_image(S, sr, output_path, title):
    # Save the spectrogram as a matplot image
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_spectrograms(segment_path, genre, track_name, segment_name, output_base_path):
    # Load the audio segment
    y, sr = sf.read(segment_path)
    
    # Create output directories for each transformation
    wavelet_dir = os.path.join(output_base_path, 'Wavelet', genre, track_name)
    os.makedirs(wavelet_dir, exist_ok=True)
    
    # Wavelet Transform
    coeffs, freqs = pywt.cwt(y, scales=np.arange(1, 65), wavelet='morl', sampling_period=1/sr)
    wavelet_output_path = os.path.join(wavelet_dir, f"{segment_name}_wavelet.npy")
    np.save(wavelet_output_path, coeffs)

def process_segments(input_base_path, output_base_path):
    for genre in os.listdir(input_base_path):
        genre_path = os.path.join(input_base_path, genre)
        if os.path.isdir(genre_path):
            for track_name in tqdm(os.listdir(genre_path)):
                track_path = os.path.join(genre_path, track_name)
                if os.path.isdir(track_path):
                    for segment_file in os.listdir(track_path):
                        segment_path = os.path.join(track_path, segment_file)
                        segment_name, _ = os.path.splitext(segment_file)
                        generate_spectrograms(segment_path, genre, track_name, segment_name, output_base_path)

input_base_path = 'segmented_genres'
output_base_path = 'transformed_spectrograms'

# Process segments to generate and save spectrograms
process_segments(input_base_path, output_base_path)
