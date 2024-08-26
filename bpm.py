import os
import librosa
import soundfile as sf
from tqdm import tqdm
import shutil

def get_bps(filename):
    try:
        y, sr = sf.read(filename)
        y = y.T
        if y.shape[0] == 2:  # if stereo, average the two channels to get mono
            y = y.mean(axis=0)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bps = tempo[0] / 60

        return bps
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def process_dataset(dataset_path):
    genres = os.listdir(dataset_path)
    bpm_data = []
    
    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        if os.path.isdir(genre_path):
            tracks = os.listdir(genre_path)
            for track in tracks:
                track_path = os.path.join(genre_path, track)
                if track_path.endswith('.wav'):
                    bpm = get_bps(track_path)
                    if bpm is not None:
                        bpm_data.append((genre, track, bpm))
                        print(f"Processed {track}, in {genre}: BPM = {bpm}")
                    else:
                        continue
    return bpm_data

def get_segments(filename, bps, segment_lenght=2):
    try:
        y, sr = librosa.load(filename)
        beat_duration = bps
        segment_duration = beat_duration * segment_lenght # Duration of each segment in seconds
        segment_samples = int(segment_duration * sr) # Number of segment samples
        
        segments = []
        for start_sample in range(0, len(y), segment_samples): # From 0 to track lenght with segment lenght steps
            end_sample = start_sample + segment_samples
            segment = y[start_sample:end_sample]
            if len(segment) == segment_samples:
                segments.append(segment)
            
        return segments, sr
    except Exception as e:
        print(f"Error processing {filename} for segmentation: {e}")
        return None, None

def process_save_segment(dataset_path, bps_data, output_path):
    for genre, track, bps in bps_data:
        if bps is None:
            continue
        
        track_path = os.path.join(dataset_path, genre, track)
        segments, sr = get_segments(track_path, bps)
        
        if segments is not None:
            track_name = os.path.splitext(track)[0] # Remove .wav extension
            track_output_path = os.path.join(output_path, genre, track_name)
            
            # Remove the dir if exists
            if os.path.exists(track_output_path):
                shutil.rmtree(track_output_path)
            
            # Create a dir
            os.makedirs(track_output_path)
            
            for i, segment in enumerate(segments):
                segment_filename = f"{track_name}_segment_{i+1}.wav"
                segment_path = os.path.join(track_output_path, segment_filename)
                sf.write(segment_path, segment, sr)
                print(f"Saved segment {i+1} for {track} in {genre}")

def bps_analisys(dataset_path):
    genres = os.listdir(dataset_path)
    bps_data = {}
    
    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        if os.path.isdir(genre_path):
            tracks = os.listdir(genre_path)
            for track in tqdm(tracks):
                track_path = os.path.join(genre_path, track)
                if track_path.endswith('.wav'):
                    bps = get_bps(track_path)
                    if genre not in bps_data:
                        bps_data[genre] = []
                    if bps is not None:
                        bps_data[genre].append(bps)

    print(bps_data)



dataset_path = "data/genres_original"
output_path = "segmented_genres"
#bps_analisys(dataset_path)
bps_data = process_dataset(dataset_path)
process_save_segment(dataset_path, bps_data, output_path)
