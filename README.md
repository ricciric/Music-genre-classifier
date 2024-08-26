# music-genre-classifier

A music-genre-classifier is a system that categorizes music tracks into various genres. This classifier analyzes the audio features of music to determine its genre.

## Design overview

Our design integrates multiple sequential steps, treated as a cohesive pipeline. The process begins with data preparation, where audio tracks are segmented based on beats per second (BPS) to ensure consistency across genres. Time-frequency spectrograms are then generated using Short-Time Fourier Transform (STFT), Wavelet Transform (WT), and Mel-Frequency Cepstral Coefficients (MFCC) methods, each offering unique insights into the audio data. These spectrograms are treated as images and processed through convolutional neural networks (CNNs) to extract meaningful features.

To effectively model temporal dependencies, we incorporate a WaveNet-based architecture with dilated convolutions. This setup enables the model to capture both local and long-range patterns within the audio sequences. Adaptive average pooling is then applied to consolidate the features into a streamlined representation, enhancing the robustness of classification.

This design ensures a comprehensive approach to genre classification by combining advanced data processing techniques with state-of-the-art neural network methods.

## Usage

- Clone the repository

```
git clone https://github.com/TheBlind11/music-genre-classifier.git
cd music-genre-classifier
```

- Install the necessary libraries

```
pip install -r requirements.txt
```

- Download the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

- Modify the name of the folder *Data* to *data*

## Testing

- Data preparation

```
python3 bpm.py
```

- If you are interested in classification using either Short-time Fourier Transform (STFT), MEL-Frequency Cepstral Coefficients (MFCC) or Wavelet transform just change the *transform type* in `classifier.py`. In particular, if you are interested in Wavelet transform it's necessary to run the `transform.py` script before classification, other transforms don't need it.

```
python3 transform.py
```

- Classification

```
python3 classifier.py
```