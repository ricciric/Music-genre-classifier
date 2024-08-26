import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torchaudio
import os
import pywt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

class MusicDataset(Dataset):
    def __init__(self, root_dir, transform=None, transform_type='wavelet'):
        self.root_dir = root_dir
        self.transform = transform
        self.transform_type = transform_type
        self.classes = os.listdir(root_dir)
        self.file_list = []
        
        if transform_type == 'wavelet':
            wavelet_transform_path = 'transformed_spectrograms/Wavelet'
            for genre in os.listdir(wavelet_transform_path):
                genre_dir = os.path.join(wavelet_transform_path, genre)
                for track in os.listdir(genre_dir):
                    track_dir = os.path.join(genre_dir, track)
                    for segment in os.listdir(track_dir):
                        file_path = os.path.join(track_dir, segment)
                        self.file_list.append((file_path, self.classes.index(genre)))
        else:                
            for c in self.classes:
                class_dir = os.path.join(root_dir, c)
                for track in os.listdir(class_dir):
                    for sample in os.listdir(os.path.join(class_dir, track)):
                        self.file_list.append((os.path.join(class_dir, track, sample), self.classes.index(c)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        if self.transform_type == 'wavelet':
            spectrogram = np.load(file_path)
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
            # Ensure the spectrogram has a consistent shape, for example (64, 690)
            target_shape = (64, 690)
            # The precomputed wavelets have different times we need to add some padding
            if spectrogram.shape != target_shape:
                spectrogram = self.pad_or_crop_spectrogram(spectrogram, target_shape)
        else:
            waveform, _ = self.load_and_pad_audio(file_path)
            if self.transform_type == 'stft':
                spectrogram = self.stft_transform(waveform)
            elif self.transform_type == 'mel':
                spectrogram = self.mel_transform(waveform)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, label
    
    # Add padding to the wavelet transformed spectrogram to make it consistent with the other spectrograms
    def pad_or_crop_spectrogram(self, spectrogram, target_shape): 
        target_channels, target_time = target_shape
        channels, time = spectrogram.shape
        
        if time < target_time:
            pad_size = target_time - time
            spectrogram = torch.nn.functional.pad(spectrogram, (0, pad_size))
        elif time > target_time:
            spectrogram = spectrogram[:, :target_time]

        if channels < target_channels:
            pad_size = target_channels - channels
            spectrogram = torch.nn.functional.pad(spectrogram, (0, 0, 0, pad_size))
        elif channels > target_channels:
            spectrogram = spectrogram[:target_channels, :]

        return spectrogram

    def load_and_pad_audio(self, file_path, max_duration=5):
        waveform, sample_rate = torchaudio.load(file_path)
        max_samples = int(max_duration * sample_rate)
        if waveform.size(1) < max_samples:
            pad_size = max_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        elif waveform.size(1) > max_samples:
            waveform = waveform[:, :max_samples]
        return waveform, sample_rate

    def mel_transform(self, waveform):
        spectrogram = torchaudio.transforms.MelSpectrogram(n_mels=64)(waveform).squeeze(0)
        return spectrogram
    
    def stft_transform(self, waveform):
        spectrogram = torchaudio.transforms.Spectrogram(n_fft=126)(waveform).squeeze(0) # Using 64 channels ((126/2) + 1)
        return spectrogram


class WaveNet(nn.Module):
    def __init__(self, num_classes, in_channels=64, channels=64, kernel_size=3, dilation_factors=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]):
        super(WaveNet, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation_factors = dilation_factors
        self.conv_layers = nn.ModuleList()
        for dilation in dilation_factors:
            self.conv_layers.append(
                nn.Conv1d(in_channels, channels, kernel_size, stride=1, padding=dilation, dilation=dilation)
            )
        self.final_conv = nn.Conv1d(channels, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        for conv in self.conv_layers:
            x = self.relu(conv(x))
        x = self.final_conv(x)
        x = self.pool(x)
        x = x.squeeze(2)
        return x

def save_metrics_plots(metrics_dict, transform_type):
    folder_path = os.path.join('metrics', transform_type + "_fixed")
    os.makedirs(folder_path, exist_ok=True)
    
    plt.figure()
    for metric_name, values in metrics_dict.items():
        plt.plot(range(1, len(values) + 1), values, marker='o', label=metric_name)
    
    plt.title(f'Metrics per Fold - {transform_type}')
    plt.xlabel('Fold Number')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'metrics.png'))
    plt.close()

if __name__ == '__main__':
    
    transform_type = 'Wavelet Transform'
    dataset = MusicDataset(root_dir='segmented_genres', transform_type="wavelet")

    k_folds = 10
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    num_classes = len(dataset.classes)
    wavenet_config = {
        'in_channels': 64,
        'channels': 64,
        'kernel_size': 3,
        'dilation_factors': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    }

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    model = WaveNet(num_classes=num_classes, **wavenet_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    output_path = "trained_nets"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    metrics_dict = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(dataset.file_list, [label for _, label in dataset.file_list])):
        print(f'Fold [{fold_idx + 1}/{k_folds}]')
        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler, num_workers=4, pin_memory=True)
        
        num_epochs = 3
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for spectrograms, labels in tqdm(train_loader, desc="Training"):
                spectrograms, labels = spectrograms.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(spectrograms)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * spectrograms.size(0)
            
            epoch_loss = running_loss / len(train_loader)
            print(f"Fold [{fold_idx + 1}/{k_folds}], Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        torch.save(model.state_dict(), os.path.join(output_path, f'wavenet_music_genre_fold_{fold_idx + 1}.pth'))

        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for spectrograms, labels in tqdm(val_loader, desc="Testing"):
                spectrograms, labels = spectrograms.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(spectrograms)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        fold_accuracy = 100 * correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        metrics_dict['Accuracy'].append(fold_accuracy)
        metrics_dict['Precision'].append(precision * 100)
        metrics_dict['Recall'].append(recall * 100)
        metrics_dict['F1-Score'].append(f1 * 100)

        print(f"Fold [{fold_idx + 1}/{k_folds}], Accuracy on validation set: {fold_accuracy:.2f}%, Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%, F1-Score: {f1 * 100:.2f}%")
        print("=" * 50)

    # Save combined metrics plot after all folds are complete
    save_metrics_plots(metrics_dict, transform_type)

    avg_accuracy = sum(metrics_dict['Accuracy']) / len(metrics_dict['Accuracy'])
    avg_precision = sum(metrics_dict['Precision']) / len(metrics_dict['Precision'])
    avg_recall = sum(metrics_dict['Recall']) / len(metrics_dict['Recall'])
    avg_f1 = sum(metrics_dict['F1-Score']) / len(metrics_dict['F1-Score'])

    print(f'Average accuracy across {k_folds} folds: {avg_accuracy:.2f}%')
    print(f'Average precision across {k_folds} folds: {avg_precision:.2f}%')
    print(f'Average recall across {k_folds} folds: {avg_recall:.2f}%')
    print(f'Average F1-score across {k_folds} folds: {avg_f1:.2f}%')

    torch.save(model.state_dict(), os.path.join(output_path, 'wavenet_music_genre_final.pth'))
