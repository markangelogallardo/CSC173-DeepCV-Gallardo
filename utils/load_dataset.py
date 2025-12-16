import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision import transforms

class ESC50Dataset(Dataset):
    def __init__(self, csv_file, audio_dir, folds, target_sample_rate=16000, duration=5):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            audio_dir (string): Directory with all the images.
            folds (list): List of folds to include (e.g., [1, 2, 3] for training).
        """
        self.annotations = pd.read_csv(csv_file)
        # Filter by fold
        self.annotations = self.annotations[self.annotations['fold'].isin(folds)]
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate * duration # 5 seconds usually

        # Audio to Mel Spectrogram Transformation
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_mels=64,  # Height of the image
            n_fft=1024,
            hop_length=512
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # Resize to fit standard model inputs (224x224 is standard for ResNet/VGG)
        self.resize = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        label = self.annotations.iloc[index, 2] # Column 2 is 'target' (0-49)
        
        # 1. Load Audio
        signal, sr = torchaudio.load(audio_sample_path)
        
        # 2. Resample if necessary
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
            
        # 3. Ensure fixed length (pad or truncate)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        elif signal.shape[1] < self.num_samples:
            padding = self.num_samples - signal.shape[1]
            signal = torch.nn.functional.pad(signal, (0, padding))

        # 4. Convert to Spectrogram
        spec = self.mel_spectrogram(signal)
        spec = self.amplitude_to_db(spec)

        # 5. Turn into 3-channel image for ImageNet models
        # Shape becomes [3, 224, 224]
        spec_image = self.resize(spec)
        spec_image = spec_image.repeat(3, 1, 1) 

        # 6. Normalize (Standard ImageNet mean/std)
        # Note: This normalization is for 0-1 range images, spectrograms are not 0-1.
        # Ideally, calculate dataset specific mean/std, but this often works fine for transfer learning.
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        spec_image = norm(spec_image)

        return spec_image, label
    
class NoAug(Dataset):
    def __init__(self, csv_file, audio_dir, folds, target_sample_rate=16000, duration=5):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            audio_dir (string): Directory with all the images.
            folds (list): List of folds to include (e.g., [1, 2, 3] for training).
        """
        self.annotations = pd.read_csv(csv_file)
        # Filter by fold
        self.annotations = self.annotations[self.annotations['fold'].isin(folds)]
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate * duration # 5 seconds usually

        # Audio to Mel Spectrogram Transformation
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_mels=64,  # Height of the image
            n_fft=1024,
            hop_length=512
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # Resize to fit standard model inputs (224x224 is standard for ResNet/VGG)
        self.resize = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        label = self.annotations.iloc[index, 2] # Column 2 is 'target' (0-49)
        
        # 1. Load Audio
        signal, sr = torchaudio.load(audio_sample_path)
        
        # 2. Resample if necessary
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
            
        # 3. Ensure fixed length (pad or truncate)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        elif signal.shape[1] < self.num_samples:
            padding = self.num_samples - signal.shape[1]
            signal = torch.nn.functional.pad(signal, (0, padding))

        # 4. Convert to Spectrogram
        spec = self.mel_spectrogram(signal)
        spec = self.amplitude_to_db(spec)

        # 5. Turn into 3-channel image for ImageNet models
        # Shape becomes [3, 224, 224]
        spec_image = self.resize(spec)
        spec_image = spec_image.repeat(3, 1, 1) 

        # 6. Normalize (Standard ImageNet mean/std)
        # Note: This normalization is for 0-1 range images, spectrograms are not 0-1.
        # Ideally, calculate dataset specific mean/std, but this often works fine for transfer learning.
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        spec_image = norm(spec_image)

        return spec_image, label