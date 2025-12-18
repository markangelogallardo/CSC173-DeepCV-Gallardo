import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DataAugmentedESC50Dataset(Dataset):
    def __init__(self, csv_file, root_dir, folds):
        """
        Args:
            csv_file (string): Path to the generated metadata CSV (e.g., 'meta_audio_aug.csv').
            root_dir (string): Directory containing the .npy files (e.g., 'data/spectrograms/audio_aug').
            folds (list): List of folds to include (e.g., [1, 2, 3] for training).
        """
        self.annotations = pd.read_csv(csv_file)
        
        # Filter by fold
        self.annotations = self.annotations[self.annotations['fold'].isin(folds)]
        
        # Reset index to avoid issues after filtering
        self.annotations = self.annotations.reset_index(drop=True)
        
        self.root_dir = root_dir

        # ImageNet Transformation Pipeline
        self.transform = transforms.Compose([
            # 1. Resize to 224x224 (Standard for MobileNet/EfficientNet/RegNet)
            transforms.Resize((224, 224)),
            
            # 2. Normalize using ImageNet stats
            # Note: This expects the input to be a float tensor.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # 1. Get Filename and Label
        npy_filename = self.annotations.iloc[index]['filename']
        # Ensure column name matches your CSV (might be 'target' or 'category')
        label = self.annotations.iloc[index]['target'] 
        
        file_path = os.path.join(self.root_dir, npy_filename)

        # 2. Load the Pre-computed Spectrogram
        try:
            # FIX: Added allow_pickle=True here
            spec_array = np.load(file_path, allow_pickle=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find .npy file: {file_path}")

        # 3. Convert to Tensor
        # If the array was saved as a jagged object, we ensure it's float32
        spec_tensor = torch.from_numpy(spec_array).float()

        # 4. Add Channel Dimension [H, W] -> [1, H, W]
        if spec_tensor.ndim == 2:
            spec_tensor = spec_tensor.unsqueeze(0)

        # 5. Convert to 3-Channel RGB [1, H, W] -> [3, H, W]
        spec_rgb = spec_tensor.repeat(3, 1, 1)

        # 6. Apply Resize and Normalization
        spec_processed = self.transform(spec_rgb)

        return spec_processed, torch.tensor(label, dtype=torch.long)