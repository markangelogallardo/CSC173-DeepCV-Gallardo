import os
import glob
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
import torch
import torchaudio
from torchvision import transforms
from tqdm import tqdm
from audiomentations import Compose, TimeStretch, PitchShift, AddBackgroundNoise
import torchaudio.transforms as T
from typing import Literal, get_args, get_origin
from sys import _getframe
# --- 1. CONFIGURATION ---

# !!! IMPORTANT: VERIFY AND ADJUST THESE PATHS !!!
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
TYPES = Literal["audio", "spectrogram", "hybrid"]

class AugmentData():
    def __init__(self, input_audio_dir, input_metadata_path, output_dir, 
                 output_metadata_path, rain_bg_path, crickets_bg_path
                 ,sample_rate=22050, aug_type : TYPES = "audio"):
        self.input_audio_dir = input_audio_dir
        self.input_metadata_path = input_metadata_path
        self.output_dir = output_dir
        self.output_metadata_path = output_metadata_path
        self.sample_rate = sample_rate
        self.rain_bg_path = rain_bg_path
        self.crickets_bg_path = crickets_bg_path
        self.og_aug_type = aug_type
        self.aug_type = aug_type
        self.num_samples = self.sample_rate * 5  # Assuming 5 seconds duration
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=128,  # Height of the image
            n_fft=1024,
            hop_length=512
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        
        # Resize to fit standard model inputs (224x224 is standard for ResNet/VGG)
        self.resize = transforms.Resize((224, 224))
        
    def generate_augmented_metadata(self):
        """
        Creates a single, large metadata file containing entries for the original 
        audio files under all defined augmentation groups.
        """
        
        input_path = os.path.join(self.input_metadata_path)
        output_path = os.path.join(self.output_metadata_path, f"ESC50_subgroup_{self.aug_type}.csv")

        # 1. Load the clean, baseline metadata
        try:
            df_master = pd.read_csv(input_path)
        except FileNotFoundError:
            print(f"Error: Base metadata file not found at {input_path}")
            return

        print(f"Loaded master metadata with {len(df_master)} clean files.")
        
        # Initialize the final list of entries
        all_entries = []
        
        # Define augmentation configurations based on type
        # Each config dict contains:
        # - 'suffix_chain': The string to append to the filename (e.g., '_rain_bg_time_mask')
        # - 'aug_label': The readable description for the new column (e.g., 'rain_bg + time_mask')
        # - 'extension': The final file extension (.wav or .npy)
        aug_configs = []

        if self.aug_type == "audio":
            # Independent Audio Augmentations
            for aug in ["rain_bg", "crickets_bg"]:
                aug_configs.append({
                    "suffix_chain": f"_{aug}",
                    "aug_label": aug,
                    "extension": ".wav"
                })

        elif self.aug_type == "spectrogram" and self.og_aug_type == "audio":
            # Independent Spectrogram Augmentations
            for aug in ["rain_bg", "rain_bg"]:
                aug_configs.append({
                    "suffix_chain": f"_{aug}",
                    "aug_label": aug,
                    "extension": ".npy"
                })
        
        elif self.aug_type == "spectrogram" and self.og_aug_type == "spectrogram":
            # Independent Spectrogram Augmentations
            for aug in ["time_mask", "freq_mask"]:
                aug_configs.append({
                    "suffix_chain": f"_{aug}",
                    "aug_label": aug,
                    "extension": ".npy"
                })        

        elif self.aug_type == "hybrid":
            # Combinatorial: Audio Augmentations + Spectrogram Augmentations
            audio_subtypes = ["rain_bg", "crickets_bg"]
            spec_subtypes = ["time_mask", "freq_mask"]
            
            for aud in audio_subtypes:
                for spec in spec_subtypes:
                    # Creates entries like: rain_bg + time_mask, rain_bg + freq_mask, etc.
                    aug_configs.append({
                        "suffix_chain": f"_{aud}_{spec}", 
                        "aug_label": f"{aud} + {spec}",
                        "extension": ".npy"
                    })

        # 2. Process each configuration
        for config in tqdm(aug_configs, desc="Generating Augmented Metadata"):
            df_aug = df_master.copy()
            
            # Store the original filename for reference
            df_aug['source_filename'] = df_master['filename']
            
            # Create the new column specifying the augmentation applied
            df_aug['augmentation_type'] = config['aug_label']
            
            # Modify the filename: Remove .wav, add suffix chain, add correct extension
            # We use regex=False for speed and safety
            base_names = df_aug['filename'].str.replace('.wav', '', regex=False)
            df_aug['filename'] = base_names + config['suffix_chain'] + config['extension']
            
            all_entries.append(df_aug)

        # 3. Concatenate and Save
        if all_entries:
            df_final_master = pd.concat(all_entries, ignore_index=True)
            df_final_master.to_csv(output_path, index=False)
            
            print("\n--- ✅ Master Augmentation Metadata Blueprint Complete ---")
            print(f"Total entries created: {len(df_final_master)}")
            print(f"Final blueprint saved to: {output_path}")
        else:
            print("Warning: No augmentation entries were generated.")

    def create_spec_augset(self):
        """
        Applies SpecAugment masking, saves as .npy, and generates a new metadata file.
        """
        # 1. Setup Masking Transforms
        if self.aug_type == "spectrogram" or self.aug_type == "hybrid":
            SPECTROGRAM_SET = {
                "time_mask": T.TimeMasking(time_mask_param=80),
                "freq_mask": T.FrequencyMasking(freq_mask_param=80)
            }
        else:
            SPECTROGRAM_SET = {
                "rain_bg": "",
                "crickets_bg": ""
            }

        # Load the base metadata
        try:
            df_metadata = pd.read_csv(self.input_metadata_path)
        except FileNotFoundError:
            print(f"ERROR: Metadata not found at {self.input_metadata_path}")
            return

        filenames_to_process = df_metadata['filename'].tolist()
        
        # Define the directory for .npy files
        output_spec_dir = os.path.join(self.output_dir, self.aug_type + "_masking")
        os.makedirs(output_spec_dir, exist_ok=True)

        # List to store new metadata entries
        new_metadata_entries = []

        for aug_name, transform in SPECTROGRAM_SET.items():
            print(f"\n--- Generating .npy Spectrogram Augmentation Set: {aug_name} ---")
            
            for index, row in tqdm(df_metadata.iterrows(), total=len(df_metadata), desc=f"Augmenting {aug_name}"):
                filename = row['filename']
                audio_path = os.path.join(self.input_audio_dir, filename)
                
                # 1. Create unique name for .npy file
                base_name = os.path.splitext(filename)[0]
                new_filename = f"{base_name}_{aug_name}.npy"
                output_path = os.path.join(output_spec_dir, new_filename)

                # Process if doesn't exist
                if not os.path.exists(output_path):
                    try:
                        # A. Extract Log-Mel using your Librosa logic
                        y, sr = librosa.load(audio_path, sr=self.sample_rate)
                        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
                        log_mel = librosa.power_to_db(mel, ref=np.max)

                        # B. Standardize sequence length (216)
                        if log_mel.shape[1] > 216:
                            log_mel = log_mel[:, :216]
                        elif log_mel.shape[1] < 216:
                            pad_width = 216 - log_mel.shape[1]
                            log_mel = np.pad(log_mel, pad_width=((0, 0), (0, pad_width)), 
                                            mode='constant', constant_values=log_mel.min())

                        # C. Apply Masking (Convert to Tensor -> Mask -> Back to Numpy)
                        # torchaudio masks expect [batch, freq, time] or [freq, time]
                        log_mel_tensor = torch.from_numpy(log_mel)
                        if self.aug_type == "spectrogram" or self.aug_type == "hybrid":
                            masked_spec = transform(log_mel_tensor)
                        
                        # D. Save as .npy
                        np.save(output_path, masked_spec.numpy())

                    except FileNotFoundError:
                        print(f"\nWarning: Source audio file not found at {audio_path}. Skipping.")
                    except Exception as e:
                        print(f"\nError processing {filename} in {aug_name}: {e}")
                        return

                # 2. Add to Metadata list regardless of whether we just created it or it existed
                new_row = row.copy()
                new_row['filename'] = new_filename # Update filename to the .npy version
                new_row['aug_name'] = aug_name     # Record which mask was used
                new_metadata_entries.append(new_row)

            print(f"  {aug_name} generation complete. Files saved to {output_spec_dir}/")
        self.aug_type = "spectrogram"
        self.generate_augmented_metadata()
    def create_audio_augset(self):
        """
        Applies augmentations and saves the results into flat, augmentation-specific folders.
        Relies on the metadata to define the file set.
        """
        CricketsAug = Compose([
            TimeStretch(min_rate=0.7, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-5.0, max_semitones=5.0, p=0.5),\
            AddBackgroundNoise(
                sounds_path=self.crickets_bg_path,
                min_snr_db=3.0, max_snr_db=25.0, p=1.0,
            ),
        ])
        RainAug = Compose([
            TimeStretch(min_rate=0.7, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-5.0, max_semitones=5.0, p=0.5),\
            AddBackgroundNoise(
                sounds_path=self.rain_bg_path,
                min_snr_db=3.0, max_snr_db=25.0, p=1.0, 
            ),
        ])
        AUDIO_SET = {
            "rain_bg": RainAug,
            "crickets_bg": CricketsAug
        }
        # Load the metadata blueprint to get the list of files to process
        try:
            df_metadata = pd.read_csv(self.input_metadata_path)
        except FileNotFoundError:
            print(f"ERROR: Metadata file not found at {self.input_metadata_path}. Cannot proceed.")
            return

        # Get the list of filenames to process
        filenames_to_process = df_metadata['filename'].tolist()
        
        if not filenames_to_process:
            print("ERROR: Metadata is empty or corrupt. No files to process.")
            return
        
        output_audio_dir = os.path.join(self.output_dir)
        if os.path.exists(output_audio_dir):
            # Using 'return' if this is inside a function to exit early
            pass 
        else:
            # 3. If it doesn't exist, create it and proceed with the generation
            os.makedirs(output_audio_dir)
        for aug_name, transform in AUDIO_SET.items():
            print(f"\n--- Generating Audio Augmentation Set: {aug_name} ---")
            # Use the file list from metadata for reliable processing
            for filename in tqdm(filenames_to_process, desc=f"Augmenting {aug_name}"):
                audio_path = os.path.join(self.input_audio_dir, filename)
                base_name, ext = os.path.splitext(filename)
                new_filename = f"{base_name}_{aug_name}{ext}"
                output_path = os.path.join(output_audio_dir, new_filename)

                # Skip if already processed
                if not os.path.exists(output_path):
                    try:
                        # Load Audio
                        samples, sr = sf.read(audio_path, dtype='float32')
                        
                        # Apply Augmentation
                        augmented_spec = transform(samples=samples, sample_rate=sr)
                        
                        # Save Augmented Audio (flat structure)
                        sf.write(output_path, augmented_spec, sr)
                        
                    except FileNotFoundError:
                        print(f"\nWarning: Source audio file not found at {audio_path}. Skipping.")
                    except Exception as e:
                        print(f"\nError processing {filename} in {aug_name}: {e}")
                        return
                    
            print(f"  {aug_name} generation complete. Files saved to {self.output_dir}/")
        self.generate_augmented_metadata()
        self.input_audio_dir = self.output_dir
        self.input_metadata_path = self.output_metadata_path
        if os.path.exists(r"data\spectrograms\audio_aug"):
            self.output_dir = r"data\spectrograms\audio_aug"
        else:
            os.makedirs(r"data\spectrograms\audio_aug")
            self.output_dir = r"data\spectrograms\audio_aug"
        self.create_spec_augset(self)    


    def run(self):
        if self.aug_type == "audio":
            self.create_audio_augset()
        elif self.aug_type == "spectrogram":
            self.create_spec_augset()
        elif self.aug_type == "hybrid":
            self.create_audio_augset()
            self.generate_augmented_metadata()
            self.input_audio_dir = os.path.join(self.output_dir)
            self.input_metadata_path = os.path.join(self.output_metadata_path)
            self.create_spec_augset()
        else:
            print(f"Error: Unknown augmentation type '{self.aug_type}'. No action taken.")
            return
        
                