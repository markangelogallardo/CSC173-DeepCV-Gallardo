import os
import glob
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
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
AUDIO_SUFFIX = ["rain_bg", "crickets_bg"]
SPEC_SUFFIX = ["time_mask", "freq_mask"]

class AugmentData():
    def __init__(self, input_audio_dir, input_metadata_path, output_dir, 
                 output_metadata_path, output_metadata_filename, rain_bg_path, crickets_bg_path
                 ,sample_rate=44100, aug_type : TYPES = "audio"):
        self.input_audio_dir = input_audio_dir
        self.input_metadata_path = input_metadata_path
        self.output_dir = output_dir
        self.output_metadata_path = output_metadata_path
        self.output_metadata_filename = output_metadata_filename
        self.sample_rate = sample_rate
        self.rain_bg_path = rain_bg_path
        self.crickets_bg_path = crickets_bg_path
        self.aug_type = aug_type
        


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
        
        output_audio_dir = os.path.join(self.output_dir, self.aug_type)
        os.makedirs(output_audio_dir, exist_ok=True)
        for aug_name, transform in AUDIO_SET.items():
            print(f"\n--- Generating Audio Augmentation Set: {aug_name} ---")
            # Use the file list from metadata for reliable processing
            for filename in tqdm(filenames_to_process, desc=f"Augmenting {aug_name}"):
                audio_path = os.path.join(self.input_audio_dir, filename)
                base_name, ext = os.path.splitext(filename)
                new_filename = f"{base_name}_{aug_name}{ext}"
                output_path = os.path.join(output_audio_dir, new_filename)

                # Skip if already processed
                if os.path.exists(output_path):
                    continue
                
                try:
                    # Load Audio
                    samples, sr = sf.read(audio_path, dtype='float32')
                    
                    # Apply Augmentation
                    augmented_samples = transform(samples=samples, sample_rate=sr)
                    
                    # Save Augmented Audio (flat structure)
                    sf.write(output_path, augmented_samples, sr)
                    
                except FileNotFoundError:
                    print(f"\nWarning: Source audio file not found at {audio_path}. Skipping.")
                except Exception as e:
                    print(f"\nError processing {filename} in {aug_name}: {e}")
                    return
                    
            print(f"  {aug_name} generation complete. Files saved to {self.output_dir}/")

    def create_spec_augset(self):
        """
        Applies augmentations and saves the results into flat, augmentation-specific folders.
        Relies on the metadata to define the file set.
        """
        TimeMask = T.TimeMasking(time_mask_param=80)
        FreqMask = T.FrequencyMasking(freq_mask_param=80)
        # Map names to the defined augmentation pipelines

        SPECTROGRAM_SET = {
            "time_mask": TimeMask,
            "freq_mask": FreqMask
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

        for aug_name, transform in SPECTROGRAM_SET.items():
            print(f"\n--- Generating Spectrogram Augmentation Set: {aug_name} ---")
            
            # Create the flat output directory for this set (e.g., data/aug_audio_flat/heavy_rain)
            output_dir_set = os.path.join(self.output_dir, aug_name)
            os.makedirs(output_dir_set, exist_ok=True)
            
            # Use the file list from metadata for reliable processing
            for filename in tqdm(filenames_to_process, desc=f"Augmenting {aug_name}"):
                
                audio_path = os.path.join(self.input_audio_dir, filename)
                output_path = os.path.join(output_dir_set, filename)

                # Skip if already processed
                if os.path.exists(output_path):
                    continue
                
                try:
                    # Load Audio
                    samples, sr = sf.read(audio_path, dtype='float32')
                    
                    # Apply Augmentation
                    augmented_samples = transform(samples=samples, sample_rate=sr)
                    
                    # Save Augmented Audio (flat structure)
                    sf.write(output_path, augmented_samples, sr)
                    
                except FileNotFoundError:
                    print(f"\nWarning: Source audio file not found at {audio_path}. Skipping.")
                except Exception as e:
                    print(f"\nError processing {filename} in {aug_name}: {e}")
                    
            print(f"  {aug_name} generation complete. Files saved to {output_dir_set}/")
    
    def create_hybrid_augset(self):
        #audio_aug -> spectrogram -> spectrogram augmentation
        pass

    def generate_augmented_metadata(self):
        """
        Creates a single, large metadata file containing entries for the original 
        audio files under all defined augmentation groups.
        """
        
        input_path = os.path.join(self.input_metadata_path)
        output_path = os.path.join(self.output_metadata_path, self.output_metadata_filename.replace('.csv', '_audio_augmented.csv'))

        # 1. Load the clean, baseline metadata
        try:
            df_master = pd.read_csv(input_path)
        except FileNotFoundError:
            print(f"Error: Base metadata file not found at {input_path}")
            return

        print(f"Loaded master metadata with {len(df_master)} clean files.")
        
        # Initialize the final list of entries
        all_entries = []

# Define the list of suffixes to iterate over based on self.aug_type
        if self.aug_type == "audio":
            # If self.aug_type is 'audio', the loop runs once with the AUDIO_SUFFIX
            suffixes_to_process = AUDIO_SUFFIX
        elif self.aug_type == "spec":
            # If self.aug_type is 'spec', the loop runs once with the SPEC_SUFFIX
            suffixes_to_process = SPEC_SUFFIX
        else:
            # Handle the case if aug_type is neither (e.g., 'both' or 'none'), though your logic implies specific flags
            # We will assume a default to prevent error, or raise an exception
            print(f"Warning: Unknown aug_type '{self.aug_type}'. Skipping augmented metadata generation.")
            suffixes_to_process = []


        for aug_suffix in tqdm(suffixes_to_process, desc="Generating Augmented Metadata"):
            df_aug = df_master.copy()
            
            # Store the original filename for reference
            df_aug['source_filename'] = df_master['filename']
            
            # Create a NEW, unique filename for this augmented version
            # The suffix value (e.g., 'audio_only' or 'spec_only') is used here
            df_aug['filename'] = df_master['filename'].str.replace('.wav', f'_{aug_suffix}.wav', regex=False)
            
            # Set the augmentation type column if needed (useful for PyTorch DataLoader lookup)
            # df_aug['augmentation_type'] = aug_suffix # Uncomment if needed
            
            all_entries.append(df_aug)

        # 4. Concatenate all entries (Baseline + the Augmented Sets generated above)
        # IMPORTANT: If your df_master already includes the baseline, this will be correct.
        # If df_master only includes the clean metadata, you need to ensure the baseline is added too.
        df_final_master = pd.concat(all_entries, ignore_index=True)

        # 5. Save the final single, appended blueprint
        df_final_master.to_csv(output_path, index=False)
        
        print("\n--- ✅ Master Augmentation Metadata Blueprint Complete ---")
        print(f"Total entries created: {len(df_final_master)}")
        print(f"Final blueprint saved to: {output_path}")

    def run(self):
        if self.aug_type == "audio":
            self.generate_augmented_metadata()
            # self.create_audio_augset()
        elif self.aug_type == "spectrogram":
            self.create_spec_augset()
        elif self.aug_type == "hybrid":
            self.create_hybrid_augset()
        else:
            print(f"Error: Unknown augmentation type '{self.aug_type}'. No action taken.")
            return
        
        
                