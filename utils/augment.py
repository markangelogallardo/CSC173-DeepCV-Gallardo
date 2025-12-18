import os
import pandas as pd
import soundfile as sf
import numpy as np
import librosa
import torch
import torchaudio.transforms as T
from tqdm import tqdm
from audiomentations import Compose, TimeStretch, PitchShift, AddBackgroundNoise

class AudioPipeline:
    def __init__(self, raw_audio_dir, metadata_path, output_dir, rain_path, crickets_path):
        self.raw_audio_dir = raw_audio_dir
        self.metadata = pd.read_csv(metadata_path)
        self.output_dir = output_dir
        self.metadata_dir = os.path.dirname(metadata_path)
        
        self.rain_bg_path = rain_path
        self.crickets_bg_path = crickets_path

    def generate_augmented_audio_dataset(self):
        """
        Step 1: 
        - Takes Raw Audio.
        - Applies Rain/Crickets augmentation.
        - Saves .wav files to `[self.output_dir]/audio_aug`.
        - Generates and saves new metadata to `[self.output_dir]/meta_audio_aug.csv`.
        """
        
        CricketsAug = Compose([
            TimeStretch(min_rate=0.7, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-5.0, max_semitones=5.0, p=0.5),
            AddBackgroundNoise(
                sounds_path=self.crickets_bg_path,
                min_snr_db=3.0, max_snr_db=25.0, p=1.0,
            ),
        ])
        
        RainAug = Compose([
            TimeStretch(min_rate=0.7, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-5.0, max_semitones=5.0, p=0.5),
            AddBackgroundNoise(
                sounds_path=self.rain_bg_path,
                min_snr_db=3.0, max_snr_db=25.0, p=1.0, 
            ),
        ])

        AUDIO_SET = {
            "rain_bg": RainAug,
            "crickets_bg": CricketsAug
        }

        output_audio_dir = os.path.join(self.output_dir, "audio_aug")
        output_meta_path = os.path.join(self.metadata_dir, "audio_aug.csv")
        
        os.makedirs(output_audio_dir, exist_ok=True)

        new_metadata_rows = []

        for index, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc="Processing Audio Augmentations"):
            original_filename = row['filename']
            audio_path = os.path.join(self.raw_audio_dir, original_filename)
            
            try:
                samples, sr = sf.read(audio_path, dtype='float32')
            except Exception as e:
                print(f"Error reading {original_filename}: {e}")
                continue

            for aug_name, transform in AUDIO_SET.items():
                
                base_name, ext = os.path.splitext(original_filename)
                new_filename = f"{base_name}_{aug_name}{ext}"
                output_path = os.path.join(output_audio_dir, new_filename)

                if not os.path.exists(output_path):
                    try:
                        augmented_samples = transform(samples=samples, sample_rate=sr)
                        sf.write(output_path, augmented_samples, sr)
                    except Exception as e:
                        print(f"Failed to augment {new_filename}: {e}")
                        continue
                
                new_row = row.copy()
                new_row['filename'] = new_filename
                new_row['source_filename'] = original_filename
                new_row['augmentation_type'] = aug_name
                
                new_metadata_rows.append(new_row)

        if new_metadata_rows:
            df_aug_metadata = pd.DataFrame(new_metadata_rows)
            df_aug_metadata.to_csv(output_meta_path, index=False)
            print(f"\n Audio Augmentation Complete.")
            print(f"   Audio saved to: {output_audio_dir}")
            print(f"   Metadata saved to: {output_meta_path}")
            return output_meta_path
        else:
            print("No metadata generated.")
            return None

    def generate_spectrogram_dataset(self, input_metadata_path, input_audio_dir, output_subdir, spec_augs=[None]):
        """
        Step 2: Generic function to convert ANY audio folder to spectrograms (.npy).
        
        Args:
            input_metadata_path: Path to CSV defining which files to process.
            input_audio_dir: Where the source .wav files are located.
            output_subdir: Folder name to save .npy files (e.g. 'log-mel_spectrograms/no_aug').
            spec_augs: List of masking to apply. 
                        - [None] = Just convert to Spectrogram (no mask).
                        - ['time_mask', 'freq_mask'] = Apply masks before saving.
        """
        try:
            df_metadata = pd.read_csv(input_metadata_path)
        except FileNotFoundError:
            print(f"ERROR: Metadata not found at {input_metadata_path}")
            return None

        output_spec_dir = os.path.join(self.output_dir, output_subdir)
        os.makedirs(output_spec_dir, exist_ok=True)
        
        safe_name = output_subdir.replace("/", "_").replace("\\", "_")
        output_meta_path = os.path.join(self.metadata_dir, f"{safe_name}.csv")

        TRANSFORM_MAP = {
            "time_mask": T.TimeMasking(time_mask_param=80),
            "freq_mask": T.FrequencyMasking(freq_mask_param=80),
            None: None 
        }

        new_metadata_entries = []

        for aug_name in spec_augs:
            
            label = aug_name if aug_name is not None else "plain"
            print(f"\n--- Generating .npy Spectrograms ({label}) ---")
            
            transform = TRANSFORM_MAP.get(aug_name)

            for _, row in tqdm(df_metadata.iterrows(), total=len(df_metadata), desc=f"Processing {label}"):
                filename = row['filename']
                audio_path = os.path.join(input_audio_dir, filename)
                
                base_name = os.path.splitext(filename)[0]
                
                if aug_name is None:
                    new_filename = f"{base_name}.npy"
                else:
                    new_filename = f"{base_name}_{aug_name}.npy"
                
                output_path = os.path.join(output_spec_dir, new_filename)

                if not os.path.exists(output_path):
                    try:
                        y, sr = librosa.load(audio_path, sr=22050) 
                        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=224)
                        log_mel = librosa.power_to_db(mel, ref=np.max)

                        target_len = 224
                        if log_mel.shape[1] > target_len:
                            log_mel = log_mel[:, :target_len]
                        elif log_mel.shape[1] < target_len:
                            pad_width = target_len - log_mel.shape[1]
                            log_mel = np.pad(log_mel, pad_width=((0, 0), (0, pad_width)), 
                                                            mode='constant', constant_values=log_mel.min())

                        if transform is not None:
                            log_mel_tensor = torch.from_numpy(log_mel)
                            masked_tensor = transform(log_mel_tensor)
                            final_array = masked_tensor.numpy()
                        else:
                            final_array = log_mel

                        np.save(output_path, final_array)

                    except FileNotFoundError:
                        print(f"Warning: Source audio {audio_path} not found. Skipping.")
                        continue
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                        continue

                new_row = row.copy()
                new_row['filename'] = new_filename      
                new_row['spec_augmentation'] = label    
                
                new_metadata_entries.append(new_row)

        if new_metadata_entries:
            df_final = pd.DataFrame(new_metadata_entries)
            df_final.to_csv(output_meta_path, index=False)
            print(f"\n Step 2 Complete ({output_subdir}).")
            print(f"   Metadata saved to: {output_meta_path}")
        else:
            print("No entries generated.")
            return None