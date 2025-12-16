import os
import glob
import librosa
import numpy as np
import cv2
from tqdm import tqdm

INPUT_DIRS = [
   r"data\raw_audio", 
   r"data\aug_audio\heavy_crickets",
   r"data\aug_audio\heavy_rain",
   r"data\aug_audio\light_crickets",
   r"data\aug_audio\light_rain",
]
OUTPUT_BASE_DIR = "data/raw_spectrograms"
# Feature Parameters
SAMPLING_RATE = 22050
N_MELS = 128         # Number of Mel bands (Height of the image)
N_FFT = 2048         # FFT Window size (Higher N_FFT gives better frequency resolution)
HOP_LENGTH = 512     # Frame shift
MAX_PAD_LENGTH = 216 # Target time steps (Width of the image)
TARGET_DURATION = 5.0  # seconds
def audio_to_melspec(y, sr, n_fft, hop_length, n_mels, target_len_samples):
    """Converts a padded/trimmed audio waveform to a normalized Mel-Spectrogram image array."""
    
        
    # 2. Generate Mel-Spectrogram (Power)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    
    # 3. Convert to Decibel Scale (Log scale, better for perception/CNNs)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # 3. Standardize sequence length (Padding/Truncation)
    if log_mel_spectrogram.shape[1] > MAX_PAD_LENGTH:
        log_mel_spectrogram = log_mel_spectrogram[:, :MAX_PAD_LENGTH]
    elif log_mel_spectrogram.shape[1] < MAX_PAD_LENGTH:
        pad_width = MAX_PAD_LENGTH - log_mel_spectrogram.shape[1]
        # Pad with a low value (like the mean or a constant)
        log_mel_spectrogram = np.pad(log_mel_spectrogram, 
                                        pad_width=((0, 0), (0, pad_width)), 
                                        mode='constant', 
                                        constant_values=log_mel_spectrogram.min()) # Padding with min/mean is common for log-spectrograms
        
    return log_mel_spectrogram

def process_all_audio_dirs():
    """Main function to loop through all input audio folders and generate spectrograms."""
    
    target_len_samples = int(SAMPLING_RATE * TARGET_DURATION)
    total_processed = 0

    for input_root in INPUT_DIRS:
        # Determine the name of the output folder (e.g., 'unsplit', 'audio_rain_heavy')
        dataset_name = os.path.basename(input_root)
        
        print(f"\n--- Processing Dataset: {dataset_name} ---")
        
        # Get all class folders (e.g., resonant, damp, common)
        class_folders = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
        
        if not class_folders:
            print(f"  Warning: No class folders found in {input_root}. Skipping.")
            continue

        for class_name in class_folders:
            source_dir = os.path.join(input_root, class_name)
            
            # Create the corresponding output structure for spectrograms:
            # data/spectrograms/unsplit/resonant/
            output_dir = os.path.join(OUTPUT_BASE_DIR, dataset_name, class_name)
            os.makedirs(output_dir, exist_ok=True)

            audio_files = glob.glob(os.path.join(source_dir, "*.wav"))
            
            # Use tqdm for a clear progress bar
            for audio_path in tqdm(audio_files, desc=f"  {dataset_name}/{class_name}"):
                filename = os.path.basename(audio_path)
                spec_filename = filename.replace(".wav", ".png") # Save as PNG
                output_path = os.path.join(output_dir, spec_filename)

                # Skip if file already exists (to resume interrupted runs)
                if os.path.exists(output_path):
                    continue

                try:
                    # 1. Load Audio
                    y, sr = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
                    
                    # 2. Generate Spectrogram Image
                    rgb_image = audio_to_melspec(y, sr, N_FFT, HOP_LENGTH, N_MELS, target_len_samples)
                    
                    # 3. Save Image (Using OpenCV's imwrite)
                    cv2.imwrite(output_path, rgb_image)
                    total_processed += 1
                
                except Exception as e:
                    print(f"\nError processing {filename}: {e}")
                    # You might want to log this error rather than printing during the loop

    print(f"\n--- Spectrogram Generation Complete! ---")
    print(f"Total images processed: {total_processed}")
    
    # Display an example of the generated image shape
    if total_processed > 0:
        spec_shape = rgb_image.shape
        print(f"Example Spectrogram Shape (H, W, C): {spec_shape}")
        # H = N_MELS (128)
        # W = Time steps (usually ~216 for 5s at these settings)
        # C = 3 (RGB Channels)
        

if __name__ == "__main__":
    process_all_audio_dirs()