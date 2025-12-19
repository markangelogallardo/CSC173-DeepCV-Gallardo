import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
# These parameters MUST match how you created the .npy file
N_MELS = 128 
SAMPLING_RATE = 22050 
HOP_LENGTH = 512 
files = librosa.util.find_files('data/ESC-50-master/audio', ext=['wav'], recurse=True)
# --- 2. INPUT FILE PATH ---
FEATURE_FILE_PATH = r'data\log-mel_spectrograms\hybrid_aug\1-20133-A-39_rain_bg_time_mask.npy' 


# --- 3. LOAD DATA ---
mel_data = np.load(FEATURE_FILE_PATH)
print(mel_data.shape)  # Should be (128, Time_steps)
print(f"Data Min: {mel_data.min()}, Data Max: {mel_data.max()}")

# --- 4. PLOT WITH PROPER MEL SCALES ---
fig, ax = plt.subplots(figsize=(10, 4))

# y_axis='mel' tells librosa to map the 128 rows to the correct Mel Frequencies (Hz)
# x_axis='time' tells librosa to map the columns to seconds
img = librosa.display.specshow(
    mel_data,            # Plotting the raw data (Linear scale)
    x_axis='time', 
    y_axis='mel',        # <--- This applies the "Proper Mel Scale" to the Y-Axis
    sr=SAMPLING_RATE, 
    hop_length=HOP_LENGTH, 
    cmap='viridis', 
    n_fft=1024,
    ax=ax
)

# Colorbar shows the raw values (Linear Power/Amplitude)
fig.colorbar(img, ax=ax, label='Power (db)')

ax.set_title('Mel Spectrogram', fontsize=12)
ax.set_ylabel('Frequency (Mel)')
ax.set_xlabel('Time (s)')

plt.tight_layout()
plt.show()