from utils.augment import AugmentData


INPUT_METADATA_PATH = r"data\metadata\ESC50_subgroup.csv" # Your master blueprint
INPUT_AUDIO_DIR = r"data\raw_audio" # The single, flat folder with all clean audio
OUTPUT_METADATA_DIR = r"data\metadata"
OUTPUT_METADATA_FILENAME = r"ESC50_subgroup_audio.csv"
OUTPUT_AUDIO_DIR = r"data\aug_data" # NEW base folder for all augmented audio sets
RAIN_NOISE_PATH = r"data\bg_noise\rain5s.wav"
CRICKET_NOISE_PATH = r"data\bg_noise\crickets5s.wav"
SAMPLE_RATE = 44100

if __name__ == "__main__":
    augmenter = AugmentData(
        input_audio_dir=INPUT_AUDIO_DIR,
        input_metadata_path=INPUT_METADATA_PATH,
        output_dir=OUTPUT_AUDIO_DIR,
        output_metadata_path=OUTPUT_METADATA_DIR,
        output_metadata_filename=OUTPUT_METADATA_FILENAME,
        rain_bg_path=RAIN_NOISE_PATH,
        crickets_bg_path=CRICKET_NOISE_PATH,
        sample_rate=SAMPLE_RATE,
        aug_type="audio"  # Choose from: "audio", "spectrogram", "hybrid"
    )
    augmenter.run()
    print("Augmentation process completed.")