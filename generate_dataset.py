from utils.augment import AudioPipeline
# from utils.visualize import AugmentData

INPUT_METADATA_PATH = r"data\metadata\ESC50_subgroup.csv" # Your master blueprint
INPUT_AUDIO_DIR = r"data\raw_audio" # The single, flat folder with all clean audio
OUTPUT_DIR = r"data" # NEW base folder for all augmented audio sets
RAIN_NOISE_PATH = r"data\bg_noise\rain5s.wav"
CRICKET_NOISE_PATH = r"data\bg_noise\crickets5s.wav"
SAMPLE_RATE = 22050

N_MELS = 128 
HOP_LENGTH = 512 
MEL_SPEC_ARRAY_PATH = r"data\mel_log-mel spectrograms"


if __name__ == "__main__":
    """Load Audio Pipeline"""
    pipeline = AudioPipeline(
        raw_audio_dir=INPUT_AUDIO_DIR,
        metadata_path=INPUT_METADATA_PATH,
        output_dir=OUTPUT_DIR,
        rain_path=RAIN_NOISE_PATH,
        crickets_path=CRICKET_NOISE_PATH
    )

    """Generate Augmented Audio Dataset, returns path to metadata CSV"""
    audio_aug_meta_path = pipeline.generate_augmented_audio_dataset()
    
    """Generate log-mel spectrograms from raw audio"""
    pipeline.generate_spectrogram_dataset(
        input_metadata_path=INPUT_METADATA_PATH,  
        input_audio_dir=INPUT_AUDIO_DIR,           
        output_subdir="log-mel_spectrograms/no_aug",
        spec_augs=[None]                             
    )
    """Generate log-mel spectrograms from augmented audio"""
    pipeline.generate_spectrogram_dataset(
        input_metadata_path=audio_aug_meta_path,  
        input_audio_dir="data/audio_aug",           
        output_subdir="log-mel_spectrograms/audio_aug",
        spec_augs=[None]                             
    )
    """Generate log-mel spectrograms with masking augmentations"""
    pipeline.generate_spectrogram_dataset(
        input_metadata_path=INPUT_METADATA_PATH,        
        input_audio_dir=INPUT_AUDIO_DIR,                  
        output_subdir="log-mel_spectrograms/spec_aug",
        spec_augs=["time_mask", "freq_mask"]         
    )
    """Generate log-mel spectrograms from augmented audio with masking augmentations"""
    pipeline.generate_spectrogram_dataset(
        input_metadata_path=audio_aug_meta_path, 
        input_audio_dir="data/audio_aug",            
        output_subdir="log-mel_spectrograms/hybrid_aug",
        spec_augs=["time_mask", "freq_mask"]         
    )