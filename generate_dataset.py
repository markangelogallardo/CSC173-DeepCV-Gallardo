from utils.data_filter import DataFilter


# --- CONFIGURATION (Adjust Paths as needed) ---
# ----------------------------------------------------------------------------------
INPUT_AUDIO_DIR = r"data/ESC-50-master/audio"       # Source folder for ALL ESC-50 audio files
INPUT_METADATA_DIR = r"data/ESC-50-master/meta/esc50.csv" # Source metadata CSV
OUTPUT_AUDIO_DIR = r"data/raw_audio"  # SINGLE, FLAT output folder for all selected audio
OUTPUT_METADATA_DIR = r"data/metadata"
OUTPUT_METADATA_FILENAME = r"ESC50_subgroup.csv"
# ----------------------------------------------------------------------------------

TARGET_CLASSES = {
    'glass_breaking': 'resonant', 
    'can_opening': 'resonant', 
    'door_wood_knock': 'damp', 
    'footsteps': 'damp',
    'hen': 'common', 
    'engine': 'common', 
    'rooster': 'common' 
}

# --- NEW MAPPING: Custom 3-Class Numerical Targets ---
# This dictionary maps your 'project_group' names to the numerical indices (0, 1, 2)
CUSTOM_TARGET_MAP = {
    'resonant': 0,
    'damp': 1,
    'common': 2
}

if __name__ == "__main__":
    select_column = DataFilter(
        input_audio_dir=INPUT_AUDIO_DIR,
        input_metadata_path=INPUT_METADATA_DIR,
        output_audio_dir=OUTPUT_AUDIO_DIR,
        output_metadata_path=OUTPUT_METADATA_DIR,
        output_metadata_filename=OUTPUT_METADATA_FILENAME,
        target_classes=TARGET_CLASSES,
        custom_target=CUSTOM_TARGET_MAP
    )
    select_column.setup_directories()
    filtered_df = select_column.filter_metadata()
    if filtered_df is not None:
        select_column.copy_audio_files()
        select_column.save_metadata()