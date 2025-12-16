import pandas as pd
import os
import shutil
from tqdm import tqdm

class DataFilter():
    def __init__(self, input_audio_dir, input_metadata_path, output_audio_dir, output_metadata_path, output_metadata_filename, 
                 target_classes, custom_target):
        self.input_audio_dir = input_audio_dir
        self.input_metadata_path = input_metadata_path
        self.output_audio_dir = output_audio_dir
        self.output_metadata_path = output_metadata_path
        self.output_metadata_filename = output_metadata_filename
        self.target_classes = target_classes
        self.custom_target = custom_target
        self.new_df = None

    def setup_directories(self):
        """Creates the single audio output folder and the metadata folder."""

        os.makedirs(self.output_audio_dir, exist_ok=True)
        os.makedirs(self.output_metadata_path, exist_ok=True)
        print(f"Output audio directory created at: {self.output_audio_dir}")
        print(f"Metadata directory created at: {self.output_metadata_path}")

    def filter_metadata(self):
        """Reads, filters, adds the 'project_group' and the new 'custom_target' columns."""
        try:
            df = pd.read_csv(self.input_metadata_path)
        except FileNotFoundError:
            print(f"Error: Metadata file not found at {self.input_metadata_path}. Cannot proceed.")
            return None

        target_names = list(self.target_classes.keys())
        
        # 1. Filter: Select only the rows corresponding to your 7 classes
        df_filtered = df[df['category'].isin(target_names)].copy()
        
        # 2. Map Project Group: Add the custom 'project_group' column
        df_filtered['project_group'] = df_filtered['category'].map(self.target_classes)
        
        # --- 3. THE ADDITION: Map Project Group to Numerical Target ---
        # Create the column your CNN will actually predict (0, 1, or 2)
        df_filtered['custom_target'] = df_filtered['project_group'].map(self.custom_target)
        
        # 4. Select final columns for the blueprint
        # We include 'target' (ESC-50 index) for reference, and the new 'custom_target'
        self.new_df = df_filtered[['filename', 'fold', 'project_group', 'custom_target', 'category', 'target']].copy()

        print(f"\nTotal files selected for processing: {len(self.new_df)}")
        print("Files per Custom Project Group:")
        print(self.new_df['project_group'].value_counts())
        print("\nNew Numerical Target Distribution:")
        print(self.new_df['custom_target'].value_counts())
    
        return self.new_df
    
    def copy_audio_files(self):
        """Copies all selected audio files to a single, flat output folder."""
        if self.new_df is None:
            return

        print("\nStarting file copying to single folder...")
        copied_count = 0
        
        # Use tqdm to show progress for file copying
        for _, row in tqdm(self.new_df.iterrows(), total=len(self.new_df), desc="Copying Audio"):
            filename = row['filename']
            
            src_path = os.path.join(self.input_audio_dir, filename)
            dest_path = os.path.join(self.output_audio_dir, filename)
            
            # Check if source file exists before copying
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_path)
                copied_count += 1
            else:
                print(f"Warning: Source audio file not found: {src_path}. Skipping.")
                
        print(f"\nFile organization complete. {copied_count} files copied successfully to {self.output_audio_dir}.")

    def save_metadata(self):
        """Saves the final metadata blueprint."""
        if self.new_df is None:
            return
            
        output_path = os.path.join(self.output_metadata_path, self.output_metadata_filename)
        
        # Save the DataFrame. It now contains the critical 'custom_target' column.
        self.new_df.to_csv(output_path, index=False)
        
        print(f"\nFinal metadata blueprint saved to: {output_path}")
