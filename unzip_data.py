#!/usr/bin/env python3
"""
Standalone script to unzip all flight data zip files.
Run this script if you prefer not to use the notebook function.
"""

import zipfile
from pathlib import Path
from tqdm import tqdm

def unzip_flight_data(data_dir='data', extract_to_subdirs=True):
    """
    Unzip all zip files in the data directory.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing zip files
    extract_to_subdirs : bool
        If True, extract each zip to its own subdirectory (recommended)
    """
    data_path = Path(data_dir)
    zip_files = list(data_path.glob('*.zip'))
    
    if not zip_files:
        print("No zip files found. Files may already be extracted.")
        return
    
    print(f"Found {len(zip_files)} zip files")
    
    for zip_file in tqdm(zip_files, desc="Unzipping"):
        try:
            # Extract to a subdirectory named after the zip file (without .zip)
            if extract_to_subdirs:
                extract_dir = data_path / zip_file.stem
                extract_dir.mkdir(exist_ok=True)
            else:
                extract_dir = data_path
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
        except zipfile.BadZipFile:
            print(f"Warning: {zip_file.name} appears to be corrupted or not a valid zip file")
        except Exception as e:
            print(f"Error extracting {zip_file.name}: {e}")

if __name__ == "__main__":
    unzip_flight_data()

