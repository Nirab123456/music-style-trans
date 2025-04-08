#!/usr/bin/env python
# coding: utf8
"""
Parse MUSDB18 MP4 stem files using FFmpeg (via a Cython module) and organize the outputs.

This script:
 1. Searches recursively under a base MUSDB18 folder for MP4 files in the "train" and "test" subfolders.
 2. For each MP4 file found, a unique track UUID is generated.
 3. Under an output base directory, two subfolders ("train" and "test") are created.
    Under each subset, five fixed folders (mixture, drums, bass, other_accompaniment, vocals) are created.
 4. Using a Cython module (imported as extract_stems), the stems for each MP4 file are extracted into a temporary folder.
 5. For each track, all extracted files are renamed using the **same track UUID** (with the component name appended) and moved into their corresponding folders.
 6. A CSV file is saved listing each song’s original file name and the corresponding track UUID,
    as well as the output paths of its components.

Requirements:
  - MUSDB18 folder structure:
       <musdb_root>/train/...
       <musdb_root>/test/...
  - Each MP4 file is a stem file.
  - The Cython module "extract_stems" is built and available.
  - FFmpeg is installed and accessible.
"""

import os
import uuid
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
import tempfile

# Import the Cython module (assumed to be built and available)
import extract_stems  # Must provide extract_all_stems(input_file, output_dir) -> dict

# Mapping of stream indices to component names.
COMPONENT_MAP = {
    0: "mixture",
    1: "drums",
    2: "bass",
    3: "other_accompaniment",
    4: "vocals"
}

# Configuration paths (adjust as needed)
MUSDB_ROOT = Path(r"D:\MUSIC_STYLE_TRANSFER\musdb18")
OUTPUT_BASE = Path(r"output_stems")
SUBSETS = ["train", "test"]

# CSV output file (timestamped)
CSV_OUTPUT = OUTPUT_BASE / f"musdb18_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# List to hold one CSV row per track.
csv_rows = []


def prepare_output_folders():
    """
    Create the fixed output folder structure for each subset:
      OUTPUT_BASE / <subset> / <component>
    Returns a dict mapping each subset to a dict mapping component to folder path.
    For example: out_dirs["train"]["mixture"] is the Path for train/mixture.
    """
    out_dirs = {}
    for subset in SUBSETS:
        subset_dir = OUTPUT_BASE / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        out_dirs[subset] = {}
        for comp in COMPONENT_MAP.values():
            comp_dir = subset_dir / comp
            comp_dir.mkdir(parents=True, exist_ok=True)
            out_dirs[subset][comp] = comp_dir
    return out_dirs


def process_track(track_mp4: Path, subset: str, output_dirs: dict):
    """
    Process one MP4 file:
      - Generate one unique track UUID.
      - Create a temporary directory for extraction.
      - Call the Cython module to extract all stems into the temporary folder.
      - For each extracted stem, rename the file as "<track_uuid>_<component>.wav" and
        move it into the fixed output folder for that component.
      - Build a dict with track-level metadata:
            track_id, subset, original_mp4, and output paths for each component.
    
    Returns the metadata dictionary for this track.
    """
    track_id = str(uuid.uuid4())
    print(f"Processing track: {track_mp4} | track_id: {track_id}")
    
    # Dictionary to store output file paths for each component
    comp_files = {comp: "" for comp in COMPONENT_MAP.values()}
    
    # Create a temporary extraction folder.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_extract_dir = Path(temp_dir)
        try:
            # extract_all_stems() should return a dict {stream_index: temp_file_path, ...}
            extracted = extract_stems.extract_all_stems(str(track_mp4), str(temp_extract_dir))
        except Exception as e:
            print(f"Error extracting stems from {track_mp4}: {e}")
            return None

        for stream_key, temp_file in extracted.items():
            try:
                stream_index = int(stream_key)
            except Exception:
                continue

            comp = COMPONENT_MAP.get(stream_index, f"stem_{stream_index}")
            # Build new file name using the same track_id and component name.
            new_file_name = f"{track_id}_{comp}.wav"
            dest_folder = output_dirs[subset].get(comp)
            if dest_folder is None:
                dest_folder = OUTPUT_BASE / subset
                dest_folder.mkdir(parents=True, exist_ok=True)
            new_file_path = dest_folder / new_file_name

            try:
                shutil.move(temp_file, new_file_path)
            except Exception as e:
                print(f"Failed to move {temp_file} to {new_file_path}: {e}")
                continue

            comp_files[comp] = str(new_file_path)
    
    # Build and return the metadata dictionary.
    track_meta = {
        "track_id": track_id,
        "subset": subset,
        "original_mp4": str(track_mp4),
        "mixture": comp_files.get("mixture", ""),
        "drums": comp_files.get("drums", ""),
        "bass": comp_files.get("bass", ""),
        "other_accompaniment": comp_files.get("other_accompaniment", ""),
        "vocals": comp_files.get("vocals", "")
    }
    return track_meta


def main():
    # Ensure output base directory exists.
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    out_dirs = prepare_output_folders()

    # Process each subset folder.
    for subset in SUBSETS:
        subset_dir = MUSDB_ROOT / subset
        if not subset_dir.exists():
            print(f"Subset folder {subset_dir} not found, skipping.")
            continue

        # Search recursively for all MP4 files.
        mp4_files = list(subset_dir.rglob("*.mp4"))
        print(f"Found {len(mp4_files)} MP4 files in subset '{subset}'.")

        for track_mp4 in mp4_files:
            meta = process_track(track_mp4, subset, out_dirs)
            if meta:
                csv_rows.append(meta)
                print(f"Processed track {track_mp4} with track_id {meta['track_id']}")

    # Save CSV with all track metadata.
    df = pd.DataFrame(csv_rows)
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"CSV index saved to: {CSV_OUTPUT}")

if __name__ == '__main__':
    main()
