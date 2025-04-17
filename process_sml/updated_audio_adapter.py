#!/usr/bin/env python
# coding: utf8
"""
This module implements an audio data preprocessing pipeline using PyTorch.
Data preprocessing includes audio loading, spectrogram computation, and chunking the full audio
into sequential segments of fixed duration, followed by optional data augmentation using a
Compose-style transformation pipeline. The dataset yields a dictionary of spectrograms computed
for each chunk of the input audio files.
"""

import csv
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T

# Import transformation functions and Compose object from transformation_utils.
from configarations import global_initial_config
from .transformation_pipeline import MyPipeline, get_shape_first_sample

# -----------------------------------------------------------------------------
# CONFIGURATION (Global Variables to be shared across modules)
# -----------------------------------------------------------------------------
USER_INPUT = {
    "sample_rate": 44000,
    "duration": 20.0,
    "input_name": "mixture",
    "perriferal_name": ["drums", "bass"],
    "is_track_id": True,
    "audio_dir": ".",
    "components": ["mixture", "drums", "bass"],
    "csv_file": "dataset_index.csv"
}

# -----------------------------------------------------------------------------
# Simple Audio Loader
# -----------------------------------------------------------------------------
class SimpleAudioIO:
    def load(
        self,
        path: Union[str, Path],
        sample_rate: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Loads an entire audio file from the given path, normalizes it,
        and resamples it if required.
        Returns the full waveform without cropping.
        """

        waveform, sr = torchaudio.load(str(path), normalize=True)

        if sample_rate is not None and sr != sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)
            sr = sample_rate

        return waveform, sr

    def save(self, path: Union[str, Path], waveform: torch.Tensor, sample_rate: int) -> None:
        """Saves the given waveform to the specified path."""
        torchaudio.save(str(path), waveform, sample_rate)

# -----------------------------------------------------------------------------
# AudioDatasetFolder with chunking
# -----------------------------------------------------------------------------
class AudioDatasetFolder(Dataset):
    def __init__(
        self,
        csv_file: str,
        audio_dir: Optional[str] = None,
        components: List[str] = None,
        sample_rate: int = 16000,  # default samplerate
        duration: float = 20.0,    # segment length in seconds
        input_name: Optional[str] = None,
        perriferal_name: Optional[List[str]] = None,
        wav_transform: Optional[Union[Callable[[torch.Tensor], torch.Tensor],
                                  List[Callable[[torch.Tensor], torch.Tensor]]]] = None,
        spec_transform: Optional[Union[Callable[[torch.Tensor], torch.Tensor],
                                  List[Callable[[torch.Tensor], torch.Tensor]]]] = None,
        is_track_id: bool = True,
    ) -> None:
        """
        Args:
            csv_file: Path to CSV index file.
            audio_dir: Base directory path to prepend to CSV file paths.
            components: List of component names to load.
            sample_rate: Sample rate used for audio loading and resampling.
            duration: Duration (in seconds) for each chunk of audio.
            input_name: Key in the dataset for the primary input spectrogram.
            perriferal_name: List of keys for peripheral sources (optional).
            is_track_id: If true, include a track identifier from the CSV.
        """
        self.is_track_id = is_track_id
        self.sample_rate = sample_rate
        self.duration = duration
        self.audio_io = SimpleAudioIO()

        # Save keys for transformation control
        self.input_name = input_name
        self.perriferal_name = perriferal_name
        self.audio_dir = Path(audio_dir) if audio_dir else None

        if components is None:
            raise ValueError("Please provide the list of components in CSV.")
        self.components = components

        # Update global configuration
        USER_INPUT.update({
            "sample_rate": sample_rate,
            "duration": duration,
            "input_name": input_name,
            "perriferal_name": perriferal_name,
            "is_track_id": is_track_id,
            "audio_dir": audio_dir,
            "components": components,
            "csv_file": csv_file,
        })
        global_initial_config.update_config(**USER_INPUT)

        # Load CSV index
        self.samples: List[Dict[str, str]] = []
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_entry = {comp: row[comp] for comp in self.components if comp in row and row[comp]}
                if len(sample_entry) != len(self.components):
                    missing = set(self.components) - set(sample_entry.keys())
                    raise ValueError(f"Missing components in CSV: {missing}")
                if self.is_track_id:
                    keys = list(row.keys())
                    sample_entry['track_id'] = row[keys[1]] if len(keys) > 1 else ''
                self.samples.append(sample_entry)

        # Determine spectrogram shape from first chunk of first sample
        first_path = Path(self.samples[0][self.components[0]])
        if self.audio_dir and not first_path.is_absolute():
            first_path = self.audio_dir / first_path
        first_wave, first_sr = self.audio_io.load(first_path, sample_rate=self.sample_rate)
        # chunk first_wave to get the shape
        chunk_samples = int(self.sample_rate * self.duration)
        first_chunk = first_wave[:, :chunk_samples]
        self.shape_of_first_nontransformed_spec_sample = get_shape_first_sample(first_chunk)

        # Build transformation pipeline
        self.pipeline = MyPipeline(
            spec_transforms=spec_transform,
            wav_transforms=wav_transform,
            shape_of_untransformed_size=self.shape_of_first_nontransformed_spec_sample
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary where each key is a component name and the value is
        a tensor of spectrograms for each sequential chunk: shape [n_chunks, freq_bins, time_steps].
        If is_track_id is True, includes 'track_id'.
        """
        sample_info = self.samples[idx]
        spectrograms: Dict[str, torch.Tensor] = {}

        for comp in self.components:
            file_path = Path(sample_info[comp])
            if self.audio_dir and not file_path.is_absolute():
                file_path = self.audio_dir / file_path

            waveform, sr = self.audio_io.load(file_path, sample_rate=self.sample_rate)
            # Calculate chunk parameters (will be converted into a seperated function eventually)
            chunk_len = int(self.sample_rate * self.duration)
            total_len = waveform.shape[1]
            n_chunks = math.ceil(total_len / chunk_len)
            pad_len = n_chunks * chunk_len - total_len
            if pad_len > 0:
                waveform = F.pad(waveform, (0, pad_len))

            # Unfold into [channels, n_chunks, chunk_len]
            chunks = waveform.unfold(dimension=1, size=chunk_len, step=chunk_len)
            # Reorder to [n_chunks, channels, chunk_len]
            chunks = chunks.permute(1, 0, 2)

            # Process each chunk through pipeline
            specs = [self.pipeline(chunk) for chunk in chunks]
            # Stack into tensor [n_chunks, freq_bins, time_steps]
            spectrograms[comp] = torch.stack(specs)

        if self.is_track_id:
            spectrograms['track_id'] = sample_info.get('track_id', '')
        return spectrograms
