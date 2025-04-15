#!/usr/bin/env python
# coding: utf8
"""
This module implements an audio data preprocessing pipeline using PyTorch.
Data preprocessing includes audio loading, spectrogram computation, cropping,
and data augmentation using a Compose-style transformation pipeline.
The dataset yields a dictionary of spectrograms computed from the input audio files.
"""

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as tf

# Import transformation functions and Compose object from transformation_utlis.
from .transformation_utlis import to_stereo, compute_spectrogram
from configarations import global_initial_config
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
        duration: Optional[float] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Loads an audio file from the given path, normalizes it, optionally crops its duration,
        and resamples it if required.
        """
        path_str: str = str(path)
        waveform, sr = torchaudio.load(path_str, normalize=True) #waveform is a pytorch tensor

        if duration is not None:
            num_samples: int = int(sr * duration)
            waveform = waveform[:, :num_samples]

        if sample_rate is not None and sr != sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)
            sr = sample_rate

        return waveform, sr

    def save(self, path: Union[str, Path], waveform: torch.Tensor, sample_rate: int) -> None:
        """Saves the given waveform to the specified path."""
        torchaudio.save(str(path), waveform, sample_rate)

# ----------------------------------------------------------------------------- 
# AudioDatasetFolder
# -----------------------------------------------------------------------------
class AudioDatasetFolder(Dataset):
    def __init__(
        self,
        csv_file: str,
        audio_dir: Optional[str] = None,
        components: List[str] = None,
        sample_rate: int = 16000, #default samplerate for musdb18
        duration: float = 20.0,
        input_name: Optional[str] = None,
        perriferal_name: Optional[List[str]] = None,
        transform: Optional[Union[Callable[[torch.Tensor], torch.Tensor],
                                  List[Callable[[torch.Tensor], torch.Tensor]]]] = None,
        is_track_id: bool = True,
    ) -> None:
        """
        Args:
            csv_file: Path to CSV index file.
            audio_dir: Base directory path to prepend to CSV file paths. If None, file paths are assumed absolute.
            components: List of component names to load.
            sample_rate: Sample rate used for audio loading and resampling.
            duration: Duration (in seconds) of audio to load from each file.
            input_name: Key in the dataset for the primary input spectrogram.
            perriferal_name: List of keys for peripheral sources (optional).
            transform: Optional transformation or a Compose-style transformation that
                       applies to the computed spectrogram. If provided, and if either input_name
                       or perriferal_name is specified, the transform will be applied only to those keys.
            is_track_id: If true, include a track identifier from the CSV.
        """
        self.is_track_id = is_track_id
        self.sample_rate = sample_rate
        self.duration = duration
        self.audio_io = SimpleAudioIO()

        # Save the specific keys for transformation control.
        self.input_name = input_name
        self.perriferal_name = perriferal_name
        self.audio_dir = Path(audio_dir) if audio_dir else None

        if components is None:
            raise ValueError("Please provide the list of components in CSV.")
        else:
            self.components = components

        # declaration of initial global variables
        USER_INPUT["input_name"]= input_name
        USER_INPUT["audio_dir"] = audio_dir
        USER_INPUT["components"] = components
        USER_INPUT["csv_file"] = csv_file
        USER_INPUT["duration"] = duration
        USER_INPUT["is_track_id"]=is_track_id
        USER_INPUT["perriferal_name"]=perriferal_name
        USER_INPUT["sample_rate"]=sample_rate
        global_initial_config.update_config(**USER_INPUT)

        # Determine how to handle transformations.
        # Accept either a single callable (which may be an instance of Compose)
        # or a list/tuple of callables.
        if transform is None:
            self.transforms: List[Callable[[torch.Tensor], torch.Tensor]] = []
        elif callable(transform):
            self.transforms = [transform]
        elif isinstance(transform, (list, tuple)):
            if all(callable(t) for t in transform):
                self.transforms = list(transform)
            else:
                raise ValueError("All elements in transform must be callable.")
        else:
            raise TypeError("transform must be either a callable or a list/tuple of callables.")


        self.samples: List[Dict[str, str]] = []
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_entry: Dict[str, str] = {}
                for comp in self.components:
                    if comp in row and row[comp]:
                        sample_entry[comp] = row[comp]
                    else:
                        raise ValueError(f"Component '{comp}' not found in CSV or is empty.")
                if self.is_track_id:
                    keys = list(row.keys())
                    sample_entry['track_id'] = row[keys[1]] if len(keys) > 1 else ""
                self.samples.append(sample_entry)
        
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary where keys are component names (e.g., 'mix', 'drums', etc.)
        and values are the computed spectrogram tensors.
        If an `input_name` or any key in `perriferal_name` is provided, transformations
        are applied only on those components; otherwise, transformations are applied on all.
        """
        sample_info: Dict[str, str] = self.samples[idx]
        spectrograms: Dict[str, torch.Tensor] = {}

        for comp in self.components:
            file_path_str: str = sample_info[comp]
            file_path = Path(file_path_str)
            if self.audio_dir and not file_path.is_absolute():
                file_path = self.audio_dir / file_path

            waveform, sr = self.audio_io.load(file_path, sample_rate=self.sample_rate, duration=self.duration)
            waveform = to_stereo(waveform)
            spec = compute_spectrogram(waveform)


            # Decide whether to apply the transforms.
            # If either input_name or perriferal_name is provided, then apply transforms only if:
            #   comp equals input_name OR comp is in perriferal_name.
            # Otherwise, if none are provided, apply transforms to all components.
            # Apply transforms selectively
            apply_transforms = (
                (self.input_name is not None and comp == self.input_name) or
                (self.perriferal_name is not None and comp in self.perriferal_name) or
                (self.input_name is None and self.perriferal_name is None)
            )
            if apply_transforms and self.transforms:
                for transform in self.transforms:
                    spec = transform(spec)
            #         waveform = transform(waveform)

            # spec = compute_spectrogram(waveform).abs()
            # # Optionally delete waveform if not needed further
            spec = spec.abs()

            del waveform  
            spectrograms[comp] = spec

        if self.is_track_id:
            spectrograms['track_id'] = sample_info.get('track_id', '')
            
        return spectrograms
