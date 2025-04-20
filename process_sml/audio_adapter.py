#!/usr/bin/env python
# coding: utf8
"""
This module implements an audio data preprocessing pipeline using PyTorch.
Data preprocessing includes audio loading, spectrogram computation, and flattening full audio
into fixed-duration chunks. Each chunk becomes its own sample, ensuring consistent tensor shapes
for batching in the DataLoader.
"""

import csv
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T

# Import transformation functions and Compose object from transformation_utils.
import configarations.global_initial_config as GI
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
            waveform = T.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
            sr = sample_rate
        return waveform, sr

    def save(self, path: Union[str, Path], waveform: torch.Tensor, sample_rate: int) -> None:
        """Saves the given waveform to the specified path."""
        torchaudio.save(str(path), waveform, sample_rate)

# -----------------------------------------------------------------------------
# AudioDatasetFolder flattened to chunks
# -----------------------------------------------------------------------------
class AudioDatasetFolder(Dataset):
    def __init__(
        self,
        csv_file: str,
        audio_dir: Optional[str] = None,
        components: List[str] = None,
        sample_rate: int = 16000,
        duration: float = 20.0,
        input_name: Optional[str] = None,
        perriferal_name: Optional[List[str]] = None,
        wav_transform: Optional[Union[Callable[[torch.Tensor], torch.Tensor],
                                     List[Callable[[torch.Tensor], torch.Tensor]]]] = None,
        spec_transform: Optional[Union[Callable[[torch.Tensor], torch.Tensor],
                                     List[Callable[[torch.Tensor], torch.Tensor]]]] = None,
        is_track_id: bool = True,
        n_fft:int = 2048,
        hop_length:int =512,

    ) -> None:
        # Save configuration
        self.sample_rate = sample_rate
        self.duration = duration
        self.is_track_id = is_track_id
        self.input_name = input_name
        self.perriferal_name = perriferal_name
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.components = components or []

        # Update global config
        USER_INPUT.update({
            "sample_rate": sample_rate,
            "duration": duration,
            "input_name": input_name,
            "perriferal_name": perriferal_name,
            "is_track_id": is_track_id,
            "audio_dir": audio_dir,
            "components": components,
            "csv_file": csv_file,
            "n_fft": n_fft,
            "hop_length" : hop_length

        })
        GI.update_config(**USER_INPUT)

        # Read CSV and store sample paths
        self.samples: List[Dict[str, str]] = []
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = {comp: row[comp] for comp in self.components if row.get(comp)}
                if len(entry) != len(self.components):
                    missing = set(self.components) - set(entry.keys())
                    raise ValueError(f"Missing components in CSV: {missing}")
                if self.is_track_id:
                    keys = list(row.keys())
                    entry['track_id'] = row[keys[1]] if len(keys) > 1 else ''
                self.samples.append(entry)

        # Build an index map: (sample_idx, chunk_idx) for each chunk in each track
        self.index_map: List[Tuple[int, int]] = []
        self.audio_io = SimpleAudioIO()
        self.chunk_len = int(self.sample_rate * self.duration)

        for i, sample in enumerate(self.samples):
            path = Path(sample[self.components[0]])
            if self.audio_dir and not path.is_absolute():
                path = self.audio_dir / path
            waveform, _ = self.audio_io.load(path, sample_rate=self.sample_rate)
            total = waveform.shape[1]
            n_chunks = math.ceil(total / self.chunk_len)
            for c in range(n_chunks):
                self.index_map.append((i, c))

        # Determine spec-shape using first chunk
        first_sample, first_chunk_idx = self.index_map[0]
        path = Path(self.samples[first_sample][self.components[0]])
        if self.audio_dir and not path.is_absolute():
            path = self.audio_dir / path
        waveform, _ = self.audio_io.load(path, sample_rate=self.sample_rate)
        pad = self.chunk_len * math.ceil(waveform.shape[1] / self.chunk_len) - waveform.shape[1]
        if pad > 0:
            waveform = F.pad(waveform, (0, pad))
        chunks = waveform.unfold(1, self.chunk_len, self.chunk_len).permute(1, 0, 2)
        first_chunk = chunks[first_chunk_idx]
        spec_shape = get_shape_first_sample(first_chunk)

        # Build pipeline
        self.pipeline = MyPipeline(
            spec_transforms=spec_transform,
            wav_transforms=wav_transform,
            shape_of_untransformed_size=spec_shape
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx, chunk_idx = self.index_map[idx]
        sample = self.samples[sample_idx]
        out: Dict[str, torch.Tensor] = {}

        for comp in self.components:
            path = Path(sample[comp])
            if self.audio_dir and not path.is_absolute():
                path = self.audio_dir / path
            waveform, _ = self.audio_io.load(path, sample_rate=self.sample_rate)
            total = waveform.shape[1]
            pad = self.chunk_len * math.ceil(total / self.chunk_len) - total
            if pad > 0:
                waveform = F.pad(waveform, (0, pad))
            chunks = waveform.unfold(1, self.chunk_len, self.chunk_len).permute(1, 0, 2)
            chunk_wave = chunks[chunk_idx]
            out[comp] = self.pipeline(chunk_wave)

        if self.is_track_id:
            out['track_id'] = sample.get('track_id', '')
        return out
