import os
import csv
import math
import sqlite3
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
from .transformation_pipeline import MyPipeline
import configarations.global_initial_config as GI
from .transformation_utlis import get_shape_first_sample
class SimpleAudioIO:
    def __init__(self):
        self._resamplers: Dict[Tuple[int,int], T.Resample] = {}

    def load(self, path: Union[str, Path], sample_rate: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        waveform, sr = torchaudio.load(str(path), normalize=True)
        if sample_rate is not None and sr != sample_rate:
            key = (sr, sample_rate)
            if key not in self._resamplers:
                self._resamplers[key] = T.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = self._resamplers[key](waveform)
            sr = sample_rate
        return waveform, sr

    def save(self, path: Union[str, Path], waveform: torch.Tensor, sample_rate: int) -> None:
        torchaudio.save(str(path), waveform, sample_rate)

class AudioDatasetFolder(Dataset):
    """
    Caches each track+component as a single .pt containing all fixed-duration chunks.
    Uses an SQLite database to map (track_idx, component) -> cache file, and a simple
    in-memory index_map of (track_idx, chunk_idx) for batch sampling across all components.
    """
    def __init__(
        self,
        csv_file: str,
        audio_dir: Optional[str] = None,
        components: List[str] = None,
        sample_rate: int = 16000,
        duration: float = 20.0,
        input_name: str = None,
        perriferal_name: Optional[List[str]] = None,
        wav_transform: Optional[Union[Callable[[torch.Tensor], torch.Tensor],
                                     List[Callable[[torch.Tensor], torch.Tensor]]]] = None,
        spec_transform: Optional[Union[Callable[[torch.Tensor], torch.Tensor],
                                     List[Callable[[torch.Tensor], torch.Tensor]]]] = None,
        is_track_id: bool = True,
        n_fft: int = 256,
        hop_length: int = 32,
        cache_dir : str = ".cache_chunks",
        cache_db_name : str = "index.db",
        input_transformation : str = "2-SPEC",
        rest_transformation : str = "2-SPEC",
    ) -> None:
        self.all_transformations = ["2-SPEC","4-SPEC","WAV"]
        if input_transformation in self.all_transformations:
            self.input_transformation = input_transformation
        else:
            raise(ValueError(f"{input_transformation} is not a valid transformation for this AudioDatasetFolder."))

        if rest_transformation in self.all_transformations:
            self.rest_transformation = rest_transformation
        else:
            raise(ValueError(f"{rest_transformation} is not a valid transformation for this AudioDatasetFolder."))

        # Basic config
        self.sample_rate = sample_rate
        self.duration = duration
        self.is_track_id = is_track_id
        self.input_name = input_name
        self.perriferal_name = perriferal_name
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.components = components or []
        self.chunk_len = int(self.sample_rate * self.duration)
        self.audio_io = SimpleAudioIO()
        self.hnn_window_cpu = torch.hann_window(n_fft, device="cpu")

        # Update global config
        USER_INPUT = {
            "sample_rate": sample_rate,
            "duration": duration,
            "input_name": input_name,
            "perriferal_name": perriferal_name,
            "is_track_id": is_track_id,
            "audio_dir": audio_dir,
            "components": components,
            "csv_file": csv_file,
            "n_fft": n_fft,
            "hop_length": hop_length,

        } 
        GI.update_config(**USER_INPUT)
  

        # 1) Read CSV index into self.samples
        self.samples: List[Dict[str, Union[Path,str]]] = []
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry: Dict[str, Union[Path,str]] = {}
                for comp in self.components:
                    if comp not in row or not row[comp]:
                        raise ValueError(f"Missing component '{comp}' in CSV row {row}")
                    p = Path(row[comp])
                    if self.audio_dir and not p.is_absolute():
                        p = self.audio_dir / p
                    entry[comp] = p
                if self.is_track_id:
                    keys = list(row.keys())
                    entry['track_id'] = row[keys[1]] if len(keys) > 1 else ''
                self.samples.append(entry)

        # 2) Prepare cache directory and SQLite mapping for component files
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        db_filepath = f"{cache_dir}/{cache_db_name}"
        init_db = os.path.exists(db_filepath)
        self.db = sqlite3.connect(str(db_filepath))
        if not init_db:
            self._create_db()
        # Ensure each component file (.pt) exists and is registered in DB
        self._ensure_cache_components()

        # 3) Build in-memory index of (track_idx, chunk_idx)
        self.index_map: List[Tuple[int,int]] = []
        self._track_lengths: Dict[int,int] = {}
        for i, sample in enumerate(self.samples):
            # load only  first component to determine chunk size
            comp0 = self.components[0]
            cache_path = self._get_cache_path(i, comp0)
            chunks = torch.load(cache_path)
            n_chunks = chunks.size(0)
            self._track_lengths[i] = n_chunks
            for c in range(n_chunks):
                self.index_map.append((i, c))

        # 4) Setup transformation pipeline & spec_shape using first chunk
        first_track, first_chunk_idx = self.index_map[0]
        cache0 = torch.load(self._get_cache_path(first_track, self.components[0]))
        first_chunk = cache0[first_chunk_idx]
        self.spec_shape , self.wav_length = get_shape_first_sample(waveform=first_chunk,hnn_window_cpu=self.hnn_window_cpu,n_fft=n_fft,hop_length=hop_length)
        self.pipeline = MyPipeline(
            spec_transforms=spec_transform,
            wav_transforms=wav_transform,
            input_name=input_name,
            peripheral_names=perriferal_name,
            n_fft=n_fft,
            hop_length=hop_length,
            input_transformation=self.input_transformation,
            rest_transformation= self.rest_transformation,
            shape_of_untransformed_size = self.spec_shape,
            hnn_window_cpu=self.hnn_window_cpu,
            
        )
        # 5) In-memory cache of loaded .pt per (track_idx, component)
        self._loaded_tracks: Dict[Tuple[int,str], torch.Tensor] = {}
        self._current_cached_track: Optional[int] = None

        GI.update_config(wav_length=self.wav_length,cache_dir_path=self.cache_dir,db_filename=cache_db_name)

    def _create_db(self):
        c = self.db.cursor()
        c.execute(
            '''CREATE TABLE components (
                   track_idx INTEGER,
                   component TEXT,
                   cache_path TEXT,
                   PRIMARY KEY (track_idx, component)
               )'''
        )
        self.db.commit()


    def _ensure_cache_components(self):
        """
        For each sample track and component, if not registered in DB,
        load audio, pad & unfold into chunks, save .pt file, and record in DB.
        Batches all INSERTs into a single transaction at the end.
        """
        cursor = self.db.cursor()

        # 1) Fetch all existing (track_idx, component)
        cursor.execute("SELECT track_idx, component FROM components")
        existing = set(cursor.fetchall())  # e.g. {(0, "mixture"), (0, "drums"), ...}

        # 2) Prepare a list to collect new rows
        to_insert: List[Tuple[int,str,str]] = []

        # 3) Loop through every (i, comp); for missing ones, do chunking + save
        for i, sample in enumerate(self.samples):
            for comp in self.components:
                key = (i, comp)
                if key in existing:
                    continue  # already cached

                # load + resample
                wav, _ = self.audio_io.load(sample[comp], sample_rate=self.sample_rate)

                # pad to multiple of chunk_len
                total = wav.size(1)
                n_chunks = math.ceil(total / self.chunk_len)
                pad_amt = n_chunks * self.chunk_len - total
                if pad_amt > 0:
                    wav = F.pad(wav, (0, pad_amt))

                # split -> (n_chunks, channels, chunk_len)
                chunks = wav.unfold(1, self.chunk_len, self.chunk_len) \
                            .permute(1, 0, 2)

                # save to disk
                cache_path = self.cache_dir / f"track_{i}_{comp}.pt"
                torch.save(chunks, str(cache_path))

                # queue up an INSERT
                to_insert.append((i, comp, str(cache_path)))

        # 4) Batch-insert all new rows in one go
        if to_insert:
            cursor.executemany(
                "INSERT INTO components (track_idx, component, cache_path) VALUES (?, ?, ?)",
                to_insert
            )
            self.db.commit()

    def _get_cache_path(self, track_idx: int, component: str) -> str:
        c = self.db.cursor()
        c.execute(
            "SELECT cache_path FROM components WHERE track_idx=? AND component=?",
            (track_idx, component)
        )
        row = c.fetchone()
        if not row:
            raise KeyError(f"No cache entry for track {track_idx}, component {component}")
        return row[0]

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        track_idx, chunk_idx = self.index_map[idx]
        out: Dict[str, torch.Tensor] = {}

        # If we’ve moved on to a new track, throw away the old one
        if self._current_cached_track is None or self._current_cached_track != track_idx:
            self._loaded_tracks.clear()
            self._current_cached_track = track_idx

        # Now load only this track’s components (and leave them until we switch again)
        for comp in self.components:
            key = (track_idx, comp)
            if key not in self._loaded_tracks:
                cache_path = self._get_cache_path(track_idx, comp)
                # this loads the entire chunk‐tensor for this component
                self._loaded_tracks[key] = torch.load(cache_path)
            chunks = self._loaded_tracks[key]
            chunk_wave = chunks[chunk_idx]
            out[comp] = self.pipeline(chunk_wave, component=comp)

        if self.is_track_id:
            out['track_id'] = self.samples[track_idx].get('track_id', '')

        return out