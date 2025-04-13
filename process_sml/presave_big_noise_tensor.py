import os
import torch
import torchaudio
import random
from process_sml import to_stereo
from configarations import global_initial_config


all_chunks = []

def save_big_noise_spec_meg_tensor(
        
        directory: str = global_initial_config.NOISE_WAV_DIR,
        save_dir : str = global_initial_config.NOISE_TENSOR_SAVE_DIR,
        target_sample_rate : int = global_initial_config.SAMPLE_RATE,
        chunk_duration_sec : int = global_initial_config.CHUNK_DURATION_RECONSTRUCTED,
        noise_tensor_name : str = global_initial_config.NOISE_TENSOR_NAME
    ):
    global_initial_config.NOISE_WAV_DIR = directory
    global_initial_config.NOISE_TENSOR_SAVE_DIR = save_dir
    global_initial_config.CHUNK_DURATION_RECONSTRUCTED = chunk_duration_sec
    global_initial_config.NOISE_TENSOR_NAME = noise_tensor_name
    """ Plese ensure that save directory exist  """

    chunk_size = target_sample_rate * chunk_duration_sec


    for filename in os.listdir(directory):
        if not filename.endswith((".mp3", ".wav", ".flac")):
            continue

        path = os.path.join(directory, filename)
        waveform, sample_rate = torchaudio.load(path)

        # Force stereo
        waveform = to_stereo(waveform)

        # Resample if needed
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        # Break into 1-second chunks
        num_chunks = waveform.shape[1] // chunk_size
        for i in range(num_chunks):
            chunk = waveform[:, i * chunk_size:(i + 1) * chunk_size]
            all_chunks.append(chunk)

    # Shuffle and stack
    random.shuffle(all_chunks)
    big_tensor = torch.cat(all_chunks, dim=1)  # [2, total_samples]
    # Save
    torch.save(big_tensor, os.path.join(save_dir, noise_tensor_name))
