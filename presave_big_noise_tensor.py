import os
import torch
import torchaudio
import random
import torchaudio.transforms as T
from process_sml import compute_waveform,to_stereo,compute_spectrogram

directory = r"sample_noise"
save_dir ="pre_saved_tensors"
target_sample_rate = 16000
chunk_duration_sec = 1
chunk_size = target_sample_rate * chunk_duration_sec

all_chunks = []

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
spec = compute_spectrogram(big_tensor)
print(f"Final tensor shape: {big_tensor.shape}")  # Should be [2, N]
print(f"shape of megnetitude {spec.abs().shape}")

# Save
torch.save(big_tensor, os.path.join(save_dir, "shuffled_big_noise_tensor.pt"))
torch.save(spec,os.path.join(save_dir, "shuffled_big_noise_spec_tensor.pt"))
