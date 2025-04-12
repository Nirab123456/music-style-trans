import os
import torch
import torchaudio
from IPython.display import Audio
from process_sml import (
    AudioDatasetFolder, Compose, RandomTimeCrop, RandomTimeStretch,
    RandomPitchShift, RandomNoise, RandomDistortion, RandomVolume,
    compute_waveform, compute_spectrogram
)

root = "pre_saved_tensors"
# Define cache path
waveform_cache_path = f"{root}/cached_waveform.pt"
spec_cache_path = f"{root}/cached_spec.pt"

os.mkdir(root)

# Check if already cached
if os.path.exists(waveform_cache_path) and os.path.exists(spec_cache_path):
    print("Loading from cache...")
    waveform = torch.load(waveform_cache_path)
    spec = torch.load(spec_cache_path)
else:
    print("Processing from raw MP3...")
    waveform, sample_rate = torchaudio.load(r"C:\Users\rifat\Downloads\Music\super-saw-bass-37512.mp3")

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Take first 5 seconds
    num_samples = sample_rate * 5
    waveform = waveform[:, :num_samples]

    # Save waveform
    torch.save(waveform, waveform_cache_path)

    # Compute spectrogram and save
    spec = compute_spectrogram(waveform)
    torch.save(spec, spec_cache_path)

# Optional: Play audio
# Audio(waveform.numpy(), rate=16000)

# Show shape
print(f"Shape of 5-second noise spectrogram: {spec.abs().shape}")
