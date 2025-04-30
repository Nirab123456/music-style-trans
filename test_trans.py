from torch.utils.data import DataLoader
import torch
import random
from typing import Dict

# Import missing modules for optimization
import torch.optim as optim
from torch.optim import lr_scheduler

# Import our custom dataset and augmentation pipeline.
from process_sml import (
    AudioDatasetFolder, Compose,
    RandomPitchShift_wav,RandomVolume_wav,RandomAbsoluteNoise_wav,RandomSpeed_wav,RandomFade_wav,RandomFrequencyMasking_spec,RandomTimeMasking_spec,RandomTimeStretch_spec,
    compute_waveform,reconstruct_waveform)
# Import the UNet model and the training function from the training module.
from train_sml import UNet, train_model_source_separation,LiteResUNet
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the component map for the dataset.
COMPONENT_MAP = ["mixture", "drums", "bass", "other_accompaniment", "vocals"]
label_names = ["drums", "bass", "other_accompaniment", "vocals"]


argS = Compose([

    #spec transformation 
    RandomTimeStretch_spec(),
    #this two working properly together
    RandomFrequencyMasking_spec(),
    RandomTimeMasking_spec(),

])
argW = Compose(
 [
    # RandomPitchShift_wav(),
    RandomVolume_wav(),
    # RandomSpeed_wav(),
    RandomAbsoluteNoise_wav(),
    RandomFade_wav(),
 ]   
)


# Set random seeds for reproducibility.
torch.manual_seed(42)
random.seed(42)

# Choose device early.
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':

    # Create the dataset.
    dataset_multi = AudioDatasetFolder(
        csv_file='output_stems/mini.csv',
        audio_dir='.',  # adjust as needed
        components=COMPONENT_MAP,
        sample_rate=16000,
        duration=5.0,
        spec_transform=argS,  # list of transforms
        wav_transform=argW,
        is_track_id=True,
        input_name= "mixture"
    )

    loader_multi = DataLoader(dataset_multi, batch_size=8, shuffle=False)
    sample_multi = next(iter(loader_multi))


    # Plot spectrogram for the 'mixture' component.
    spec = sample_multi['mixture'][0]  # select first sample and first channel
