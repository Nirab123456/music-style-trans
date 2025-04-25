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
    RandomPitchShift_wav,RandomVolume_wav,RandomAbsoluteNoise_wav,RandomSpeed_wav,RandomFade_wav,RandomFrequencyMasking_spec,RandomTimeMasking_spec,RandomTimeStretch_spec)
# Import the UNet model and the training function from the training module.
from train_sml import UNet, train_model_source_separation,LiteResUNet
import torch.nn as nn

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

# Create the dataset.
dataset_multi = AudioDatasetFolder(
    csv_file='output_stems/musdb18_index_20250408_121813.csv',
    audio_dir='.',  # adjust as needed
    components=COMPONENT_MAP,
    sample_rate=16000,
    duration=5.0,
    spec_transform=argS,  # list of transforms
    wav_transform=argW,
    is_track_id=True,
    input_name= "mixture"
)




# Split dataset into train and validation (e.g., 80/20 split).
dataset_size = len(dataset_multi)
indices = list(range(dataset_size))
split = int(0.8 * dataset_size)
train_indices, val_indices = indices[:split], indices[split:]
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
train_loader = DataLoader(dataset_multi, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(dataset_multi, batch_size=32, sampler=val_sampler)
dataloaders: Dict[str, DataLoader] = {"train": train_loader, "val": val_loader}

# -------------------------------
# Model Integration
# -------------------------------

# model = UNet(in_channels=2)

# # Define the label names (target keys) for source separation.

# # Prepare the final convolution layers for each target output.
# for key in label_names:
#     model.final_convs[key] = nn.Conv2d(16, 2, kernel_size=1)

model = LiteResUNet(backbone="resnet18",source_names=label_names,pretrained=True,in_channels=4)


# IMPORTANT: Move the entire model to the device after adding the final conv layers.
model = model.to(device)

# -------------------------------
# Loss Function, Optimizer, Scheduler
# -------------------------------
# Use L1 loss for source separation.
criterion = nn.L1Loss()
# Create the optimizer using the model parameters.
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# Create a learning rate scheduler.
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1,)


if __name__ == '__main__':

    # -------------------------------
    # Train the Model
    # -------------------------------
    # Here, the input key is "mixture" and label names are defined as above.
    best_model = train_model_source_separation(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=3,
        device=device,
        log_dir='./logs',
        checkpoint_dir='./checkpoints',
        input_name="mixture",  # use "mixture" for the input spectrogram from the batch
        label_names=label_names,  # list of target keys for separated sources
        print_freq=10,
    )
