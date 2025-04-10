from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import random
from typing import Dict

# Import missing modules for optimization
import torch.optim as optim
from torch.optim import lr_scheduler

# Import our custom dataset and augmentation pipeline.
from process_sml import AudioDatasetFolder, augmentation_pipeline
# Import the UNet model from the training module.
from train_sml import UNet , train_model_source_separation
import torch.nn as nn

# Define the component map for the dataset.
COMPONENT_MAP = ["mixture", "drums", "bass", "other_accompaniment", "vocals"]
IS_TRACK_ID = True

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
    sample_rate=44100,
    duration=5.0,
    transform=[augmentation_pipeline],  # list of transforms
    is_track_id=IS_TRACK_ID,
)

# Create a loader for visualization or debugging if needed.
loader_multi = DataLoader(dataset_multi, batch_size=32, shuffle=False)

# Split dataset into train and validation (e.g., 80/20 split).
dataset_size = len(dataset_multi)
indices = list(range(dataset_size))
split = int(0.8 * dataset_size)
train_indices, val_indices = indices[:split], indices[split:]
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
train_loader = DataLoader(dataset_multi, batch_size=8, sampler=train_sampler, num_workers=4)
val_loader = DataLoader(dataset_multi, batch_size=8, sampler=val_sampler, num_workers=4)
dataloaders: Dict[str, DataLoader] = {"train": train_loader, "val": val_loader}

# -------------------------------
# Model Integration
# -------------------------------
# For source separation, we assume that the model takes a mix spectrogram as input
# and outputs separated sources for each target.
# Here we create a UNet-based model.
# Instantiate the UNet. Since the UNet in train_sml accepts only in_channels,
# we omit any unsupported parameter (like out_channels).
model = UNet(in_channels=1)
# Move the model to the chosen device.
model = model.to(device)

# Prepare the final convolution layers for each source.
# Based on the architecture, we assume the output feature maps have 16 channels.
model.final_convs["vocals_spectrogram"] = nn.Conv2d(16, 1, kernel_size=1)
model.final_convs["accompaniment_spectrogram"] = nn.Conv2d(16, 1, kernel_size=1)

# -------------------------------
# Loss Function, Optimizer, Scheduler
# -------------------------------
# Use L1 loss for source separation.
criterion = nn.L1Loss()
# Create the optimizer using the model parameters.
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# Create a learning rate scheduler.
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# (Now you can pass the dataloaders, model, optimizer, etc. to your training function.)
# For example:
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
    y_true_names=["vocals_spectrogram", "accompaniment_spectrogram"],
    y_pred_names=["vocals_spectrogram", "accompaniment_spectrogram"],
    print_freq=10,
)
