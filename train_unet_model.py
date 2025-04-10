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
# Import the UNet model and the training function from the training module.
from train_sml import UNet, train_model_source_separation
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
# For source separation, the model is expected to take the mixture spectrogram as input,
# and output separated source spectrograms corresponding to each target.
# --- Since the mixture is stereo, we initialize the UNet with in_channels=2 ---
model = UNet(in_channels=2)

# Define the label names (target keys) for source separation.
label_names = ["drums", "bass", "other_accompaniment", "vocals"]

# Prepare the final convolution layers for each target output.
# Here we assume that the decoder produces feature maps with 16 channels.
for key in label_names:
    model.final_convs[key] = nn.Conv2d(16, 1, kernel_size=1)

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
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

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
