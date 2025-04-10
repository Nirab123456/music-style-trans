#!/usr/bin/env python
# coding: utf8
"""
This script implements an end-to-end training and testing pipeline for source separation.
It:
  - Uses an AudioDatasetFolder-based pipeline to load and pre-process audio spectrograms.
  - Defines a simple UNet-based model (replaceable with any other model).
  - Implements training with learning rate decay, TensorBoard logging, and optional checkpoint saving.
  - Computes losses for source separation using provided y_true (ground truth) and y_pred (model output) keys.
  
Adjust file paths, hyperparameters, and model architecture as needed.
"""

import os
import time
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchaudio.transforms as T

# -----------------------------------------------------------------------------
# Model Definition: Simple UNet for Source Separation
# -----------------------------------------------------------------------------
class UNet(nn.Module):
    """
    A simplified UNet architecture for source separation.
    This model takes a mixture spectrogram (B, 1, F, T) and outputs a dictionary with 
    separated source spectrograms.
    """
    def __init__(self, in_channels: int = 1, features: List[int] = [16, 32, 64]) -> None:
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder.append(self.double_conv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self.double_conv(features[-1], features[-1] * 2)

        # Decoder (reverse)
        rev_features = features[::-1]
        for feature in rev_features:
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self.double_conv(feature * 2, feature))

        # Final output conv layers for each source.
        # For source separation, we output one channel per source.
        # These are initialized externally after model instantiation.
        self.final_convs = nn.ModuleDict()

    def double_conv(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        skip_connections = []
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](x)
        
        # Generate individual source spectrograms via separate final conv layers.
        outputs: Dict[str, torch.Tensor] = {}
        for key, conv in self.final_convs.items():
            outputs[key] = conv(x)
        return outputs

# -----------------------------------------------------------------------------
# Training Function for Source Separation
# -----------------------------------------------------------------------------
def train_model_source_separation(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler,
    num_epochs: int,
    device: torch.device,
    log_dir: str,
    checkpoint_dir: Optional[str],
    y_true_names: List[str],
    y_pred_names: List[str],
    print_freq: int = 10,
) -> nn.Module:
    """
    Trains the model for a source separation task.
    The model is assumed to accept a mixture spectrogram as input (under key "mixture") and output a dictionary
    with keys corresponding to y_pred_names. The dataset dictionary contains ground truth spectrograms
    with keys from y_true_names.
    
    Args:
        model: Source separation model.
        dataloaders: Dictionary with 'train' and 'val' DataLoaders.
        criterion: Loss function (e.g., L1Loss) applied per source.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        num_epochs: Number of training epochs.
        device: Device for training.
        log_dir: TensorBoard log directory.
        checkpoint_dir: Directory to save model checkpoints.
        y_true_names: List of keys for ground truth sources 
                      (e.g., ["vocals_spectrogram", "accompaniment_spectrogram"]).
        y_pred_names: List of keys that the model outputs.
        print_freq: Frequency (in batches) to print updates.
    
    Returns:
        The best model (with lowest validation loss).
    """
    writer = SummaryWriter(log_dir)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 40)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            num_samples = 0
            data_loader = dataloaders[phase]

            for batch_idx, batch in enumerate(data_loader):
                # Get the mixture input and ensure a channel dimension.
                inputs = batch["mixture"]
                if inputs.dim() == 3:
                    inputs = inputs.unsqueeze(1)
                inputs = inputs.to(device)

                # Get ground truth spectrograms for each source.
                y_true_dict = {}
                for key in y_true_names:
                    y = batch[key]
                    if y.dim() == 3:
                        y = y.unsqueeze(1)
                    y_true_dict[key] = y.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs: Dict[str, torch.Tensor] = model(inputs)
                    # Compute loss as the sum of per-source losses.
                    loss = 0.0
                    for y_pred_key, y_true_key in zip(y_pred_names, y_true_names):
                        loss += criterion(outputs[y_pred_key], y_true_dict[y_true_key])

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                num_samples += batch_size

                if batch_idx % print_freq == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(f"{phase.capitalize()} Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(data_loader)}] "
                          f"Loss: {loss.item():.4f} LR: {current_lr:.6f}")
                    global_step = epoch * len(data_loader) + batch_idx
                    writer.add_scalar(f"{phase}/Batch_Loss", loss.item(), global_step)
                    writer.add_scalar("LR", current_lr, global_step)

            epoch_loss = running_loss / num_samples
            print(f"{phase.capitalize()} Epoch Loss: {epoch_loss:.4f}")
            writer.add_scalar(f"{phase}/Epoch_Loss", epoch_loss, epoch)

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            checkpoint_state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": epoch_loss,
            }
            torch.save(checkpoint_state, checkpoint_path)
            print(f"Checkpoint saved at: {checkpoint_path}")

    time_elapsed = time.time() - since
    print(f"Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s")
    print(f"Best val loss: {best_loss:.4f}")

    model.load_state_dict(best_model_wts)
    writer.close()
    return model

# -----------------------------------------------------------------------------
# Testing Function for Source Separation
# -----------------------------------------------------------------------------
def test_model_source_separation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    y_true_names: List[str],
    y_pred_names: List[str],
    criterion: nn.Module
) -> None:
    """
    Evaluates the source separation model on a validation set and prints loss.
    
    Args:
        model: Source separation model.
        dataloader: DataLoader for the test/validation set.
        device: Device for evaluation.
        y_true_names: List of ground truth keys.
        y_pred_names: List of keys that the model outputs.
        criterion: Loss function.
    """
    model.eval()
    running_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            # Get the mixture input and ensure it has a channel dimension.
            inputs = batch["mixture"]
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            inputs = inputs.to(device)
            
            # Get ground truth spectrograms.
            y_true = {}
            for key in y_true_names:
                y = batch[key]
                if y.dim() == 3:
                    y = y.unsqueeze(1)
                y_true[key] = y.to(device)

            outputs: Dict[str, torch.Tensor] = model(inputs)
            loss = 0.0
            for y_pred_key, y_true_key in zip(y_pred_names, y_true_names):
                loss += criterion(outputs[y_pred_key], y_true[y_true_key])
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

    avg_loss = running_loss / num_samples
    print(f"Test Loss: {avg_loss:.4f}")
