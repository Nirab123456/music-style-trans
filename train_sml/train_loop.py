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
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


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
    input_name: str,
    label_names: List[str],
    print_freq: int = 10,
) -> nn.Module:
    """
    Trains the model for a source separation task.
    The model is assumed to accept an input spectrogram (under key input_name) and output a dictionary
    with keys corresponding to label_names. The dataset dictionary contains ground truth spectrograms
    with keys from label_names.
    
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
        input_name: Key in the batch for the input mixture.
        label_names: List of keys for ground truth sources 
                     (e.g., ["vocals_spectrogram", "accompaniment_spectrogram"]).
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
                # Get the input tensor using the provided input_name key and ensure a channel dimension.
                inputs = batch[input_name]
                if inputs.dim() == 3:
                    inputs = inputs.unsqueeze(1)
                inputs = inputs.to(device)

                # Get ground truth spectrograms for each source using label_names.
                y_true_dict = {}
                for key in label_names:
                    y = batch[key]
                    if y.dim() == 3:
                        y = y.unsqueeze(1)
                    y_true_dict[key] = y.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs: Dict[str, torch.Tensor] = model(inputs)
                    # Compute loss as the sum of per-source losses.
                    loss = 0.0
                    for key in label_names:
                        loss += criterion(outputs[key], y_true_dict[key])

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
    input_name: str,
    label_names: List[str],
    criterion: nn.Module
) -> None:
    """
    Evaluates the source separation model on a test/validation set and prints the loss.
    
    Args:
        model: Source separation model.
        dataloader: DataLoader for the test/validation set.
        device: Device for evaluation.
        input_name: Key in the batch for the input spectrogram.
        label_names: List of ground truth keys which should match with model output keys.
        criterion: Loss function.
    """
    model.eval()
    running_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            # Get the input tensor using the provided input_name key and ensure it has a channel dimension.
            inputs = batch[input_name]
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            inputs = inputs.to(device)
            
            # Get ground truth spectrograms for each source from label_names.
            y_true = {}
            for key in label_names:
                y = batch[key]
                if y.dim() == 3:
                    y = y.unsqueeze(1)
                y_true[key] = y.to(device)

            # Run the model forward pass.
            outputs: Dict[str, torch.Tensor] = model(inputs)
            loss = 0.0
            for key in label_names:
                loss += criterion(outputs[key], y_true[key])
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

    avg_loss = running_loss / num_samples
    print(f"Test Loss: {avg_loss:.4f}")
