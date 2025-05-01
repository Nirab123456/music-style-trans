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
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from process_sml import batch_reconstruct_waveform
import torchaudio


# -----------------------------------------------------------------------------
# Training Function for Source Separation
# -----------------------------------------------------------------------------
def train_model_source_separation(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: torch.device,
    scheduler: Optional[lr_scheduler._LRScheduler] = None,
    log_dir: str = "logs",
    checkpoint_dir: str = "checkpoints",
    input_name: str = None,
    label_names: List[str] = None,
    print_freq: int = 10,
) -> nn.Module:
    """
    Trains the model for a source separation task.

    Args:
        model: source separation model (returns Dict[str, Tensor])
        train_dataset: training set
        val_dataset: validation set
        batch_size: batch size
        optimizer: optimizer
        criterion: loss function
        scheduler: learning rate scheduler (optional)
        num_epochs: number of epochs
        device: torch.device
        log_dir: TensorBoard logging directory
        checkpoint_dir: directory to save checkpoints
        input_name: key in batch for input tensor
        label_names: keys in batch for target tensors
        print_freq: batches between status prints
    Returns:
        model with best validation loss loaded
    """
    assert criterion is not None, "criterion must be provided"
    assert label_names, "label_names must be provided"

    model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    best_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    since = time.time()

    with SummaryWriter(log_dir) as writer:
        for epoch in range(1, num_epochs+1):
            print(f"Epoch {epoch}/{num_epochs}")
            print("-" * 40)

            for phase, loader in [('train', train_loader), ('val', val_loader)]:
                is_train = (phase == 'train')
                model.train() if is_train else model.eval()

                running_loss = 0.0
                total_samples = 0

                for batch_idx, batch in enumerate(loader):
                    x = batch[input_name]
                    if x.ndim == 3:
                        x = x.unsqueeze(0)
                    x = x.to(device)

                    y_dict = {}
                    for key in label_names:
                        y = batch[key]
                        if y.ndim == 3:
                            y = y.unsqueeze(0)
                        y_dict[key] = y.to(device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(is_train):
                        outputs: Dict[str, torch.Tensor] = model(x)
                        loss = sum(criterion(outputs[k], y_dict[k]) for k in label_names)
                        if is_train:
                            loss.backward()
                            optimizer.step()

                    bsz = x.size(0)
                    running_loss += loss.item() * bsz
                    total_samples += bsz

                    if batch_idx % print_freq == 0:
                        lr = optimizer.param_groups[0]['lr']
                        print(f"{phase.capitalize()} [{batch_idx}/{len(loader)}] "
                              f"Loss: {loss.item():.4f} LR: {lr:.6f}")
                        step = (epoch-1) * len(loader) + batch_idx
                        writer.add_scalar(f"{phase}/Batch_Loss", loss.item(), step)
                        writer.add_scalar("LR", lr, step)

                epoch_loss = running_loss / total_samples
                print(f"{phase.capitalize()} Epoch Loss: {epoch_loss:.4f}")
                writer.add_scalar(f"{phase}/Epoch_Loss", epoch_loss, epoch)

                if not is_train and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_wts = copy.deepcopy(model.state_dict())

            # Step scheduler after validation
            if scheduler is not None:
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(best_loss)
                else:
                    scheduler.step()

            # Save checkpoint every epoch
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_loss': best_loss,
            }, os.path.join(checkpoint_dir, f"epoch_{epoch}.pth"))
            print(f"Checkpoint saved: epoch_{epoch}.pth")

        time_elapsed = time.time() - since
        print(f"Training complete in {int(time_elapsed//60)}m {int(time_elapsed%60)}s. Best val loss: {best_loss:.4f}")

    # Load best weights
    model.load_state_dict(best_wts)
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
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(0)
            inputs = inputs.to(device)
            
            # Get ground truth spectrograms for each source from label_names.
            y_true = {}
            for key in label_names:
                y = batch[key]
                if y.dim() == 2:
                    y = y.unsqueeze(0)
                y_true[key] = y.to(device)

            # Run the model forward pass.
            outputs: Dict[str, torch.Tensor] = model(inputs)
            loss = 0.0
            for key in label_names:

                output = outputs[key]


                loss += criterion(output, y_true[key])
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

    avg_loss = running_loss / num_samples
    print(f"Test Loss: {avg_loss:.4f}")


def infer_and_save(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: str,
    input_name: str,
    label_names: list,
    sample_rate: int
):
    """
    Run inference on the dataloader and save both the reconstructed mixture
    and each separated source as .wav files.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.to(device).eval()

    with torch.no_grad():
        for batch in dataloader:
            # Get the input tensor using the provided input_name key and ensure it has a channel dimension.
            inputs = batch[input_name]
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(0)
            inputs = inputs.to(device)
            
            # Get ground truth spectrograms for each source from label_names.
            y_true = {}
            for key in label_names:
                y = batch[key]
                if y.dim() == 2:
                    y = y.unsqueeze(0)
                y_true[key] = y.to(device)

            # Run the model forward pass.
            outputs: Dict[str, torch.Tensor] = model(inputs)

            # ---- 3) Reconstruct & save each source ----
            for key in label_names:
                specphase = outputs[key]   # [B, 2*C, F, T]
                mag, phase = torch.chunk(specphase, 2, dim=1)
                complex_spec = torch.polar(mag, phase)   # [B, C, F, T]
                wavs = batch_reconstruct_waveform(complex_spec)  # [B, C, L]
                wavs = wavs.cpu()

                for i in range(wavs.size(0)):
                    torchaudio.save(
                        os.path.join(output_dir, f"{key}_b_i{i}.wav"),
                        wavs[i],
                        sample_rate
                    )
            mix_mag, mix_phase = torch.chunk(inputs, 2, dim=1)
            mix_complex = torch.polar(mix_mag, mix_phase)
            mix_wavs = batch_reconstruct_waveform(mix_complex).cpu()  # [B, C, L]
            for i in range(mix_wavs.size(0)):
                torchaudio.save(
                    os.path.join(output_dir, f"mixture_b{i}.wav"),
                    mix_wavs[i],
                    sample_rate
                )
    print(f"✅ All inference outputs saved to {output_dir}")
