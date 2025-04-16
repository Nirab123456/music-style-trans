import torch
import random
from typing import Callable, List, Tuple
import torch.nn.functional as F
import configarations.global_initial_config as global_initial_config
from torchvision.transforms import RandomCrop

# ------------------------------
# Individual Transformation Classes
# ------------------------------

class RandomTimeCrop:
    def __init__(self, target_time: int):
        self.target_time = target_time
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.dim() == 2:
            freq, time = spec.shape
            if time > self.target_time:
                start = random.randint(0, time - self.target_time)
                return spec[:, start:start+self.target_time]
            return spec
        elif spec.dim() == 3:
            channels, freq, time = spec.shape
            if time > self.target_time:
                start = random.randint(0, time - self.target_time)
                return spec[:, :, start:start+self.target_time]
            return spec
        else:
            raise ValueError("Unsupported tensor shape for RandomTimeCrop.")

class RandomTimeStretch:
    def __init__(self, factor_range: Tuple[float, float] = (0.9, 1.1)):
        self.factor_range = factor_range
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        factor: float = random.uniform(*self.factor_range)
        if spec.dim() != 3:
            raise ValueError("RandomTimeStretch expects a tensor with shape (channels, freq, time)")
        channels, freq, time = spec.shape
        new_time: int = int(time * factor)
        stretched = F.interpolate(spec.unsqueeze(0),
                                  size=(freq, new_time),
                                  mode='bilinear',
                                  align_corners=False).squeeze(0)
        if stretched.size(2) < time:
            return F.pad(stretched, (0, time - stretched.size(2)))
        else:
            return stretched[:, :, :time]

class RandomPitchShift:
    def __init__(self, shift_range: Tuple[float, float] = (-1.0, 1.0)):
        self.shift_range = shift_range
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.dim() != 3:
            raise ValueError("RandomPitchShift expects a tensor with shape (channels, freq, time)")
        semitones: float = random.uniform(*self.shift_range)
        factor: float = 2 ** (semitones / 12.0)
        channels, freq, time = spec.shape
        new_freq: int = int(freq * factor)
        shifted = F.interpolate(spec.unsqueeze(0),
                                size=(new_freq, time),
                                mode='bilinear',
                                align_corners=False).squeeze(0)
        if shifted.size(1) < freq:
            return F.pad(shifted, (0, 0, 0, freq - shifted.size(1)))
        else:
            return shifted[:, :freq, :]
        
class RandomNoise:
    def __init__(self, noise_std: float = 0.05):
        self.noise_std = noise_std
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        noise_scale: float = self.noise_std * spec.max().item()
        noise = torch.randn_like(spec) * noise_scale
        return spec + noise


def random_crop_3d(
    tensor: torch.Tensor,
    size: torch.Size
) -> torch.Tensor:
    """
    Randomly crop a tensor to the given size.

    Args:
        tensor (torch.Tensor): The input tensor to crop.
        size (torch.Size): The desired output size.

    Returns:
        torch.Tensor: The randomly cropped tensor.
    """
    first_dim : int = size[1]
    second_dim : int = size[2]

    crop_size = (first_dim,second_dim)
    # Create the transform
    transform = RandomCrop(size=crop_size)

    # Apply it directly
    cropped_tensor = transform(tensor)

    return cropped_tensor

#updated Random noise for absolute realworld noise simulation 
class RandomAbsoluteNoise:
    def __init__(
        self,
        noise_std: float = 0.05,
        noise_tensor_path=f"{global_initial_config.NOISE_TENSOR_SAVE_DIR}/{global_initial_config.NOISE_TENSOR_NAME}"
    ):
        """updated Random noise for absolute realworld noise simulation"""
        self.noise_std = noise_std
        self.noise_tensor = torch.load(noise_tensor_path)

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        shape_of_cur_spec: torch.Size = spec.shape

        noise_tensor: torch.Tensor = random_crop_3d(self.noise_tensor, shape_of_cur_spec)

        # Scale noise by noise_std and add to spec directly
        wav: torch.Tensor = spec + noise_tensor * self.noise_std
        # sape = wav.shape

        return wav



class RandomDistortion:
    def __init__(self, gamma_range: Tuple[float, float] = (0.8, 1.2)):
        self.gamma_range = gamma_range
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        gamma: float = random.uniform(*self.gamma_range)
        return torch.pow(spec, gamma)

class RandomVolume:
    def __init__(self, volume_range: Tuple[float, float] = (0.8, 1.2)):
        self.volume_range = volume_range
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        scale: float = random.uniform(*self.volume_range)
        return spec * scale

# ------------------------------
# RandomSubsetCompose Class for Transformations
# ------------------------------

class RandomSubsetCompose:
    def __init__(self, transforms: List[Callable[[torch.Tensor], torch.Tensor]], num_transforms: int):
        """
        Randomly selects `num_transforms` from the list and applies them in random order.
        
        Args:
            transforms: List of transformation callables.
            num_transforms: Number of transformations to apply each time.
        """
        if num_transforms > len(transforms):
            raise ValueError("num_transforms cannot be greater than the number of available transforms.")
        self.transforms = transforms
        self.num_transforms = num_transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        selected_transforms = random.sample(self.transforms, self.num_transforms)
        for transform in selected_transforms:
            x = transform(x)
        return x
