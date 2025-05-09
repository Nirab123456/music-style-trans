import torch
import torchaudio
import torch.nn as nn

from .transformation_utlis import to_stereo, adjust_phase_shape, adjust_spec_shape
from .spec_transform import (
    RandomFrequencyMasking_spec,
    RandomTimeMasking_spec,
    RandomTimeStretch_spec
)
from .sdct_ext import stdct,istdct

class MyPipeline(nn.Module):
    def __init__(
        self,
        wav_transforms: nn.Module = None,
        spec_transforms: nn.Module = None,
        shape_of_untransformed_size: torch.Size = None,
        input_name: str = None,
        peripheral_names: list[str] = None,
        n_fft: int = 2048,
        hop_length: int = 512,
        input_transformation: str = "2-SPEC",
        rest_transformation: str = "2-SPEC",
        hnn_window_cpu : torch.Tensor = None,
    ):
        """
        A unified pipeline that applies both waveform- and spectrogram-level transforms.

        Args:
            wav_transforms: waveform-level transforms module (or None to skip).
            spec_transforms: spectrogram-level transforms (or None to skip).
            shape_of_untransformed_size: target spec shape for resizing.
            input_name: name of the primary component.
            peripheral_names: list of other component names to treat like input.
            n_fft: FFT size for spectrogram.
            hop_length: hop length for spectrogram.
            input_transformation: one of ['WAV', '2-SPEC', '4-SPEC'] for input component.
            rest_transformation: one of ['WAV', '2-SPEC', '4-SPEC'] for  peripheral components.
        """
        super().__init__()
        self.input_transformation = input_transformation
        self.rest_transformation = rest_transformation
        self.input_name = input_name
        self.peripheral_names = peripheral_names or []
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.wav_transforms = wav_transforms
        self.spec_transforms = list(spec_transforms) if spec_transforms else []

        # split complex (time-stretch) vs. real-only transforms
        self.complex_transforms = [
            t for t in self.spec_transforms if isinstance(t, RandomTimeStretch_spec)
        ]
        self.real_transforms = [
            t for t in self.spec_transforms if not isinstance(t, RandomTimeStretch_spec)
        ]
        self.use_complex = bool(self.complex_transforms)
        self.hnn_window_cpu = hnn_window_cpu
        self.shape_of_spec = shape_of_untransformed_size



    def forward(self, waveform: torch.Tensor, component: str) -> torch.Tensor:
        # determine if we process transforms or bypass
        is_input = component == self.input_name
        is_peripheral = component in self.peripheral_names
        if not (is_input or is_peripheral):
            return self._bypass(waveform)

        # apply waveform-level transforms
        if self.wav_transforms:
            waveform = self.wav_transforms(waveform)

        # choose which transformation setting to use
        transform_type = (
            self.input_transformation if is_input else self.rest_transformation
        )
        if transform_type == "WAV":
            return waveform


        # --- NEW 2‑STDC branch: real‐only DCT “spectrogram” ---
        if transform_type == "2-STDC":
            # 1) frame + window + DCT
            coeffs = stdct(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.hnn_window_cpu,
                norm=None
            )                           # (batch, n_frames, n_fft)
            # 2) any “real” spec transforms
            for t in self.real_transforms:
                coeffs = t(coeffs)

            mean = coeffs.mean(dim=[1,2], keepdim=True)       # [C,1,1]
            std  = coeffs.std(dim=[1,2], keepdim=True) + 1e-6 # avoid div by zero
            coeffs = (coeffs - mean) / std

            return coeffs


        # compute complex spectrogram if needed
        if self.use_complex:
            spec_complex = torchaudio.functional.spectrogram(
                    waveform=waveform,
                    pad=0,
                    window=self.hnn_window_cpu,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    power=None,  
                    normalized=False,
                    win_length=self.n_fft,
                )
            for t in self.complex_transforms:
                spec_complex = t(spec_complex)
        else:
            spec_complex = torchaudio.functional.spectrogram(
                    waveform=waveform,
                    pad=0,
                    window=self.hnn_window_cpu,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    power=None,  
                    normalized=False,
                    win_length=self.n_fft,
                )

        # magnitude and optional phase
        spec = spec_complex.abs()
        phase = spec_complex.angle() if transform_type == "4-SPEC" else None

        # apply real-only spectrogram transforms on magnitude
        for t in self.real_transforms:
            spec = t(spec)

        # resize if a target shape is provided
        if self.shape_of_spec is not None:
            target = self.shape_of_spec[-2:]
            spec = adjust_spec_shape(spec, target)
            if phase is not None:
                phase = adjust_phase_shape(phase, target)

        # return according to type
        if transform_type == "2-SPEC":
            return spec
        if transform_type == "4-SPEC":
            return torch.cat((spec, phase), dim=0)

        raise ValueError(f"Unknown transformation type: {transform_type}")

    def _bypass(self, waveform: torch.Tensor) -> torch.Tensor:
        # stereo conversion and basic spectrogram without any transforms
        waveform = to_stereo(waveform)
        spec_complex = torchaudio.functional.spectrogram(
                waveform=waveform,
                pad=0,
                window=self.hnn_window_cpu,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                power=None,  
                normalized=False,
                win_length=self.n_fft,
            )
        spec = spec_complex.abs()
        t = self.rest_transformation
        if t == "2-SPEC":
            return spec
        if t == "4-SPEC":
            return torch.cat((spec, spec_complex.angle()), dim=0)
        if t == "WAV":
            return waveform
        if t == "2-STDC":
                # DCT-based “spectrogram” on stereo signal
                # waveform shape: (2, time) → stdct returns (2, n_frames, n_fft)
                coeffs = stdct(
                    waveform,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    window=self.hnn_window_cpu,
                    norm=None
                )
                return coeffs
        raise ValueError(f"Unknown rest_transformation: {t}")

