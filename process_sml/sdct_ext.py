import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn


import torch.fft

def dct1_rfft_impl(x):
    return torch.view_as_real(torch.fft.rfft(x, dim=1))

def dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))

def idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)

def stdct(waveform: torch.Tensor,
          n_fft: int,
          hop_length: int,
          window: torch.Tensor,
          norm: str = None) -> torch.Tensor:
    """
    waveform: (batch, time)
    returns: (batch, n_frames, n_fft) real DCT-II coefficients
    """
    # 1) frame into [B, n_fft, n_frames]
    frames = waveform.unfold(1, n_fft, hop_length)                # [B, n_frames, n_fft]
    frames = frames * window.view(1, 1, n_fft)                   # apply window
    # 2) DCT along last dim
    B, T, F = frames.shape
    frames = frames.contiguous().view(-1, F)                     # [B*T, F]
    coeffs = dct(frames, norm=norm)                              # your dct-II → [B*T, F]
    return coeffs.view(B, T, F)                                  # [B, n_frames, n_fft]

def istdct(coeffs: torch.Tensor,
           n_fft: int,
           hop_length: int,
           window: torch.Tensor,
           length: int,
           norm: str = None) -> torch.Tensor:
    """
    coeffs: (batch, n_frames, n_fft)
    returns: (batch, length) reconstructed waveform
    """
    B, T, F = coeffs.shape
    # 1) inverse DCT each frame
    v = coeffs.contiguous().view(-1, F)
    frames = idct(v, norm=norm).view(B, T, F)                    # [B, n_frames, n_fft]
    # 2) apply window
    frames = frames * window.view(1, 1, n_fft)
    # 3) overlap–add
    out = torch.zeros(B, (T-1)*hop_length + n_fft, device=frames.device)
    for i in range(T):
        start = i * hop_length
        out[:, start:start+n_fft] += frames[:, i]
    # 4) trim/pad to exactly `length`
    if out.size(1) > length:
        out = out[:, :length]
    elif out.size(1) < length:
        out = F.pad(out, (0, length - out.size(1)))
    return out
