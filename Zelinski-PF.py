"""
Zelinski Postfilter Implementation for Multichannel Audio Processing.

This module implements the Zelinski postfilter algorithm for multichannel noise reduction.
The algorithm combines coherence-based filtering in low/high frequency bands with
midband preservation, using GCC-PHAT for channel alignment and STFT-based processing.

Key Features:
- Multichannel audio processing with automatic alignment
- Hybrid frequency band processing (coherence-based + midband pass-through)
- GCC-PHAT delay estimation for temporal alignment
- Quality assessment using STOI and mel spectrogram distance

"""

import numpy as np
import librosa
import soundfile as sf
import os
import scipy.io.wavfile as wav
from scipy.signal import wiener
from typing import Union, Tuple, Optional, List
import numpy.typing as npt


def align_signals(clean: npt.NDArray[np.float64], denoised: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Align clean and denoised signals to have identical length.

    Crops or pads the clean signal to match the denoised signal length.
    Ensures both signals have the same temporal length for comparison.

    Args:
        clean: Clean reference signal array
        denoised: Processed/denoised signal array

    Returns:
        Tuple of (aligned_clean, aligned_denoised) with identical lengths
    """
    min_len = min(len(clean), len(denoised))
    max_len = max(len(clean), len(denoised))
    
    if len(clean) > len(denoised):
        clean = clean[:len(denoised)]
    elif len(clean) < len(denoised):
        pad = np.zeros(len(denoised) - len(clean))
        clean = np.concatenate([clean, pad])
    
    return clean, denoised[:len(clean)]

def calculate_stoi_multichannel(ref: Union[str, npt.NDArray[np.float64]],
                               deg: Union[str, npt.NDArray[np.float64]],
                               sr: int,
                               extended: bool = False) -> float:
    """
    Compute average STOI (Short-Time Objective Intelligibility) or ESTOI for multichannel signals.

    Ensures reference and degraded signals are aligned and have the same length.
    Calculates STOI for each channel and returns the average score.

    Args:
        ref: Reference signal (clean) - either file path or numpy array
        deg: Degraded signal (processed) - either file path or numpy array
        sr: Sample rate in Hz
        extended: Whether to use extended STOI (ESTOI) instead of standard STOI

    Returns:
        Average STOI score across all channels (0-1, higher is better)
        Returns NaN if no valid channels found
    """
    # Load if file paths
    if isinstance(ref, str):
        ref, _ = librosa.load(ref, sr=sr, mono=False)
    if isinstance(deg, str):
        deg, _ = librosa.load(deg, sr=sr, mono=False)
    
    # Ensure 2D shape
    if ref.ndim == 1:
        ref = ref[:, np.newaxis]
    if deg.ndim == 1:
        deg = deg[:, np.newaxis]

    # Match channels
    min_ch = min(ref.shape[1], deg.shape[1])
    ref, deg = ref[:, :min_ch], deg[:, :min_ch]

    # Align time length (crop or pad clean to match denoised)
    ref, deg = align_signals(ref, deg)

    scores = []
    for ch in range(min_ch):
        try:
            assert len(ref) == len(deg), f"Length mismatch after alignment: {len(ref)} vs {len(deg)}"

            score = stoi(ref[:, ch], deg[:, ch], sr, extended=extended)
            scores.append(score)
        except Exception as e:
            print(f"Warning: STOI error on channel {ch}: {e}")

    return float(np.mean(scores)) if scores else np.nan


def gcc_phat_delay(sig: npt.NDArray[np.float64],
                  ref: npt.NDArray[np.float64],
                  fs: int,
                  max_tau: Optional[float] = None) -> float:
    """
    Estimate time delay between two signals using Generalized Cross-Correlation with PHase Transform (GCC-PHAT).

    This method provides robust delay estimation even in noisy environments by using
    the phase transform to whiten the cross-correlation function.

    Args:
        sig: Signal array (1D numpy array)
        ref: Reference signal array (1D numpy array)
        fs: Sampling frequency in Hz
        max_tau: Maximum expected delay in seconds (search window limit)

    Returns:
        Estimated delay in seconds. Positive means sig lags ref (sig occurs later).
    """
    # Make sure inputs are 1D numpy arrays
    sig = np.asarray(sig, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    n = sig.shape[0] + ref.shape[0]

    # FFT length: next power of two for efficient computation
    N = 1 << (int(np.ceil(np.log2(n))))
    SIG = np.fft.rfft(sig, n=N)
    REF = np.fft.rfft(ref, n=N)

    # Compute cross-power spectrum
    R = SIG * np.conj(REF)
    denom = np.abs(R)
    # Avoid division by zero
    denom[denom == 0] = 1e-16
    # Apply PHAT weighting (phase transform) - whiten the spectrum
    R_phat = R / denom

    # Compute cross-correlation via inverse FFT
    cc = np.fft.irfft(R_phat, n=N)
    # Shift so that zero lag is at center
    max_shift = int(N // 2)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # Search within max_tau if provided
    if max_tau is not None:
        max_shift_samples = int(np.minimum(max_shift, np.ceil(max_tau * fs)))
    else:
        max_shift_samples = max_shift

    mid = max_shift
    search = cc[mid - max_shift_samples: mid + max_shift_samples + 1]
    peak = np.argmax(np.abs(search))
    peak_index = peak + (mid - max_shift_samples)

    # Convert to lag (samples), center offset:
    lag = peak_index - mid

    # Parabolic interpolation for sub-sample precision
    # Fit parabola to peak and neighbors: y(-1), y(0), y(+1)
    if 1 <= (peak + (mid - max_shift_samples)) < (len(cc) - 1):
        i = peak_index
        y_m1, y0, y_p1 = cc[i-1], cc[i], cc[i+1]
        # Parabolic interpolation formula: delta = 0.5 * (y[-1] - y[+1]) / (2*y[0] - y[-1] - y[+1])
        denom_par = (y_m1 - 2*y0 + y_p1)
        if denom_par != 0:
            delta = 0.5 * (y_m1 - y_p1) / denom_par
        else:
            delta = 0.0
    else:
        delta = 0.0

    lag_refined = lag + delta
    delay_seconds = lag_refined / float(fs)
    return delay_seconds

def fractional_shift_in_time(x: npt.NDArray[np.float64],
                           delay_seconds: float,
                           fs: int) -> npt.NDArray[np.float64]:
    """
    Apply fractional time delay to a signal using frequency domain processing.

    Shifts the signal by a fractional number of samples by applying a linear
    phase shift in the frequency domain. This method supports sub-sample delays.

    Args:
        x: Input signal array (1D)
        delay_seconds: Delay amount in seconds. Positive means shift right (delay).
        fs: Sampling frequency in Hz

    Returns:
        Time-shifted signal with same length as input
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    # FFT
    X = np.fft.fft(x)
    # Frequency bins
    k = np.arange(N)
    # Angular frequency for bin k: 2*pi*k/N (rad/sample)
    # Phase shift = exp(-j * 2*pi * k * delay_in_samples / N)
    delay_samples = delay_seconds * fs
    # Linear phase ramp in frequency domain
    phase = np.exp(-1j * 2.0 * np.pi * k * delay_samples / N)
    X_shifted = X * phase
    x_shifted = np.fft.ifft(X_shifted)
    # Return real part (should be nearly real due to input being real)
    return np.real(x_shifted)
def align_multichannel_gccphat(data: npt.NDArray[np.float64],
                              fs: int,
                              ref_ch: int = 0,
                              max_delay: float = 0.02,
                              verbose: bool = False) -> npt.NDArray[np.float64]:
    """
    Align multichannel time-domain signals using GCC-PHAT delay estimation.

    Estimates the relative delays between each channel and a reference channel,
    then compensates for these delays to temporally align all channels.

    Args:
        data: Multichannel signal array of shape (samples, channels)
        fs: Sampling frequency in Hz
        ref_ch: Reference channel index (default: 0)
        max_delay: Maximum expected delay in seconds for GCC-PHAT search
        verbose: Whether to print delay estimates for each channel

    Returns:
        Aligned multichannel signal array with same shape as input
    """
    # Ensure 2D shape (samples, channels)
    arr = np.asarray(data)
    if arr.ndim == 1:
        return arr[:, np.newaxis]  # Convert to (N, 1)
    if arr.shape[1] == 1:
        return arr.copy()  # Already single channel

    ref = arr[:, ref_ch]
    out = np.zeros_like(arr)
    out[:, ref_ch] = ref.copy()  # Reference channel unchanged

    delays = []
    for ch in range(arr.shape[1]):
        if ch == ref_ch:
            delays.append(0.0)
            continue
        sig = arr[:, ch]
        # Estimate delay of current channel relative to reference
        delay = gcc_phat_delay(sig, ref, fs, max_tau=max_delay)
        delays.append(delay)
        # Apply negative delay to align: shift left if delay > 0
        shifted = fractional_shift_in_time(sig, -delay, fs)
        out[:, ch] = shifted

    if verbose:
        print(f"GCC-PHAT delays (sec) per channel relative to ch{ref_ch}: {delays}")
    return out

def zelinski_postfilter(input_file: str,
                       output_file: str,
                       n_fft: int = 512,
                       hop: int = 96,
                       eps: float = 1e-12,
                       floor_db: float = -10,
                       alpha: float = 0.6,
                       verbose: bool = False,
                       midband_low: float = 800,
                       midband_high: float = 3200) -> None:
    """
    Apply Zelinski postfilter to multichannel audio for noise reduction.

    Implements a hybrid postfilter combining coherence-based gain calculation
    in low/high frequency bands with a pass-through in the midband (800-3200 Hz).
    Uses GCC-PHAT for multichannel alignment and STFT-based processing.

    Args:
        input_file: Path to input multichannel audio file
        output_file: Path to save filtered output audio
        n_fft: FFT size for STFT analysis
        hop: Hop length for STFT (overlap)
        eps: Small epsilon value to avoid division by zero
        floor_db: Minimum gain floor in dB
        alpha: Temporal smoothing factor (0-1)
        verbose: Whether to print processing information
        midband_low: Lower frequency boundary of midband in Hz
        midband_high: Upper frequency boundary of midband in Hz
    """

    # === Load and Preprocess Audio ===
    data, sr = sf.read(input_file)
    if data.ndim == 1:
        data = data[:, None]  # Convert mono to (N, 1)
    C = data.shape[1]  # Number of channels
    n_samples = len(data)

    # Align channels temporally using GCC-PHAT
    data_aligned = align_multichannel_gccphat(data, sr, ref_ch=0, max_delay=0.12, verbose=False)

    # === Short-Time Fourier Transform ===
    # Compute STFT for each channel: shape (C, F, T)
    X = np.stack([
        librosa.stft(data_aligned[:, ch], n_fft=n_fft, hop_length=hop, win_length=n_fft)
        for ch in range(C)
    ], axis=0)
    F, T = X.shape[1], X.shape[2]  # Frequency bins, time frames
    freqs = np.linspace(0, sr/2, F)

    # === Zelinski Gain Calculation ===
    # Compute coherence-based gain for each frequency bin
    # Based on cross-correlation between channel pairs
    G = np.zeros((F, T), dtype=np.float32)
    for f in range(F):
        num = np.zeros(T, dtype=np.float64)   # numerator: sum of real cross-correlations
        denom = np.zeros(T, dtype=np.float64) # denominator: sum of power spectra

        # Compute coherence between all channel pairs
        for i in range(C):
            Xi = X[i, f, :]  # STFT coefficients for channel i, frequency f
            denom += np.abs(Xi)**2  # accumulate power spectrum
            for j in range(i+1, C):
                Xj = X[j, f, :]  # STFT coefficients for channel j, frequency f
                # Real part of cross-correlation (coherence magnitude)
                num += np.real(Xi * np.conj(Xj))

        # Sanitize values to handle NaN/inf
        num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
        denom = np.nan_to_num(denom, nan=0.0, posinf=0.0, neginf=0.0)

        # Avoid division by zero and numerical underflow
        denom = np.maximum(denom, eps * 10)

        # Compute coherence ratio: |E[X_i * conj(X_j)]| / E[|X_i|^2]
        ratio = num / ((C - 1) * denom)
        ratio = np.clip(ratio, 0.0, 1.0)  # Ensure ratio is in [0, 1]
        ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply temporal smoothing to reduce fluctuations
        for t in range(1, T):
            ratio[t] = alpha * ratio[t-1] + (1 - alpha) * ratio[t]

        # Convert coherence to gain with minimum floor
        G[f, :] = np.maximum(ratio, 10 ** (floor_db / 20.0))

    # === Hybrid Frequency Band Mask ===
    # Create frequency mask: 0 in midband (preserve), 1 in low/high bands (apply coherence gain)
    hybrid_mask = np.ones(F)
    hybrid_mask[(freqs >= midband_low) & (freqs <= midband_high)] = 0.0

    # Smooth transitions at band edges to avoid artifacts
    ramp = 100  # Transition width in Hz
    idx_low = np.logical_and(freqs >= midband_low - ramp, freqs < midband_low)
    idx_high = np.logical_and(freqs > midband_high, freqs <= midband_high + ramp)
    if np.any(idx_low):
        hybrid_mask[idx_low] = np.linspace(1, 0, np.sum(idx_low))  # Fade from 1 to 0
    if np.any(idx_high):
        hybrid_mask[idx_high] = np.linspace(0, 1, np.sum(idx_high))  # Fade from 0 to 1

    # === Apply Hybrid Gain ===
    # Compute magnitude and phase from average across channels
    X_mag = np.mean(np.abs(X), axis=0)  # Average magnitude across channels
    X_phase = np.angle(X[0])  # Use phase from first channel

    # Combine coherence gain (low/high) with pass-through (midband)
    G_total = (hybrid_mask[:, None] * G) + (1 - hybrid_mask[:, None]) * 1.0

    # Apply gain to magnitude and reconstruct complex STFT
    Y = G_total * X_mag * np.exp(1j * X_phase)

    # === Inverse STFT and Save ===
    y = librosa.istft(Y, hop_length=hop, win_length=n_fft)
    y = np.clip(y, -1.0, 1.0)  # Ensure output is in valid range
    sf.write(output_file, y, sr)

    if verbose:
        print(f"[Hybrid Zelinski] midband preserved {midband_low}-{midband_high} Hz, "
              f"gain floor={floor_db} dB, alpha={alpha}, C={C}")
def compute_mel_distance(ref: npt.NDArray[np.float32],
                        deg: npt.NDArray[np.float32],
                        sr: int,
                        n_mels: int = 128,
                        fmax: float = 8000) -> float:
    """
    Compute mel spectrogram distance between reference and degraded signals.

    Calculates the mean squared error between mel spectrograms converted to dB scale.
    This provides a perceptual distance measure that correlates with human perception.

    Args:
        ref: Reference (clean) signal array
        deg: Degraded (processed) signal array
        sr: Sample rate in Hz
        n_mels: Number of mel bands for spectrogram
        fmax: Maximum frequency for mel spectrogram in Hz

    Returns:
        Mean squared error between mel spectrograms in dB scale.
        Lower values indicate better quality. Returns NaN on error.
    """
    ref = np.asarray(ref, dtype=np.float32)
    deg = np.asarray(deg, dtype=np.float32)

    # Flatten multi-dimensional arrays if needed (take mean across channels)
    if ref.ndim > 1:
        ref = ref.mean(axis=-1)
    if deg.ndim > 1:
        deg = deg.mean(axis=-1)

    # Ensure same length
    min_len = min(len(ref), len(deg))
    ref, deg = ref[:min_len], deg[:min_len]

    try:
        # Compute mel spectrograms (perceptually motivated frequency scaling)
        mel_ref = librosa.feature.melspectrogram(y=ref, sr=sr, n_mels=n_mels, fmax=fmax)
        mel_deg = librosa.feature.melspectrogram(y=deg, sr=sr, n_mels=n_mels, fmax=fmax)

        # Convert to log scale (dB) for better perceptual representation
        mel_ref_db = librosa.power_to_db(mel_ref, ref=np.max)
        mel_deg_db = librosa.power_to_db(mel_deg, ref=np.max)

        # Compute mean squared error between spectrograms
        distance = np.mean((mel_ref_db - mel_deg_db)**2)
        return float(distance)
    except Exception as e:
        print(f"Warning: Mel distance computation failed: {e}")
        return np.nan



input_folder = f"oo/ReTM2.5_office_0.7/"
output_folder = "zel/"
