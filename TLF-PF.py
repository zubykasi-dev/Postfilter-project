import os
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import stft, istft
from tqdm import tqdm
import warnings

def safe_audio_load(path: str, sr: int = 16000, mono: bool = False) -> np.ndarray:
    """Load audio robustly and return 2D array (samples, channels).

    Args:
        path: Path to audio file.
        sr: Target sample rate in Hz.
        mono: If True, collapse multi-channel signal to mono.

    Returns:
        2D NumPy array of shape (n_samples, n_channels).
    """
    try:
        data, file_sr = sf.read(path, always_2d=True)  # Always returns 2D
        if file_sr != sr:
            data = librosa.resample(data.T, orig_sr=file_sr, target_sr=sr).T
        if mono:
            data = np.mean(data, axis=1, keepdims=True)
        return data
    except Exception:
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        y, _ = librosa.load(path, sr=sr, mono=False)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        return y


# ------------------------------
# Temporal Low-Pass Filter (unchanged)
# ------------------------------
def temporal_lowpass_filter_multichannel(
    signal: np.ndarray,
    fs: int,
    nperseg: int = 512,
    noverlap: int = 256,
    filter_length: int = 3,
) -> np.ndarray:
    """Apply temporal low-pass smoothing in the STFT magnitude domain (per-channel).

    Args:
        signal: Input audio waveform (1D or 2D array with channels in axis 1).
        fs: Sampling frequency in Hz.
        nperseg: Window length for STFT.
        noverlap: Overlap length for STFT.
        filter_length: Temporal smoothing kernel width (in frames).

    Returns:
        Denoised audio waveform with same shape as input.
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    filtered_signal = np.zeros_like(signal)

    for channel in range(signal.shape[1]):
        f, t, Zxx = stft(signal[:, channel], fs, nperseg=nperseg, noverlap=noverlap)
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        filtered_mag = np.copy(magnitude)
        for k in range(magnitude.shape[0]):
            filtered_mag[k, :] = np.convolve(magnitude[k, :],
                                             np.ones(filter_length) / filter_length,
                                             mode='same')
        filtered_Zxx = filtered_mag * np.exp(1j * phase)
        _, filtered_channel_signal = istft(filtered_Zxx, fs, nperseg=nperseg, noverlap=noverlap)

        # Adjust length
        if len(filtered_channel_signal) > signal.shape[0]:
            filtered_channel_signal = filtered_channel_signal[:signal.shape[0]]
        elif len(filtered_channel_signal) < signal.shape[0]:
            filtered_channel_signal = np.pad(filtered_channel_signal,
                                             (0, signal.shape[0] - len(filtered_channel_signal)))

        filtered_signal[:, channel] = filtered_channel_signal

    return filtered_signal


# ------------------------------
# Paths and config
# ------------------------------
input_folder = f"oo/ReTM2.5_office_0.7/4.9/"
output_folder = "lowpass/"
output_prefix = "lowpass"
os.makedirs(output_folder, exist_ok=True)

