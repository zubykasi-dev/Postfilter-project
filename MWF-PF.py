"""Multi-channel Wiener Filter Post-processor for Audio Denoising.

This module implements frequency-domain Wiener filtering for audio denoising,
supporting both single and multi-channel audio files. It includes STFT-based
filtering with automatic noise spectrum estimation and output normalization.

Configuration:
    - Single-core CPU enforcement for deterministic processing
    - Scipy signal processing (STFT/ISTFT)
    - NumPy for numerical operations
"""

import os

# Enforce single-core CPU for deterministic RTF measurements.
# Set BLAS/OMP environment variables BEFORE importing heavy numeric libs.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

# Best-effort: limit thread pools from common libraries when available.
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
except Exception:
    pass

from typing import Tuple

import soundfile as sf
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import wiener
from scipy.signal import stft, istft
import librosa
import torch

# Force single-threaded PyTorch for reproducible single-core runs
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass
def enforce_single_core() -> None:
    """Enforce single-core CPU usage across numeric libraries.
    
    This function configures environment variables and library settings to
    ensure single-threaded execution. Useful for deterministic and
    reproducible audio processing.
    
    Configured libraries:
        - OpenMP (OMP_NUM_THREADS)
        - MKL (Intel Math Kernel Library)
        - OpenBLAS
        - NumExpr
        - vecLib (macOS)
        - PyTorch
        - threadpoolctl (if available)
    """
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # Print verification for the run
    print(f"Thread settings: OMP={os.environ.get('OMP_NUM_THREADS')}, MKL={os.environ.get('MKL_NUM_THREADS')}, OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}")
    try:
        print(f"PyTorch threads: {torch.get_num_threads()}")
    except Exception:
        pass
def improved_wiener(
    data: np.ndarray,
    fs: int,
    window_size: int = 2048,
    overlap: float = 0.75,
    noise_reduction_factor: float = 9.0
) -> np.ndarray:
    """Apply frequency-domain Wiener filtering with automatic noise estimation.
    
    Uses Short-Time Fourier Transform (STFT) to decompose the signal into
    frequency bins, estimates noise from initial frames, and applies optimal
    Wiener filter gain in the frequency domain.
    
    Args:
        data: Input audio signal (1D array of samples)
        fs: Sampling frequency in Hz
        window_size: FFT window size (default: 2048 samples)
        overlap: Window overlap fraction, 0-1 (default: 0.75 = 75%)
        noise_reduction_factor: Regularization factor for SNR calculation
                              (default: 9.0, higher = more aggressive filtering)
    
    Returns:
        Denoised audio signal, same length as input
    
    Processing steps:
        1. Compute STFT decomposition
        2. Estimate noise spectrum from first 5 frames
        3. Calculate a priori SNR and Wiener filter gain
        4. Apply gain in frequency domain
        5. Reconstruct time-domain signal via inverse STFT
    """
    # Compute STFT decomposition: converts time-domain signal to frequency representation
    # f: frequency bins, t: time frames, Zxx: complex STFT coefficients
    f, t, Zxx = stft(data, fs=fs, nperseg=window_size, noverlap=int(window_size * overlap))
    
    # Estimate noise spectrum from initial silence/noise frames
    # Assumes first 5 frames contain mostly noise; takes mean magnitude across frequencies
    noise_estimate = np.mean(np.abs(Zxx[:, :5]), axis=1)
    
    # Compute power spectral density: |X(f,t)|^2
    sig_power = np.abs(Zxx) ** 2
    
    # Compute noise power spectrum: replicate across time frames
    noise_power = noise_estimate[:, np.newaxis] ** 2
    
    # Calculate a priori SNR (Signal-to-Noise Ratio) in frequency bins
    # Clipped at minimum 6 dB to prevent over-suppression at low SNR
    xi = np.maximum(sig_power / noise_power - 0, 6)
    
    # Compute Wiener filter gain: G(f,t) = SNR / (SNR + noise_reduction_factor)
    # Range: [0, 1], approaches 1 at high SNR (pass signal), 0 at low SNR (suppress noise)
    G = xi / (xi + noise_reduction_factor)
    
    # Apply frequency-domain filtering: multiply STFT by gain
    Zxx_denoised = Zxx * G
    
    # Convert denoised frequency representation back to time domain
    _, denoised_data = istft(Zxx_denoised, fs=fs, nperseg=window_size, noverlap=int(window_size * overlap))
    
    return denoised_data
def multichannel_wiener_filter(input_file: str, output_file: str) -> None:
    """Denoise an audio file using multi-channel Wiener filtering.
    
    Supports both mono and multi-channel audio files. For mono files,
    converts to single-channel format. Automatically handles channel-wise
    processing, length alignment, and output normalization.
    
    Args:
        input_file: Path to input WAV file (mono or multi-channel)
        output_file: Path to save denoised WAV file
    
    Processing steps:
        1. Load audio and detect channel configuration
        2. Convert mono to channel dimension if needed
        3. Process each channel through improved_wiener() filter
        4. Align output lengths with input (handle STFT edge effects)
        5. Normalize output to [-1, 1] range
        6. Write denoised audio to output file
    
    Raises:
        FileNotFoundError: If input file does not exist
        RuntimeError: If audio processing fails
    """
    # Load audio file and sample rate
    data, sample_rate = sf.read(input_file)
    
    # Handle single-channel vs multi-channel audio format
    if len(data.shape) == 1:
        # Convert 1D array (mono) to 2D array with shape (samples, 1)
        data = data.reshape(-1, 1)
    
    # Initialize output array with the same shape and dtype as input
    denoised_data = np.zeros_like(data)
    
    # Process each audio channel independently
    for i in range(data.shape[1]):
        # Extract single channel data
        channel_data = data[:, i]
        
        # Skip processing for silent/zero-variance channels (avoid degenerate filter behavior)
        if np.var(channel_data) < 1e-10:
            denoised_data[:, i] = channel_data
            continue
        
        # Apply Wiener filter to current channel
        denoised_channel = improved_wiener(channel_data, sample_rate)
        
        # Align output length with input (STFT may produce different lengths)
        if len(denoised_channel) > len(channel_data):
            # Truncate if too long
            denoised_channel = denoised_channel[:len(channel_data)]
        elif len(denoised_channel) < len(channel_data):
            # Zero-pad if too short (silence at end)
            denoised_channel = np.pad(denoised_channel, (0, len(channel_data) - len(denoised_channel)), 'constant')
        
        # Store processed channel in output array
        denoised_data[:, i] = denoised_channel
    
    # Normalize output to standard audio range [-1.0, 1.0] to prevent clipping
    max_val = np.max(np.abs(denoised_data))
    if max_val > 0:
        denoised_data = denoised_data / max_val
    
    # Write denoised audio to output file, preserving original sample rate
    sf.write(output_file, denoised_data, sample_rate)

def custom_wiener(
    data: np.ndarray,
    noise_level: float = 0.05,
    epsilon: float = 1e-10
) -> np.ndarray:
    """Apply time-domain Wiener filtering with local variance estimation.
    
    This is an alternative to frequency-domain filtering, using a sliding window
    to estimate local mean and variance, then applying the Wiener filter formula
    in the time domain.
    
    Args:
        data: Input audio signal (1D array of samples)
        noise_level: Estimated noise power (default: 0.05)
        epsilon: Minimum variance floor to prevent division by zero (default: 1e-10)
    
    Returns:
        Filtered audio signal, same length as input
    
    Note:
        Generally less effective for broadband noise compared to frequency-domain
        methods, but computationally simpler.
    """
    # Estimate local mean and variance using a sliding window
    window_size = 5
    local_mean = np.convolve(data, np.ones(window_size) / window_size, mode='same')
    local_var = np.convolve((data - local_mean) ** 2, np.ones(window_size) / window_size, mode='same')
    
    # Prevent zero variance by adding epsilon to avoid division by zero
    local_var = np.maximum(local_var, epsilon)
    
    # Apply Wiener filter formula: output = mean + (variance - noise) / variance * (signal - mean)
    return local_mean + (local_var - noise_level) / local_var * (data - local_mean)



# ============================================================================
# CONFIGURATION
# ============================================================================

# Input and output file paths
INPUT_FOLDER: str = "oo/ReTM2.5_office_0.7/4.9/"
OUTPUT_FOLDER: str = "mwf/"
OUTPUT_PREFIX: str = "mwf"


def process_folder(start_num: int = 102, end_num: int = 163) -> None:
    """Process a batch of numbered audio files with Wiener filtering.
    
    Iterates over a range of file numbers, loads each input file,
    applies multi-channel Wiener filtering, and saves the denoised output.
    Errors during processing are caught and logged without stopping execution.
    
    Args:
        start_num: Starting file number (inclusive, default: 100)
        end_num: Ending file number (exclusive, default: 160)
    
    File naming convention:
        - Input: {INPUT_FOLDER}/denoised_{num}.wav
        - Output: {OUTPUT_FOLDER}/{OUTPUT_PREFIX}_denoised_{num}.wav
    
    Example:
        process_folder(111, 129) processes denoised_111.wav to denoised_128.wav
    """
    for num in range(start_num, end_num):
        try:
            # Construct input and output file paths
            input_file: str = f'{INPUT_FOLDER}denoised_{num}.wav'
            output_file: str = f"{OUTPUT_FOLDER}{OUTPUT_PREFIX}_denoised_{num}.wav"
            
            # Apply multi-channel Wiener filtering
            multichannel_wiener_filter(input_file, output_file)
            
            # Log successful processing
            print(f"Processed {os.path.basename(input_file)}")
            
        except Exception as e:
            # Log error and continue to next file
            print(f"Error processing {num}: {e}")
            continue

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Print processing banner
print("====================================")
print("🔄 Running MWF Denoising")
print("====================================")
print()

# Enforce single-core execution for reproducibility
enforce_single_core()

# Process all files in the specified range
process_folder()
