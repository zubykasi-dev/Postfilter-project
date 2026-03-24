import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import stft, istft
import soundfile as sf
import pesq_test as pt
from scipy.ndimage import gaussian_filter
import scipy.io.wavfile as wav
import stoi_test as st
import rtf as rtf
import librosa
import time
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pypapi
    from pypapi import events
    PYPAPI_AVAILABLE = True
except Exception:
    PYPAPI_AVAILABLE = False


def rmm_denoise(
    data: Union[np.ndarray, List[float]],
    rate: int,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Apply RMM-based denoising on audio signal.

    Args:
        data: Input audio samples (numpy array or list).
        rate: Sample rate in Hz.
        nperseg: STFT window length (samples). If None defaults to 0.128*rate.
        noverlap: STFT overlap length (samples). If None defaults to 0.85*nperseg.

    Returns:
        Denoised audio signal as 1D numpy array, or None on failure.
    """
    # Ensure data is a numpy array and not empty
    data = np.array(data)
    
    if data.size == 0:
        print("Error: Input data is empty")
        return None, None
    
    # If the input is 2D and has more than one channel, take the first channel
    if data.ndim > 1 and data.shape[1] > 1:
        data = data[:, 0]
    
    # Ensure data is 1D
    data = data.flatten()
    n_samples = len(data)

    # Set default STFT parameters if not provided
    if nperseg is None:
        nperseg = int(rate * 0.128)
    if noverlap is None:
        noverlap = int(nperseg * 0.85)
        
    # Start timing
    t_start = time.time()

    try:
        # Compute the STFT of the input signal
        f, t, Zxx = stft(data, fs=rate, nperseg=nperseg, noverlap=noverlap)
        
        # Compute the magnitude spectrogram
        magnitude_spectrogram = np.abs(Zxx)
        
        # 2. Implement true RMM mask calculation
        def relative_max(x, size=(1, 7)):
            from scipy.ndimage import maximum_filter
            return maximum_filter(x, size=size)
        
        local_max = relative_max(magnitude_spectrogram)
        
        # Calculate RMM mask with a small constant to avoid division by zero
        epsilon = 1e-10
        rmm_mask = magnitude_spectrogram / (local_max + epsilon)
        
        # 3. Apply adaptive smoothing to the mask
        def adaptive_smoothing(mask, magnitude):
            energy = np.sum(magnitude**2, axis=0)
            max_energy = np.max(energy)
            smoothing_factor = 1 - (energy / max_energy)
            for i in range(mask.shape[1]):
                mask[:, i] = gaussian_filter(mask[:, i], sigma=smoothing_factor[i])
            return mask
        
        rmm_mask = adaptive_smoothing(rmm_mask, magnitude_spectrogram)
        
        # 4. Implement an adaptive noise floor
        def estimate_noise_floor(spectrogram):
            # Estimate noise floor as the 10th percentile of magnitude
            return np.percentile(spectrogram, 5, axis=1)[:, np.newaxis]
        
        noise_floor = estimate_noise_floor(magnitude_spectrogram)
        rmm_mask = np.maximum(rmm_mask, noise_floor / (local_max + epsilon))
        
        # Apply the RMM mask to the magnitude spectrogram
        denoised_magnitude_spectrogram = magnitude_spectrogram * rmm_mask
        
        # 5. Implement phase reconstruction
        def phase_reconstruction(mag, angles, iterations=10):
            for _ in range(iterations):
                _, x = istft(mag * np.exp(1j * angles), fs=rate, nperseg=nperseg, noverlap=noverlap)
                _, _, Zxx_new = stft(x, fs=rate, nperseg=nperseg, noverlap=noverlap)
                angles = np.angle(Zxx_new)
            return angles
        
        reconstructed_phase = phase_reconstruction(denoised_magnitude_spectrogram, np.angle(Zxx))
        
        # Reconstruct the denoised signal using the inverse STFT
        _, denoised_signal = istft(denoised_magnitude_spectrogram * np.exp(1j * reconstructed_phase), 
                                   fs=rate, nperseg=nperseg, noverlap=noverlap)

        return denoised_signal

    except Exception as e:
        print(f"Error in STFT/iSTFT processing: {str(e)}")
        return None
input_folder = f"oo/ReTM2.5_0.7_4.9/"
output_folder = "RTM/"
output_prefix = 'RTM'
