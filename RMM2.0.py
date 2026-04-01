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


def compute_mel_distance(
    ref: Union[np.ndarray, List[float]],
    deg: Union[np.ndarray, List[float]],
    sr: int,
    n_mels: int = 128,
    fmax: int = 8000,
) -> float:
    """Compute mel spectrogram distance between reference and degraded signals.

    Arguments:
        ref: Reference audio signal as 1D array-like.
        deg: Degraded audio signal as 1D array-like.
        sr: Sample rate in Hz.
        n_mels: Number of Mel bands.
        fmax: Maximum frequency for Mel filterbank.

    Returns:
        Mean squared error between Mel spectrograms in dB scale.
        Lower values indicate better perceptual similarity.
    """
    ref = np.asarray(ref, dtype=np.float32)
    deg = np.asarray(deg, dtype=np.float32)

    # Flatten multi-dimensional arrays if needed
    if ref.ndim > 1:
        ref = ref.mean(axis=-1)
    if deg.ndim > 1:
        deg = deg.mean(axis=-1)

    min_len = min(len(ref), len(deg))
    ref, deg = ref[:min_len], deg[:min_len]

    try:
        # Compute mel spectrograms
        mel_ref = librosa.feature.melspectrogram(y=ref, sr=sr, n_mels=n_mels, fmax=fmax)
        mel_deg = librosa.feature.melspectrogram(y=deg, sr=sr, n_mels=n_mels, fmax=fmax)
        
        # Convert to log scale
        mel_ref_db = librosa.power_to_db(mel_ref, ref=np.max)
        mel_deg_db = librosa.power_to_db(mel_deg, ref=np.max)
        
        # Compute mean squared error
        distance = np.mean((mel_ref_db - mel_deg_db)**2)
        return float(distance)
    except Exception as e:
        print(f"⚠️ Mel distance computation failed: {e}")
        return np.nan


# ---------------------------
# GFLOP/GMAC measurement helpers
# ---------------------------
def start_gflop_counter() -> bool:
    """Start PAPI FP operations counter if available.

    Returns:
        True if PAPI counter started, False otherwise.
    """
    if PYPAPI_AVAILABLE:
        try:
            pypapi.start_counters([events.PAPI_FP_OPS])
            return True
        except Exception:
            return False
    return False


def stop_gflop_counter() -> int:
    """Stop PAPI counter and return FLOPs count.

    Returns:
        FLOPs count (rounded to int), or 0 when unavailable.
    """
    if PYPAPI_AVAILABLE:
        try:
            counters = pypapi.stop_counters()
            if counters and len(counters) > 0:
                return int(counters[0])
        except Exception:
            pass
    return 0


def estimate_rmm_flops(n_samples: int, nperseg: int, noverlap: int) -> int:
    """Estimate FLOPs for RMM denoising: STFT + mask ops + ISTFT.

    Args:
        n_samples: Number of audio samples.
        nperseg: STFT window size.
        noverlap: STFT overlap size.

    Returns:
        Estimated total FLOPs as int.
    """
    if nperseg <= 0:
        return 0
    frames = max(1, (n_samples - noverlap) // (nperseg - noverlap))
    freq_bins = nperseg // 2 + 1

    # STFT: ~5 ops per FFT butterfly
    fft_ops = 5.0 * nperseg * np.log2(nperseg) if nperseg > 0 else 0
    stft_flops = frames * fft_ops

    # Mask calculation: element-wise ops on (freq_bins, frames)
    mask_flops = freq_bins * frames * 6.0  # relative max, division, smoothing

    # Phase reconstruction (10 iterations of STFT/ISTFT)
    phase_flops = 10 * (frames * fft_ops + frames * fft_ops)

    # ISTFT
    istft_flops = frames * fft_ops

    total = stft_flops + mask_flops + phase_flops + istft_flops
    return max(int(total), 0)


def estimate_rmm_gmacs(n_samples: int, nperseg: int, noverlap: int) -> int:
    """Estimate MACs for RMM denoising.

    Args:
        n_samples: Number of audio samples.
        nperseg: STFT window size.
        noverlap: STFT overlap size.

    Returns:
        Estimated total MACs as int.
    """
    if nperseg <= 0:
        return 0
    frames = max(1, (n_samples - noverlap) // (nperseg - noverlap))
    freq_bins = nperseg // 2 + 1

    # FFT MACs: (n_fft/2) * log2(n_fft) per FFT
    fft_macs = (nperseg / 2.0) * np.log2(nperseg) if nperseg > 0 else 0
    stft_macs = frames * fft_macs

    # Mask MACs
    mask_macs = freq_bins * frames * 2.0

    # Phase reconstruction MACs
    phase_macs = 10 * (frames * fft_macs + frames * fft_macs)

    istft_macs = frames * fft_macs

    total = stft_macs + mask_macs + phase_macs + istft_macs
    return max(int(total), 0)


def compute_gmacs_per_second(macs: int, elapsed_time_sec: float) -> float:
    """Convert MACs and elapsed time to GMACs/s.

    Args:
        macs: Number of multiply-accumulate operations.
        elapsed_time_sec: Elapsed time in seconds.

    Returns:
        GMACs per second.
    """
    if elapsed_time_sec <= 0:
        return 0.0
    return (macs / max(elapsed_time_sec, 1e-9)) / 1e9


def rmm_denoise(
    data: Union[np.ndarray, List[float]],
    rate: int,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]]]:
    """Apply RMM-based denoising on audio signal.

    Args:
        data: Input audio samples (numpy array or list).
        rate: Sample rate in Hz.
        nperseg: STFT window length (samples). If None defaults to 0.128*rate.
        noverlap: STFT overlap length (samples). If None defaults to 0.85*nperseg.

    Returns:
        Tuple of (denoised_signal, metrics_dict) or (None, None) on failure.
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
        
    # Start timing and GFLOP counter
    t_start = time.time()
    gflop_t0 = time.time()
    start_gflop_counter()

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

        # Stop GFLOP counter
        gflop_t1 = time.time()
        measured_flops = stop_gflop_counter()
        gflop_elapsed = gflop_t1 - gflop_t0

        # Estimate FLOPs/MACs if not measured
        if measured_flops == 0:
            measured_flops = estimate_rmm_flops(n_samples, nperseg, noverlap)
        measured_macs = estimate_rmm_gmacs(n_samples, nperseg, noverlap)

        t_end = time.time()
        processing_time = t_end - t_start
        audio_duration = n_samples / float(rate)
        rtf = processing_time / max(audio_duration, 1e-9)
        gflops = (measured_flops / max(gflop_elapsed, 1e-9)) / 1e9 if gflop_elapsed > 0 else 0.0
        gmacs_per_sec = compute_gmacs_per_second(measured_macs, gflop_elapsed)

        metrics = {
            'processing_time': processing_time,
            'audio_duration': audio_duration,
            'rtf': rtf,
            'flops': measured_flops,
            'gflops': gflops,
            'macs': measured_macs,
            'gmacs_per_sec': gmacs_per_sec
        }

        return denoised_signal, metrics

    except Exception as e:
        print(f"Error in STFT/iSTFT processing: {str(e)}")
        return None, None
input_folder = f"oo/ReTM2.5_0.7_4.9/"
output_folder = "RTM/"
output_prefix = 'RTM'

def process_folder(
    input_folder: str,
    start_index: int = 100,
    num_files: int = 60,
) -> None:
    """Process a folder of denoised files through RMM and gather metrics.

    Args:
        input_folder: Path to the input folder containing WAV files.
        start_index: Index of first file to process.
        num_files: Number of sequential files to process.

    Returns:
        None (prints metrics and writes output files).
    """
    rtf_list: List[float] = []
    gflop_list: List[float] = []
    gmacs_per_sec_list: List[float] = []
    flops_total: int = 0

    for i in range(start_index, start_index + num_files):
        num = i  # Start from 1441 and increment
        
        input_file = f'{input_folder}denoised_{num}.wav'
        
        output_file = f"{output_folder}{output_prefix}_denoised_noisy_speech{num}.wav"
        audio, sample_rate = librosa.load(input_file, sr=None, mono=True)
        global input_duration
        input_duration = len(audio) / sample_rate
        try:
        
            signal, sample_rate = sf.read(input_file)
        
            filtered_signal, metrics = rmm_denoise(signal, sample_rate, nperseg=1024, noverlap=128)
            
            if filtered_signal is not None:
                
                sf.write(output_file, filtered_signal, sample_rate)
                
                # Collect metrics
                if metrics is not None:
                    rtf_list.append(metrics['rtf'])
                    gflop_list.append(metrics['gflops'])
                    gmacs_per_sec_list.append(metrics['gmacs_per_sec'])
                    flops_total += metrics['flops']
                    #print(f"File {num:03d} | RTF {metrics['rtf']:.4f} | GFLOPs {metrics['gflops']:.2f} | GMACs/s {metrics['gmacs_per_sec']:.4f}")
            else:
                print(f"Failed to process file: {input_file}")
        
        except Exception as e:
            print(f"Error processing file {input_file}: {str(e)}")
"""
    # Print summary statistics
    print("\n" + "="*60)
    print("RMM DENOISE - PERFORMANCE METRICS SUMMARY")
    print("="*60)
    if rtf_list:
        print(f"Real-Time Factor (RTF):")
        print(f"  Mean RTF:    {np.mean(rtf_list):.6f}")
        print(f"  Min RTF:     {np.min(rtf_list):.6f}")
        print(f"  Max RTF:     {np.max(rtf_list):.6f}")

    if gflop_list:
        print(f"\nGFLOPs/s:")
        print(f"  Mean GFLOPs: {np.mean(gflop_list):.2f}")
        print(f"  Min GFLOPs:  {np.min(gflop_list):.2f}")
        print(f"  Max GFLOPs:  {np.max(gflop_list):.2f}")
        print(f"  Total FLOPs: {flops_total:.2e}")

    if gmacs_per_sec_list:
        print(f"\nGMACs/s:")
        print(f"  Mean GMACs/s: {np.mean(gmacs_per_sec_list):.4f}")
        print(f"  Min GMACs/s:  {np.min(gmacs_per_sec_list):.4f}")
        print(f"  Max GMACs/s:  {np.max(gmacs_per_sec_list):.4f}")

    print("="*60)
   
#rtf_score = rtf.calculate_rtf(process_folder)
#rtf_score=rtf_score / input_duration
#print(f"RTF Score: {rtf_score:.6f}")
"""
for distance in [4.9]:
    input_folder = f"oo/Retm2.5_office_0.7/{distance}/"
    print(f"Processing folder: {input_folder} with RT: 0.4 and Distance: {distance}")
    process_folder(input_folder,100, 60)
    num = 99
    print("---------------------------------")
    print("---------*-Mel Distance-*----------")
    for i in range(60):
        num += 1
        ref_file = f'D:/CND/clean/clean{num}.wav'
        deg_file = f'{output_folder}{output_prefix}_denoised_noisy_speech{num}.wav'
        sr = 16000  # Sample rate
        try:
            ref, rate = pt.read_audio(ref_file)
            deg, _ = pt.read_audio(deg_file)
            mel_dist = compute_mel_distance(ref, deg, sr)
            print(f'{mel_dist:.4f}')
        except Exception as e:
            print(f"Error processing: {e}")
            continue

"""
num = 99
print ("---------------------------------")
print ("---------*-PESQ Score-*----------")
print ("---------------------------------")

for i in range(60):
    num += 1
    ref_file = f'D:/CND/clean/clean{num}.wav'
    #ref_file = f'H:/malay-clean/clean{num}.wav'
    deg_file = f'{output_folder}{output_prefix}_denoised_noisy_speech{num}.wav'
    sr = 16000  # Sample rate
    try:
        # Pesq and PESQi
        ref, rate = pt.read_audio(ref_file)
        deg, _ = pt.read_audio(deg_file)
        pesq_score = pt.pesq(rate, ref, deg, 'nb')
        print(pesq_score)
    except Exception as e:
        print(f"Error processing")
        continue


num = 99
print("---------------------------------")
print("---------*-stoi score-*----------")
for i in range(60):
    num += 1
    ref_file = f'D:/CND/clean/clean{num}.wav'
    #ref_file = f'H:/malay-clean/clean{num}.wav'
    deg_file = f'{output_folder}{output_prefix}_denoised_noisy_speech{num}.wav'
    sr = 16000  # Sample rate
    try:
        avg_stoi_score = st.calculate_stoi_multichannel(ref_file, deg_file, sr, False)
        print(f'{avg_stoi_score}')
    except Exception as e:
        print(f"Error processing: {e}")
        continue
num = 99
print("---------------------------------")
print("---------*-Estoi score-*----------")
for i in range(60):
    num += 1
    ref_file = f'D:/CND/clean/clean{num}.wav'
    #ref_file = f'H:/malay-clean/clean{num}.wav'
    deg_file = f'{output_folder}{output_prefix}_denoised_noisy_speech{num}.wav'
    sr = 16000  # Sample rate
    try:
        avg_stoi_score = st.calculate_stoi_multichannel(ref_file, deg_file, sr, True)
        print(f'{avg_stoi_score}')
    except Exception as e:
        print(f"Error processing: {e}")
        continue


"""