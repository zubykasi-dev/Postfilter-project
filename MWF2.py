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

import soundfile as sf
import pesq_test as pt
import stoi_test as st
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import wiener
import numpy as np
from scipy.signal import stft, istft
import librosa
import rtf as rtf
import time
import csv
import torch

# Force single-threaded PyTorch for reproducible single-core runs
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass
def enforce_single_core():
    """Re-apply single-core settings at runtime and report configuration."""
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
def improved_wiener(data, fs, window_size=2048, overlap=0.75, noise_reduction_factor=9.0):
    # Perform Short-time Fourier Transform (STFT)
    f, t, Zxx = stft(data, fs=fs, nperseg=window_size, noverlap=int(window_size * overlap))
    
    # Estimate noise spectrum (assuming first few frames are noise)
    noise_estimate = np.mean(np.abs(Zxx[:, :5]), axis=1)
    
    # Compute signal power
    sig_power = np.abs(Zxx) ** 2
    
    # Compute noise power
    noise_power = noise_estimate[:, np.newaxis] ** 2
    
    # Compute a priori SNR
    xi = np.maximum(sig_power / noise_power - 0, 6)
    
    # Compute Wiener filter gain
    G = xi / (xi + noise_reduction_factor)
    
    # Apply Wiener filter
    Zxx_denoised = Zxx * G
    
    # Inverse STFT
    _, denoised_data = istft(Zxx_denoised, fs=fs, nperseg=window_size, noverlap=int(window_size * overlap))
    
    return denoised_data
def multichannel_wiener_filter(input_file, output_file):
    # Read the audio file using soundfile
    data, sample_rate = sf.read(input_file)
    # Check if the audio file is single-channel or multi-channel
    if len(data.shape) == 1:
        # Single-channel audio
        data = data.reshape(-1, 1)
    
    # Apply Wiener filter to each channel
    denoised_data = np.zeros_like(data)
    
    for i in range(data.shape[1]):
        channel_data = data[:, i]
        
        # Check for zero variance
        if np.var(channel_data) < 1e-10:
            denoised_data[:, i] = channel_data
            continue
        
        # Apply improved Wiener filter
        denoised_channel = improved_wiener(channel_data, sample_rate)
        
        # Ensure denoised_channel has the same length as the original channel
        if len(denoised_channel) > len(channel_data):
            denoised_channel = denoised_channel[:len(channel_data)]
        elif len(denoised_channel) < len(channel_data):
            denoised_channel = np.pad(denoised_channel, (0, len(channel_data) - len(denoised_channel)), 'constant')
        
        denoised_data[:, i] = denoised_channel
    
    # Normalize the output to the range [-1, 1]
    max_val = np.max(np.abs(denoised_data))
    if max_val > 0:
        denoised_data = denoised_data / max_val
    
    # Save the denoised audio file
    sf.write(output_file, denoised_data, sample_rate)

def custom_wiener(data, noise_level=0.05, epsilon=1e-10):
    # Estimate local mean and variance using a sliding window
    window_size = 5
    local_mean = np.convolve(data, np.ones(window_size) / window_size, mode='same')
    local_var = np.convolve((data - local_mean) ** 2, np.ones(window_size) / window_size, mode='same')
    
    # Prevent zero variance by adding epsilon
    local_var = np.maximum(local_var, epsilon)
    
    # Apply Wiener filter formula
    return local_mean + (local_var - noise_level) / local_var * (data - local_mean)


# ===== Measurement helpers =====
try:
    import pypapi
    from pypapi import events
    PYPAPI_AVAILABLE = True
except Exception:
    PYPAPI_AVAILABLE = False


def start_gflop_counter():
    if PYPAPI_AVAILABLE:
        try:
            pypapi.start_counters([events.PAPI_FP_OPS])
            return True
        except Exception:
            return False
    return False


def stop_gflop_counter():
    if PYPAPI_AVAILABLE:
        try:
            counters = pypapi.stop_counters()
            if counters and len(counters) > 0:
                return int(counters[0])
        except Exception:
            pass
    return 0


def estimate_mwf_flops(signal_len, sr, window_size=2048, overlap=0.75, n_channels=1):
    """Estimate FLOPs for MWF (Wiener filter via STFT)."""
    if window_size <= 0:
        return 0
    overlap_len = int(window_size * overlap)
    frames = max(1, (signal_len - overlap_len) // (window_size - overlap_len))
    freq_bins = window_size // 2 + 1
    
    # STFT: FFT cost per frame ~ 5 * n_fft * log2(n_fft)
    fft_ops = 5.0 * window_size * np.log2(window_size) if window_size > 0 else 0
    stft_flops = n_channels * frames * fft_ops
    
    # Wiener filtering in frequency domain: element-wise operations
    wiener_flops = n_channels * frames * freq_bins * 5  # rough estimate: SNR calc, gain, multiply
    
    # ISTFT: FFT cost per frame
    istft_flops = n_channels * frames * fft_ops
    
    total = stft_flops + wiener_flops + istft_flops
    return max(int(total), 0)


def estimate_mwf_gmacs(signal_len, sr, window_size=2048, overlap=0.75, n_channels=1):
    """Estimate MACs for MWF."""
    if window_size <= 0:
        return 0
    overlap_len = int(window_size * overlap)
    frames = max(1, (signal_len - overlap_len) // (window_size - overlap_len))
    freq_bins = window_size // 2 + 1
    
    # FFT MACs per frame ~ (n_fft/2) * log2(n_fft)
    fft_macs = (window_size / 2.0) * np.log2(window_size) if window_size > 0 else 0
    stft_macs = n_channels * frames * fft_macs
    
    # Wiener: element-wise multiply and add per bin
    wiener_macs = n_channels * frames * freq_bins * 2  # multiply + add estimate
    
    # ISTFT FFT
    istft_macs = n_channels * frames * fft_macs
    
    total = stft_macs + wiener_macs + istft_macs
    return max(int(total), 0)


def compute_gmacs_per_second(macs, elapsed_time_sec):
    """Convert MACs and time to GMACs/s."""
    if elapsed_time_sec <= 0:
        return 0.0
    return (macs / max(elapsed_time_sec, 1e-9)) / 1e9
input_folder = f"oo/ReTM2.5_office_0.7/4.9/"
output_folder = "mwf/"
output_prefix = 'mwf'
i=0


def process_folder(start_num=100, end_num=160):
    """Process folder with instrumented denoising for RTF/GFLOP/GMAC measurement.
    Processes files numbered from `start_num` (inclusive) to `end_num` (exclusive).
    """
    metrics_list = []

    for num in range(start_num, end_num):
        try:
            input_file = f'{input_folder}denoised_{num}.wav'
            output_file = f"{output_folder}{output_prefix}_denoised_{num}.wav"
            
            # Load audio
            audio, sample_rate = librosa.load(input_file, sr=None, mono=True)
            input_duration = len(audio) / sample_rate
            
            # Determine channel count for estimators (audio is mono from librosa.load)
            n_channels = 1
            
            # Start timing and optional hardware counter
            start = time.time()
            counter_started = start_gflop_counter()
            
            # Run the denoising (multichannel_wiener_filter)
            multichannel_wiener_filter(input_file, output_file)
            
            # Stop counters / estimate
            if counter_started:
                flops = stop_gflop_counter()
            else:
                flops = estimate_mwf_flops(len(audio), sample_rate, n_channels=n_channels)
            
            end = time.time()
            elapsed = end - start
            rtf = elapsed / max(input_duration, 1e-9)
            gflops = (flops / max(elapsed, 1e-9)) / 1e9
            macs = estimate_mwf_gmacs(len(audio), sample_rate, n_channels=n_channels)
            gmacs_per_sec = compute_gmacs_per_second(macs, elapsed)
            
            metrics = {
                'file': os.path.basename(input_file),
                'processing_time': elapsed,
                'audio_duration': input_duration,
                'rtf': rtf,
                'flops': int(flops),
                'gflops': float(gflops),
                'macs': int(macs),
                'gmacs_per_sec': float(gmacs_per_sec)
            }
            metrics_list.append(metrics)
            
            print(f"{metrics['file']}: RTF={metrics['rtf']:.4f}, GFLOPs/s={metrics['gflops']:.3f}, GMACs/s={metrics['gmacs_per_sec']:.3f}")
            
        except Exception as e:
            print(f"Error processing {num}: {e}")
            continue
    
    return metrics_list

# Run instrumented filtering with metrics
os.makedirs(output_folder, exist_ok=True)
print("====================================")
print("🔄 Running MWF Denoising")
print("====================================\n")

# Re-apply single-core enforcement here to be safe
enforce_single_core()

metrics_list = process_folder()

# Save CSV
csv_path = os.path.join(output_folder, f"{output_prefix}_metrics.csv")
with open(csv_path, 'w', newline='', encoding='utf-8') as csvf:
    fieldnames = ['file', 'processing_time', 'audio_duration', 'rtf', 'flops', 'gflops', 'macs', 'gmacs_per_sec']
    writer = csv.DictWriter(csvf, fieldnames=fieldnames)
    writer.writeheader()
    for m in metrics_list:
        writer.writerow(m)

print(f"\nCSV saved: {csv_path}")

# Compute and print averages for the collected metrics
if metrics_list:
    n = len(metrics_list)
    avg_processing_time = float(np.mean([m['processing_time'] for m in metrics_list]))
    avg_audio_duration = float(np.mean([m['audio_duration'] for m in metrics_list]))
    avg_rtf = float(np.mean([m['rtf'] for m in metrics_list]))
    avg_flops = float(np.mean([m['flops'] for m in metrics_list]))
    avg_gflops = float(np.mean([m['gflops'] for m in metrics_list]))
    avg_macs = float(np.mean([m['macs'] for m in metrics_list]))
    avg_gmacs = float(np.mean([m['gmacs_per_sec'] for m in metrics_list]))

    print('\n=== Average Metrics ===')
    print(f"Files processed : {n}")
    print(f"Avg processing_time : {avg_processing_time:.4f} s")
    print(f"Avg audio_duration  : {avg_audio_duration:.4f} s")
    print(f"Avg RTF             : {avg_rtf:.4f}")
    print(f"Avg FLOPs           : {avg_flops:.0f}")
    print(f"Avg GFLOPs/s        : {avg_gflops:.6f}")
    print(f"Avg MACs            : {avg_macs:.0f}")
    print(f"Avg GMACs/s         : {avg_gmacs:.6f}")
else:
    print('\nNo metrics collected; no files processed.')
