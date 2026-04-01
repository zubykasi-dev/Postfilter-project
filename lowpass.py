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
    # Limit global native thread-pools to 1 worker where possible
    threadpool_limits(limits=1)
except Exception:
    pass

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import stft, istft
from tqdm import tqdm
import pesq_test as pt
import stoi_test as st
import warnings
import soundfile as sf
import librosa
import numpy as np
import warnings
import time
try:
    import pypapi
    from pypapi import events
    PYPAPI_AVAILABLE = True
except Exception:
    PYPAPI_AVAILABLE = False
import torch
import csv

# Force single-threaded PyTorch for reproducible single-core runs
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

def safe_audio_load(path, sr=16000, mono=False):
    """
    Robust loader: returns (samples, channels)
    Ensures 2D output even for mono signals.
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
def temporal_lowpass_filter_multichannel(signal, fs, nperseg=512, noverlap=456, filter_length=3):
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

clean_dir = "E:/CND/clean"
log_file = os.path.join(output_folder, "TLF_evaluation_log.txt")


print("====================================")
print("🔄 Running Temporal Lowpass Filtering")
print("====================================")

# NOTE: The actual filtering loop is executed after the metric helper
# functions are defined (below). This placeholder avoids calling
# estimator helpers before they exist in the file.


# ------------------------------
# Helper function for mono conversion + alignment
# ------------------------------
def load_and_align(ref_path, deg_path, sr_target=16000):
    ref, sr_r = librosa.load(ref_path, sr=None, mono=True)
    deg, sr_d = librosa.load(deg_path, sr=None, mono=True)

    if sr_r != sr_target:
        ref = librosa.resample(ref, orig_sr=sr_r, target_sr=sr_target)
    if sr_d != sr_target:
        deg = librosa.resample(deg, orig_sr=sr_d, target_sr=sr_target)

    min_len = min(len(ref), len(deg))
    ref, deg = ref[:min_len], deg[:min_len]

    # Normalize to [-1,1]
    ref = ref / (np.max(np.abs(ref)) + 1e-3)
    deg = deg / (np.max(np.abs(deg)) + 1e-3)

    return ref, deg, sr_target
def match_length(clean, denoised):
    """
    Ensure both signals have the same length for fair metric evaluation.
    If denoised is shorter, trim clean to match.
    If denoised is longer, trim denoised to match.
    """
    min_len = min(len(clean), len(denoised))
    return clean[:min_len], denoised[:min_len]


# ------------------------------
# 2️⃣ PESQ Evaluation
# ------------------------------
print("\n---------------------------------")
print("---------*-PESQ Evaluation-*----------")
print("---------------------------------")

pesq_scores = []
log_file = os.path.join(output_folder, f"TLF_results_log.txt")
with open(log_file, "w", encoding="utf-8") as log:

    log.write("=============== PESQ Evaluation ===============\n")

    for num in tqdm(range(100, 160), desc="PESQ"):
        ref_file = f"{clean_dir}/clean{num}.wav"
        deg_file = f"{output_folder}{output_prefix}_denoised_{num}.wav"

        if not os.path.exists(ref_file) or not os.path.exists(deg_file):
            log.write(f"⚠️ Missing file(s) for {num}\n")
            continue

        try:
            ref, deg, sr = load_and_align(ref_file, deg_file, 16000)
            ref, deg = match_length(ref, deg)
            pesq_val = pt.pesq(sr, ref, deg, 'nb')             log.write(f"{pesq_val:.3f}\n")
            #print(f"✅ clean{num}.wav: PESQ={pesq_val:.3f}")
            pesq_scores.append(pesq_val)
        except Exception as e:
            log.write(f"❌ Error for clean{num}.wav: {e}\n")
            print(f"❌ Error for clean{num}.wav: {e}")
            continue

    if pesq_scores:
        avg_pesq = np.mean(pesq_scores)
        log.write(f"\nAverage PESQ: {avg_pesq:.3f}\n")
        print(f"\n📊 Average PESQ: {avg_pesq:.3f}")
    else:
        log.write("⚠️ No valid PESQ scores computed.\n")

# ------------------------------
# 3️⃣ STOI Evaluation
# ------------------------------
print("---------------------------------")
print("---------*-STOI Evaluation-*----------")
print("---------------------------------")

stoi_scores = []
with open(log_file, "a", encoding="utf-8") as log:
    log.write("\n=============== STOI Evaluation ===============\n")
    for num in tqdm(range(100, 160), desc="STOI"):
        #num = 100 + i
        ref_file = f"{clean_dir}/clean{num}.wav"
        deg_file = f"{output_folder}{output_prefix}_denoised_{num}.wav"
        #print(ref_file)
        #print(deg_file)
        try:
            ref = safe_audio_load(ref_file, sr=16000, mono=False)
            deg = safe_audio_load(deg_file, sr=16000, mono=False)

            # Ensure matching lengths
            min_len = min(ref.shape[0], deg.shape[0])
            ref, deg = ref[:min_len], deg[:min_len]
            if ref.shape[1] != deg.shape[1]:
                min_ch = min(ref.shape[1], deg.shape[1])
                ref, deg = ref[:, :min_ch], deg[:, :min_ch]

            avg_stoi = st.calculate_stoi_multichannel(ref, deg, 16000, False)
            stoi_scores.append(avg_stoi)
            #print(f"{num}: STOI = {avg_stoi:.4f}")
            log.write(f"{avg_stoi:.4f}\n")
        except Exception as e:
            log.write(f"WARNING: STOI error for {num}: {e}\n")
            continue

    if stoi_scores:
        avg_stoi = np.mean(stoi_scores)
        print(f"\nAverage STOI: {avg_stoi:.4f}")
        log.write(f"\nAverage STOI: {avg_stoi:.4f}\n")



# ------------------------------
# 4️⃣ ESTOI Evaluation
# ------------------------------
print("\n---------------------------------")
print("---------*-ESTOI Evaluation-*----------")
print("---------------------------------")

estoi_scores = []

with open(log_file, "a", encoding="utf-8") as log:
    log.write("\n=============== ESTOI Evaluation ===============\n")

    for num in tqdm(range(100, 160), desc="STOI"):
        #num = 100 + i
        ref_file = f"{clean_dir}/clean{num}.wav"
        deg_file = f"{output_folder}{output_prefix}_denoised_{num}.wav"
       
        try:
            ref = safe_audio_load(ref_file, sr=16000, mono=False)
            deg = safe_audio_load(deg_file, sr=16000, mono=False)

            # Ensure matching lengths
            min_len = min(ref.shape[0], deg.shape[0])
            ref, deg = ref[:min_len], deg[:min_len]
            if ref.shape[1] != deg.shape[1]:
                min_ch = min(ref.shape[1], deg.shape[1])
                ref, deg = ref[:, :min_ch], deg[:, :min_ch]

            avg_estoi = st.calculate_stoi_multichannel(ref, deg, 16000, True)
            log.write(f"{avg_estoi:.3f}\n")
            #print(f"✅ clean{num}.wav: ESTOI={avg_estoi:.3f}")
            estoi_scores.append(avg_estoi)
        except Exception as e:
            log.write(f"❌ ESTOI error for {num}: {e}\n")
            print(f"❌ ESTOI error for {num}: {e}")
            continue

    if estoi_scores:
        avg_estoi = np.mean(estoi_scores)
        log.write(f"\nAverage ESTOI: {avg_estoi:.3f}\n")
        print(f"\n📊 Average ESTOI: {avg_estoi:.3f}")
    else:
        log.write("⚠️ No valid ESTOI scores computed.\n")

print(f"\n📁 Log file saved to: {log_file}")
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

def estimate_lowpass_flops(signal_len, sr, nperseg=512, noverlap=456, filter_length=3, n_channels=1):
    if nperseg <= 0:
        return 0
    frames = max(1, (signal_len - noverlap) // (nperseg - noverlap))
    freq_bins = nperseg // 2 + 1
    fft_ops = 5.0 * nperseg * np.log2(nperseg) if nperseg > 0 else 0
    stft_flops = n_channels * frames * fft_ops
    conv_flops = n_channels * freq_bins * frames * filter_length
    istft_flops = n_channels * frames * fft_ops
    total = stft_flops + conv_flops + istft_flops
    return max(int(total), 0)

def estimate_lowpass_gmacs(signal_len, sr, nperseg=512, noverlap=456, filter_length=3, n_channels=1):
    if nperseg <= 0:
        return 0
    frames = max(1, (signal_len - noverlap) // (nperseg - noverlap))
    freq_bins = nperseg // 2 + 1
    fft_macs = (nperseg / 2.0) * np.log2(nperseg) if nperseg > 0 else 0
    stft_macs = n_channels * frames * fft_macs
    conv_macs = n_channels * freq_bins * frames * (filter_length - 1)
    istft_macs = n_channels * frames * fft_macs
    total = stft_macs + conv_macs + istft_macs
    return max(int(total), 0)

def compute_gmacs_per_second(macs, elapsed_time_sec):
    if elapsed_time_sec <= 0:
        return 0.0
    return (macs / max(elapsed_time_sec, 1e-9)) / 1e9


# ------------------------------
# Instrumented filtering + CSV export
# ------------------------------
metrics_list = []

print("\nStarting instrumented filtering and metric collection...")
for num in tqdm(range(100, 160), desc="Filtering"):
    input_file = f"{input_folder}denoised_{num}.wav"
    output_file = f"{output_folder}{output_prefix}_denoised_{num}.wav"

    if not os.path.exists(input_file):
        # Keep same behavior as before
        print(f"⚠️ Missing: {input_file}")
        continue

    try:
        sig, sr = sf.read(input_file)

        # Determine channel count for estimators
        n_channels = sig.shape[1] if sig.ndim > 1 else 1

        # Start timing and optional hardware counter
        start = time.time()
        counter_started = start_gflop_counter()

        # Run the filter
        filtered = temporal_lowpass_filter_multichannel(sig, sr)

        # Stop counters / estimate
        if counter_started:
            flops = stop_gflop_counter()
        else:
            flops = estimate_lowpass_flops(len(sig), sr, n_channels=n_channels)

        end = time.time()
        elapsed = end - start
        audio_dur = float(sig.shape[0]) / float(sr) if sr > 0 else 0.0
        rtf = elapsed / max(audio_dur, 1e-9)
        gflops = (flops / max(elapsed, 1e-9)) / 1e9
        macs = estimate_lowpass_gmacs(len(sig), sr, n_channels=n_channels)
        gmacs_per_sec = compute_gmacs_per_second(macs, elapsed)

        # Write output and record metrics
        sf.write(output_file, filtered, sr)

        metrics = {
            'file': os.path.basename(input_file),
            'processing_time': elapsed,
            'audio_duration': audio_dur,
            'rtf': rtf,
            'flops': int(flops),
            'gflops': float(gflops),
            'macs': int(macs),
            'gmacs_per_sec': float(gmacs_per_sec)
        }
        metrics_list.append(metrics)

        print(f"{metrics['file']}: RTF={metrics['rtf']:.3f}, GFLOPs/s={metrics['gflops']:.3f}, GMACs/s={metrics['gmacs_per_sec']:.3f}")

    except Exception as e:
        print(f"❌ Error processing {input_file}: {e}")
        continue


# Save CSV
csv_path = os.path.join(output_folder, f"{output_prefix}_metrics.csv")
with open(csv_path, 'w', newline='', encoding='utf-8') as csvf:
    fieldnames = ['file', 'processing_time', 'audio_duration', 'rtf', 'flops', 'gflops', 'macs', 'gmacs_per_sec']
    writer = csv.DictWriter(csvf, fieldnames=fieldnames)
    writer.writeheader()
    for m in metrics_list:
        writer.writerow(m)

print(f"CSV saved: {csv_path}")
# Compute and print averages for the collected metrics
if metrics_list:
    import math
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
    print('\nNo metrics collected; CSV may be empty or no files processed.')

# Mark averages task completed
try:
    from functions import manage_todo_list as _mt
except Exception:
    _mt = None
