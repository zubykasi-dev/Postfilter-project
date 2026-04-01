import numpy as np
import librosa
import soundfile as sf
import pesq_test as pt
import os
import scipy.io.wavfile as wav
from scipy.signal import wiener
from pystoi import stoi
import soundfile as sf
import time
try:
    import pypapi
    from pypapi import events
    PYPAPI_AVAILABLE = True
except ImportError:
    PYPAPI_AVAILABLE = False


# ---------------------------
# GFLOP/GMAC measurement helpers
# ---------------------------
def start_gflop_counter():
    """Start PAPI FP operations counter if available."""
    if PYPAPI_AVAILABLE:
        try:
            pypapi.start_counters([events.PAPI_FP_OPS])
            return True
        except Exception:
            return False
    return False


def stop_gflop_counter():
    """Stop PAPI counter and return FLOPs count."""
    if PYPAPI_AVAILABLE:
        try:
            counters = pypapi.stop_counters()
            if counters and len(counters) > 0:
                return int(counters[0])
        except Exception:
            pass
    return 0


def estimate_zelinski_flops(n_channels, n_samples, n_fft, hop_length):
    """
    Estimate FLOPs for Zelinski postfilter:
    - STFT: n_channels * frames * n_fft * log(n_fft) 
    - Zelinski gain computation: C * (C-1)/2 * F * T multiply-adds
    - ISTFT: frames * n_fft * log(n_fft)
    """
    frames = (n_samples - n_fft) // hop_length + 1
    freq_bins = n_fft // 2 + 1
    
    # STFT: ~5 ops per FFT butterfly
    fft_ops = 5.0 * n_fft * np.log2(n_fft) if n_fft > 0 else 0
    stft_flops = n_channels * frames * fft_ops
    
    # Zelinski gain: for each freq bin and frame, compute cross-correlations
    # C channels -> C*(C-1)/2 pairs, plus magnitude computation
    zelinski_flops = n_channels * (n_channels - 1) / 2 * freq_bins * frames * 4.0
    
    # ISTFT
    istft_flops = frames * fft_ops
    
    total = stft_flops + zelinski_flops + istft_flops
    return max(int(total), 0)


def estimate_zelinski_gmacs(n_channels, n_samples, n_fft, hop_length):
    """
    Estimate MACs for Zelinski postfilter:
    - Similar structure but counting multiply-accumulates specifically
    """
    frames = (n_samples - n_fft) // hop_length + 1
    freq_bins = n_fft // 2 + 1
    
    # FFT: (n_fft/2) * log2(n_fft) MACs per FFT
    fft_macs = (n_fft / 2.0) * np.log2(n_fft) if n_fft > 0 else 0
    stft_macs = n_channels * frames * fft_macs
    
    # Zelinski gain computation MACs
    # For each pair of channels: compute cross-correlation
    zelinski_macs = n_channels * (n_channels - 1) / 2 * freq_bins * frames * 2.0
    
    # ISTFT
    istft_macs = frames * fft_macs
    
    total = stft_macs + zelinski_macs + istft_macs
    return max(int(total), 0)


def compute_gmacs_per_second(macs, elapsed_time_sec):
    """Convert MACs and elapsed time to GMACs/s."""
    if elapsed_time_sec <= 0:
        return 0.0
    gmacs = (macs / max(elapsed_time_sec, 1e-9)) / 1e9
    return gmacs



    """
    Align clean and denoised signals to have identical length.
    Crops or pads the clean signal to match denoised length.
    """
    min_len = min(len(clean), len(denoised))
    max_len = max(len(clean), len(denoised))
    
    if len(clean) > len(denoised):
        clean = clean[:len(denoised)]
    elif len(clean) < len(denoised):
        pad = np.zeros(len(denoised) - len(clean))
        clean = np.concatenate([clean, pad])
    
    return clean, denoised[:len(clean)]

def calculate_stoi_multichannel(ref, deg, sr, extended=False):
    """
    Compute average STOI or ESTOI for multichannel signals.
    Ensures ref and deg are aligned and same length.
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
            print(f"⚠️ STOI error on channel {ch}: {e}")

    return float(np.mean(scores)) if scores else np.nan


def gcc_phat_delay(sig, ref, fs, max_tau=None):
    """
    Estimate time delay (seconds) between sig and ref using GCC-PHAT (whole-signal).
    Returns a float delay (seconds). Positive means sig lags ref (sig occurs later).
    """
    # make sure 1D numpy arrays
    sig = np.asarray(sig, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    n = sig.shape[0] + ref.shape[0]

    # FFT length: next power of two for efficiency
    N = 1 << (int(np.ceil(np.log2(n))))
    SIG = np.fft.rfft(sig, n=N)
    REF = np.fft.rfft(ref, n=N)

    R = SIG * np.conj(REF)
    denom = np.abs(R)
    # avoid divide-by-zero
    denom[denom == 0] = 1e-16
    R_phat = R / denom

    cc = np.fft.irfft(R_phat, n=N)  # cross-correlation
    # shift so that zero lag is center
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

    # Parabolic interpolation for sub-sample refinement
    # use neighbors: y[-1], y[0], y[+1]
    if 1 <= (peak + (mid - max_shift_samples)) < (len(cc) - 1):
        i = peak_index
        y_m1, y0, y_p1 = cc[i-1], cc[i], cc[i+1]
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

def fractional_shift_in_time(x, delay_seconds, fs):
    """
    Apply fractional delay to 1D signal x (numpy) by multiplying its FFT
    by a phase ramp: H(k) = exp(-j*omega*delay). This supports fractional delays.
    Positive delay_seconds means shift right (signal delayed).
    Returns a real-valued signal of same length as x (IFFT of windowed FFT).
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    # FFT
    X = np.fft.fft(x)
    # frequency bins
    k = np.arange(N)
    # angular frequency for bin k: 2*pi*k/N (rad/sample)
    # phase shift = exp(-j * 2*pi * k * delay_in_samples / N)
    delay_samples = delay_seconds * fs
    phase = np.exp(-1j * 2.0 * np.pi * k * delay_samples / N)
    X_shifted = X * phase
    x_shifted = np.fft.ifft(X_shifted)
    # return real part (should be nearly real)
    return np.real(x_shifted)
def align_multichannel_gccphat(data, fs, ref_ch=0, max_delay=0.02, verbose=False):
    """
    Align multichannel time-domain array `data` (samples x channels) to reference channel `ref_ch`.
    - max_delay : maximum expected delay in seconds (search window).
    Returns aligned copy of data (same shape).
    """
    # ensure 2D (samples, channels)
    arr = np.asarray(data)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    if arr.shape[1] == 1:
        return arr.copy()

    ref = arr[:, ref_ch]
    out = np.zeros_like(arr)
    out[:, ref_ch] = ref.copy()

    delays = []
    for ch in range(arr.shape[1]):
        if ch == ref_ch:
            delays.append(0.0)
            continue
        sig = arr[:, ch]
        # estimate delay of sig relative to ref
        delay = gcc_delay = None
        try:
            delay = gcc_phat_delay(sig, ref, fs, max_tau=max_delay)
        except Exception:
            # fallback: zero delay
            delay = 0.0
        delays.append(delay)
        # apply negative of delay to align sig to ref:
        # if delay > 0, sig occurs later than ref, so shift left by delay (advance)
        # we apply fractional_shift_in_time(x, -delay, fs) to compensate
        shifted = fractional_shift_in_time(sig, -delay, fs)
        out[:, ch] = shifted

    if verbose:
        print(f"GCC-PHAT delays (sec) per channel relative to ch{ref_ch}: {delays}")
    return out

def zelinski_postfilter(input_file, output_file,
                               n_fft=512, hop=96,
                               eps=1e-12, floor_db=-10,
                               alpha=0.6, verbose=False,
                               midband_low=800, midband_high=3200):

    # Start timing and GFLOP counter
    t_start = time.time()
    gflop_start = start_gflop_counter()
    gflop_t0 = time.time()

    # === Load ===
    data, sr = sf.read(input_file)
    if data.ndim == 1:
        data = data[:, None]
    C = data.shape[1]
    n_samples = len(data)
    data_aligned = align_multichannel_gccphat(data, sr, ref_ch=0, max_delay=0.12, verbose=False)

    # === STFT (C, F, T) ===
    X = np.stack([
        librosa.stft(data_aligned[:, ch], n_fft=n_fft, hop_length=hop, win_length=n_fft)
        for ch in range(C)
    ], axis=0)
    F, T = X.shape[1], X.shape[2]
    freqs = np.linspace(0, sr/2, F)

    # === Zelinski Gain ===
    G = np.zeros((F, T), dtype=np.float32)
    for f in range(F):
        num = np.zeros(T, dtype=np.float64)
        denom = np.zeros(T, dtype=np.float64)

        for i in range(C):
            Xi = X[i, f, :]
            denom += np.abs(Xi)**2
            for j in range(i+1, C):
                Xj = X[j, f, :]
                num += np.real(Xi * np.conj(Xj))

        # sanitize before division
        num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
        denom = np.nan_to_num(denom, nan=0.0, posinf=0.0, neginf=0.0)

        # avoid divide-by-zero and underflow
        denom = np.maximum(denom, eps * 10)

        ratio = num / ((C - 1) * denom)
        ratio = np.clip(ratio, 0.0, 1.0)
        ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

        # temporal smoothing
        for t in range(1, T):
            ratio[t] = alpha * ratio[t-1] + (1 - alpha) * ratio[t]

        G[f, :] = np.maximum(ratio, 10 ** (floor_db / 20.0))

    # === Hybrid mask ===
    hybrid_mask = np.ones(F)
    hybrid_mask[(freqs >= midband_low) & (freqs <= midband_high)] = 0.0

    # smooth transitions
    ramp = 100
    idx_low = np.logical_and(freqs >= midband_low - ramp, freqs < midband_low)
    idx_high = np.logical_and(freqs > midband_high, freqs <= midband_high + ramp)
    if np.any(idx_low):
        hybrid_mask[idx_low] = np.linspace(1, 0, np.sum(idx_low))
    if np.any(idx_high):
        hybrid_mask[idx_high] = np.linspace(0, 1, np.sum(idx_high))

    # === Apply hybrid gain ===
    X_mag = np.mean(np.abs(X), axis=0)
    X_phase = np.angle(X[0])
    G_total = (hybrid_mask[:, None] * G) + (1 - hybrid_mask[:, None]) * 1.0
    Y = G_total * X_mag * np.exp(1j * X_phase)

    # === ISTFT ===
    y = librosa.istft(Y, hop_length=hop, win_length=n_fft)
    y = np.clip(y, -1.0, 1.0)
    sf.write(output_file, y, sr)

    # Stop timing and measure FLOPs
    gflop_t1 = time.time()
    measured_flops = stop_gflop_counter()
    gflop_elapsed = gflop_t1 - gflop_t0
    
    # Estimate FLOPs/MACs if not measured
    if measured_flops == 0:
        measured_flops = estimate_zelinski_flops(C, n_samples, n_fft, hop)
    measured_macs = estimate_zelinski_gmacs(C, n_samples, n_fft, hop)
    
    t_end = time.time()
    processing_time = t_end - t_start
    audio_duration = n_samples / float(sr)
    rtf = processing_time / max(audio_duration, 1e-9)
    gflops = (measured_flops / max(gflop_elapsed, 1e-9)) / 1e9 if gflop_elapsed > 0 else 0.0
    gmacs_per_sec = compute_gmacs_per_second(measured_macs, gflop_elapsed)

    if verbose:
        print(f"[Hybrid Zelinski] midband preserved {midband_low}-{midband_high} Hz, "
              f"gain floor={floor_db} dB, alpha={alpha}, C={C}")
    
    # Return metrics
    return {
        'processing_time': processing_time,
        'audio_duration': audio_duration,
        'rtf': rtf,
        'flops': measured_flops,
        'gflops': gflops,
        'macs': measured_macs,
        'gmacs_per_sec': gmacs_per_sec
    }
def compute_mel_distance(ref, deg, sr, n_mels=128, fmax=8000):
    """
    Compute mel spectrogram distance between reference and degraded signals.
    Returns the mean squared error between mel spectrograms in dB scale.
    Lower values indicate better quality.
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



input_folder = f"oo/ReTM2.5_office_0.7/"
output_folder = "zel/"
i=0

# Initialize metrics tracking lists
rtf_list = []
gflop_list = []
gmacs_per_sec_list = []
flops_total = 0
for distance in [4.9]:
    folder = f"{input_folder}{distance}/"
    for i in range(60):
        num = 100 + i  # Start from 1441 and increment
        
        input_file = f'{folder}denoised_{num}.wav'
        output_prefix = 'zelinski'
        output_file= f"zel/{output_prefix}_denoised_noisy_speech{num}.wav"
        # Read the audio file
        signal, sample_rate = sf.read(input_file)
        
        # Process and collect metrics
        metrics = zelinski_postfilter(input_file, output_file)
        
        # Accumulate metrics
        #rtf_list.append(metrics['rtf'])
        #gflop_list.append(metrics['gflops'])
        #gmacs_per_sec_list.append(metrics['gmacs_per_sec'])
        #flops_total += metrics['flops']
        
        # Print per-file metrics
        #print(f"File {num:03d} | RTF {metrics['rtf']:.4f} | GFLOPs {metrics['gflops']:.2f} | GMACs/s {metrics['gmacs_per_sec']:.4f}")


    num = 99
    print(f"Distance {distance:.1f}-----------")
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
    ref_file = f'E:/CND/clean/clean{num}.wav'
    #ref_file = f'E:/malay-clean/clean{num}.wav'
    deg_file = f'{output_folder}{output_prefix}_denoised_noisy_speech{num}.wav'
    sr = 16000  # Sample rate
    try:
        # Pesq and PESQi
        ref, sr_c = pt.read_audio(ref_file)
        deg, sr_d = pt.read_audio(deg_file)
        
        if sr_c != sr_d:
            raise ValueError(f"Sample rate mismatch: clean={sr_c}, denoised={sr_d}")

        # Align lengths
        clean, denoised = align_signals(ref, deg)
        pesq_score = pt.pesq(sr_c, clean, denoised, 'nb')
        print(pesq_score)
    except Exception as e:
        print(f"Error processing")
        continue
print ("---------------------------------")
print ("---------*-STOI Score-*----------")
print ("---------------------------------")
num = 99
for i in range(60):
    num += 1
    
    ref_file = f'E:/CND/clean/clean{num}.wav'
    #ref_file = f'E:/malay-clean/clean{num}.wav'
    deg_file = f'{output_folder}{output_prefix}_denoised_noisy_speech{num}.wav'
    
    sr = 16000  # Sample rate

    # Check if files exist before processing
    if not os.path.exists(ref_file) or not os.path.exists(deg_file):
        print(f"Skipping files: not found.")
        continue

    try:
        # Stoi ESTOI / PESQ test
        avg_stoi_score = calculate_stoi_multichannel(ref_file, deg_file, sr)
        print(avg_stoi_score)

    except Exception as e:
        print(f"Error processing")
        continue
print ("---------------------------------")
print ("---------*-ESTOI Score-*----------")
print ("---------------------------------")
num = 99
for i in range(60):
    num += 1
    
    ref_file = f'E:/CND/clean/clean{num}.wav'
    #ref_file = f'E:/malay-clean/clean{num}.wav'
    deg_file = f'{output_folder}{output_prefix}_denoised_noisy_speech{num}.wav'
    
    sr = 16000  # Sample rate

    # Check if files exist before processing
    if not os.path.exists(ref_file) or not os.path.exists(deg_file):
        print(f"Skipping files: not found.")
        continue

    try:
        # Stoi ESTOI / PESQ test
        avg_stoi_score = calculate_stoi_multichannel(ref_file, deg_file, sr,extended=True)
        print(avg_stoi_score)

    except Exception as e:
        print(f"Error processing")
        continue

# Print summary statistics for Zelinski metrics
print("\n" + "="*60)
print("ZELINSKI FILTER - PERFORMANCE METRICS SUMMARY")
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
"""