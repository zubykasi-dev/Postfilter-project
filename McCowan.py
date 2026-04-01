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
except Exception:
    PYPAPI_AVAILABLE = False
def align_signals(clean, denoised):
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

def compute_adaptive_min_gain(X, min_gain_floor=0.75, max_gain_floor=0.98, eps=1e-12):
    # X: (C, F, T)
    mag = np.mean(np.abs(X), axis=0)  # (F,T)
    p = np.mean(mag**2, axis=1)       # (F,)
    p10 = np.percentile(mag**2, 10, axis=1)
    snr = np.log10((p + eps) / (p10 + eps))  # unitless
    # normalize snr to 0..1 roughly
    smin, smax = np.percentile(snr, 5), np.percentile(snr, 95)
    snr_norm = np.clip((snr - smin) / max((smax - smin), 1e-6), 0.0, 1.0)
    # map to min_gain: noisy bins -> lower min_gain; high-snr bins -> close to 1
    min_gain = min_gain_floor + (max_gain_floor - min_gain_floor) * snr_norm
    return min_gain  # shape (F,)

# blend fraction
def blend_with_original(y_filtered, input_file, blend=0.12):
    # load original denoised (same file you passed into postfilter)
    orig, sr = sf.read(input_file)
    # ensure same length
    if len(orig) > len(y_filtered):
        orig = orig[:len(y_filtered)]
    elif len(orig) < len(y_filtered):
        orig = np.concatenate([orig, np.zeros(len(y_filtered) - len(orig))])
    y_blend = blend * orig + (1 - blend) * y_filtered
    y_blend = np.clip(y_blend, -1.0, 1.0)
    return y_blend

def compute_filter_diagnostics(G, X, eps=1e-12):
    """
    G: (F, T) gain applied
    X: (C, F, T) stft of channels used by filter (complex)
    Prints and returns diagnostics dict.
    """
    F, T = G.shape
    total_bins = F * T
    zero_bins = np.sum(G < 1e-3)
    near_zero_pct = 100.0 * zero_bins / total_bins

    mean_gain = float(np.mean(G))
    med_gain = float(np.median(G))
    std_gain = float(np.std(G))

    # estimate SNR per freq: mean power / 10th percentile
    mag = np.mean(np.abs(X), axis=0)  # (F, T)
    power = mag**2 + eps
    snr_est = 10 * np.log10(np.mean(power, axis=1) / (np.percentile(power, 10, axis=1) + eps) + eps)
    mean_snr = float(np.mean(snr_est))
    low_snr_frac = float(np.mean(snr_est < 0.0))  # fraction of freq bins with <0 dB

    diag = {
        "mean_gain": mean_gain,
        "median_gain": med_gain,
        "std_gain": std_gain,
        "near_zero_pct": near_zero_pct,
        "mean_snr_db": mean_snr,
        "low_snr_frac": low_snr_frac
    }
    print("=== Filter diagnostics ===")
    for k,v in diag.items():
        print(f"{k}: {v}")
    return diag


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


def estimate_mccowan_flops(n_channels, n_samples, n_fft, hop_length):
    """Estimate FLOPs for McCowan postfilter.

    Heuristic estimate including STFT/ISTFT, pairwise cross-operations
    and some per-bin/frame arithmetic.
    """
    if n_fft <= 0 or hop_length <= 0:
        return 0
    frames = max(1, (n_samples - n_fft) // hop_length + 1)
    freq_bins = n_fft // 2 + 1

    # FFT ops approximation: ~5 * n_fft * log2(n_fft) per FFT
    fft_ops = 5.0 * n_fft * np.log2(n_fft)
    stft_flops = n_channels * frames * fft_ops

    # Pairwise cross-power: C*(C-1)/2 complex multiplies per freq/frame
    pairs = n_channels * (n_channels - 1) / 2.0
    # count ~6 FLOPs per complex multiply/add + magnitude ops
    pairwise_flops = pairs * freq_bins * frames * 6.0

    # Per-bin temporal smoothing and other scalar ops
    additional = freq_bins * frames * 10.0

    # ISTFT
    istft_flops = frames * fft_ops

    total = stft_flops + pairwise_flops + additional + istft_flops
    return max(int(total), 0)


def estimate_mccowan_gmacs(n_channels, n_samples, n_fft, hop_length):
    """Estimate MACs (multiply-accumulates) for McCowan postfilter."""
    if n_fft <= 0 or hop_length <= 0:
        return 0
    frames = max(1, (n_samples - n_fft) // hop_length + 1)
    freq_bins = n_fft // 2 + 1

    # FFT MACs approximation: (n_fft/2) * log2(n_fft) per FFT
    fft_macs = (n_fft / 2.0) * np.log2(n_fft)
    stft_macs = n_channels * frames * fft_macs

    # Pairwise MACs: assume 2 MACs per complex multiply-accumulate
    pairs = n_channels * (n_channels - 1) / 2.0
    pairwise_macs = pairs * freq_bins * frames * 2.0

    istft_macs = frames * fft_macs

    total = stft_macs + pairwise_macs + istft_macs
    return max(int(total), 0)


def compute_gmacs_per_second(macs, elapsed_time_sec):
    """Convert MACs and elapsed time to GMACs/s."""
    if elapsed_time_sec <= 0:
        return 0.0
    gmacs = (macs / max(elapsed_time_sec, 1e-9)) / 1e9
    return gmacs


def mccowan_postfilter(
                        input_file, output_file,
                        n_fft=512, hop=96,
                        alpha=0.90,
                        coherence_floor=0.15,
                        min_gain_floor=0.55,
                        blend_orig=0.18,
                        midband_preserve=1.0,
                        # selective protection params
                        selective_protect=False,
                        protect_threshold_db=2.0,
                        protect_blend=0.6,
                        protect_persist_pct=0.25,
                        protect_snr_thresh=0.5,
                        env_corr_limit_db=2.0,
                        env_corr_blend=0.3,
                        verbose=False
                    ):
    data, sr = sf.read(input_file)
    if data.ndim == 1:
        data = data[:, None]
    C = data.shape[1]
    n_samples = data.shape[0]

    # Start timing and optional GFLOP counter
    t_start = time.time()
    gflop_t0 = time.time()
    start_gflop_counter()

    # STFT
    X = np.stack([librosa.stft(data[:, ch], n_fft=n_fft, hop_length=hop, win_length=n_fft)
                  for ch in range(C)], axis=0)  # (C,F,T)
    F, T = X.shape[1], X.shape[2]

    G = np.zeros((F, T), dtype=np.float32)
    prev_G = np.ones(F, dtype=np.float32)

    for f in range(F):
        Xi = X[:, f, :]
        P = np.abs(Xi)**2
        Px = np.mean(P, axis=0) + 1e-12

        # pairwise cross-power & denom
        num = np.zeros(T, dtype=np.float64)
        denom = np.zeros(T, dtype=np.float64)
        coh_sum = np.zeros(T, dtype=np.float64)
        count = 0
        for i in range(C):
            for j in range(i+1, C):
                Xij = Xi[i] * np.conj(Xi[j])
                num += np.real(Xij)
                denom += np.abs(Xi[i]) * np.abs(Xi[j]) + 1e-12
                coh_sum += np.abs(Xij) / (np.abs(Xi[i]) * np.abs(Xi[j]) + 1e-12)
                count += 1
        denom = np.maximum(denom, 1e-12)

        gamma = np.clip(num / denom, 0.0, 5.0)  # normalized cross-energy ratio
        coh_mean = np.clip(coh_sum / max(count,1), coherence_floor, 1.0)
        spp = np.clip((coh_mean - coherence_floor) / (1.0 - coherence_floor), 0.0, 1.0)

        # basic gamma->gain mapping but with a more contrast-preserving nonlinearity
        # map gamma -> [0,1] via 1-exp(-gamma) to increase dynamic range for larger gamma
        G_post = (1.0 - np.exp(-gamma)) * spp + (1.0 - spp) * min_gain_floor

        # temporal smoothing
        G[f,:] = alpha * prev_G[f] + (1 - alpha) * G_post
        prev_G[f] = np.mean(G[f,:])

    # adaptive freq-wise min_gain
    min_gain_vector = compute_adaptive_min_gain(X, min_gain_floor=min_gain_floor, max_gain_floor=0.98)
    for f in range(F):
        G[f,:] = np.maximum(G[f,:], min_gain_vector[f])

    # optionally preserve midband by blending G towards unity in mid frequencies
    if midband_preserve < 1.0:
        FREQ_LOW, FREQ_HIGH = 500, 3500
        freqs = np.linspace(0, sr/2, F)
        idx_mid = np.where((freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH))[0]
        for f in idx_mid:
            G[f,:] = midband_preserve * G[f,:] + (1.0 - midband_preserve) * 1.0

    # selective per-frequency protection and mild envelope correction
    if selective_protect:
        eps = 1e-12
        # mean magnitude across channels (F, T)
        orig_mag = np.mean(np.abs(X), axis=0)
        # filtered magnitude estimate after applying G (approx)
        filtered_mag = orig_mag * G

        # per-frequency long-term means
        orig_mean = np.mean(orig_mag, axis=1) + eps  # (F,)
        filt_mean = np.mean(filtered_mag, axis=1) + eps  # (F,)

        # per-frame dB deviations |filtered - orig|
        per_frame_db = 20.0 * np.log10((filtered_mag + eps) / (orig_mag + eps))  # (F,T)
        # fraction of frames exceeding threshold per freq
        frac_exceed = np.mean(np.abs(per_frame_db) > protect_threshold_db, axis=1)

        # overall mean deviation per freq (post - pre) in dB
        delta_db = 20.0 * np.log10(filt_mean / orig_mean)

        # get an snr-like proxy from min_gain_vector (higher -> likely higher SNR)
        # normalize between its nominal floor and 0.98
        mgf = np.copy(min_gain_vector)
        mgf_floor = min_gain_floor
        mgf_max = 0.98
        snr_proxy = np.clip((mgf - mgf_floor) / max((mgf_max - mgf_floor), 1e-6), 0.0, 1.0)

        protect_bins = (np.abs(delta_db) > protect_threshold_db) & (frac_exceed >= protect_persist_pct) & (snr_proxy >= protect_snr_thresh)

        # apply blending toward unity for protected bins
        for f in np.where(protect_bins)[0]:
            G[f, :] = protect_blend * G[f, :] + (1.0 - protect_blend) * 1.0

        # mild envelope correction: compute limited correction (orig vs filtered mean)
        gain_corr_db = 20.0 * np.log10(orig_mean / (filt_mean + eps))
        gain_corr_db = np.clip(gain_corr_db, -env_corr_limit_db, env_corr_limit_db)
        gain_corr = 10 ** (gain_corr_db / 20.0)

        # apply envelope correction only where snr_proxy is sufficiently high
        corr_bins = snr_proxy >= protect_snr_thresh
        for f in np.where(corr_bins)[0]:
            # blend multiplicative correction into G
            G[f, :] = (1.0 - env_corr_blend) * G[f, :] + env_corr_blend * (G[f, :] * gain_corr[f])

    # diagnostics
    if verbose:
        compute_filter_diagnostics(G, X)

    # apply gains per-channel and preserve channel phase (reconstruct per-channel)
    # Yc: (C, F, T)
    Yc = np.empty_like(X, dtype=np.complex64)
    for c in range(C):
        X_c = X[c]
        Yc[c] = G * np.abs(X_c) * np.exp(1j * np.angle(X_c))

    # preserve short-term envelopes in midband per-channel
    X_orig_mag = np.mean(np.abs(X), axis=0)
    FREQ_LOW, FREQ_HIGH = 600, 3200
    freqs = np.linspace(0, sr/2, F)
    idx_mid = np.where((freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH))[0]
    if len(idx_mid) > 0:
        window = 5
        for f in idx_mid:
            orig_rms = np.sqrt(np.convolve(X_orig_mag[f]**2, np.ones(window)/window, mode='same') + 1e-12)
            for c in range(C):
                filt_rms_c = np.sqrt(np.convolve((np.abs(Yc[c, f])**2), np.ones(window)/window, mode='same') + 1e-12)
                ratio = np.clip((orig_rms / (filt_rms_c + 1e-12)), 0.75, 1.25)
                Yc[c, f] *= ratio

    # ISTFT per-channel and time-domain combine (average)
    y_channels = []
    for c in range(C):
        y_c = librosa.istft(Yc[c], hop_length=hop, win_length=n_fft)
        y_channels.append(y_c)
    # pad/truncate to same length
    min_len = min(len(yc) for yc in y_channels)
    y_stack = np.stack([yc[:min_len] for yc in y_channels], axis=1)
    y = np.mean(y_stack, axis=1)
    y = np.clip(y, -1.0, 1.0)

    # optional blend with original denoised to recover envelope
    if blend_orig and blend_orig > 0:
        orig, _ = sf.read(input_file)
        if len(orig) > len(y):
            orig = orig[:len(y)]
        elif len(orig) < len(y):
            orig = np.concatenate([orig, np.zeros(len(y)-len(orig))])
        y = blend_orig * orig + (1 - blend_orig) * y
        y = np.clip(y, -1.0, 1.0)

    sf.write(output_file, y, sr)
    if verbose:
        print(f"[McCowan-final] wrote {output_file}, blend={blend_orig}, alpha={alpha}, min_gain_floor={min_gain_floor}")
    # Stop GFLOP counter
    gflop_t1 = time.time()
    measured_flops = stop_gflop_counter()
    gflop_elapsed = gflop_t1 - gflop_t0

    # Estimate FLOPs/MACs if not measured
    if measured_flops == 0:
        measured_flops = estimate_mccowan_flops(C, n_samples, n_fft, hop)
    measured_macs = estimate_mccowan_gmacs(C, n_samples, n_fft, hop)

    t_end = time.time()
    processing_time = t_end - t_start
    audio_duration = n_samples / float(sr)
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

    return y, sr, metrics



if __name__ == '__main__':
    input_folder = f"oo/Retm2.5_office_0.7/4.9/"
    output_folder = "mcc/"
    i=0
    # Initialize metrics tracking
    rtf_list = []
    gflop_list = []
    gmacs_per_sec_list = []
    flops_total = 0

    for i in range(60):
        num = 100 + i  # Start from 100 and increment
        input_file = f'{input_folder}denoised_{num}.wav'
        output_prefix = 'mccowen'
        output_file= f"mcc/{output_prefix}_denoised_noisy_speech{num}.wav"
        # Read the audio file (will also be read inside function)
        try:
            signal, sample_rate = sf.read(input_file)
        except Exception:
            print(f"Skipping missing input: {input_file}")
            continue
        # Default processing (can be overridden by calling function directly)
        try:
            out = mccowan_postfilter(input_file, output_file, blend_orig=0, alpha=0.5, min_gain_floor=0.35)
            # function returns (y, sr, metrics)
            if isinstance(out, tuple) and len(out) == 3:
                y_out, sr_out, metrics = out
            else:
                # backward compatible: if only (y, sr) returned
                y_out, sr_out = out
                metrics = None
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue

        if metrics is not None:
            rtf_list.append(metrics['rtf'])
            gflop_list.append(metrics['gflops'])
            gmacs_per_sec_list.append(metrics['gmacs_per_sec'])
            flops_total += metrics['flops']
            print(f"File {num:03d} | RTF {metrics['rtf']:.4f} | GFLOPs {metrics['gflops']:.2f} | GMACs/s {metrics['gmacs_per_sec']:.4f}")
        else:
            print(f"Processed: {input_file}")

    num = 99
    print ("---------------------------------")
    print ("---------*-PESQ Score-*----------")
    print ("---------------------------------")
    for i in range(60):
        num += 1
        ref_file = f'E:/CND/clean/clean{num}.wav'
        deg_file = f'{output_folder}{output_prefix}_denoised_noisy_speech{num}.wav'
        sr = 16000  # Sample rate
        try:
            ref, sr_c = pt.read_audio(ref_file)
            deg, sr_d = pt.read_audio(deg_file)
            if sr_c != sr_d:
                raise ValueError(f"Sample rate mismatch: clean={sr_c}, denoised={sr_d}")
            clean, denoised = align_signals(ref, deg)
            pesq_score = pt.pesq(sr_c, clean, denoised, 'nb')
            print(pesq_score)
        except Exception as e:
            print(f"Error processing")
            continue
    # Print summary statistics for McCowan metrics
    print("\n" + "="*60)
    print("MCCOWAN FILTER - PERFORMANCE METRICS SUMMARY")
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
    print ("---------------------------------")
    print ("---------*-STOI Score-*----------")
    print ("---------------------------------")
    num = 99
    for i in range(60):
        num += 1
        ref_file = f'E:/CND/clean/clean{num}.wav'
        deg_file = f'{output_folder}{output_prefix}_denoised_noisy_speech{num}.wav'
        sr = 16000  # Sample rate
        if not os.path.exists(ref_file) or not os.path.exists(deg_file):
            print(f"Skipping files: not found.")
            continue
        try:
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
        deg_file = f'{output_folder}{output_prefix}_denoised_noisy_speech{num}.wav'
        sr = 16000  # Sample rate
        if not os.path.exists(ref_file) or not os.path.exists(deg_file):
            print(f"Skipping files: not found.")
            continue
        try:
            avg_stoi_score = calculate_stoi_multichannel(ref_file, deg_file, sr,extended=True)
            print(avg_stoi_score)
        except Exception as e:
            print(f"Error processing")
            continue