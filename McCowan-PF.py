import numpy as np
import librosa
import soundfile as sf
import pesq_test as pt
import os
import scipy.io.wavfile as wav
from scipy.signal import wiener
from pystoi import stoi
from typing import Dict, Tuple, Optional, Union
def align_signals(clean: np.ndarray, denoised: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align clean and denoised signals to identical length.

    Crops or pads the clean signal to match denoised length. Return values are
    (aligned_clean, aligned_denoised) with equal length.
    """
    min_len = min(len(clean), len(denoised))
    max_len = max(len(clean), len(denoised))
    
    if len(clean) > len(denoised):
        clean = clean[:len(denoised)]
    elif len(clean) < len(denoised):
        pad = np.zeros(len(denoised) - len(clean))
        clean = np.concatenate([clean, pad])
    
    return clean, denoised[:len(clean)]

def calculate_stoi_multichannel(ref: Union[str, np.ndarray], deg: Union[str, np.ndarray], sr: int, extended: bool=False) -> float:
    """Compute average STOI/ESTOI for multichannel signals.

    Accepts file paths or numpy arrays for reference and degraded signals.
    Aligns signals on time and channels before computing voice intelligibility metric.
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

def compute_adaptive_min_gain(X: np.ndarray, min_gain_floor: float=0.75, max_gain_floor: float=0.98, eps: float=1e-12) -> np.ndarray:
    """Compute frequency-dependent adaptive minimum gain for postfilter."""
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
def blend_with_original(y_filtered: np.ndarray, input_file: str, blend: float=0.12) -> np.ndarray:
    """Blend filtered output with original signal to preserve natural envelope.

    Args:
        y_filtered: Post-filtered audio signal
        input_file: Reference original audio file path
        blend: Blend ratio applied to original signal (0.0..1.0)
    """
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

def compute_filter_diagnostics(G: np.ndarray, X: np.ndarray, eps: float=1e-12) -> Dict[str, float]:
    """Compute simple diagnostics for filter gain and SNR behavior.

    Args:
        G: Gain matrix applied to STFT bins (F, T)
        X: STFT complex matrix for channels (C, F, T)
        eps: Small value for numerical stability

    Returns:
        Dictionary of diagnostic floats including means, medians, and ratio metrics.
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




def mccowan_postfilter(
    input_file: str,
    output_file: str,
    n_fft: int = 512,
    hop: int = 96,
    alpha: float = 0.90,
    coherence_floor: float = 0.15,
    min_gain_floor: float = 0.55,
    blend_orig: float = 0.18,
    midband_preserve: float = 1.0,
    # selective protection params
    selective_protect: bool = False,
    protect_threshold_db: float = 2.0,
    protect_blend: float = 0.6,
    protect_persist_pct: float = 0.25,
    protect_snr_thresh: float = 0.5,
    env_corr_limit_db: float = 2.0,
    env_corr_blend: float = 0.3,
    verbose: bool = False
) -> Tuple[np.ndarray, int]:
    """Apply McCowan postfilter to a denoised audio file.

    This function performs coherence-based multi-microphone post-filtering in the
    STFT domain, optional selective protection and envelope correction, and then
    reconstructs time-domain output using ISTFT.

    Args:
        input_file: Path to input audio waveform (mono or multi-channel).
        output_file: Path for filtered output.
        n_fft: FFT size.
        hop: STFT hop length.
        alpha: Temporal smoothing factor for gain updates.
        coherence_floor: Coherence lower bound to avoid unstable ratios.
        min_gain_floor: Minimum gain after filtering (per-frequency floor).
        blend_orig: Blend fraction to combine output with original input.
        midband_preserve: Midband preservation factor (1.0=no preserve, 0=full preserve).
        selective_protect: Enable per-frequency selective protection mode.
        protect_threshold_db: Threshold for selective protection in dB.
        protect_blend: Blend weight for protected bins.
        protect_persist_pct: Minimum fraction of frames required for protection.
        protect_snr_thresh: SNR proxy threshold for protection eligibility.
        env_corr_limit_db: Envelope correction limit in dB.
        env_corr_blend: Envelope correction blending factor.
        verbose: Enable console diagnostics.

    Returns:
        y: Filtered time-domain waveform (mono mix of channels).
        sr: Sample rate of output signal.
    """
    data, sr = sf.read(input_file)
    if data.ndim == 1:
        data = data[:, None]
    C = data.shape[1]
    n_samples = data.shape[0]

    # Start timing and optional GFLOP counter
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

    return y, sr



if __name__ == '__main__':
    input_folder = f"oo/Retm2.5_office_0.7/4.9/"
    output_folder = "mcc/"
