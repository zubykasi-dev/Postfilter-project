import os
import numpy as np
import librosa
import soundfile as sf
import torch
from scipy import linalg
from scipy.io import wavfile
from scipy.signal import medfilt
from pystoi import stoi
from scipy.signal import stft, istft
from pesq import pesq as pesq_backend  # ✅ use library PESQ safely
import time
from multiprocessing import Pool, cpu_count, get_context
import stoi_test as st
from typing import Any, Dict, List, Optional, Tuple, Union
try:
    import pypapi
    from pypapi import events
    PYPAPI_AVAILABLE = True
except ImportError:
    PYPAPI_AVAILABLE = False
    print("⚠️  pypapi not available — GFLOP measurement disabled")

N_FFT = 1024
HOP = 128
WIN = 'hann'  # used by librosa.stft / istft
SAMPLE_RATE = 16000
MAX_FILES_TO_PROCESS = 100  # used in folder processing

REF_CHANNELS = [0, 1]
TARGET_CHANNELS = [2, 3, 4, 5]  # 4 targets

try:
    vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    (get_speech_timestamps, _, _, _, _) = vad_utils
except Exception as e:
    print(f"⚠️ Could not load silero-vad (will fallback): {e}")
    vad_model = None
    get_speech_timestamps = None


def compute_pesq(ref: Union[np.ndarray, List[float]],
                  deg: Union[np.ndarray, List[float]],
                  sr: int) -> float:
    """Compute PESQ score, returning float or np.nan if unavailable."""
    mode = 'wb' if sr > 48000 else 'nb'

    ref = np.asarray(ref, dtype=np.float32)
    deg = np.asarray(deg, dtype=np.float32)

    # Flatten multi-dimensional arrays if needed
    if ref.ndim > 1:
        ref = ref.mean(axis=-1)
    if deg.ndim > 1:
        deg = deg.mean(axis=-1)

    min_len = min(len(ref), len(deg))
    if min_len < int(0.25 * sr):
        print(f"⚠️ PESQ skipped: signal too short ({min_len} samples)")
        return np.nan

    ref, deg = ref[:min_len], deg[:min_len]
    ref = np.clip(ref, -1, 1)
    deg = np.clip(deg, -1, 1)

    try:
        score = pesq_backend(sr, ref, deg, mode)
        return float(score)
    except Exception as e:
        print(f"⚠️ PESQ computation failed: {e}")
        return np.nan

# Helpers: STFT / ISTFT wrappers
# ---------------------------
def stft_multi_from_array(audio_arr: Union[np.ndarray, List[float]],
                          n_fft: int = N_FFT,
                          hop_length: int = HOP) -> np.ndarray:
    """Compute multi-channel STFT and return (C, F, T) complex array."""
    audio = np.asarray(audio_arr)
    if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        # assume shape (C, samples)
        pass
    elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
        # if user passed (samples, C) -> transpose
        audio = audio.T
    elif audio.ndim == 1:
        audio = np.expand_dims(audio, 0)

    C = audio.shape[0]
    X_list = []
    for ch in range(C):
        X = librosa.stft(audio[ch], n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=WIN, center=True)
        X_list.append(X)
    return np.stack(X_list, axis=0)  # (C, F, T)

def istft_from_stft_matrix(S: np.ndarray,
                            hop_length: int = HOP,
                            win_length: int = N_FFT) -> np.ndarray:
    """Inverse STFT on a single channel matrix (F, T)."""
    # S shape (F, T) (librosa uses shape (n_fft/2+1, frames))
    return librosa.istft(S, hop_length=hop_length, win_length=win_length, window=WIN, center=True)

# ---------------------------
# VAD wrapper (returns non-speech segments in samples)
# ---------------------------
def perform_vad(audio_path: str,
                vad_model: Optional[Any],
                vad_threshold: float = 4.0,
                min_silence_duration: float = 0.05) -> Tuple[List[Dict[str, int]], int]:
    """Run VAD and return list of non-speech segments and sample rate."""
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    audio_int16 = (audio * 32767).astype(np.int16)
    if vad_model is None or get_speech_timestamps is None:
        # fallback to first 0.5s as noise segment
        segs = [{'start': 0, 'end': int(0.5 * sr)}]
        return segs, sr
    try:
        speech_ts = get_speech_timestamps(audio_int16, vad_model, sampling_rate=sr, threshold=vad_threshold)
        non_speech = []
        if speech_ts:
            # use initial non-speech before first speech segment
            if speech_ts[0]['start'] > 0:
                non_speech.append({'start': 0, 'end': speech_ts[0]['start']})
        else:
            non_speech.append({'start': 0, 'end': len(audio_int16)})
        # merge very short gaps
        min_samples = int(min_silence_duration * sr)
        merged = []
        for seg in non_speech:
            if not merged or seg['start'] - merged[-1]['end'] >= min_samples:
                merged.append(seg)
            else:
                merged[-1]['end'] = seg['end']
        return merged, sr
    except Exception:
        # fallback
        return [{'start': 0, 'end': int(0.5 * sr)}], sr


# ---------------------------
def estimate_retm_freq(A_in: Union[np.ndarray, List[float]],
                       B_in: Union[np.ndarray, List[float]],
                       sr: int = SAMPLE_RATE,
                       window_dur: float = 2.0,
                       overlap: float = 0.75,
                       reg_scale: float = 1e-1,
                       min_frames_per_window: int = 4,
                       debug: bool = False) -> List[np.ndarray]:
    """Estimate frequency-dependent ReTM gain Wf windows for multi-channel input."""
    A_in = np.asarray(A_in)
    B_in = np.asarray(B_in)
    if A_in.ndim != 2 or B_in.ndim != 2:
        raise ValueError("A_in / B_in must be 2D arrays (n_chan, samples)")

    n_ref, n_samp_ref = A_in.shape
    n_err, n_samp_err = B_in.shape
    n_samp = min(n_samp_ref, n_samp_err)
    A = A_in[:, :n_samp]
    B = B_in[:, :n_samp]

    # --- Compute STFTs ---
    A_stft = stft_multi_from_array(A, n_fft=N_FFT, hop_length=HOP)  # (n_ref, F, T)
    B_stft = stft_multi_from_array(B, n_fft=N_FFT, hop_length=HOP)  # (n_err, F, T)

    n_ref2, F, T = A_stft.shape
    n_err2 = B_stft.shape[0]

    frames_per_sec = sr / float(HOP)
    window_frames = max(min_frames_per_window, int(window_dur * frames_per_sec))
    hop_frames = max(1, int(window_frames * (1 - overlap)))

    Wf_windows = []
    for start in range(0, T - window_frames + 1, hop_frames):
        end = start + window_frames
        A_win = A_stft[:, :, start:end]
        B_win = B_stft[:, :, start:end]

        if A_win.shape[-1] < min_frames_per_window:
            continue

        Wf_list = []
        for f in range(F):
            A_f = A_win[:, f, :]
            B_f = B_win[:, f, :]

            if A_f.shape[-1] < 2:
                continue

            try:
                PAA = (A_f @ A_f.conj().T) / float(max(1, A_f.shape[-1]))
                PBA = (B_f @ A_f.conj().T) / float(max(1, A_f.shape[-1]))
                PAA_reg = PAA + reg_scale * np.eye(PAA.shape[0], dtype=np.complex64)

                Wf = np.linalg.solve(PAA_reg, PBA.T).T.astype(np.complex64)
                if np.isfinite(Wf).all() and Wf.shape == (n_err, n_ref):
                    Wf_list.append(Wf)
            except Exception as e:
                if debug:
                    print(f"⚠️ Skipped frequency bin {f}: {e}")
                continue

        if not Wf_list:
            continue

        # ✅ Average across valid frequency bins for this window
        try:
            Wf_stack = np.stack(Wf_list, axis=0)  # (valid_F, n_err, n_ref)
        except Exception as e:
            if debug:
                print(f"❌ Skipping invalid Wf_list: inconsistent shapes ({[w.shape for w in Wf_list]})")
            continue

        Wf_avg = np.mean(Wf_stack, axis=0)
        Wf_full = np.tile(Wf_avg[np.newaxis, :, :], (F, 1, 1))  # (F, n_err, n_ref)

        Wf_windows.append(Wf_full)


    # --- Fallback broadband mode ---
    if not Wf_windows:
        if debug:
            print("⚠️ No valid sliding windows found — using broadband fallback.")
        try:
            PAA_acc = np.zeros((n_ref, n_ref), dtype=np.complex64)
            PBA_acc = np.zeros((n_err, n_ref), dtype=np.complex64)
            for f in range(F):
                A_f = A_stft[:, f, :]
                B_f = B_stft[:, f, :]
                if A_f.shape[-1] < 1:
                    continue
                PAA_acc += (A_f @ A_f.conj().T) / float(max(1, A_f.shape[-1]))
                PBA_acc += (B_f @ A_f.conj().T) / float(max(1, A_f.shape[-1]))
            PAA_acc /= float(max(1, F))
            PBA_acc /= float(max(1, F))
            PAA_reg = PAA_acc + reg_scale * np.eye(PAA_acc.shape[0], dtype=np.complex64)
            Wf_broad = (np.linalg.solve(PAA_reg, PBA_acc.T).T).astype(np.complex64)
            Wf_windows = [(np.tile(Wf_broad[np.newaxis, :, :], (F, 1, 1)), F)]
        except Exception as e:
            if debug:
                print("❌ Broadband fallback failed:", e)
            return []

    if debug:
        print(f"✅ Estimated {len(Wf_windows)} valid Wf windows; shape per Wf: {Wf_windows[0][0].shape}")
    return Wf_windows


# ---------------------------
# Denoising: apply Wf (averaged) to full file STFT
#   Wf: either (F, n_err, n_ref) or list of such - we'll accept both; if list, we average
# ---------------------------
def denoise_signal_freq_from_Wf(audio_arr: Union[np.ndarray, List[float]],
                                Wf: Union[np.ndarray, List[np.ndarray]],
                                ref_channels: List[int] = REF_CHANNELS,
                                target_channels: List[int] = TARGET_CHANNELS,
                                n_fft: int = N_FFT,
                                hop_length: int = HOP) -> np.ndarray:
    """Apply ReTM frequency-domain filter Wf to multi-channel audio and return denoised targets."""
    audio = np.asarray(audio_arr)
    # convert to (C, samples)
    if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        audio = audio.T
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    # If Wf is a list, average them
    if isinstance(Wf, list):
        if len(Wf) == 0:
            raise ValueError("Empty Wf list")
        Wf = np.mean(np.stack(Wf, axis=0), axis=0)  # (F, n_err, n_ref)

    F = Wf.shape[0]
    n_err = Wf.shape[1]
    n_ref = Wf.shape[2]

    # compute full STFTs
    X_stft = stft_multi_from_array(audio, n_fft=n_fft, hop_length=hop_length)  # (C, F, T)
    C, F2, T = X_stft.shape
    if F2 != F:
        # If STFT frequency bins mismatch (rare), adjust by trimming or repeating
        minF = min(F, F2)
        Wf = Wf[:minF]
        F = minF

    # reference & target slices
    try:
        ref = X_stft[ref_channels, :F, :].astype(np.complex64)    # (n_ref, F, T)
        tar = X_stft[target_channels, :F, :].astype(np.complex64) # (n_err, F, T)
    except Exception as e:
        raise RuntimeError(f"Channel index error while slicing STFTs: {e}")

    # Build Y_hat
    Y_hat = np.zeros_like(tar, dtype=np.complex64)  # (n_err, F, T)
    for f in range(F):
        Wf_f = Wf[f]  # (n_err, n_ref)
        Ref_f = ref[:, f, :]  # (n_ref, T)
        est_noise = Wf_f @ Ref_f    # (n_err, T)
        Y_hat[:, f, :] = tar[:, f, :] - est_noise

    # ISTFT per target channel
    denoised_list = []
    for ch in range(n_err):
        S_ch = Y_hat[ch]   # shape (F, T)
        y = istft_from_stft_matrix(S_ch, hop_length=hop_length, win_length=n_fft)
        denoised_list.append(y)

    # Stack into (n_err, samples)
    maxlen = max(len(x) for x in denoised_list)
    denoised = np.zeros((n_err, maxlen), dtype=np.float32)
    for i, s in enumerate(denoised_list):
        denoised[i, :len(s)] = s

    # Normalize by peak to avoid clipping
    peak = np.max(np.abs(denoised)) + 1e-12
    denoised = denoised / peak

    return denoised
