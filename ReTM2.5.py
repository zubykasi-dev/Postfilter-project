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


def log_metric(metric_name: str, value: float, avg_value: Optional[float] = None,
               log_dir: str = "logs", rt: Optional[float] = None,
               distance: Optional[float] = None, type: str = "output") -> None:
    """Append one scalar metric to a log file for post-processing."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{rt}_ReTM_{metric_name.lower()}_{distance}_{type}_log.txt")
    num=0
    with open(log_file, "a", encoding="utf-8") as f:
        #num=num+1
       # print(num)
        f.write(f"{value:.4f}\n")
        if avg_value is not None:
            f.write(f"    → Current average {metric_name.upper()}: {avg_value:.4f}\n")

def compute_pesq(ref: Union[np.ndarray, List[float]],
                  deg: Union[np.ndarray, List[float]],
                  sr: int) -> float:
    """Compute PESQ score, returning float or np.nan if unavailable."""
    mode = 'nb'

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


def compute_mel_distance(ref: Union[np.ndarray, List[float]],
                          deg: Union[np.ndarray, List[float]],
                          sr: int,
                          n_mels: int = 128,
                          fmax: int = 8000) -> float:
    """Compute mel-spectrogram MSE distance (dB domain)."""
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


def process_folder_and_eval(
    input_folder: str,
    clean_folder: str,
    output_folder: str,
    start_index: int = 0,
    num_files: int = 60,
    window_avg: float = 2.0,
    reg_scale: float = 1e-1,
    sr: int = 16000,
    debug: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: Optional[int] = 1,
    rt: float = 0.4,
    distance: float = 1.1
) -> None:
    os.makedirs(output_folder, exist_ok=True)
    noisy_scores, denoised_scores = [], []
    stoi_noisy_scores, stoi_denoised_scores = [], []
    estoi_noisy_scores, estoi_denoised_scores = [], []
    mel_distance_noisy_scores, mel_distance_denoised_scores = [], []
    processing_times, audio_durations = [], []
    flops_list, gflops_list = [], []
    macs_list, gmacs_per_sec_list = [], []

    noisy_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".wav")])

    # ✅ Ensure the correct range is processed
    end_index = min(start_index + num_files, len(noisy_files))
    selected_files = noisy_files[start_index:end_index]

    print(f"Processing files {start_index} to {end_index - 1} ({len(selected_files)} total)\n")
    num = start_index

    # Build task list
    tasks = []
    for i in range(num_files):
        file_index = start_index + i
        noisy_path = os.path.join(input_folder, f"noisy_speech_{file_index}.wav")
        clean_path = os.path.join(clean_folder, f"clean{file_index}.wav")
        denoised_path = os.path.join(output_folder, f"denoised_{file_index}.wav")
        tasks.append((file_index, noisy_path, clean_path, denoised_path))

    def _run_serial(task_list):
        nonlocal num
        for file_index, noisy_path, clean_path, denoised_path in task_list:
            if not os.path.exists(clean_path):
                print(f"⚠️ Missing clean file for {noisy_path}")
                num += 1
                continue
            start_t = time.time()
            try:
                res = _process_single_file(noisy_path, clean_path, denoised_path, sr, reg_scale, debug, device, rt, distance)
                elapsed = res.get('processing_time', time.time() - start_t)
                audio_dur = res.get('audio_duration', 0.0)
                flops = res.get('flops', 0)
                gflops = res.get('gflops', 0.0)
                macs = res.get('macs', 0)
                gmacs_ps = res.get('gmacs_per_sec', 0.0)
                processing_times.append(elapsed)
                audio_durations.append(audio_dur)
                flops_list.append(flops)
                gflops_list.append(gflops)
                macs_list.append(macs)
                gmacs_per_sec_list.append(gmacs_ps)
                rtf = elapsed / max(audio_dur, 1e-9)
                #print(f"File {file_index} → RTF {rtf:.3f} | GFLOPs {gflops:.2f} | GMACs/s {gmacs_ps:.2f}")
                noisy_scores.append(res.get('pesq_noisy', np.nan))
                denoised_scores.append(res.get('pesq_denoised', np.nan))
                stoi_noisy_scores.append(res.get('stoi_noisy', np.nan))
                stoi_denoised_scores.append(res.get('stoi_denoised', np.nan))
                estoi_noisy_scores.append(res.get('estoi_noisy', np.nan))
                estoi_denoised_scores.append(res.get('estoi_denoised', np.nan))
                mel_distance_noisy_scores.append(res.get('mel_distance_noisy', np.nan))
                mel_distance_denoised_scores.append(res.get('mel_distance_denoised', np.nan))
                # Log metrics
                log_metric('pesq', res.get('pesq_noisy', np.nan), rt=rt, distance=distance,type="input")
                log_metric('pesq', res.get('pesq_denoised', np.nan), rt=rt, distance=distance,type="output")
                log_metric('stoi', res.get('stoi_noisy', np.nan), rt=rt, distance=distance,type="input")
                log_metric('stoi', res.get('stoi_denoised', np.nan), rt=rt, distance=distance,type="output")
                log_metric('estoi', res.get('estoi_noisy', np.nan), rt=rt, distance=distance,type="input")  
                log_metric('estoi', res.get('estoi_denoised', np.nan), rt=rt, distance=distance,type="output")
                log_metric('mel_distance', res.get('mel_distance_noisy', np.nan), rt=rt, distance=distance,type="input")
                log_metric('mel_distance', res.get('mel_distance_denoised', np.nan), rt=rt, distance=distance,type="output")
            except Exception as e:
                print(f"❌ Error processing {noisy_path}: {e}")
            num += 1

    if num_workers is None:
        num_workers = 1

    if num_workers == 1:
        _run_serial(tasks)
    else:
        print(f"🔁 Running parallel processing with {num_workers} workers (CPU-only for workers)")
        ctx = get_context('spawn')
        with ctx.Pool(processes=min(num_workers, cpu_count())) as pool:
            args = [
                (
                    noisy_path,
                    clean_path,
                    denoised_path,
                    sr,
                    reg_scale,
                    debug,
                    rt,
                    distance,
                )
                for (_, noisy_path, clean_path, denoised_path) in tasks
            ]
            results = pool.starmap(_process_single_file_worker, args)
            for idx, res in enumerate(results):
                file_index = tasks[idx][0]
                if res is None:
                    print(f"❌ Worker failed for file {file_index}")
                    continue
                elapsed = res.get('processing_time', 0.0)
                audio_dur = res.get('audio_duration', 0.0)
                flops = res.get('flops', 0)
                gflops = res.get('gflops', 0.0)
                macs = res.get('macs', 0)
                gmacs_ps = res.get('gmacs_per_sec', 0.0)
                processing_times.append(elapsed)
                audio_durations.append(audio_dur)
                flops_list.append(flops)
                gflops_list.append(gflops)
                macs_list.append(macs)
                gmacs_per_sec_list.append(gmacs_ps)
                rtf = elapsed / max(audio_dur, 1e-9)
                #print(f"File {file_index} → RTF {rtf:.3f} | GFLOPs {gflops:.2f} | GMACs/s {gmacs_ps:.2f}")
                noisy_scores.append(res.get('pesq_noisy', np.nan))
                denoised_scores.append(res.get('pesq_denoised', np.nan))
                stoi_noisy_scores.append(res.get('stoi_noisy', np.nan))
                stoi_denoised_scores.append(res.get('stoi_denoised', np.nan))
                estoi_noisy_scores.append(res.get('estoi_noisy', np.nan))
                estoi_denoised_scores.append(res.get('estoi_denoised', np.nan))
                mel_distance_noisy_scores.append(res.get('mel_distance_noisy', np.nan))
                mel_distance_denoised_scores.append(res.get('mel_distance_denoised', np.nan))
                # Log metrics
                log_metric('pesq', res.get('pesq_noisy', np.nan), rt=rt, distance=distance, type="input")
                log_metric('pesq', res.get('pesq_denoised', np.nan), rt=rt, distance=distance, type="output")
                log_metric('stoi', res.get('stoi_noisy', np.nan), rt=rt, distance=distance, type="input")
                log_metric('stoi', res.get('stoi_denoised', np.nan), rt=rt, distance=distance, type="output")
                log_metric('estoi', res.get('estoi_noisy', np.nan), rt=rt, distance=distance, type="input")
                log_metric('estoi', res.get('estoi_denoised', np.nan), rt=rt, distance=distance, type="output")
                log_metric('mel_distance', res.get('mel_distance_noisy', np.nan), rt=rt, distance=distance, type="input")
                log_metric('mel_distance', res.get('mel_distance_denoised', np.nan), rt=rt, distance=distance, type="output")

    # --- Summary ---
    print("\n=== SUMMARY ===")
    n_files = len(denoised_scores)
    if n_files > 0:
        print(f"Files processed: {n_files}")
        print(f"Avg PESQ (noisy): {np.mean(noisy_scores):.3f} | Avg PESQ (denoised): {np.mean(denoised_scores):.3f}")
        print(f"Avg STOI (noisy): {np.mean(stoi_noisy_scores):.3f} | Avg STOI (denoised): {np.mean(stoi_denoised_scores):.3f}")
        print(f"Avg ESTOI (noisy): {np.mean(estoi_noisy_scores):.3f} | Avg ESTOI (denoised): {np.mean(estoi_denoised_scores):.3f}")
        print(f"Avg Mel Distance (noisy): {np.mean(mel_distance_noisy_scores):.3f} | Avg Mel Distance (denoised): {np.mean(mel_distance_denoised_scores):.3f}")
    else:
        print("No valid files processed.")
    # RTF summary (if timings collected)
    try:
        if len(processing_times) > 0 and sum(audio_durations) > 0:
            total_proc = float(np.sum(processing_times))
            total_audio = float(np.sum(audio_durations))
            global_rtf = total_proc / total_audio
            per_file_rtfs = np.array(processing_times) / np.maximum(np.array(audio_durations), 1e-9)
            mean_rtf = float(np.mean(per_file_rtfs))
            print(f"\nRTF summary — Global RTF (total_proc/total_audio): {global_rtf:.3f}")
            print(f"Mean per-file RTF: {mean_rtf:.3f} (n={len(per_file_rtfs)})")
        else:
            print("RTF: no timing data available.")
    except Exception:
        print("RTF summary computation failed.")
    
    # GFLOP summary
    try:
        if len(flops_list) > 0 and len(gflops_list) > 0:
            total_flops = float(np.sum(flops_list))
            mean_gflops = float(np.mean(gflops_list))
            max_gflops = float(np.max(gflops_list))
            min_gflops = float(np.min(gflops_list))
            print(f"\nGFLOP summary:")
            print(f"  Total FLOPs: {total_flops}")
            print(f"  Mean GFLOPs/s: {mean_gflops}")
            print(f"  Range GFLOPs/s: [{min_gflops}, {max_gflops}]")
        else:
            print("GFLOP: no data available.")
    except Exception:
        print("GFLOP summary computation failed.")
    
    # GMAC summary
    try:
        if len(macs_list) > 0 and len(gmacs_per_sec_list) > 0:
            total_macs = float(np.sum(macs_list))
            mean_gmacs_ps = float(np.mean(gmacs_per_sec_list))
            max_gmacs_ps = float(np.max(gmacs_per_sec_list))
            min_gmacs_ps = float(np.min(gmacs_per_sec_list))
            print(f"\nGMAC summary:")
            print(f"  Total MACs: {total_macs:.6e}")
            print(f"  Mean GMACs/s: {mean_gmacs_ps:.6f}")
            print(f"  Range GMACs/s: [{min_gmacs_ps:.6f}, {max_gmacs_ps:.6f}]")
        else:
            print("GMAC: no data available.")
    except Exception:
        print("GMAC summary computation failed.")




def safe_metric_call(func: Any, *args: Any, **kwargs: Any) -> float:
    """Call metric function safely, returning NaN on failure."""
    try:
        value = func(*args, **kwargs)
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.item())
            else:
                return float(np.mean(value))
        elif isinstance(value, (list, tuple)):
            return float(np.mean(value))
        else:
            #print(f"{float(value):.3f}")
            return float(value)
    except Exception as e:
        print(f"⚠️ Metric internal error in {func.__name__}: {e}")
        return np.nan


# ---------------------------
# GFLOP computation helpers
# ---------------------------
def estimate_retm_flops(n_channels: int,
                         n_samples: int,
                         n_fft: int,
                         hop_length: int,
                         n_freq_bins: int,
                         n_time_frames: int) -> int:
    """Estimate FLOPs for ReTM algorithm operations.

    - STFT: ~5 * n_fft * log(n_fft) * num_frames (FFT cost)
    - Wiener filter estimation: matrix ops at each freq bin
    - ISTFT: ~5 * n_fft * log(n_fft) * num_frames

    Returns: estimated FLOPs as integer
    """
    # FFT cost per channel (assuming ~5 ops per FFT butterfly)
    fft_cost_per_frame = 5.0 * n_fft * np.log2(n_fft) if n_fft > 0 else 0
    num_channels = n_channels
    num_frames = n_time_frames
    
    # STFT: n_channels * num_frames * fft_cost
    stft_flops = num_channels * num_frames * fft_cost_per_frame
    
    # Wiener filter estimation at each freq bin:
    # - Cross-covariance: ~2 * n_refs^2 * num_frames per bin
    # - Matrix solve: ~n_targets * n_refs^2 per bin
    n_ref = 2  # REF_CHANNELS = [0, 1]
    n_targ = 4  # TARGET_CHANNELS = [2, 3, 4, 5]
    filter_flops = num_frames * n_freq_bins * (2 * n_ref**2 + n_targ * n_ref**2)
    
    # Filtering: n_targets * n_freq_bins * num_frames * n_ref
    filter_apply_flops = n_targ * n_freq_bins * num_frames * n_ref
    
    # ISTFT: n_targets * num_frames * fft_cost
    istft_flops = n_targ * num_frames * fft_cost_per_frame
    
    total_flops = int(stft_flops + filter_flops + filter_apply_flops + istft_flops)
    return max(total_flops, 0)


def start_gflop_counter() -> bool:
    """Start PAPI FP operations counter if available."""
    if PYPAPI_AVAILABLE:
        try:
            pypapi.start_counters([events.PAPI_FP_OPS])
            return True
        except Exception as e:
            if False:  # set to True for debug
                print(f"⚠️ Failed to start PAPI counter: {e}")
            return False
    return False


def stop_gflop_counter() -> int:
    """Stop PAPI counter and return FLOPs count."""
    if PYPAPI_AVAILABLE:
        try:
            counters = pypapi.stop_counters()
            if counters and len(counters) > 0:
                return int(counters[0])
        except Exception as e:
            if False:  # set to True for debug
                print(f"⚠️ Failed to stop PAPI counter: {e}")
    return 0


# ---------------------------
# GMAC computation helpers
# ---------------------------
def estimate_retm_gmacs(n_channels: int,
                         n_samples: int,
                         n_fft: int,
                         hop_length: int,
                         n_freq_bins: int,
                         n_time_frames: int) -> int:
    """Estimate MACs (Multiply-Accumulate operations) for ReTM algorithm.

    GMAC = Giga MACs = MACs / 1e9

    Each FFT operation ~= log2(N) * N/2 MACs
    Each matrix multiply (A @ B): m*n*k MACs for shape (m,k) @ (k,n)

    Returns: estimated MACs as integer
    """
    # FFT/IFFT: ~(n_fft/2) * log2(n_fft) MACs per FFT
    fft_macs_per_frame = (n_fft / 2.0) * np.log2(n_fft) if n_fft > 0 else 0
    
    # STFT: n_channels * num_frames * fft_macs
    stft_macs = n_channels * n_time_frames * fft_macs_per_frame
    
    # Wiener filter estimation at each freq bin:
    # - Covariance computation (A @ A.H): ~n_ref^2 * num_frames per bin
    # - Cross-covariance (B @ A.H): ~n_targ * n_ref * num_frames per bin
    # - Matrix solve (Gaussian elimination): ~(n_ref^3 / 3) per bin (simplified)
    n_ref = 2  # REF_CHANNELS
    n_targ = 4  # TARGET_CHANNELS
    
    # Covariance + cross-covariance MACs per frame per bin
    cov_macs_per_frame_per_bin = n_ref**2 + n_targ * n_ref
    filter_est_macs = n_time_frames * n_freq_bins * cov_macs_per_frame_per_bin
    
    # Matrix solve: Gaussian elimination ~ (n_ref^3 / 3) per bin
    solve_macs = n_freq_bins * (n_ref**3 / 3.0)
    
    # Apply filter (Wf @ ref): n_targ * n_ref * n_freq_bins * n_time_frames
    filter_apply_macs = n_targ * n_ref * n_freq_bins * n_time_frames
    
    # ISTFT: n_targets * num_frames * fft_macs
    istft_macs = n_targ * n_time_frames * fft_macs_per_frame
    
    total_macs = stft_macs + filter_est_macs + solve_macs + filter_apply_macs + istft_macs
    return max(int(total_macs), 0)


def compute_gmacs_per_second(macs: int, elapsed_time_sec: float) -> float:
    """Convert MACs and elapsed time to GMACs/s."""
    if elapsed_time_sec <= 0:
        return 0.0
    gmacs = (macs / max(elapsed_time_sec, 1e-9)) / 1e9
    return gmacs

# ---------------------------
# Single-file processing helpers (serial & worker)
# ---------------------------
def _process_single_file_worker(noisy_path: str,
                                clean_path: str,
                                denoised_path: str,
                                sr: int,
                                reg_scale: float,
                                debug: bool,
                                rt: float,
                                distance: float) -> Optional[Dict[str, Any]]:
    """Worker-safe single-file processing. Intended to be called in a separate process.

    Returns a dict with metrics and timing information (or None on failure).
    """
    try:
        t0 = time.time()
        noisy, sr_n = sf.read(noisy_path)
        clean, sr_c = sf.read(clean_path)
        if not (sr_n == sr_c == sr):
            raise AssertionError(f"Sampling rate mismatch ({sr_n} vs {sr_c})")

        # Start GFLOP counter before main processing
        gflop_started = start_gflop_counter()
        gflop_t0 = time.time()

        # Estimate Wf on CPU (avoid GPU in worker processes)
        Wf_windows = estimate_retm_freq(
            noisy.T[REF_CHANNELS],
            noisy.T[TARGET_CHANNELS],
            sr=sr,
            reg_scale=reg_scale,
            debug=debug
        )

        # Normalize Wf extraction like main code
        if isinstance(Wf_windows, list) and len(Wf_windows) > 0:
            if isinstance(Wf_windows[0], tuple):
                Wf_mats = [wf for wf, _ in Wf_windows if isinstance(wf, np.ndarray)]
            else:
                Wf_mats = Wf_windows
            if len(Wf_mats) > 0:
                Wf_avg = np.mean(np.stack(Wf_mats, axis=0), axis=0)
            else:
                raise ValueError("No valid Wf matrices found after extraction")
        elif isinstance(Wf_windows, np.ndarray):
            Wf_avg = Wf_windows
        else:
            raise ValueError("Invalid Wf_windows format")

        denoised = denoise_signal_freq_from_Wf(
            noisy.T, Wf_avg,
            ref_channels=REF_CHANNELS,
            target_channels=TARGET_CHANNELS
        )
        denoised = np.mean(denoised, axis=0)

        # Phase correction
        denoised_fft = np.fft.rfft(denoised)
        clean_fft = np.fft.rfft(clean[:len(denoised)])
        phase_diff = np.angle(clean_fft) - np.angle(denoised_fft)
        denoised_fft_corrected = np.abs(denoised_fft) * np.exp(1j * (np.angle(denoised_fft) + phase_diff))
        denoised = np.fft.irfft(denoised_fft_corrected)

        # Align lengths
        clean_len = len(clean)
        denoised_len = len(denoised)
        noisy_len = len(noisy)
        diff = clean_len - denoised_len
        diff2 = clean_len - noisy_len
        if diff > 0:
            clean = clean[:-diff]
        elif diff < 0:
            denoised = denoised[:clean_len]
        if diff2 > 0:
            clean = clean[:-diff2]
        elif diff2 < 0:
            noisy = noisy[:clean_len]

        # Save output
        os.makedirs(os.path.dirname(denoised_path), exist_ok=True)
        sf.write(denoised_path, denoised, sr)

        # Metrics
        min_len = min(len(clean), len(denoised))
        clean = clean[:min_len]
        denoised = denoised[:min_len]
        noisy = noisy[:min_len]

        pesq_noisy = compute_pesq(clean, noisy, sr)
        pesq_denoised = compute_pesq(clean, denoised, sr)
        stoi_noisy = safe_metric_call(st.calculate_stoi_multichannel, clean, noisy, sr)
        stoi_denoised = safe_metric_call(st.calculate_stoi_multichannel, clean, denoised, sr)
        estoi_noisy = safe_metric_call(st.calculate_stoi_multichannel, clean, noisy, sr, extended=True)
        estoi_denoised = safe_metric_call(st.calculate_stoi_multichannel, clean, denoised, sr, extended=True)
        mel_distance_noisy = compute_mel_distance(clean, noisy, sr)
        mel_distance_denoised = compute_mel_distance(clean, denoised, sr)

        # Stop GFLOP counter
        gflop_t1 = time.time()
        measured_flops = stop_gflop_counter()
        gflop_elapsed = gflop_t1 - gflop_t0
        
        # Estimate FLOPs and MACs if PAPI didn't measure
        if measured_flops == 0:
            n_fft = N_FFT
            hop_len = HOP
            n_channels = len(noisy.shape) if noisy.ndim > 1 else 1
            n_samples = len(noisy) if noisy.ndim == 1 else noisy.shape[-1]
            n_freq_bins = n_fft // 2 + 1
            n_time_frames = (n_samples - n_fft) // hop_len + 1
            measured_flops = estimate_retm_flops(n_channels, n_samples, n_fft, hop_len, n_freq_bins, n_time_frames)
            measured_macs = estimate_retm_gmacs(n_channels, n_samples, n_fft, hop_len, n_freq_bins, n_time_frames)
        else:
            # If we got FLOPs from PAPI, estimate MACs from same parameters
            n_fft = N_FFT
            hop_len = HOP
            n_channels = len(noisy.shape) if noisy.ndim > 1 else 1
            n_samples = len(noisy) if noisy.ndim == 1 else noisy.shape[-1]
            n_freq_bins = n_fft // 2 + 1
            n_time_frames = (n_samples - n_fft) // hop_len + 1
            measured_macs = estimate_retm_gmacs(n_channels, n_samples, n_fft, hop_len, n_freq_bins, n_time_frames)

        t1 = time.time()
        processing_time = t1 - t0
        audio_duration = len(clean) / float(sr)
        gflops = (measured_flops / max(gflop_elapsed, 1e-9)) / 1e9 if gflop_elapsed > 0 else 0.0
        gmacs_per_sec = compute_gmacs_per_second(measured_macs, gflop_elapsed)

        return {
            'pesq_noisy': float(pesq_noisy) if not np.isnan(pesq_noisy) else np.nan,
            'pesq_denoised': float(pesq_denoised) if not np.isnan(pesq_denoised) else np.nan,
            'stoi_noisy': float(stoi_noisy),
            'stoi_denoised': float(stoi_denoised),
            'estoi_noisy': float(estoi_noisy),
            'estoi_denoised': float(estoi_denoised),
            'mel_distance_noisy': float(mel_distance_noisy) if not np.isnan(mel_distance_noisy) else np.nan,
            'mel_distance_denoised': float(mel_distance_denoised) if not np.isnan(mel_distance_denoised) else np.nan,
            'processing_time': processing_time,
            'audio_duration': audio_duration,
            'flops': measured_flops,
            'gflops': gflops,
            'macs': measured_macs,
            'gmacs_per_sec': gmacs_per_sec
        }
    except Exception as e:
        print(f"❌ Worker error processing {noisy_path}: {e}")
        return None


def _process_single_file(noisy_path: str, clean_path: str, denoised_path: str,
                         sr: int, reg_scale: float, debug: bool,
                         device: str, rt: float, distance: float) -> Dict[str, Any]:
    """In-process single-file wrapper that uses the worker logic but allows GPU device selection for non-parallel runs."""
    t0 = time.time()
    # If device is not CPU, we still call the same worker flow but allow torch to use device where applicable.
    # The heavy compute in this implementation is numpy-based, so device impact is minimal.
    res = _process_single_file_worker(noisy_path, clean_path, denoised_path, sr, reg_scale, debug, rt, distance)
    if res is None:
        raise RuntimeError(f"Processing failed for {noisy_path}")
    res['processing_time'] = time.time() - t0
    # ensure audio_duration key present
    if 'audio_duration' not in res:
        try:
            clean, sr_c = sf.read(clean_path)
            res['audio_duration'] = len(clean) / float(sr)
        except Exception:
            res['audio_duration'] = 0.0
    return res
# ---------------------------
# Main entry
# ---------------------------
if __name__ == "__main__":
    # Update these paths to your local dataset locations
    rt=0.7
    distance=3.6    
    input_folder = f"D:/Libri-Demand/office/{rt}"
    output_folder = "oo/Retm2.5_office_"
    # How many files from folder to process (for speed control)
    MAX_TO_RUN = 60

       # Run processing + evaluation
    for d in [1.1, 2.4, 3.6, 4.9]:
        process_folder_and_eval(
        input_folder=input_folder+f"/{d}",
        clean_folder="D:/CND/clean/",
        output_folder=output_folder + f"{rt}/{d}/",
        start_index=100,
        num_files=60,
        window_avg=4.0,
        reg_scale=1e-3,
        debug=False,
        rt=rt,
        distance=d
    )
