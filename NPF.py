# ----------------------------
# Model: simplified Grouped Temporal CRN(GTCRN)-inspired dual-branch PF
import os
import sys
import math
import glob
import time
import numpy as np
import soundfile as sf
import librosa
from pystoi import stoi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pesq_test as pt  

# NEW: Modern AMP
from torch.amp import autocast, GradScaler
try:
    import pypapi
    from pypapi import events
    PYPAPI_AVAILABLE = True
except Exception:
    PYPAPI_AVAILABLE = False

def compute_pesq_from_files(ref_path, deg_path, mode="wb"):
    """
    Compute PESQ from two file paths using pesq_test (pt).
    mode: "wb" (wideband) or "nb" (narrowband). For SR=16000 use "wb".
    Returns float PESQ or np.nan on error.
    """
    try:
        # read_audio returns (signal, sr)
        ref, sr_r = pt.read_audio(ref_path)
        deg, sr_d = pt.read_audio(deg_path)
        if sr_r != sr_d:
            raise ValueError(f"Sample rate mismatch: {sr_r} vs {sr_d}")

        # Align lengths (crop longer to shorter)
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]

        # Compute PESQ
        return float(pt.pesq(sr_r, ref, deg, mode))
    except Exception as e:
        print(f"⚠️ PESQ error for {ref_path} vs {deg_path}: {e}")
        return np.nan
def compute_stoi_from_files(ref_path, deg_path, sr=16000, extended=False):
    """
    Compute STOI or ESTOI from file paths.
    extended=False → STOI
    extended=True  → ESTOI
    """
    try:
        ref, sr_r = sf.read(ref_path)
        deg, sr_d = sf.read(deg_path)
        
        # Align
        ref, deg = align_signals(ref, deg)
        if sr_r != sr_d:
            raise ValueError(f"SR mismatch: {sr_r} vs {sr_d}")

        # Convert stereo → mono (if any)
        if ref.ndim > 1:
            ref = ref.mean(axis=1)
        if deg.ndim > 1:
            deg = deg.mean(axis=1)

        # Align lengths
        L = min(len(ref), len(deg))
        ref = ref[:L]
        deg = deg[:L]

        return float(stoi(ref, deg, sr, extended=extended))

    except Exception as e:
        print(f"⚠️ STOI/ESTOI error ({ref_path} vs {deg_path}): {e}")
        return np.nan


############################
#  Padding Collate
############################
def pad_collate(batch):
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)

    xs = [F.pad(x, (0, max_len - x.shape[0])) for x in xs]
    ys = [F.pad(y, (0, max_len - y.shape[0])) for y in ys]

    xs = torch.stack(xs).unsqueeze(1)  # (B,1,T)
    ys = torch.stack(ys).unsqueeze(1)
    return xs, ys


############################
#  STFT Module
############################
class STFT(nn.Module):
    def __init__(self, n_fft=512, hop_length=128, win_length=512, window_fn=torch.hann_window):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", window_fn(win_length))

    def forward(self, x):
        # Accepts (B,1,T) or (B,T)
        if x.dim() == 3:
            x = x.squeeze(1)  # → (B,T)

        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )

    def inverse(self, X, length=None):
        return torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,  # FIXED (was self.hop)
            win_length=self.win_length,
            window=self.window,
            length=length,
        )


# ---------------------------
# GFLOP/GMAC measurement helpers
# ---------------------------
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


def estimate_npf_flops(model, gst, n_samples, n_fft, hop_length):
    """Heuristic FLOP estimate for NPF inference: STFT + model parameter usage + ISTFT."""
    if n_fft <= 0 or hop_length <= 0:
        return 0
    frames = max(1, (n_samples - n_fft) // hop_length + 1)
    fft_ops = 5.0 * n_fft * np.log2(n_fft)
    # STFT (mono)
    stft_flops = frames * fft_ops

    # model param-based heuristic: number of params * frames * small factor
    try:
        param_count = sum(p.numel() for p in model.parameters())
    except Exception:
        param_count = 0
    try:
        gst_params = sum(p.numel() for p in gst.parameters())
    except Exception:
        gst_params = 0

    model_flops = (param_count + gst_params) * frames * 2.0 * 1e-3

    istft_flops = frames * fft_ops

    total = stft_flops + model_flops + istft_flops
    return max(int(total), 0)


def estimate_npf_gmacs(model, gst, n_samples, n_fft, hop_length):
    if n_fft <= 0 or hop_length <= 0:
        return 0
    frames = max(1, (n_samples - n_fft) // hop_length + 1)
    fft_macs = (n_fft / 2.0) * np.log2(n_fft)
    stft_macs = frames * fft_macs
    try:
        param_count = sum(p.numel() for p in model.parameters())
    except Exception:
        param_count = 0
    try:
        gst_params = sum(p.numel() for p in gst.parameters())
    except Exception:
        gst_params = 0

    model_macs = (param_count + gst_params) * frames * 1e-3
    istft_macs = frames * fft_macs
    total = stft_macs + model_macs + istft_macs
    return max(int(total), 0)


def compute_gmacs_per_second(macs, elapsed_time_sec):
    if elapsed_time_sec <= 0:
        return 0.0
    return (macs / max(elapsed_time_sec, 1e-9)) / 1e9


############################
#  GST Module
############################
class GST(nn.Module):
    def __init__(self, n_fft=512, num_heads=8, token_dim=128):
        super().__init__()
        freq_bins = n_fft // 2 + 1

        self.embed = nn.Linear(freq_bins, token_dim)
        self.attn = nn.Linear(token_dim, num_heads)
        self.tokens = nn.Parameter(torch.randn(num_heads, token_dim))

    def forward(self, logmag):
        # logmag: (B,1,F,T)
        x = logmag.squeeze(1).mean(dim=-1)  # (B,F)

        e = torch.tanh(self.embed(x))  # (B,128)
        w = torch.softmax(self.attn(e), dim=-1)  # (B,H)
        style = w @ self.tokens  # (B,128)

        return style


############################
#  Gated Layer
############################
import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################
# Gated Convolution Layer
###############################################
class GLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel, stride, padding=kernel // 2)

    def forward(self, x):
        h = self.conv(x)
        c, g = torch.chunk(h, 2, dim=1)
        return c * torch.sigmoid(g)


###############################################
# Clean UpConv: 2D Upsampling (F × T)
###############################################
class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Upsample frequency AND time
        self.up = nn.Upsample(scale_factor=(2, 2), mode="nearest")
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(self.up(x)))


###############################################
# Fully Fixed DualBranchPF
###############################################
import torch
import torch.nn as nn
import torch.nn.functional as F

class DualBranchPF(nn.Module):
    def __init__(self, n_mics=1, gst_dim=128, n_fft=512, gru_hidden=1024):
        super().__init__()
        self.n_fft = n_fft
        self.freq_bins = n_fft // 2 + 1
        self.gst_dim = gst_dim
        self.gru_hidden = gru_hidden

        # Encoder
        self.enc1 = GLayer(n_mics, 32, 3, 1)
        self.enc2 = GLayer(32, 64, 3, 2)
        self.enc3 = GLayer(64, 128, 3, 2)

        # We'll determine GRU input size dynamically during first forward
        self.cgru = None
        self.proj = None

        # Decoder
        self.dec3 = UpConv(128 + gst_dim, 64)
        self.dec2 = UpConv(64, 32)
        self.dec1 = nn.Conv2d(32, 2, 3, padding=1)  # real + imag

    def forward(self, X_multi, mag_ref, gst_vec):
        B, _, F, T = X_multi.shape
        X_real = X_multi.real

        # ---- Encoder ----
        e1 = self.enc1(X_real)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        B, C3, F3, T3 = e3.shape

        # ---- GRU ----
        gru_in = e3.permute(0, 3, 1, 2).reshape(B, T3, C3 * F3)

        # Create GRU on first forward if not yet initialized
        if self.cgru is None:
            self.cgru = nn.GRU(
                input_size=C3 * F3,
                hidden_size=self.gru_hidden,
                batch_first=True
            ).to(e3.device)
            self.proj = nn.Linear(self.gru_hidden, C3 * F3).to(e3.device)

        gru_out, _ = self.cgru(gru_in)
        gru_out = self.proj(gru_out)
        gru_out = gru_out.reshape(B, T3, C3, F3).permute(0, 2, 3, 1)

        # ---- GST expansion ----
        gst_ex = gst_vec[:, :, None, None].expand(-1, -1, F3, T3)

        # ---- Decoder ----
        d3 = self.dec3(torch.cat([gru_out, gst_ex], dim=1))
        d2 = self.dec2(d3)
        out = self.dec1(d2)

        # Split real and imag
        real, imag = torch.chunk(out, 2, dim=1)

        # ---- Align frequency ----
        if real.size(2) > self.freq_bins:
            real = real[:, :, :self.freq_bins, :]
            imag = imag[:, :, :self.freq_bins, :]
        elif real.size(2) < self.freq_bins:
            pad = self.freq_bins - real.size(2)
            real = F.pad(real, (0, 0, 0, pad))
            imag = F.pad(imag, (0, 0, 0, pad))

        # ---- Align time ----
        if real.size(3) > T:
            real = real[:, :, :, :T]
            imag = imag[:, :, :, :T]
        elif real.size(3) < T:
            pad = T - real.size(3)
            real = F.pad(real, (0, pad))
            imag = F.pad(imag, (0, pad))

        m_complex = torch.complex(real.float(), imag.float())
        return m_complex, None

############################
#  Dataset
############################
class PFTrainingDataset(Dataset):
    def __init__(self, noisy_files=None, clean_files=None, sr=16000):
        """
        Accepts either:
        - noisy_files + clean_files: lists of file paths
        """
        if noisy_files is None or clean_files is None:
            raise ValueError("noisy_files and clean_files must be provided")
        
        self.retm_files = noisy_files
        self.clean_files = clean_files
        self.sr = sr

    def __len__(self):
        return len(self.retm_files)

    def __getitem__(self, idx):
        noisy = self.retm_files[idx]
        clean = self.clean_files[idx]

        x, _ = sf.read(noisy)
        y, _ = sf.read(clean)

        if x.ndim > 1: x = x[:, 0]
        if y.ndim > 1: y = y[:, 0]

        L = min(len(x), len(y))
        x, y = x[:L], y[:L]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


############################
#  TRAINING LOOP
############################
from tqdm import tqdm
import random
def train_postfilter(
        file_list,           # list of noisy files
        clean_files,         # list of clean files (must match file_list)
        save_path="pf_model.pt",
        epochs=1,
        batch_size=1,
        lr=1e-4,
        n_fft=512,
        hop=128,
        device="cuda"
):
    # Build dataset
    dataset = PFTrainingDataset(noisy_files=file_list, clean_files=clean_files)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate,
        num_workers=0
    )

    stft = STFT(n_fft, hop).to(device)
    gst = GST(n_fft=512).to(device)
    model = DualBranchPF().to(device)

    opt = torch.optim.Adam(list(model.parameters()) + list(gst.parameters()), lr=lr)
    scaler = GradScaler("cuda")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")

        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            T = x.size(-1)

            with autocast("cuda"):
                X = stft(x)
                Y = stft(y)

                mag = X.abs().unsqueeze(1)
                logmag = torch.log1p(mag)
                gst_vec = gst(logmag)

                m_complex, _ = model(X.unsqueeze(1), mag, gst_vec)
                S_hat = m_complex.squeeze(1) * X

                y_hat = stft.inverse(S_hat, length=T)

                loss = F.l1_loss(S_hat.abs(), Y.abs()) + 0.5 * F.l1_loss(y_hat, y.squeeze(1))

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        print(f"[Epoch {epoch}] avg loss={total_loss/len(loader):.4f}")

        torch.save({"model": model.state_dict(), "gst": gst.state_dict()}, save_path)

    print("Training complete.")


############################
#  LOAD MODEL
############################
def load_postfilter(path, device="cpu"):
    """
    Load a trained DualBranchPF + GST checkpoint safely.
    Works even if GRU weights were missing in older checkpoints.
    """
    ck = torch.load(path, map_location=device)

    # Initialize model and GST
    model = DualBranchPF().to(device)
    gst = GST().to(device)

    # --- Load GST safely ---
    if "gst" in ck:
        gst.load_state_dict(ck["gst"])

    # --- Load model safely ---
    model_state = ck.get("model", {})
    model_keys = set(model_state.keys())
    model_expected_keys = set(model.state_dict().keys())

    # Filter out unexpected keys (e.g., cgru in older checkpoints)
    filtered_state = {k: v for k, v in model_state.items() if k in model_expected_keys}

    # Load state dict
    model.load_state_dict(filtered_state, strict=False)

    model.eval()
    gst.eval()
    return model, gst


############################
#  APPLY POSTFILTER
############################
def apply_postfilter_to_file(model, gst, input_wav, output_wav,
                             n_fft=512, hop=128, device="cpu"):
    # Read input
    x, sr = sf.read(input_wav)
    if x.ndim > 1:
        x = x[:, 0]
    n_samples = len(x)

    # Start timing and optional GFLOP counter
    t_start = time.time()
    gflop_t0 = time.time()
    start_gflop_counter()

    x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    T = x.size(-1)

    stft = STFT(n_fft, hop).to(device)
    X = stft(x)

    mag = X.abs().unsqueeze(1)
    logmag = torch.log1p(mag)

    with torch.no_grad():
        gst_vec = gst(logmag)
        m_complex, _ = model(X.unsqueeze(1), mag, gst_vec)
        S_hat = m_complex.squeeze(1) * X
        S_hat_istft = S_hat  # already (F,T)
        y_hat = stft.inverse(S_hat_istft, length=T)


    sf.write(output_wav, y_hat.squeeze().cpu().numpy(), sr)
    print(f"Saved → {output_wav}")

    # Stop GFLOP counter and compute metrics
    gflop_t1 = time.time()
    measured_flops = stop_gflop_counter()
    gflop_elapsed = gflop_t1 - gflop_t0

    # Fallback estimates if hardware counters not available
    if measured_flops == 0:
        measured_flops = estimate_npf_flops(model, gst, n_samples, n_fft, hop)
    measured_macs = estimate_npf_gmacs(model, gst, n_samples, n_fft, hop)

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

    return metrics




############################
#  STOI
############################
def align_signals(a, b):
    L = min(len(a), len(b))
    return a[:L], b[:L]

def calculate_stoi_multichannel(ref, deg, sr=16000, extended=False):
    """
    Compute STOI / ESTOI for mono or multichannel signals.
    - ref, deg may be file paths or numpy arrays.
    - Returns a single float.
    """
    # Load if needed
    if isinstance(ref, str):
        ref, _ = librosa.load(ref, sr=sr, mono=True)
    if isinstance(deg, str):
        deg, _ = librosa.load(deg, sr=sr, mono=True)

    # Align
    ref, deg = align_signals(ref, deg)

    # Compute STOI
    try:
        return float(stoi(ref, deg, sr, extended=extended))
    except Exception as e:
        print(f"[STOI ERROR] {e}")
        return np.nan

def calculate_pesq(ref_file, deg_file, sr=16000, mode="wb"):
    """
    Compute PESQ wideband (wb) or narrowband (nb)
    using your pesq_test module.
    """
    try:
        ref, sr_r = pt.read_audio(ref_file)
        deg, sr_d = pt.read_audio(deg_file)

        if sr_r != sr_d:
            raise ValueError("Sample rate mismatch")

        ref, deg = align_signals(ref, deg)

        return float(pt.pesq(sr_r, ref, deg, mode))

    except Exception as e:
        print(f"[PESQ ERROR] {ref_file}: {e}")
        return np.nan

import re

def extract_index(fname):
    """
    Extract the last integer found in the filename.
    Example:
        noisy_speech_120.wav → 120
        office_0.4_1.1_155.wav → 155
    """
    nums = re.findall(r"\d+", fname)
    return int(nums[-1]) if nums else None
def get_clean_filename(noisy_fname):
    idx = extract_index(noisy_fname)
    return f"clean{idx}.wav" if idx is not None else None


############################
#  MAIN
############################
if __name__ == "__main__":
    #noisy_root_dirs = (
    #    [f"D:/CND/noisy/train/cn-set/{rt}/{snr}" for rt in ["200", "400", "600", "800"] for snr in ["-10","-5","0","5","10"]] +
    #    [f"D:/CND/noisy/train/env-set/{rt}/{snr}" for rt in ["200", "400", "600", "800"] for snr in ["-10","-5","0","5","10"]] +
    #    [f"D:/CND/noisy/train/ego-set/{rt}/{snr}" for rt in ["200", "400", "600", "800"] for snr in ["-10","-5","0","5","10"]]
    #)
    noisy_root_dirs = (
        [f"D:/Libri-DEMAND/train/office/{rt}/{snr}" for rt in ["0.4", "0.7"] for snr in ["1.1","2.4","3.6","4.9"]] +
        [f"D:/Libri-DEMAND/train/rest/{rt}/{snr}" for rt in ["0.4", "0.7"] for snr in ["1.1","2.4","3.6","4.9"]] 
        )
    clean_root_dirs = ["D:/CND/clean/"] * len(noisy_root_dirs)

    nfiles_per_folder = 350  # NEW: number of files per folder
    all_noisy_files = []
    all_clean_files = []

    for noisy_dir, clean_dir in zip(noisy_root_dirs, clean_root_dirs):

        noisy_files = sorted(os.listdir(noisy_dir))

        # Optional sampling
        if nfiles_per_folder < len(noisy_files):
            noisy_files = random.sample(noisy_files, nfiles_per_folder)

        for noisy_fname in noisy_files:
            noisy_path = os.path.join(noisy_dir, noisy_fname)

            # 🔥 Extract numeric index from noisy filename
            clean_fname = get_clean_filename(noisy_fname)

            if clean_fname is None:
                continue  # skip if no valid index
            print(f"{noisy_fname} → {clean_fname}")
            clean_path = os.path.join(clean_dir, clean_fname)

            if os.path.exists(noisy_path) and os.path.exists(clean_path):
                all_noisy_files.append(noisy_path)
                all_clean_files.append(clean_path)



    print(f"Total files selected: {len(all_noisy_files)}")
    model_path = f"pf_model_{nfiles_per_folder}_CND.pt"

    if not os.path.exists(model_path):
        print("No pretrained model found. Starting training...")
        train_postfilter(
            file_list=all_noisy_files,
            clean_files=all_clean_files,
            save_path=model_path,
            epochs=50,
            batch_size=3,
            lr=1e-4,
            n_fft=512,
            hop=128,
            device="cuda"
        )
    else:
        print(f"Pretrained model '{model_path}' found. Skipping training.")

    # Limit to single CPU core for single-core RTF measurement
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Load model & apply postfilter (on CPU)
    model, gst = load_postfilter(model_path, device="cpu")

    input_folder = "oo/ReTM2.5_office_0.7/4.9/"
    output_folder = "pf_out/"

    os.makedirs(output_folder, exist_ok=True)

    # Collect performance metrics while applying postfilter
    rtf_list = []
    gflop_list = []
    gmacs_per_sec_list = []
    flops_total = 0

    for i in range(100, 160):
        inp = f"{input_folder}/denoised_{i}.wav"
        out = f"{output_folder}/pf_denoised_{i}.wav"
        if not os.path.exists(inp):
            continue
        metrics = None
        try:
            metrics = apply_postfilter_to_file(model, gst, inp, out, hop=96)
        except Exception as e:
            print(f"Error applying postfilter to {inp}: {e}")
            continue

        if metrics is not None:
            rtf_list.append(metrics['rtf'])
            gflop_list.append(metrics['gflops'])
            gmacs_per_sec_list.append(metrics['gmacs_per_sec'])
            flops_total += metrics['flops']
            print(f"File {i:03d} | RTF {metrics['rtf']:.4f} | GFLOPs {metrics['gflops']:.2f} | GMACs/s {metrics['gmacs_per_sec']:.4f}")
    # Print summary statistics for NPF postfilter
    print("\n" + "="*60)
    print("NPF POSTFILTER - PERFORMANCE METRICS SUMMARY")
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

    print("----------- INPUT PESQ SCORES -----------")
    sr = 16000

    for i in range(100, 160):
        ref = f"D:/CND/clean/clean{i}.wav"          # clean speech
        inp = f"{input_folder}/denoised_{i}.wav"                 # ReTM output (input to PF)

        if os.path.exists(ref) and os.path.exists(inp):
            score = compute_pesq_from_files(ref, inp, mode="nb")      # or "nb"
            print(f"{score:.4f}")


    print("----------- PESQ SCORES -----------")
    sr = 16000
    for i in range(100, 160):
        ref = f"D:/CND/clean/clean{i}.wav"
        deg = f"{output_folder}/pf_denoised_{i}.wav"
        if os.path.exists(ref) and os.path.exists(deg):
            score = compute_pesq_from_files(ref, deg, mode="nb")   # or "nb"
            print(f"{score:.4f}")

    print("----------- STOI SCORES -----------")
    sr = 16000
    for i in range(100, 160):
        ref = f"D:/CND/clean/clean{i}.wav"
        deg = f"{output_folder}/pf_denoised_{i}.wav"
        if os.path.exists(ref) and os.path.exists(deg):
            score = compute_stoi_from_files(ref, deg, sr)
            print(f"{score:.4f}")
    print("----------- ESTOI SCORES -----------")
    sr = 16000
    for i in range(100, 160):
        ref = f"D:/CND/clean/clean{i}.wav"
        deg = f"{output_folder}/pf_denoised_{i}.wav"
        if os.path.exists(ref) and os.path.exists(deg):
            score = compute_stoi_from_files(ref, deg, sr, extended=True)
            print(f"{score:.4f}")
