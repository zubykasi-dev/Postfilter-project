# ----------------------------
# Model: simplified Grouped Temporal CRN(GTCRN)-inspired dual-branch PF
import os
import sys
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# NEW: Modern AMP
from torch.amp import autocast, GradScaler

############################
#  Padding Collate
############################
def pad_collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad batch tensors to the same length and return (inputs, targets)."""
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
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        window_fn: callable = torch.hann_window,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", window_fn(win_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def inverse(self, X: torch.Tensor, length: int | None = None) -> torch.Tensor:
        return torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,  # FIXED (was self.hop)
            win_length=self.win_length,
            window=self.window,
            length=length,
        )


############################
#  GST Module
############################
class GST(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        num_heads: int = 8,
        token_dim: int = 128,
    ):
        super().__init__()
        freq_bins = n_fft // 2 + 1

        self.embed = nn.Linear(freq_bins, token_dim)
        self.attn = nn.Linear(token_dim, num_heads)
        self.tokens = nn.Parameter(torch.randn(num_heads, token_dim))

    def forward(self, logmag: torch.Tensor) -> torch.Tensor:
        """Create global style token from spectral log magnitude."""
        # logmag: (B,1,F,T)
        x = logmag.squeeze(1).mean(dim=-1)  # (B,F)

        e = torch.tanh(self.embed(x))  # (B,128)
        w = torch.softmax(self.attn(e), dim=-1)  # (B,H)
        style = w @ self.tokens  # (B,128)

        return style


############################
#  Gated Layer
############################


###############################################
# Gated Convolution Layer
###############################################
class GLayer(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel, stride, padding=kernel // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gated convolution: half channels for value, half for gate."""
        h = self.conv(x)
        c, g = torch.chunk(h, 2, dim=1)
        return c * torch.sigmoid(g)


###############################################
# Clean UpConv: 2D Upsampling (F × T)
###############################################
class UpConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # Upsample frequency AND time
        self.up = nn.Upsample(scale_factor=(2, 2), mode="nearest")
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample and smooth with 2D convolution."""
        return F.relu(self.conv(self.up(x)))


###############################################
# Fully Fixed DualBranchPF
###############################################

class DualBranchPF(nn.Module):
    def __init__(
        self,
        n_mics: int = 1,
        gst_dim: int = 128,
        n_fft: int = 512,
        gru_hidden: int = 1024,
    ):
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
        self.cgru: nn.GRU | None = None
        self.proj: nn.Linear | None = None

        # Decoder
        self.dec3 = UpConv(128 + gst_dim, 64)
        self.dec2 = UpConv(64, 32)
        self.dec1 = nn.Conv2d(32, 2, 3, padding=1)  # real + imag

    def forward(
        self,
        X_multi: torch.Tensor,
        mag_ref: torch.Tensor,
        gst_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Decode dynamic complex mask from multi-channel STFT features."""
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

#  Dataset
class PFTrainingDataset(Dataset):
    def __init__(
        self,
        noisy_files: list[str],
        clean_files: list[str],
        sr: int = 16000,
    ):
        """Dataset for paired noisy/clean waveform training files."""
        if noisy_files is None or clean_files is None:
            raise ValueError("noisy_files and clean_files must be provided")

        self.retm_files: list[str] = noisy_files
        self.clean_files: list[str] = clean_files
        self.sr: int = sr

    def __len__(self) -> int:
        return len(self.retm_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        noisy = self.retm_files[idx]
        clean = self.clean_files[idx]

        x, _ = sf.read(noisy)
        y, _ = sf.read(clean)

        if x.ndim > 1: x = x[:, 0]
        if y.ndim > 1: y = y[:, 0]

        L = min(len(x), len(y))
        x, y = x[:L], y[:L]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


#  TRAINING LOOP

from tqdm import tqdm
def train_postfilter(
        file_list: list[str],           # list of noisy files
        clean_files: list[str],         # list of clean files (must match file_list)
        save_path: str = "pf_model.pt",
        epochs: int = 1,
        batch_size: int = 1,
        lr: float = 1e-4,
        n_fft: int = 512,
        hop: int = 128,
        device: str = "cuda",
) -> None:
    """Train post-filter model with paired dataset and save checkpoint."""
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


#  LOAD MODEL

def load_postfilter(path: str, device: str = "cpu") -> dict[str, object]:
    """Load a trained DualBranchPF + GST checkpoint safely.

    Args:
        path: Path to saved checkpoint.
        device: Torch device string.

    Returns:
        Loaded model and GST module.
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


#  APPLY POSTFILTER
def apply_postfilter_to_file(model, gst, input_wav, output_wav,
                             n_fft=512, hop=128, device="cpu"):
    # Read input
    x, sr = sf.read(input_wav)
    if x.ndim > 1:
        x = x[:, 0]

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
