import numpy as np
import soundfile as sf
from pesq import pesq

def read_audio(file_path):
    data, rate = sf.read(file_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data / np.iinfo(data.dtype).max  # Normalize to [-1, 1]
    return data.astype(np.float32), rate

def pesq_score(ref, deg, sr):
    # Ensure correct PESQ mode
    mode = 'wb' if sr >= 48000 else 'nb'

    ref = np.asarray(ref, dtype=np.float32)
    deg = np.asarray(deg, dtype=np.float32)

    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]

    # Safety: PESQ expects values in [-1, 1]
    ref = np.clip(ref, -1, 1)
    deg = np.clip(deg, -1, 1)

    try:
        score = pesq(sr, ref, deg, mode)
    except Exception as e:
        print(f"PESQ computation failed: {e}")
        score = np.nan
    return score


# Example usage
#ref_file = 'MCEoutput_dereverberated.wav'
#deg_file = 'MC_reverb6.wav'

#ref, rate = read_audio(ref_file)
#deg, _ = read_audio(deg_file)

# Calculate PESQ score
#pesq_score_wb = pesq(rate, ref, deg, 'wb')
#print("PESQ Score (Wideband):", pesq_score_wb)
