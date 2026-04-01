import numpy as np
import soundfile as sf
import librosa
from pystoi import stoi

import numpy as np
import librosa
from pystoi import stoi

def calculate_stoi_multichannel(ref, deg, sr, extended=False):
    """
    Compute average STOI or ESTOI for multichannel signals.
    Accepts either file paths or numpy arrays.
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

    # Match length
    min_len = min(ref.shape[0], deg.shape[0])
    ref, deg = ref[:min_len], deg[:min_len]

    # Match channels
    min_ch = min(ref.shape[1], deg.shape[1])
    ref, deg = ref[:, :min_ch], deg[:, :min_ch]

    scores = []
    for ch in range(min_ch):
        try:
            score = stoi(ref[:, ch], deg[:, ch], sr, extended=extended)
            scores.append(score)
        except Exception as e:
            print(f"⚠️ STOI error on channel {ch}: {e}")

    return float(np.mean(scores)) if scores else np.nan



# Example usage
#ref_file = 'MC1.wav'
#deg_file = 'MCE_output_dereverberated.wav'
#sr = 16000  # Sample rate
#avg_stoi_score = calculate_stoi_multichannel(ref_file, deg_file, sr)
#print(f'Average STOI Score: {avg_stoi_score}')

