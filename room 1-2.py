"""Room acoustics simulation pipeline (pyroomacoustics).

This module generates reverberant multichannel mixtures from clean and environmental
acoustics source files. It defines amplitude/distance compensation, signal length
matching, and room simulation processing functions.

The main script parameters define room RT60, dimensions, and source/microphone setup.
"""

from random import randint
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import fftconvolve
from pydub import AudioSegment
import os
import time

start_time = time.process_time()

# reverberation time and dimensions of the room
rt = 0.5
snr = 0
rtfolder = int(rt * 1000)
room_dim = [7.0, 6.5, 3.4]  # meters
rt60_tgt = rt  # seconds



def adjust_amplitude(audio: np.ndarray, distance: float, fs: int) -> np.ndarray:
    """Apply distance-based attenuation and air absorption across spectrum.

    Uses a simple frequency-dependent attenuation model in the frequency domain.

    Args:
        audio: Input time-domain signal
        distance: Source-to-microphone distance in meters (m)
        fs: Sample rate in Hz

    Returns:
        Adjusted audio with attenuation and air absorption applied
    """
    c = 343  # Speed of sound in m/s
    alpha = 0.005  # Air absorption coefficient (can be adjusted)
    freqs = np.fft.rfftfreq(len(audio), d=1/fs)
    attenuation = np.exp(-alpha * distance * freqs / c)
    audio_fft = np.fft.rfft(audio)
    audio_fft_attenuated = audio_fft * attenuation / distance
    adjusted_audio = np.fft.irfft(audio_fft_attenuated)
    return adjusted_audio

def match_length(source: np.ndarray, target_length: int) -> np.ndarray:
    """Trim or zero-pad a 1D signal to exactly target_length samples."""
    if len(source) > target_length:
        result = source[:target_length]
    else:
        result = np.pad(source, (0, target_length - len(source)), 'constant')
    
    # Print the output length in terms of seconds
    
    #print(f"Output length: {output_length_seconds} seconds")
    
    return result

def process_files(clean_folder: str, env_folder: str, output_folder: str) -> None:
    """Create noisy multichannel reverberant mixtures and save to output folder.

    Args:
        clean_folder: Path to clean speech WAV files
        env_folder: Path to environment/background WAV files (recursively scanned)
        output_folder: Path to save generated noisy microphone mixture files
    """
    clean_files = [f for f in os.listdir(clean_folder) if f.endswith(".wav")]
    
    if not clean_files:
        print(f"Error: No .wav files found in the clean folder: {clean_folder}")
        return

    env_files = []
    for root, _, files in os.walk(env_folder):
        for file in files:
            if file.endswith(".wav"):
                env_files.append(os.path.join(root, file))

    if not env_files:
        print(f"Error: No .wav files found in the environment folder: {env_folder}")
        return

    max = 300
    num = 0
    for clean_file in clean_files:
        if num >= len(env_files):
            num3 = randint(0, len(env_files) - 1)
        else:
            num3 = num
        fs2, audio = wavfile.read(os.path.join(clean_folder, clean_files[num]))
        _, audio3 = wavfile.read(os.path.join(env_folder, env_files[num3]))
        
        # Print the files being mixed
        #print(f"Mixing clean file: {clean_files[num]} with environment file: {env_files[num3]}")
        
        num += 1
        if num > max:
            break
        if audio.ndim > 1:
            audio = audio[:, 0]
        if audio3.ndim > 1:
            audio3 = audio3[:, 0]

        audio3 = match_length(audio3, len(audio))
        e_absorption, _ = pra.inverse_sabine(rt60_tgt, room_dim)

        room = pra.ShoeBox(
            room_dim, fs=fs2, materials=pra.Material(e_absorption), max_order=7
        )
        
        room.add_source([1.8 + source_distance, 1.10, 1.70], signal=audio, delay=0.0)
        room.add_source([1.6, 2.10, 1.70], signal=audio3, delay=0.0)

        # Define the locations of the microphones
        mic_locs = np.c_[
            [1.4, 1.10, 1.5], [1.6, 1.10, 1.5], [1.8, 1.10, 1.5],  # mic 1, mic 2, mic 3
            [1.4, 1.10, 1.7], [1.6, 1.10, 1.7], [1.8, 1.10, 1.7]
        ]

        # Finally place the array in the room
        room.add_microphone_array(mic_locs)

        # Run the simulation (this will also build the RIR automatically)
        sig = room.simulate()

        # Match the length of the simulated microphone signals to the length of the clean speech file
        matched_signals = np.zeros((room.mic_array.signals.shape[0], len(audio)))
        for i in range(room.mic_array.signals.shape[0]):
            matched_signals[i] = match_length(room.mic_array.signals[i], len(audio))
        room.mic_array.signals = matched_signals

        # Ensure the length of the signals matches the length of audio
        for i in range(room.mic_array.signals.shape[0]):
            room.mic_array.signals[i] = match_length(room.mic_array.signals[i], len(audio))

        # Save the simulated microphone signals to a wav file
        output_file = os.path.join(output_folder, f"noisy_speech_{num}.wav")

        room.mic_array.to_wav(
            output_file,
            norm=True,
            bitdepth=np.int16,
        )

        # Read and print the sampling rate of the output file
        #fs, _ = wavfile.read(output_file)
        #print(f"Sampling rate: {fs} Hz")

        #output_snr = calculate_output_snr(output_file)
        #print(f"{output_snr}")

    rt60 = room.measure_rt60()
    print("The desired RT60 was {}".format(rt60_tgt))
    print("The measured RT60 is {}".format(rt60[1, 0]))

source_distance = 5.1
clean_folder = "h:/malay-clean"
env_folder = "H:/Demand-rest/"  # This is the root folder

output_folder = f"h:/Malay-Demand/rest/{source_distance}"
process_files(clean_folder, env_folder, output_folder)
print(output_folder)
end_time = time.process_time()
print(f"CPU time used: {(end_time - start_time)/60} minutes")