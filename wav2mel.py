import torch
import os
import torchaudio
import numpy as np
from os.path import join
from tqdm import tqdm

def wav2mel(filename, filename2=None, mix_lambda=-1, target_length=1024):
    """
    Extract a mel spectrogram from .wav audio. If filename2 is provided and mix_lambda >= 0, 
    performs mixup between two waveforms. Returns a tensor of shape (time, n_mels).
    
    :param filename: Path to first waveform.
    :param filename2: Path to second waveform (for mixup), or None.
    :param mix_lambda: Weight for mixup (0 <= mix_lambda <= 1). If < 0, no mixup is done.
    :param target_length: Number of frames to pad/crop the resulting spectrogram.
    
    :return: A torch.FloatTensor of shape (target_length, n_mels).
    """
    
    # 1. Load audio (single or mixup)
    if filename2 is None or mix_lambda < 0:
        # No mixup
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

    
    # 2. Define mel spectrogram transform
    sample_rate = 16000
    # Increase n_fft for finer frequency resolution
    n_fft = 2048  
    hop_length = 512
    win_length = 2048

    n_mels = 128

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window_fn=torch.hann_window,
        n_mels=n_mels,
        f_min=0.0,
        f_max=sample_rate // 2  # or something smaller if your data doesn't have energy in very high freqs
    )
    
    # 3. Convert waveform -> mel spectrogram
    #    Expect shape = (1, time), so mel_spectrogram = (1, n_mels, frames)
    mel_spectrogram = mel_transform(waveform)
    
    # 4. Convert to decibel scale (optional but common)
    mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
    # Now shape = (1, n_mels, frames)
    
    # 5. Rearrange shape to (frames, n_mels)
    mel_spectrogram_db = mel_spectrogram_db.squeeze(0).transpose(0, 1)
    # shape now (frames, n_mels)
    
    # 6. Pad or truncate to target_length
    n_frames = mel_spectrogram_db.shape[0]
    p = target_length - n_frames

    if p > 0:
        # pad
        padding = torch.nn.ZeroPad2d((0, 0, 0, p))  # left=0, right=0, top=0, bottom=p
        mel_spectrogram_db = padding(mel_spectrogram_db)
    elif p < 0:
        # truncate
        mel_spectrogram_db = mel_spectrogram_db[:target_length, :]

    return mel_spectrogram_db

if __name__ == "__main__":
    root_dir = "./"
    sets = ["meld-train-muse/audio", "meld-dev-muse/audio", "meld-test-muse/audio"]
    for i, source_set in enumerate(sets):
        allwavs = [wav for wav in os.listdir(join(root_dir, source_set)) if wav.endswith(".wav")]
        for wav in tqdm(allwavs):
            wav_path = join(root_dir, source_set, wav)

            # Extract mel spectrogram
            mel_spec = wav2mel(wav_path, None, -1, 1024)

            # Decide where to save
            if i == 0:
                save_path = join(root_dir, "train_mel", wav.replace(".wav", ".npy"))
            elif i == 1:
                save_path = join(root_dir, "dev_mel", wav.replace(".wav", ".npy"))
            else:
                save_path = join(root_dir, "test_mel", wav.replace(".wav", ".npy"))
            
            # Save mel spectrogram to .npy
            np.save(save_path, mel_spec.numpy(), allow_pickle=True)
