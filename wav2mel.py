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
    
    
    if filename2 is None or mix_lambda < 0:
        
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

    
    
    sample_rate = 16000
    
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
        f_max=sample_rate // 2  
    )
    
    
    
    mel_spectrogram = mel_transform(waveform)
    
    
    mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
    
    
    
    mel_spectrogram_db = mel_spectrogram_db.squeeze(0).transpose(0, 1)
    
    
    
    n_frames = mel_spectrogram_db.shape[0]
    p = target_length - n_frames

    if p > 0:
        
        padding = torch.nn.ZeroPad2d((0, 0, 0, p))  
        mel_spectrogram_db = padding(mel_spectrogram_db)
    elif p < 0:
        
        mel_spectrogram_db = mel_spectrogram_db[:target_length, :]

    return mel_spectrogram_db

if __name__ == "__main__":
    root_dir = "./"
    sets = ["meld-train-muse/audio", "meld-dev-muse/audio", "meld-test-muse/audio"]
    for i, source_set in enumerate(sets):
        allwavs = [wav for wav in os.listdir(join(root_dir, source_set)) if wav.endswith(".wav")]
        for wav in tqdm(allwavs):
            wav_path = join(root_dir, source_set, wav)

            
            mel_spec = wav2mel(wav_path, None, -1, 1024)

            
            if i == 0:
                save_path = join(root_dir, "train_mel", wav.replace(".wav", ".npy"))
            elif i == 1:
                save_path = join(root_dir, "dev_mel", wav.replace(".wav", ".npy"))
            else:
                save_path = join(root_dir, "test_mel", wav.replace(".wav", ".npy"))
            
            
            np.save(save_path, mel_spec.numpy(), allow_pickle=True)
