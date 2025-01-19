import torch
import torchaudio
from tqdm import tqdm
import os
from os.path import join as join
import numpy as np

def wav2fbank(filename, filename2=None, mix_lambda=-1):
        
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, 
                                                      use_energy=False, window_type='hanning', 
                                                      num_mel_bins=128, dither=0.0, frame_shift=10)
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            print('there is a loading error')

        target_length = 1024
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

if __name__=="__main__":
    root_dir = "./"
    sets = ["meld-train-muse/audio", "meld-dev-muse/audio", "meld-test-muse/audio"]
    for i, source_set in enumerate(sets):
        allwavs = [wav for wav in os.listdir(join(root_dir, source_set)) if wav.endswith(".wav")]
        for wav in tqdm(allwavs):
            wav_path = join(root_dir, source_set, wav)

            fbank = wav2fbank(wav_path, None, 0)

            if i == 0:
                save_path = join(root_dir, "train_fbank", wav.replace(".wav", ".npy"))
            elif i == 1:
                save_path = join(root_dir, "dev_fbank", wav.replace(".wav", ".npy"))
            else:
                save_path = join(root_dir, "test_fbank", wav.replace(".wav", ".npy"))
            
            np.save(save_path, fbank, allow_pickle = True)