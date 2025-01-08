from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd


class MELDDataset(Dataset):
    def __init__(self, csv_file, root_dir='./data', mode="train", transform=None, target_sr=16000):
        self.samples = []
        self.transform = transform
        self.target_sr = target_sr

        df = pd.read_csv(f'{root_dir}/{csv_file}')
        for _, row in df.iterrows():
            file_name = f'dia{row["Dialogue_ID"]}_utt{row["Utterance_ID"]}'
            video_path = f'{root_dir}/video/{file_name}.mp4'
            audio_path = f'{root_dir}/audio/{file_name}.wav'
            npy_path   = f'{mode}_fbank/{file_name}.npy'

            self.samples.append((
                npy_path,
                row["Utterance"], 
                row["Emotion"],
                row["Sentiment"]
            ))

    def get_emotions_dicts(self):
        labels = list(set([sample[2] for sample in self.samples]))
        int_to_str = {idx: label for idx, label in enumerate(labels)}
        str_to_int = {emotion: idx for idx, emotion in int_to_str.items()}
        return int_to_str, str_to_int

    def get_sentiments_dicts(self):
        labels = list(set([sample[3] for sample in self.samples]))
        int_to_str = {idx: label for idx, label in enumerate(labels)}
        str_to_int = {sentiment: idx for idx, sentiment in int_to_str.items()}
        return int_to_str, str_to_int

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, transcript, emotion, sentiment = self.samples[idx]
        emotion_to_int = self.get_emotions_dicts()[1]
        sentiment_to_int = self.get_sentiments_dicts()[1]

        # Load the resampled audio from .npy
        audio_array = np.load(npy_path)

        if self.transform:
            audio_array = self.transform(audio_array)

        audio_tensor = torch.tensor(audio_array, dtype=torch.float)

        return (
            audio_tensor,         # [T]
            transcript,
            emotion_to_int[emotion],
            sentiment_to_int[sentiment]
        )