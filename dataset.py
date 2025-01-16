from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd


class MELDDataset(Dataset):
    def __init__(self, csv_file, root_dir='./data', mode="train", transform=None, target_sr=16000):
        self.samples = []
        self.transform = transform
        self.target_sr = target_sr

        self.emotion_class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        self.sentiment_class_counts = {0: 0, 1: 0, 2: 0}

        df = pd.read_csv(f'{root_dir}/{csv_file}')
        for _, row in df.iterrows():
            file_name = f'dia{row["Dialogue_ID"]}_utt{row["Utterance_ID"]}'
            video_path = f'{root_dir}/video/{file_name}.mp4'
            audio_path = f'{root_dir}/audio/{file_name}.wav'
            npy_path   = f'{mode}_fbank/{file_name}.npy'

            emotion_int = self.emotion_to_int(row["Emotion"])
            sentiment_int = self.sentiment_to_int(row["Sentiment"])

            self.emotion_class_counts[emotion_int] += 1
            self.sentiment_class_counts[sentiment_int] += 1

            self.samples.append((
                npy_path,
                row["Utterance"], 
                emotion_int,
                sentiment_int
            ))

    def emotion_to_int(self, str):
        str_to_int = {"neutral": 0, "joy": 1, "surprise": 2, "anger": 3, "sadness": 4, "fear": 5, "disgust": 6}
        return str_to_int[str]

    def emotion_to_str(self, int):
        int_to_str = {0: "neutral", 1: "joy", 2: "surprise", 3: "anger", 4: "sadness", 5: "fear", 6: "disgust"}
        return int_to_str[int]

    def sentiment_to_int(self, str):
        str_to_int = {"neutral": 0, "positive": 1, "negative": 2}
        return str_to_int[str]
    
    def sentiment_to_str(self, int):
        int_to_str = {0: "neutral", 1: "positive", 2: "negative"}
        return int_to_str[int]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, transcript, emotion, sentiment = self.samples[idx]

        # Load the resampled audio from .npy
        audio_array = np.load(npy_path)

        if self.transform:
            audio_array = self.transform(audio_array)

        audio_tensor = torch.tensor(audio_array, dtype=torch.float)

        return (
            audio_tensor,
            transcript,
            emotion,
            sentiment
        )
