from scipy.io.wavfile import write
from moviepy.editor import VideoFileClip
import os
from torch.utils.data import Dataset
import torchaudio
import os
import pandas as pd
import numpy as np
# import ffmpeg

torchaudio.set_audio_backend("sox_io")  # Modern and recommended backend

# def extract_audio(input_video_path, output_audio_path):
#     (
#         ffmpeg
#         .input(input_video_path)
#         .output(output_audio_path, **{'q:a': 0})
#         .run()
#     )

def extract_audio(input_video_path, output_audio_path):
    # Ensure that the output directory exists
    output_dir = os.path.dirname(output_audio_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the video file
    video = VideoFileClip(input_video_path)

    # Write audio directly to a file
    video.audio.write_audiofile(
        output_audio_path, fps=44100, verbose=False, logger=None)

    # Close the video clip
    video.close()

class MELDDataset(Dataset):
    """
    A simple Dataset for IEMOCAP-like data, where each entry has:
      - audio_path: Path to the .wav file
      - transcript: The text transcript
      - label: The emotion label (int)
    """

    def __init__(self, csv_file, root_dir='./data', split_type='train', transform=None):
        """
        :param csv_file: Path to a CSV file (or txt) listing [audio_path, transcript, label].
        :param transform: Optional audio transform (torchaudio transforms or custom).
        """
        self.samples = []
        self.transform = transform

        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
          file_name = f'dia{row["Dialogue_ID"]}_utt{row["Utterance_ID"]}'
          video_path = f'{root_dir}/{split_type}_splits/{file_name}.mp4'
          audio_path = f'{root_dir}/train_splits/audio/{file_name}.wav'
          
          # if not os.path.exists(audio_path):
          #   extract_audio(video_path, audio_path)
          
          self.samples.append((
            audio_path,
            row["Utterance"], 
            row["Emotion"],
            row['Sentiment']
          ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, transcript, emotion, sentiment = self.samples[idx]

        waveform, sample_rate = torchaudio.load(audio_path)
        if self.transform:
            waveform = self.transform(waveform)

        # Return raw waveform, transcript string, and label
        return waveform, sample_rate, transcript, emotion, sentiment
