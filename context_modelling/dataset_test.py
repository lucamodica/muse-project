from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T
from concurrent.futures import ThreadPoolExecutor
import os

class AudioTransformPipeline:
    """
    Custom pipeline for chaining multiple audio transformations with dynamic resampling.
    """

    def __init__(self, target_sample_rate=16000, n_mels=128):
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels

    def __call__(self, waveform, orig_sample_rate):
        
        if orig_sample_rate != self.target_sample_rate:
            resampler = T.Resample(
                orig_freq=orig_sample_rate,
                new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)
        
        return waveform


def meld_collate_fn(batch):
    """
    Custom collate function to handle variable numbers of utterances per dialogue.
    """
    dialog_ids = []
    audio_lists = []
    text_lists = []
    emotion_lists = []
    sentiment_lists = []

    for conv in batch:
        dialog_ids.append(conv["dialog_id"])

        
        fbank_tensors = [torch.tensor(fbank) for fbank in conv["audio_list"]]
        
        fbank_padded = pad_sequence(fbank_tensors, batch_first=True)
        audio_lists.append(fbank_padded)

        text_lists.append(conv["text_list"])
        emotion_lists.append(torch.tensor(conv["emotion_list"]))
        sentiment_lists.append(torch.tensor(conv["sentiment_list"]))

    return {
        "dialog_ids": dialog_ids,
        "audio_lists": audio_lists,
        "text_lists": text_lists,
        "emotion_lists": emotion_lists,
        "sentiment_lists": sentiment_lists
    }


class MELDConversationDataset(Dataset):
    """
    Dataset class for MELD conversations with optional torchaudio transforms 
    and Hugging Face audio feature extraction.

    Loads data from a CSV and organizes utterances by dialogue.
    """

    def __init__(
        self,
        csv_file,
        root_dir='./data',
        audio_processor=None,
        audio_transform=None,
        text_transform=None,
        sampling_rate=16000,
        target_length=1024,
        max_workers=16
    ):
        """
        :param csv_file: Path to the CSV file with conversation metadata (relative to root_dir).
        :param root_dir: Root directory containing audio files.
        :param audio_processor: A Hugging Face AudioProcessor (e.g., Whisper feature extractor).
        :param audio_transform: A custom torchaudio transform pipeline.
        :param text_transform: (Optional) Text transform (unused in this example).
        :param sampling_rate: Target sampling rate for audio.
        :param target_length: Number of time frames to pad or truncate to (currently not used).
        :param max_workers: Maximum number of threads to use for parallel processing.
        """
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.audio_processor = audio_processor
        self.audio_transform = audio_transform
        self.text_transform = text_transform
        self.sampling_rate = sampling_rate
        self.target_length = target_length

        
        df = pd.read_csv(os.path.join(root_dir, csv_file))
        df = df.sort_values(by=['Dialogue_ID', 'Utterance_ID'])

        
        self.emotion_class_counts = {i: 0 for i in range(7)}
        self.sentiment_class_counts = {i: 0 for i in range(3)}
        self.max_utterance_size = 0

        
        
        self.dialogues = {}

        
        tasks = []
        prev_dia_id = None
        utt_count = 0

        for _, row in df.iterrows():
            dia_id = row["Dialogue_ID"]
            utt_id = row["Utterance_ID"]

            
            if prev_dia_id == dia_id:
                utt_count += 1
            else:
                if utt_count > self.max_utterance_size:
                    self.max_utterance_size = utt_count
                utt_count = 1

            audio_file = os.path.join(root_dir, f'audio/dia{dia_id}_utt{utt_id}.wav')
            tasks.append((dia_id, utt_id, audio_file, row))
            prev_dia_id = dia_id

        
        max_workers = min(max_workers, len(tasks)) if tasks else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self._process_utterance, tasks))

        
        for dia_id, utt_data in results:
            if dia_id not in self.dialogues:
                self.dialogues[dia_id] = []
            self.dialogues[dia_id].append(utt_data)

        
        self.dialogues = [(k, v) for k, v in self.dialogues.items()]

    def _process_utterance(self, task):
        """
        Internal method to process a single utterance (audio + metadata).
        Designed for parallel execution via ThreadPoolExecutor.

        :param task: (dia_id, utt_id, audio_file, row)
        :return: (dialogue_id, utterance_dict)
        """
        dia_id, utt_id, audio_file, row = task

        
        waveform, sr = torchaudio.load(audio_file)

        
        if waveform.size(0) == 2:
            waveform = waveform.mean(dim=0, keepdim=True)

        
        if self.audio_transform:
            waveform = self.audio_transform(waveform, sr)

        
        audio_features = waveform  
        if self.audio_processor:
            inputs = self.audio_processor(
                waveform.squeeze(0),
                sampling_rate=self.sampling_rate,
                return_tensors="pt"
            )
            if "input_values" in inputs:
                
                audio_features = inputs.input_values[0]
            elif "input_features" in inputs:
                
                audio_features = inputs.input_features[0]
            else:
                raise ValueError("Unsupported processor output format.")

        
        emotion_int = self.emotion_to_int(row["Emotion"].lower())
        sentiment_int = self.sentiment_to_int(row["Sentiment"].lower())

        
        self.emotion_class_counts[emotion_int] += 1
        self.sentiment_class_counts[sentiment_int] += 1

        utter_dict = {
            "audio_features": audio_features,
            "transcript": row["Utterance"],
            "emotion": emotion_int,
            "sentiment": sentiment_int
        }

        return dia_id, utter_dict

    def __len__(self):
        """Return the number of unique dialogues."""
        return len(self.dialogues)

    def __getitem__(self, idx):
        """
        Return a dictionary containing:
          - dialog_id
          - audio_list
          - text_list
          - emotion_list
          - sentiment_list
        """
        dialog_id, utterances = self.dialogues[idx]

        audio_list = []
        text_list = []
        emotion_list = []
        sentiment_list = []

        for utt in utterances:
            audio_list.append(utt["audio_features"])
            text_list.append(utt["transcript"])
            emotion_list.append(utt["emotion"])
            sentiment_list.append(utt["sentiment"])

        return {
            "dialog_id": dialog_id,
            "audio_list": audio_list,
            "text_list": text_list,
            "emotion_list": emotion_list,
            "sentiment_list": sentiment_list
        }

    @staticmethod
    def emotion_to_int(emotion):
        """
        Map emotion labels to integers.
        """
        str_to_int = {
            "neutral": 0, "joy": 1, "surprise": 2,
            "anger": 3, "sadness": 4, "fear": 5, "disgust": 6
        }
        return str_to_int[emotion]

    @staticmethod
    def sentiment_to_int(sentiment):
        """
        Map sentiment labels to integers.
        """
        str_to_int = {"neutral": 0, "positive": 1, "negative": 2}
        return str_to_int[sentiment]
