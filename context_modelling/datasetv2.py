from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T
from concurrent.futures import ThreadPoolExecutor

import torch.multiprocessing as mp


class MELDConversationDataset(Dataset):
    def __init__(self, csv_file, 
        root_dir='./data',
        audio_processor=None,
        audio_transform=None,
        text_transform=None,
        sampling_rate=16000, 
        target_length=1024):
        """
        Dataset class for MELD conversations with support for torchaudio transforms.

        :param csv_file: Path to the CSV file with conversation metadata.
        :param root_dir: Root directory containing audio files (default: './data').
        :param audio_transform: A torchaudio transform to apply to the waveform (default: None).
                                If None, the raw waveform is used.
        :param target_length: Number of time frames to pad or truncate to (default: 1024).
        """
        df = pd.read_csv(f'{root_dir}/{csv_file}')

        
        df = df.sort_values(by=['Dialogue_ID', 'Utterance_ID'])

        
        self.emotion_class_counts = {i: 0 for i in range(7)}
        self.sentiment_class_counts = {i: 0 for i in range(3)}
        self.max_utterance_size = 0

        self.dialogues = {}  
        self.audio_transform = audio_transform
        self.text_transform = text_transform
        self.target_length = target_length 
        self.audio_processor = audio_processor
        self.sampling_rate = sampling_rate
        
        prev_dia_id = None
        utt_count = 0

        
        tasks = []

        for _, row in df.iterrows():
            dia_id = row["Dialogue_ID"]
            utt_id = row["Utterance_ID"]

            
            if prev_dia_id == dia_id:
                utt_count += 1
            else:
                if utt_count > self.max_utterance_size:
                    self.max_utterance_size = utt_count
                utt_count = 1

            
            audio_file = f'{root_dir}/audio/dia{dia_id}_utt{utt_id}.wav'

            
            tasks.append((dia_id, utt_id, audio_file, row))
            prev_dia_id = dia_id

        
        
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(self.process_utterance, tasks)

        
        for dia_id, utt_data in results:
            if dia_id not in self.dialogues:
                self.dialogues[dia_id] = []
            self.dialogues[dia_id].append(utt_data)

        
        self.dialogues = [(k, v) for k, v in self.dialogues.items()]
        
    def process_utterance(self, task):
        """
        Process a single utterance. This function is designed to be used in parallel processing.
        
        :param task: A tuple containing dia_id, utt_id, audio_file, and row metadata.
        :return: A tuple (dialogue_id, utterance_data).
        """
        dia_id, utt_id, audio_file, row = task

        
        waveform, sr = torchaudio.load(audio_file)
        
        
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        
        if self.audio_transform:
            audio_features = self.audio_transform(waveform, sr)
        else:
            audio_features = waveform
            
        
        if self.audio_processor:
            
            inputs = self.audio_processor(
                waveform.squeeze(0),  
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
            )
            if "input_values" in inputs:
                
                audio_features = inputs.input_values[0]
            elif "input_features" in inputs:
                
                audio_features = inputs.input_features[0]
            else:
                raise ValueError("Unsupported processor output format.")
        else:
            
            audio_features = waveform

        
        emotion = row["Emotion"].lower()
        emotion_int = self.emotion_to_int(emotion)
        sentiment = row["Sentiment"].lower()
        sentiment_int = self.sentiment_to_int(sentiment)

        
        self.emotion_class_counts[emotion_int] += 1
        self.sentiment_class_counts[sentiment_int] += 1

        utter_dict = {
            "audio_features": audio_features,
            "transcript": row["Utterance"],
            "emotion": emotion_int,
            "sentiment": sentiment_int
        }

        return dia_id, utter_dict

    def emotion_to_int(self, emotion):
        """
        Map emotion labels to integers.
        """
        str_to_int = {"neutral": 0, "joy": 1, "surprise": 2,
                      "anger": 3, "sadness": 4, "fear": 5, "disgust": 6}
        return str_to_int[emotion]

    def emotion_to_str(self, emotion_int):
        """
        Map integers to emotion labels.
        """
        int_to_str = {0: "neutral", 1: "joy", 2: "surprise",
                      3: "anger", 4: "sadness", 5: "fear", 6: "disgust"}
        return int_to_str[emotion_int]

    def sentiment_to_int(self, sentiment):
        """
        Map sentiment labels to integers.
        """
        str_to_int = {"neutral": 0, "positive": 1, "negative": 2}
        return str_to_int[sentiment]

    def sentiment_to_str(self, sentiment_int):
        """
        Map integers to sentiment labels.
        """
        int_to_str = {0: "neutral", 1: "positive", 2: "negative"}
        return int_to_str[sentiment_int]

    def __len__(self):
        """
        Return the number of dialogues.
        """
        return len(self.dialogues)

    def __getitem__(self, idx):
        """
        Get a dialogue by index.
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
        
def meld_collate_fn(batch):
    
    
    

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
    
class AudioTransformPipeline:
    """
    Custom pipeline for chaining multiple audio transformations with dynamic resampling.
    """

    def __init__(self, target_sample_rate=16000, n_mels=128):
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels

    def __call__(self, waveform, orig_sample_rate):
        audio = waveform
        
        
        if orig_sample_rate != self.target_sample_rate:
            resampler = T.Resample(
                orig_freq=orig_sample_rate, new_freq=self.target_sample_rate)
            audio = resampler(audio)
        
        return audio