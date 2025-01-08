
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

class MELDConversationDataset(Dataset):
    def __init__(self, csv_file, root_dir='./data', mode="train"):
        """
        We'll store a list of (dialog_id, [list_of_utterance_dicts]).
        Each utterance_dict might contain:
          {
            "fbank_path": str,
            "transcript": str,
            "emotion": int,
            "sentiment": int
          }
        """
        df = pd.read_csv(f'{root_dir}/{csv_file}')

        self.emotion_class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        self.sentiment_class_counts = {0: 0, 1: 0, 2: 0}
        
        self.dialogues = {}  # key: dialogue_id, value: list of utterance dicts
        for _, row in df.iterrows():
            dia_id = row["Dialogue_ID"]
            utt_id = row["Utterance_ID"]
            
            fbank_path = f'../{mode}_fbank/dia{dia_id}_utt{utt_id}.npy'

            emotion = row["Emotion"]
            emotion = emotion.lower()
            emotion_int = self.emotion_to_int(emotion)
            sentiment = row["Sentiment"]
            sentiment = sentiment.lower()
            sentiment_int = self.sentiment_to_int(sentiment)

            self.emotion_class_counts[emotion_int] += 1
            self.sentiment_class_counts[sentiment_int] += 1
            
            utter_dict = {
                "fbank_path": fbank_path,
                "transcript": row["Utterance"],
                "emotion": emotion_int,
                "sentiment": sentiment_int
            }
            
            if dia_id not in self.dialogues:
                self.dialogues[dia_id] = []
            self.dialogues[dia_id].append(utter_dict)
        
        # Convert to list of (dialog_id, list_of_utterances)
        self.dialogues = [(k, sorted(v, key=lambda x: x["fbank_path"])) for k, v in self.dialogues.items()]
        # The sorting step ensures the utterances are in ascending order of utt_id if needed.

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
        return len(self.dialogues)
    
    def __getitem__(self, idx):
        dialog_id, utterances = self.dialogues[idx]
        
        # For each utterance, load filterbanks, transcript, emotion, sentiment
        fbank_list = []
        text_list = []
        emotion_list = []
        sentiment_list = []
        
        for utt in utterances:
            fbank = np.load(utt["fbank_path"])          # shape e.g. (T, fbank_dim)
            fbank_list.append(fbank)
            
            text_list.append(utt["transcript"])
            emotion_list.append(utt["emotion"])         # or mapped to int
            sentiment_list.append(utt["sentiment"])     # or mapped to int
        
        return {
            "dialog_id": dialog_id,
            "fbank_list": fbank_list,
            "text_list": text_list,
            "emotion_list": emotion_list,
            "sentiment_list": sentiment_list
        }



def meld_collate_fn(batch):
    # batch is a list of conversation dicts (one per item in dataset)
    # We can combine them into a single batch, 
    # but each conversation may have different # of utterances.
    
    dialog_ids = []
    fbank_lists = []
    text_lists = []
    emotion_lists = []
    sentiment_lists = []
    
    for conv in batch:
        dialog_ids.append(conv["dialog_id"])
        
        # Convert fbank_list (list of numpy arrays) to tensors and pad
        fbank_tensors = [torch.tensor(fbank) for fbank in conv["fbank_list"]]
        fbank_padded = pad_sequence(fbank_tensors, batch_first=True)  # Pad along the time dim (T)
        fbank_lists.append(fbank_padded)
        
        text_lists.append(conv["text_list"])
        emotion_lists.append(torch.tensor(conv["emotion_list"]))
        sentiment_lists.append(torch.tensor(conv["sentiment_list"]))
    
    # Return them "as is", or do further padding if needed.
    return {
        "dialog_ids": dialog_ids,
        "fbank_lists": fbank_lists,
        "text_lists": text_lists,
        "emotion_lists": emotion_lists,
        "sentiment_lists": sentiment_lists
    }
