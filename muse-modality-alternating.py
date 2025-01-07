


import tarfile
import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, Wav2Vec2Processor, Wav2Vec2Model
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
from pprint import pprint
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import warnings

from backbone_alternating import resnet18



# ## Data

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

            # Precompute & save .npy if doesn't exist
            # if not os.path.exists(npy_path):
            #     audio_array, sr = librosa.load(audio_path, sr=None)
            #     if sr != self.target_sr:
            #         # Use named arguments for librosa.resample in case of librosa 0.10+
            #         audio_array = librosa.resample(
            #             y=audio_array,
            #             orig_sr=sr,
            #             target_sr=self.target_sr
            #         )
            #     # Create folder if needed
            #     os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            #     np.save(npy_path, audio_array)

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
            self.target_sr,       # already at target_sr
            transcript,
            emotion_to_int[emotion],
            sentiment_to_int[sentiment]
        )

def collate_fn(batch):
    """
    Custom collate function to handle variable-length audio inputs.
    Pads audio waveforms in the batch to the length of the longest waveform.
    """
    audio_arrays, sample_rates, texts, emotions, sentiments = zip(*batch)

    # Pad audio waveforms to the maximum length in the batch
    audio_arrays = [audio.clone().detach() for audio in audio_arrays]
    audio_arrays_padded = pad_sequence(audio_arrays, batch_first=True, padding_value=0)

    # Convert labels to tensors
    emotions = torch.tensor(emotions)
    sentiments = torch.tensor(sentiments)

    return audio_arrays_padded, sample_rates, texts, emotions, sentiments

class AudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", 
                 target_sr=16000, 
                 fine_tune=False, 
                 unfreeze_last_n=2,  # Number of last layers to unfreeze
                 device='cuda'):
        super(AudioEncoder, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self.target_sr = target_sr

        # Freeze entire model
        for param in self.model.parameters():
            param.requires_grad = False

        if fine_tune:
            # Unfreeze only the last `unfreeze_last_n` encoder layers
            total_layers = len(self.model.encoder.layers)
            for layer_idx in range(total_layers - unfreeze_last_n, total_layers):
                for param in self.model.encoder.layers[layer_idx].parameters():
                    param.requires_grad = True

    def forward(self, waveforms):
        """
        :param waveforms: Tensor of shape [B, T] (already at self.target_sr)
        :return: A tensor of shape [B, hidden_dim] (audio embeddings for the batch)
        """
        # Ensure the waveforms are on the correct device
        waveforms = waveforms.to(self.model.device)
        
        # Prepare inputs for Wav2Vec2Processor
        inputs = self.processor(
            waveforms.cpu(),
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True
        ).input_values.squeeze((0, 1)).to(self.model.device)
    
        # Forward pass
        with torch.no_grad() if not self.training or not any(
            p.requires_grad for p in self.model.parameters()
        ) else torch.enable_grad():
            outputs = self.model(inputs)
            hidden_states = outputs.last_hidden_state  # shape [B, T, D]
    
        # Average pooling
        audio_emb = hidden_states.mean(dim=1)  # shape [B, D]
    
        return audio_emb

        
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class TextEncoder(nn.Module):
    """
    Encodes text using a pretrained BERT model from Hugging Face.
    """

    def __init__(self,
                 model_name="bert-base-uncased",
                 fine_tune=False,
                 unfreeze_last_n_layers=2,
                 device='cuda'):
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)

        # If we're not fine-tuning at all, freeze everything.
        # Otherwise, freeze everything first, then selectively unfreeze layers.
        for param in self.model.parameters():
            param.requires_grad = False

        if fine_tune and unfreeze_last_n_layers > 0:
            # For BERT-base, there are 12 encoder layers: encoder.layer[0] ... encoder.layer[11].
            # Unfreeze the last N layers:
            total_layers = len(self.model.encoder.layer)
            for layer_idx in range(total_layers - unfreeze_last_n_layers, total_layers):
                for param in self.model.encoder.layer[layer_idx].parameters():
                    param.requires_grad = True

            # Optionally unfreeze the pooler layer if you use it
            for param in self.model.pooler.parameters():
                param.requires_grad = True

    def forward(self, text_list):
        """
        :param text_list: A list of strings (or a single string) to encode.
        :return: A tensor of shape [batch_size, hidden_dim] with text embeddings
        """
        device = self.model.device

        # If a single string is passed, wrap it into a list
        if isinstance(text_list, str):
            text_list = [text_list]

        encodings = self.tokenizer(
            text_list, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)

        # If all parameters are frozen (no grad), then no_grad() is fine.
        # But if some layers are unfrozen, we want torch.enable_grad()
        # so that backprop can proceed for those layers.
        use_grad = any(p.requires_grad for p in self.model.parameters())
        with torch.enable_grad() if use_grad else torch.no_grad():
            outputs = self.model(
                input_ids=encodings.input_ids,
                attention_mask=encodings.attention_mask
            )

        # outputs.last_hidden_state -> shape [batch_size, seq_len, hidden_dim]
        # Typically we use the [CLS] token embedding as a single representation
        cls_emb = outputs.last_hidden_state[:, 0, :]  # shape [B, hidden_dim]
        return cls_emb


from torch.profiler import profile, record_function, ProfilerActivity
import time

class MultimodalClassifier(nn.Module):
    def __init__(self,
                 fusion_dim=768, 
                 audio_model_name="facebook/wav2vec2-base",
                 text_model_name="bert-base-uncased",
                 audio_fine_tune=False,
                 text_fine_tune=False,
                 unfreeze_last_n_audio=2,
                 unfreeze_last_n_text=2,
                 hidden_dim=768,
                 num_emotions=7,
                 num_sentiments=3):
        """
        :param audio_model_name: e.g., "facebook/wav2vec2-base"
        :param text_model_name: e.g., "bert-base-uncased"
        :param audio_fine_tune: whether to fine-tune the audio encoder
        :param text_fine_tune: whether to fine-tune the text encoder
        :param hidden_dim: dimension of the embeddings (depends on the model)
        :param num_classes: number of emotion classes (e.g., 4 for IEMOCAP: angry, happy, sad, neutral)
        """
        super(MultimodalClassifier, self).__init__()

        # Build encoders
        self.audio_model_name = audio_model_name
        self.text_model_name = text_model_name
        #self.audio_encoder = AudioEncoder(model_name=audio_model_name, fine_tune=audio_fine_tune, unfreeze_last_n=unfreeze_last_n_audio)
        self.audio_encoder = resnet18(modality='audio')
        self.text_encoder = TextEncoder(model_name=text_model_name, fine_tune=text_fine_tune, unfreeze_last_n_layers=unfreeze_last_n_text)

        # If Wav2Vec2 base has 768 dims and BERT base has 768 dims -> total is 1536
        # If you use average pooling / CLS, that might remain 768 for each

        self.e_shared_classifier = nn.Linear(fusion_dim, num_emotions)
        self.s_shared_classifier = nn.Linear(fusion_dim, num_sentiments)

    def forward(self, audio_array, text):
        """
        :param waveform: Tensor of shape [1, T] or [B, 1, T] with audio waveforms
        :param sample_rate: int or list of ints (for multiple samples)
        :param text: list of strings or a single string
        :return: logits of shape [B, num_classes]
        """

        audio_array = audio_array.unsqueeze(1)
        audio_emb = self.audio_encoder(audio_array)  # shape [1, D]
        audio_emb = F.adaptive_avg_pool2d(audio_emb, 1)
        audio_emb = torch.flatten(audio_emb, 1)

        text_emb = self.text_encoder(text)  # shape [B, D]

        return audio_emb, text_emb


def process_task(model, m, e_labels, s_labels, e_reg, s_reg, e_criterion, s_criterion, device='cuda'):

    softmax = nn.Softmax(dim=1)

    e_logits = model.e_shared_classifier(m)
    s_logits = model.s_shared_classifier(m)

    # --- Compute losses ---
    e_loss = e_criterion(e_logits, e_labels)
    s_loss = s_criterion(s_logits, s_labels)

    combined_loss = e_reg * e_loss + s_reg * s_loss

    # --- Predictions ---

    preds_e = torch.argmax(softmax(e_logits), dim=1)
    preds_s = torch.argmax(softmax(s_logits), dim=1)

    return combined_loss, e_loss, s_loss, preds_e, preds_s, e_logits, s_logits

def train_one_epoch(model, dataloader, optimizer, criterions,
                    emotion_reg=0.6, sentiment_reg=0.4, device='cuda'):
    model.train()

    softmax = nn.Softmax(dim=1)

    # tracked measures
    losses = {'combined_audio': 0.0, 'audio_emotion': 0.0, 'audio_sentiment': 0.0, 
              'combined_text': 0.0, 'text_emotion': 0.0, 'text_sentiment': 0.0, 
              'fused_combined': 0.0, 'fused_emotion': 0.0, 'fused_sentiment': 0.0}

    metrics = {'emotion': {'fused': [], 'audio': [], 'text': [], 'labels': []},
               'sentiment': {'fused': [], 'audio': [], 'text': [], 'labels': []}}

    # Wrap dataloader with tqdm
    loop = tqdm(dataloader, desc="Training", leave=False)
    
    for audio_arrays, sr, texts, emotion_labels, sentiment_labels in loop:
        # Move data to device
        audio_arrays = audio_arrays.to(device)
        emotion_labels = emotion_labels.to(device)
        sentiment_labels = sentiment_labels.to(device)

        optimizer.zero_grad()

        # Forward pass for audio and text
        a, t = model(audio_arrays, texts)

        # ---------------------------- AUDIO MODALITY TRAINING ----------------------------

        combined_audio_loss, e_audio_loss, s_audio_loss, audio_e_pred, audio_s_pred, audio_e_logits, audio_s_logits = process_task(
            model=model, 
            m=a, 
            e_labels=emotion_labels,
            s_labels=sentiment_labels,
            e_reg=emotion_reg,
            s_reg=sentiment_reg,
            e_criterion=criterions['emotion'],
            s_criterion=criterions['sentiment']
        )

        optimizer.zero_grad()
        combined_audio_loss.backward()
        optimizer.step()


        losses['combined_audio'] += combined_audio_loss.item()
        losses['audio_emotion'] += e_audio_loss.item()
        losses['audio_sentiment'] += s_audio_loss.item()
        metrics['emotion']['audio'].extend(audio_e_pred)
        metrics['sentiment']['audio'].extend(audio_s_pred)

        # ---------------------------- TEXT MODALITY TRAINING ----------------------------

        combined_text_loss, e_text_loss, s_text_loss, text_e_pred, text_s_pred, text_e_logits, text_s_logits = process_task(
            model=model, 
            m=t, 
            e_labels=emotion_labels,
            s_labels=sentiment_labels,
            e_reg=emotion_reg,
            s_reg=sentiment_reg,
            e_criterion=criterions['emotion'],
            s_criterion=criterions['sentiment']
        )

        optimizer.zero_grad()
        combined_text_loss.backward()
        optimizer.step()

        losses['combined_text'] += combined_text_loss.item()
        losses['text_emotion'] += e_text_loss.item()
        losses['text_sentiment'] += s_text_loss.item()
        metrics['emotion']['text'].extend(text_e_pred)
        metrics['sentiment']['text'].extend(text_s_pred)

        # ---------------------------- MODALITY FUSION ----------------------------
        # Combine the predictions
        e_logits = (audio_e_logits + text_e_logits) / 2
        s_logits = (audio_s_logits + text_s_logits) / 2

        # Compute the combined loss
        e_loss = criterions['emotion'](e_logits, emotion_labels)
        s_loss = criterions['sentiment'](s_logits, sentiment_labels)
        combined_loss = emotion_reg * e_loss + sentiment_reg * s_loss

        predictions_e = torch.argmax(softmax(e_logits), dim=1)
        predictions_s = torch.argmax(softmax(s_logits), dim=1)

        losses['fused_combined'] += combined_loss.item()
        losses['fused_emotion'] += e_loss.item()
        losses['fused_sentiment'] += s_loss.item()

        metrics['emotion']['fused'].extend(predictions_e)
        metrics['sentiment']['fused'].extend(predictions_s)

        # ---------------------------- END ----------------------------


        metrics['emotion']['labels'].extend(emotion_labels.cpu().numpy())
        metrics['sentiment']['labels'].extend(sentiment_labels.cpu().numpy())
    
    # Average losses
    for key in losses.keys():
        losses[key] /= len(dataloader)

    # compute metrics per modality 
    emotion_metrics = {
        'fused': compute_metrics(metrics['emotion']['labels'], metrics['emotion']['fused']),
        'audio': compute_metrics(metrics['emotion']['labels'], metrics['emotion']['audio']),
        'text': compute_metrics(metrics['emotion']['labels'], metrics['emotion']['text']),
    }
    sentiment_metrics = {
        'fused': compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['fused']),
        'audio': compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['audio']),
        'text': compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['text']),
    }
    
    return losses, {'emotion': emotion_metrics, 'sentiment': sentiment_metrics}


def validate_one_epoch(model, dataloader, criterions, emotion_reg, sentiment_reg, device='cuda'):
    model.eval()

    losses = {'combined_audio': 0.0, 'audio_emotion': 0.0, 'audio_sentiment': 0.0, 
              'combined_text': 0.0, 'text_emotion': 0.0, 'text_sentiment': 0.0, 
              'fused_combined': 0.0, 'fused_emotion': 0.0, 'fused_sentiment': 0.0}

    metrics = {'emotion': {'fused': [], 'audio': [], 'text': [], 'labels': []},
               'sentiment': {'fused': [], 'audio': [], 'text': [], 'labels': []}}

    with torch.no_grad():
        # Also wrap validation dataloader with tqdm
        loop = tqdm(dataloader, desc="Validation", leave=False)
        
        for audio_arrays, sr, texts, emotion_labels, sentiment_labels in loop:
            audio_arrays = audio_arrays.to(device)
            emotion_labels = emotion_labels.to(device)
            sentiment_labels = sentiment_labels.to(device)

            # Forward pass for audio and text
            a, t = model(audio_arrays, texts)

            # ---------------------------- AUDIO MODALITY VALIDATION ----------------------------
            combined_audio_loss, e_audio_loss, s_audio_loss, audio_e_pred, audio_s_pred, audio_e_logits, audio_s_logits = process_task(
                model=model, 
                m=a, 
                e_labels=emotion_labels,
                s_labels=sentiment_labels,
                e_reg=emotion_reg,
                s_reg=sentiment_reg,
                e_criterion=criterions['emotion'],
                s_criterion=criterions['sentiment']
            )

            losses['combined_audio'] += combined_audio_loss.item()
            losses['audio_emotion'] += e_audio_loss.item()
            losses['audio_sentiment'] += s_audio_loss.item()
            metrics['emotion']['audio'].extend(audio_e_pred)
            metrics['sentiment']['audio'].extend(audio_s_pred)

            # ---------------------------- TEXT MODALITY VALIDATION ----------------------------
            combined_text_loss, e_text_loss, s_text_loss, text_e_pred, text_s_pred, text_e_logits, text_s_logits = process_task(
                model=model, 
                m=t, 
                e_labels=emotion_labels,
                s_labels=sentiment_labels,
                e_reg=emotion_reg,
                s_reg=sentiment_reg,
                e_criterion=criterions['emotion'],
                s_criterion=criterions['sentiment']
            )

            losses['combined_text'] += combined_text_loss.item()
            losses['text_emotion'] += e_text_loss.item()
            losses['text_sentiment'] += s_text_loss.item()
            metrics['emotion']['text'].extend(text_e_pred)
            metrics['sentiment']['text'].extend(text_s_pred)

            # ---------------------------- MODALITY FUSION ----------------------------
            # Combine the predictions
            e_logits = (audio_e_logits + text_e_logits) / 2
            s_logits = (audio_s_logits + text_s_logits) / 2

            # Compute the combined loss
            e_loss = criterions['emotion'](e_logits, emotion_labels)
            s_loss = criterions['sentiment'](s_logits, sentiment_labels)
            combined_loss = emotion_reg * e_loss + sentiment_reg * s_loss

            predictions_e = torch.argmax(e_logits, dim=1)
            predictions_s = torch.argmax(s_logits, dim=1)

            losses['fused_combined'] += combined_loss.item()
            losses['fused_emotion'] += e_loss.item()
            losses['fused_sentiment'] += s_loss.item()
            metrics['emotion']['fused'].extend(predictions_e)
            metrics['sentiment']['fused'].extend(predictions_s)

            # ---------------------------- END ----------------------------
            metrics['emotion']['labels'].extend(emotion_labels.cpu().numpy())
            metrics['sentiment']['labels'].extend(sentiment_labels.cpu().numpy())

        # Average losses
        for key in losses.keys():
            losses[key] /= len(dataloader)

        # compute metrics per modality
        emotion_metrics = {
            'fused': compute_metrics(metrics['emotion']['labels'], metrics['emotion']['fused']),
            'audio': compute_metrics(metrics['emotion']['labels'], metrics['emotion']['audio']),
            'text': compute_metrics(metrics['emotion']['labels'], metrics['emotion']['text']),
        }
        sentiment_metrics = {
            'fused': compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['fused']),
            'audio': compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['audio']),
            'text': compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['text']),
        }
    return losses, {'emotion': emotion_metrics, 'sentiment': sentiment_metrics}

def compute_metrics(true_labels, predictions):
    """
    Compute classification metrics including accuracy, per-class F1 scores, and weighted average F1 score.
    """
    # Compute overall accuracy
    overall_accuracy = accuracy_score(true_labels, predictions)

    # Compute F1 scores
    report = classification_report(
        true_labels, predictions, output_dict=True, zero_division=0
    )
    per_class_f1 = {label: values["f1-score"] for label, values in report.items() if label.isdigit()}
    macro_f1 = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]

    # Compile metrics into a dictionary
    metrics = {
        "acc": overall_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1,
    }

    return metrics

def task_result_to_table(task_result):
    df = pd.DataFrame([
        {
            'epoch': res['epoch'],
            'train_loss': res['train_loss'],
            'val_loss': res['val_loss'],
            **flatten_metrics(res['train_metrics'], prefix='train'),
            **flatten_metrics(res['val_metrics'], prefix='val')
        }
        for res in task_result
    ])
    
    return df

def flatten_metrics(metrics, prefix=''):
    """
    Flatten metrics for easier storage in a tabular format.

    :param metrics: Dictionary of metrics (e.g., {'fused': {...}, 'audio': {...}, 'text': {...}})
    :param prefix: Prefix for column names (optional, e.g., 'train' or 'val').
    :return: Flattened dictionary.
    """
    flattened = {}
    for modality, modality_metrics in metrics.items():
        for metric_name, value in modality_metrics.items():
            flattened[f"{prefix}_{modality}_{metric_name}"] = value
    return flattened

def train_and_validate(model, train_loader, val_loader, optimizer, criterions, num_epochs, experiment_name, device='cuda', save_dir='./results'):
    """
    Train and validate the model for multiple epochs, and display results in a summary table at the end.

    :param model: The model to train.
    :param train_dataloader: Training DataLoader.
    :param val_dataloader: Validation DataLoader.
    :param optimizer: Optimizer.
    :param criterion: Loss function.
    :param num_epochs: Number of epochs.
    :param model_name: Name of the model (for display in results).
    :param device: Device to use for training ('cuda' or 'cpu').
    """
    os.makedirs(save_dir, exist_ok=True)
    
    results = {
        'experiment_name': experiment_name,
        'model_state_dict': None,
        'results_emotions': [],
        'results_sentiments': []
    }

    for epoch in range(num_epochs):
        train_losses, train_metrics = train_one_epoch(model, train_loader, optimizer, criterions, device=device)
        val_losses, val_metrics = validate_one_epoch(model, val_loader, criterions, device=device)

        print(f"Epoch [{epoch+1}/{num_epochs}]")

        # Print losses
        print("Losses:")
        for key, value in losses.items():
            print(f"\t{key.capitalize().replace('_', ' ')}: {value:.4f}")

        # Print metrics
        print("\nMetrics:")
        for task in ['emotion', 'sentiment']:
            print(f"\t{task.capitalize()} Metrics:")
            for modality in ['fused', 'audio', 'text']:
                # Check if there are metrics for the modality
                if metrics[task][modality]:
                    # Assuming the list contains accuracy and F1 scores
                    values = metrics[task][modality]
                    print(f"\t\t{modality.capitalize()} Metrics: {values}")
            # Print the labels for the task
            if metrics[task]['labels']:
                print(f"\t\tLabels: {metrics[task]['labels']}")


        print("---------------------------------------------------------------\n")

        # # Save results for both tasks
        # results['results_emotions'].append({
        #     'epoch': epoch + 1,
        #     'train_loss': train_losses['emotion'],
        #     'val_loss': val_losses['emotion'],
        #     'train_metrics': train_metrics['emotion'],
        #     'val_metrics': val_metrics['emotion']
        # })
        # results['results_sentiments'].append({
        #     'epoch': epoch + 1,
        #     'train_loss': train_losses['sentiment'],
        #     'val_loss': val_losses['sentiment'],
        #     'train_metrics': train_metrics['sentiment'],
        #     'val_metrics': val_metrics['sentiment']
        # })

    # Save model state
    results['model_state_dict'] = model.state_dict()
    results['optimizer'] = optimizer

    # save all the results as pkl file
    results_joblib_path = os.path.join(save_dir, f"{experiment_name}_results.pkl")
    joblib.dump(results, results_joblib_path)

    # display the summary of the training as a Datframe (a table per task)
    # print('\n\n SUMMARY OF THE RESULTS\n')
    # for task in ['emotions', 'sentiments']:
    #     print(f'Results for the {task} task:')
    #     display(task_result_to_table(results[f'results_{task}_bsize{train_dataloader.batch_size}']))
    # print(f"Results saved as Joblib file: {results_joblib_path}.")
    
    # Display final results in a table
    return model, results

def plot_metrics(results, task='emotions', metric='acc', modality='fused'):
    """
    Plots metrics (e.g., accuracy or F1 score) for a specific task and modality over epochs.
    
    :param results: Dictionary containing training and validation results.
    :param task: Task name ('emotion' or 'sentiment').
    :param metric: Metric to plot (e.g., 'acc', 'macro_f1', 'weighted_f1').
    :param modality: Modality to plot ('fused', 'audio', 'text').
    """
    train_values = [
        epoch_results['train_metrics'][modality][metric] 
        for epoch_results in results[f'results_{task}']
    ]
    val_values = [
        epoch_results['val_metrics'][modality][metric] 
        for epoch_results in results[f'results_{task}']
    ]
    epochs = list(range(1, len(train_values) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_values, label=f'Train {metric.capitalize()}', marker='o')
    plt.plot(epochs, val_values, label=f'Val {metric.capitalize()}', marker='o')
    plt.title(f'{task.capitalize()} {metric.capitalize()} ({modality.capitalize()})')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()

def test_and_save_labels(model, test_loader, dataset, criterions, save_path, device='cuda'):
    """
    Perform inference on the test set and save the true and predicted labels to a file.

    :param model: The trained model.
    :param test_loader: DataLoader for the test set (the embedding one)
    :param criterions: Dictionary of loss functions for each task.
    :param save_path: Path to save the true and predicted labels.
    :param device: Device to use for inference ('cuda' or 'cpu').
    """
    model.eval()

    true_and_pred_labels = {
        'emotion': {'true': [], 'pred': []},
        'sentiment': {'true': [], 'pred': []},
    }

    e_int_to_str, e_str_to_int = dataset.get_emotions_dicts()
    s_int_to_str, s_str_to_int = dataset.get_emotions_dicts()

    with torch.no_grad():
        for a, t, emotion_labels, sentiment_labels in test_loader:
            # Move data to device
            a = a.to(device)
            t = t.to(device)
            emotion_labels = emotion_labels.to(device)
            sentiment_labels = sentiment_labels.to(device)

            # ------------------- EMOTION TASK -------------------
            _, e_fused, _, _ = process_task(
                model=model, a=a, t=t, classifier=model.emotion_classifier,
                weight_size=model.emotion_classifier.weight.size(1),
                labels=emotion_labels, criterion=criterions['emotion']
            )
            true_and_pred_labels['emotion']['true'].extend(emotion_labels.cpu().numpy())
            true_and_pred_labels['emotion']['pred'].extend(e_fused)

            # ------------------ SENTIMENT TASK ------------------
            _, s_fused, _, _ = process_task(
                model=model, a=a, t=t, classifier=model.sentiment_classifier,
                weight_size=model.sentiment_classifier.weight.size(1),
                labels=sentiment_labels, criterion=criterions['sentiment']
            )
            true_and_pred_labels['sentiment']['true'].extend(sentiment_labels.cpu().numpy())
            true_and_pred_labels['sentiment']['pred'].extend(s_fused)

    return true_and_pred_labels

def analyze_results_per_class(true_labels, predicted_labels, class_names, task_name="Sentiment", mode="confusion_matrix"):
    """
    Analyze results per class with confusion matrix, classification report, or ROC curves.

    Args:
        true_labels (list or np.ndarray): True labels for the task.
        predicted_labels (list or np.ndarray): Predicted labels from the model.
        class_names (list): List of class names.
        task_name (str): Name of the task (e.g., "Sentiment" or "Emotion").
        mode (str): The type of analysis. Options: "confusion_matrix", "classification_report", "roc_curve".
    """
    if mode == "confusion_matrix":
        # Plot confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(class_names)))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Confusion Matrix for {task_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.show()

    elif mode == "classification_report":
        # Print classification report
        report = classification_report(true_labels, predicted_labels, target_names=class_names, zero_division=0)
        print(f"Classification Report for {task_name}:\n")
        print(report)

    elif mode == "roc_curve":
        # Compute and plot ROC curves
        true_binarized = label_binarize(true_labels, classes=range(len(class_names)))
        predicted_binarized = label_binarize(predicted_labels, classes=range(len(class_names)))
        plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(true_binarized[:, i], predicted_binarized[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")  # Random baseline
        plt.title(f"ROC Curve for {task_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

def main():

    #warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    #warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    device = 'cuda'

    print("Loading data...")

    # setup the dataset
    train_set = MELDDataset(
        csv_file="train_sent_emo.csv",
        root_dir="./meld-train-muse",
        mode="train"
    )

    dev_set = MELDDataset(
        csv_file="dev_sent_emo.csv",
        root_dir="./meld-dev-muse",
        mode="dev"
    )

    test_set = MELDDataset(
        csv_file='test_sent_emo.csv',
        root_dir='./meld-test-muse',
        mode='test'
    )
    
    print("Data loaded.")

    train_loader = DataLoader(
    train_set,
    batch_size=64,         # You can adjust the batch size
    shuffle=True,          # Shuffle data during training
    num_workers=16,         # Number of workers for parallel data loading
    collate_fn=collate_fn
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=64,         # You can adjust the batch size
        shuffle=True,          # Shuffle data during training
        num_workers=16,         # Number of workers for parallel data loading
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_set,
        batch_size=64,         # You can adjust the batch size
        shuffle=True,          # Shuffle data during training
        num_workers=16,         # Number of workers for parallel data loading
        collate_fn=collate_fn
    )

    print("Data loaders created.")

    audio_model_name = "facebook/wav2vec2-base"
    text_model_name = "bert-base-uncased"

    # Gather all emotion labels in the train set
    emotion_labels = [sample[2] for sample in train_set.samples]

    # Count how many samples of each class (returns unique labels and their counts)
    unique_classes, counts = np.unique(emotion_labels, return_counts=True)
    print("Class labels:", unique_classes)
    print("Class counts:", counts)

    _, str_to_int = train_set.get_emotions_dicts()

    num_classes = len(str_to_int)
    ordered_counts = [0] * num_classes

    for class_label, count in zip(unique_classes, counts):
        class_idx = str_to_int[class_label]   # Convert 'neutral' → 0, 'joy' → 1, etc.
        ordered_counts[class_idx] = count

    ordered_counts = np.array(ordered_counts)
    print("Ordered counts:", ordered_counts)

    # Avoid division by zero
    inverse_freq = 1.0 / np.maximum(ordered_counts, 1)

    emotions_class_weights = torch.tensor(inverse_freq, dtype=torch.float32)
    # or normalize them
    emotions_class_weights = emotions_class_weights / emotions_class_weights.sum() #* num_classes

    print("Class weights:", emotions_class_weights)
    emotions_class_weights = emotions_class_weights.to(device)

    lr = 0.0001

    criterions = {
        'emotion': nn.CrossEntropyLoss(weight=emotions_class_weights),
        'sentiment': nn.CrossEntropyLoss()
    }

    num_epochs = 50
    num_emotions = len(train_set.get_emotions_dicts()[0].values())
    num_sentiments = len(train_set.get_sentiments_dicts()[0].values())

    model = MultimodalClassifier(
        text_fine_tune=True,
        #audio_fine_tune=True,
        #unfreeze_last_n_audio=2,
        unfreeze_last_n_text=12,
        num_emotions=num_emotions,
        num_sentiments=num_sentiments
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    experiment_name = 'TEST'

    print("Training model...")

    model, results = train_and_validate(
        model, train_loader, dev_loader, 
        optimizer, criterions, num_epochs, 
        experiment_name=experiment_name, 
        device='cuda', save_dir='./saved_results'
    )
    
    print("Training complete.")


    #plot_metrics(loaded_results, metric='macro_f1', modality='fused', task='emotions')
    #true_and_pred_labels = test_and_save_labels(model, test_emb_loader, test_set, criterions, save_path='/kaggle/working')


    # for mode in ['confusion_matrix', 'classification_report', 'roc_curve']:
    # analyze_results_per_class(
    #     true_and_pred_labels['emotion']['true'], 
    #     true_and_pred_labels['emotion']['pred'], 
    #     unique_classes,
    #     task_name="Emotions",
    #     mode=mode
    # )

if __name__ == "__main__":
    main()

