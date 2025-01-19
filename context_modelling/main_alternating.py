


import tarfile
import os
from torch.utils.data import Dataset
import numpy as np
import librosa
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, Wav2Vec2Processor, Wav2Vec2Model
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score
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
from tabulate import tabulate
from pprint import pprint

from backbone import resnet18

from utils import plot_metrics, analyze_results_per_class, compute_metrics, task_result_to_table, flatten_metrics, compute_emotion_class_weights
from dataset import MELDConversationDataset, meld_collate_fn
from model import MultimodalClassifierWithLSTM, TextEncoder, AudioEncoder, FusionLSTM

import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_audio", action="store_true", help="Use audio modality")
    parser.add_argument("--use_text", action="store_true", help="Use text modality")
    parser.add_argument("--emotion_task", action="store_true", help="Train emotion task")
    parser.add_argument("--sentiment_task", action="store_true", help="Train sentiment task")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--emotion_reg", type=float, default=0.6, help="Regularization for emotion task")
    parser.add_argument("--sentiment_reg", type=float, default=0.4, help="Regularization for sentiment task")
    parser.add_argument("--experiment_name", type=str, default="Experiment", help="Name of the experiment")
    parser.add_argument("--alternating", action="store_true", help="Train tasks in an alternating fashion")

    return parser.parse_args()

def get_modality_losses_preds(model, modality, batch, optimizer, criterions, emotion_reg, sentiment_reg, device):

    softmax = nn.Softmax(dim=-1)

    batch_fused_emotion_loss = 0.0
    batch_fused_sentiment_loss = 0.0
        
    batch_fused_emotion_preds = []
    batch_fused_sentiment_preds = []

    batch_modality_emotion_loss = 0.0
    batch_modality_sentiment_loss = 0.0

    batch_combined_modality_loss = 0.0 

    batch_modality_emotion_preds = []
    batch_modality_sentiment_preds = []

    total_utt_in_batch = 0

    modality_full_emotion_logits = torch.zeros(0, 7).to(device)
    modality_full_sentiment_logits = torch.zeros(0, 3).to(device)

    for b_idx in range(len(batch["dialog_ids"])):
        fbank_list      = batch["fbank_lists"][b_idx]       
        text_list       = batch["text_lists"][b_idx]        
        emotion_list    = batch["emotion_lists"][b_idx]
        sentiment_list  = batch["sentiment_lists"][b_idx]

        total_utt_in_batch += len(emotion_list)

        actual_len = len(emotion_list)

        audio_array = torch.as_tensor(fbank_list, dtype=torch.float, device=device)
        audio_array = audio_array.unsqueeze(1)

        modality_emotion_tensor = torch.as_tensor(emotion_list, dtype=torch.long, device=device)
        modality_sentiment_tensor = torch.as_tensor(sentiment_list, dtype=torch.long, device=device)

        optimizer.zero_grad()

        if modality == 'audio':
            a, _ = model(audio_array, text_list, alternating=True)
            modality_lstm_out, (modality_hn, modality_cn) = model.audio_lstm(a)
        else:
            _, t = model(audio_array, text_list, alternating=True)
            modality_lstm_out, (modality_hn, modality_cn) = model.text_lstm(t)

        modality_lstm_out = modality_lstm_out.squeeze(0)

        modality_emotion_logits = model.emotion_head(modality_lstm_out)
        modality_sentiment_logits = model.sentiment_head(modality_lstm_out)

        if len(modality_emotion_logits.shape) == 3:
            modality_emotion_logits = modality_emotion_logits.squeeze(0) 
            modality_sentiment_logits = modality_sentiment_logits.squeeze(0)

        modality_emotion_logits = modality_emotion_logits[:actual_len]
        modality_sentiment_logits = modality_sentiment_logits[:actual_len]

        modality_emotion_tensor = modality_emotion_tensor[:actual_len]
        modality_sentiment_tensor = modality_sentiment_tensor[:actual_len]

        modality_e_loss = criterions['emotion'](modality_emotion_logits, modality_emotion_tensor)
        modality_s_loss = criterions['sentiment'](modality_sentiment_logits, modality_sentiment_tensor)

        batch_modality_emotion_loss += modality_e_loss
        batch_modality_sentiment_loss += modality_s_loss
        modality_dialogue_loss = emotion_reg * modality_e_loss + sentiment_reg * modality_s_loss
        batch_combined_modality_loss += modality_dialogue_loss

        modality_full_emotion_logits = torch.cat((modality_full_emotion_logits, modality_emotion_logits), dim=0)
        modality_full_sentiment_logits = torch.cat((modality_full_sentiment_logits, modality_sentiment_logits), dim=0)

        modality_e_preds = torch.argmax(softmax(modality_emotion_logits), dim=-1).detach().cpu().numpy()
        modality_s_preds = torch.argmax(softmax(modality_sentiment_logits), dim=-1).detach().cpu().numpy()

        batch_modality_emotion_preds.extend(modality_e_preds)
        batch_modality_sentiment_preds.extend(modality_s_preds)

    batch_modality_emotion_loss /= len(batch["dialog_ids"])
    batch_modality_sentiment_loss /= len(batch["dialog_ids"]) 
    batch_combined_modality_loss /= len(batch["dialog_ids"])

    optimizer.zero_grad()
    batch_combined_modality_loss.backward()
    optimizer.step()

    return batch_modality_emotion_loss, batch_modality_sentiment_loss, batch_combined_modality_loss, \
              batch_modality_emotion_preds, batch_modality_sentiment_preds, \
                modality_full_emotion_logits, modality_full_sentiment_logits



def train_one_epoch(args, model, dataloader, optimizers, criterions, emotion_reg=0.5, sentiment_reg=0.5, device='cuda'):
    model.train()

    softmax = nn.Softmax(dim=-1)

    losses = {'fused_emotion': 0.0, 'fused_sentiment': 0.0}
    metrics = {
        'emotion': {'fused': [], 'labels': []},
        'sentiment': {'fused': [], 'labels': []}
    }


    optimizer_a, optimizer_t = optimizers
    losses['audio_emotion'] = 0.0
    losses['audio_sentiment'] = 0.0
    losses['text_emotion'] = 0.0
    losses['text_sentiment'] = 0.0
    metrics['emotion']['audio'] = []
    metrics['emotion']['text'] = []
    metrics['sentiment']['audio'] = []
    metrics['sentiment']['text'] = []
        

    loop = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in loop:

        batch_emotion_labels = []
        batch_sentiment_labels = []

        batch_audio_emotion_loss, batch_audio_sentiment_loss, batch_combined_audio_loss, \
        batch_audio_emotion_preds, batch_audio_sentiment_preds, \
        audio_emotion_logits, audio_sentiment_logits, \
         = get_modality_losses_preds(model, "audio", batch, optimizer_a, criterions, emotion_reg, sentiment_reg, device)

        batch_text_emotion_loss, batch_text_sentiment_loss, batch_combined_text_loss, \
        batch_text_emotion_preds, batch_text_sentiment_preds, \
        text_emotion_logits, text_sentiment_logits, \
            = get_modality_losses_preds(model, "text", batch, optimizer_t, criterions, emotion_reg, sentiment_reg, device)

        for b_idx in range(len(batch["dialog_ids"])):
            emotion_list    = batch["emotion_lists"][b_idx]
            sentiment_list  = batch["sentiment_lists"][b_idx]

            batch_emotion_labels.extend(emotion_list)
            batch_sentiment_labels.extend(sentiment_list)

        batch_fused_emotion_loss = batch_audio_emotion_loss * 0.5 + batch_text_emotion_loss * 0.5
        batch_fused_sentiment_loss = batch_audio_sentiment_loss * 0.5 + batch_text_sentiment_loss * 0.5

        batch_fused_emotion_preds = torch.argmax(softmax(audio_emotion_logits * 0.5 + text_emotion_logits * 0.5), dim=-1).cpu().numpy()
        batch_fused_sentiment_preds = torch.argmax(softmax(audio_sentiment_logits * 0.5 + text_sentiment_logits * 0.5), dim=-1).cpu().numpy()

        losses['fused_emotion'] += batch_fused_emotion_loss.item()
        losses['fused_sentiment'] += batch_fused_sentiment_loss.item()
        losses['audio_emotion'] += batch_audio_emotion_loss.item()
        losses['audio_sentiment'] += batch_audio_sentiment_loss.item()
        losses['text_emotion'] += batch_text_emotion_loss.item()
        losses['text_sentiment'] += batch_text_sentiment_loss.item()

        metrics['emotion']['fused'].extend(batch_fused_emotion_preds)
        metrics['sentiment']['fused'].extend(batch_fused_sentiment_preds)
        metrics['emotion']['audio'].extend(batch_audio_emotion_preds)
        metrics['emotion']['text'].extend(batch_text_emotion_preds)
        metrics['sentiment']['audio'].extend(batch_audio_sentiment_preds)
        metrics['sentiment']['text'].extend(batch_text_sentiment_preds)
        metrics['emotion']['labels'].extend(batch_emotion_labels)
        metrics['sentiment']['labels'].extend(batch_sentiment_labels)

    losses['fused_emotion'] /= len(dataloader)
    losses['fused_sentiment'] /= len(dataloader)
    losses['audio_emotion'] /= len(dataloader)
    losses['audio_sentiment'] /= len(dataloader)
    losses['text_emotion'] /= len(dataloader)
    losses['text_sentiment'] /= len(dataloader)

    print("fused metrics shape:", len(metrics['emotion']['labels']), len(metrics['emotion']['fused']))
    print("audio metrics shape:", len(metrics['emotion']['labels']), len(metrics['emotion']['audio']))
    print("text metrics shape:", len(metrics['emotion']['labels']), len(metrics['emotion']['text']))

    fused_emotion_metrics = compute_metrics(metrics['emotion']['labels'], metrics['emotion']['fused'])
    fused_sentiment_metrics = compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['fused'])
    audio_emotion_metrics = compute_metrics(metrics['emotion']['labels'], metrics['emotion']['audio'])
    audio_sentiment_metrics = compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['audio'])
    text_emotion_metrics = compute_metrics(metrics['emotion']['labels'], metrics['emotion']['text'])
    text_sentiment_metrics = compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['text'])

    return losses, {
        'audio': {'emotion': audio_emotion_metrics, 'sentiment': audio_sentiment_metrics},
        'text': {'emotion': text_emotion_metrics, 'sentiment': text_sentiment_metrics},
        'fused': {'emotion': fused_emotion_metrics, 'sentiment': fused_sentiment_metrics}
    }

def validate_one_epoch(args, model, dataloader, criterions, emotion_reg=0.5, sentiment_reg=0.5, device='cuda'):
    """
    Validate for one epoch. Mirrors the structure of train_one_epoch without gradient 
    computations, backward passes, or optimizer steps.
    """
    model.eval()
    softmax = nn.Softmax(dim=-1)

    losses = {'fused_emotion': 0.0, 'fused_sentiment': 0.0}
    metrics = {
        'emotion': {'fused': [], 'labels': []},
        'sentiment': {'fused': [], 'labels': []}
    }

    losses.update({
        'audio_emotion': 0.0, 
        'audio_sentiment': 0.0, 
        'text_emotion': 0.0, 
        'text_sentiment': 0.0
    })
    metrics['emotion'].update({'audio': [], 'text': []})
    metrics['sentiment'].update({'audio': [], 'text': []})

    loop = tqdm(dataloader, desc="Validating", leave=False)

    with torch.no_grad():
        for batch in loop:
            batch_fused_emotion_loss = 0.0
            batch_fused_sentiment_loss = 0.0

            batch_fused_emotion_preds = []
            batch_fused_sentiment_preds = []

            batch_emotion_labels = []
            batch_sentiment_labels = []

            batch_audio_emotion_loss = 0.0
            batch_audio_sentiment_loss = 0.0
            batch_text_emotion_loss = 0.0
            batch_text_sentiment_loss = 0.0

            batch_audio_emotion_preds = []
            batch_audio_sentiment_preds = []
            batch_text_emotion_preds = []
            batch_text_sentiment_preds = []
            total_utt_in_batch = 0

            for b_idx in range(len(batch["dialog_ids"])):
                fbank_list      = batch["fbank_lists"][b_idx]
                text_list       = batch["text_lists"][b_idx]
                emotion_list    = batch["emotion_lists"][b_idx]
                sentiment_list  = batch["sentiment_lists"][b_idx]
                
                total_utt_in_batch += len(emotion_list)

                batch_emotion_labels.extend(emotion_list)
                batch_sentiment_labels.extend(sentiment_list)

                actual_len = len(emotion_list)

                audio_array = torch.as_tensor(fbank_list, dtype=torch.float, device=device).unsqueeze(1)

                audio_emotion_tensor = torch.as_tensor(emotion_list, dtype=torch.long, device=device)[:actual_len]
                audio_sentiment_tensor = torch.as_tensor(sentiment_list, dtype=torch.long, device=device)[:actual_len]
                text_emotion_tensor = torch.as_tensor(emotion_list, dtype=torch.long, device=device)[:actual_len]
                text_sentiment_tensor = torch.as_tensor(sentiment_list, dtype=torch.long, device=device)[:actual_len]

                a, t = model(audio_array, text_list, alternating=True)

                audio_lstm_out, _ = model.audio_lstm(a)
                audio_lstm_out = audio_lstm_out.squeeze(0)

                audio_emotion_logits = model.emotion_head(audio_lstm_out)[:actual_len]
                audio_sentiment_logits = model.sentiment_head(audio_lstm_out)[:actual_len]

                audio_e_loss = criterions['emotion'](audio_emotion_logits, audio_emotion_tensor)
                audio_s_loss = criterions['sentiment'](audio_sentiment_logits, audio_sentiment_tensor)

                batch_audio_emotion_loss += audio_e_loss
                batch_audio_sentiment_loss += audio_s_loss

                audio_e_preds = torch.argmax(softmax(audio_emotion_logits), dim=-1).cpu().numpy()
                audio_s_preds = torch.argmax(softmax(audio_sentiment_logits), dim=-1).cpu().numpy()
                batch_audio_emotion_preds.extend(audio_e_preds)
                batch_audio_sentiment_preds.extend(audio_s_preds)

                text_lstm_out, _ = model.text_lstm(t)
                text_lstm_out = text_lstm_out.squeeze(0)

                text_emotion_logits = model.emotion_head(text_lstm_out)[:actual_len]
                text_sentiment_logits = model.sentiment_head(text_lstm_out)[:actual_len]

                text_e_loss = criterions['emotion'](text_emotion_logits, text_emotion_tensor)
                text_s_loss = criterions['sentiment'](text_sentiment_logits, text_sentiment_tensor)

                batch_text_emotion_loss += text_e_loss
                batch_text_sentiment_loss += text_s_loss

                text_e_preds = torch.argmax(softmax(text_emotion_logits), dim=-1).cpu().numpy()
                text_s_preds = torch.argmax(softmax(text_sentiment_logits), dim=-1).cpu().numpy()
                batch_text_emotion_preds.extend(text_e_preds)
                batch_text_sentiment_preds.extend(text_s_preds)

                batch_fused_emotion_loss += (audio_e_loss + text_e_loss) * 0.5
                batch_fused_sentiment_loss += (audio_s_loss + text_s_loss) * 0.5

                fused_e_pred = torch.argmax(
                    softmax(audio_emotion_logits * 0.5 + text_emotion_logits * 0.5),
                    dim=-1
                ).cpu().numpy()
                fused_s_pred = torch.argmax(
                    softmax(audio_sentiment_logits * 0.5 + text_sentiment_logits * 0.5),
                    dim=-1
                ).cpu().numpy()

                batch_fused_emotion_preds.extend(fused_e_pred)
                batch_fused_sentiment_preds.extend(fused_s_pred)

            losses['fused_emotion'] += (batch_fused_emotion_loss.item() / len(batch["dialog_ids"]))
            losses['fused_sentiment'] += (batch_fused_sentiment_loss.item() / len(batch["dialog_ids"]))

            metrics['emotion']['fused'].extend(batch_fused_emotion_preds)
            metrics['sentiment']['fused'].extend(batch_fused_sentiment_preds)

            losses['audio_emotion'] += (batch_audio_emotion_loss.item() / len(batch["dialog_ids"]))
            losses['audio_sentiment'] += (batch_audio_sentiment_loss.item() / len(batch["dialog_ids"]))
            losses['text_emotion'] += (batch_text_emotion_loss.item() / len(batch["dialog_ids"]))
            losses['text_sentiment'] += (batch_text_sentiment_loss.item() / len(batch["dialog_ids"]))

            metrics['emotion']['audio'].extend(batch_audio_emotion_preds)
            metrics['emotion']['text'].extend(batch_text_emotion_preds)
            metrics['sentiment']['audio'].extend(batch_audio_sentiment_preds)
            metrics['sentiment']['text'].extend(batch_text_sentiment_preds)

            metrics['emotion']['labels'].extend(batch_emotion_labels)
            metrics['sentiment']['labels'].extend(batch_sentiment_labels)

    losses['fused_emotion'] /= len(dataloader)
    losses['fused_sentiment'] /= len(dataloader)

    fused_emotion_metrics = compute_metrics(metrics['emotion']['labels'], metrics['emotion']['fused'])
    fused_sentiment_metrics = compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['fused'])

    losses['audio_emotion'] /= len(dataloader)
    losses['audio_sentiment'] /= len(dataloader)
    losses['text_emotion'] /= len(dataloader)
    losses['text_sentiment'] /= len(dataloader)

    audio_emotion_metrics = compute_metrics(metrics['emotion']['labels'], metrics['emotion']['audio'])
    audio_sentiment_metrics = compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['audio'])
    text_emotion_metrics = compute_metrics(metrics['emotion']['labels'], metrics['emotion']['text'])
    text_sentiment_metrics = compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['text'])

    return losses, {
        'audio': {'emotion': audio_emotion_metrics, 'sentiment': audio_sentiment_metrics},
        'text': {'emotion': text_emotion_metrics, 'sentiment': text_sentiment_metrics},
        'fused': {'emotion': fused_emotion_metrics, 'sentiment': fused_sentiment_metrics}
    }      

def train_and_validate(args, model, train_loader, val_loader, optimizers, criterions, num_epochs, experiment_name, device='cuda', save_dir='./results'):
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

    best_epoch = 0
    best_emotion_f1 = 0.0
    best_emotion_weighted_acc = 0.0
    best_sentiment_f1 = 0.0
    best_sentiment_weighted_acc = 0.0
    best_model_state = None

    tasks = ['emotion', 'sentiment']

    modalities = ['audio', 'text', 'fused']

    for epoch in range(num_epochs):

        train_losses, train_metrics = train_one_epoch(args, model, train_loader, optimizers, criterions, device=device)
        val_losses, val_metrics = validate_one_epoch(args, model, val_loader, criterions, device=device)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        
        for modality in modalities:
            print(f"xxxxx {modality.capitalize()} MODALITY xxxxx")
            for task in tasks:
                print(f"\t{task.capitalize()} Train Loss: {train_losses[modality + '_' + task]:.4f}")
                print(f"\t{task.capitalize()} Val Loss:   {val_losses[modality + '_' + task]:.4f}")
                train_acc = train_metrics[modality][task]['balanced_acc'] * 100
                val_acc = val_metrics[modality][task]['balanced_acc'] * 100
                train_f1 = train_metrics[modality][task]['weighted_f1'] * 100
                val_f1 = val_metrics[modality][task]['weighted_f1'] * 100
                print(f"\t\tTrain Balanced Acc: {train_acc:.2f}%, Train F1 (weighted): {train_f1:.2f}%")
                print(f"\t\tVal Balanced Acc:   {val_acc:.2f}%, Val F1 (weighted):   {val_f1:.2f}%")
                print('\n')
        if val_metrics['fused']['emotion']['weighted_f1'] > best_emotion_f1:
            best_emotion_f1 = val_metrics['fused']['emotion']['weighted_f1']
            best_emotion_weighted_acc = val_metrics['fused']['emotion']['balanced_acc']
            best_sentiment_f1 = val_metrics['fused']['sentiment']['weighted_f1']
            best_sentiment_weighted_acc = val_metrics['fused']['sentiment']['balanced_acc']
            best_epoch = epoch
            best_model_state = model.state_dict()



        print("\n\nBest epoch so far:", best_epoch)
        print("\nEMOTION:")
        print("\t- Best emotion F1:", best_emotion_f1)
        print("\t- Best emotion weighted acc:", best_emotion_weighted_acc)
        print("\nSENTIMENT:")
        print("\t- Best sentiment F1:", best_sentiment_f1)
        print("\t- Best sentiment weighted acc:", best_sentiment_weighted_acc)
        print("\n---------------------------------------------------------------\n\n")

        model.load_state_dict(best_model_state)

    return model, results

def test_inference(args, model, test_loader, criterions, experiment_name, device='cuda'):
    """
    Perform inference on the test set and save the true and predicted labels to a file.

    :param model: The trained model.
    :param test_loader: DataLoader for the test set (the embedding one)
    :param criterions: Dictionary of loss functions for each task.
    :param save_path: Path to save the true and predicted labels.
    :param device: Device to use for inference ('cuda' or 'cpu').
    """
    model.eval()

    softmax = nn.Softmax(dim=-1)

    true_and_pred_labels = {
        'emotion': {'true': [], 'pred': []},
        'sentiment': {'true': [], 'pred': []},
    }

    with torch.no_grad():
        for batch in test_loader:
            for b_idx in range(len(batch["dialog_ids"])):
                fbank_list      = batch["fbank_lists"][b_idx]
                text_list       = batch["text_lists"][b_idx]
                emotion_list    = batch["emotion_lists"][b_idx]
                sentiment_list  = batch["sentiment_lists"][b_idx]

                actual_len = len(emotion_list)

                audio_array = torch.as_tensor(fbank_list, dtype=torch.float, device=device).unsqueeze(1)

                texts = text_list

                emotion_labels = torch.as_tensor(emotion_list, dtype=torch.long, device=device)
                sentiment_labels = torch.as_tensor(sentiment_list, dtype=torch.long, device=device)

                audio_emotion_tensor = emotion_labels
                audio_sentiment_tensor = sentiment_labels
                text_emotion_tensor = emotion_labels
                text_sentiment_tensor = sentiment_labels

                a, t = model(audio_array, texts, alternating=True)

                audio_lstm_out, _ = model.audio_lstm(a)
                audio_lstm_out = audio_lstm_out.squeeze(0)

                audio_emotion_logits = model.emotion_head(audio_lstm_out)[:actual_len]
                audio_sentiment_logits = model.sentiment_head(audio_lstm_out)[:actual_len]

                audio_e_preds = torch.argmax(softmax(audio_emotion_logits), dim=-1).cpu().numpy()
                audio_s_preds = torch.argmax(softmax(audio_sentiment_logits), dim=-1).cpu().numpy()

                text_lstm_out, _ = model.text_lstm(t)
                text_lstm_out = text_lstm_out.squeeze(0)

                text_emotion_logits = model.emotion_head(text_lstm_out)[:actual_len]
                text_sentiment_logits = model.sentiment_head(text_lstm_out)[:actual_len]

                text_e_preds = torch.argmax(softmax(text_emotion_logits), dim=-1).cpu().numpy()
                text_s_preds = torch.argmax(softmax(text_sentiment_logits), dim=-1).cpu().numpy()

                fused_e_preds = torch.argmax(softmax(audio_emotion_logits * 0.5 + text_emotion_logits * 0.5), dim=-1).cpu().numpy()
                fused_s_preds = torch.argmax(softmax(audio_sentiment_logits * 0.5 + text_sentiment_logits * 0.5), dim=-1).cpu().numpy()

                true_and_pred_labels['emotion']['true'].extend(emotion_list)
                true_and_pred_labels['emotion']['pred'].extend(fused_e_preds)

                true_and_pred_labels['sentiment']['true'].extend(sentiment_list)
                true_and_pred_labels['sentiment']['pred'].extend(fused_s_preds)
    
    return true_and_pred_labels

def main():
    args = get_arguments()

    device = 'cuda'

    print("Loading data...")

    train_set = MELDConversationDataset(csv_file="train_sent_emo.csv", root_dir="../meld-train-muse", mode="train")
    dev_set = MELDConversationDataset(csv_file="dev_sent_emo.csv", root_dir="../meld-dev-muse", mode="dev")
    test_set = MELDConversationDataset(csv_file='test_sent_emo.csv', root_dir='../meld-test-muse', mode='test')
    
    print("Data loaded.")

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=16, collate_fn=meld_collate_fn)
    dev_loader = DataLoader(dev_set, batch_size=8, shuffle=True, num_workers=16, collate_fn=meld_collate_fn)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=True, num_workers=16, collate_fn=meld_collate_fn)

    print("Data loaders created.")

    num_emotions = len(train_set.emotion_class_counts)
    num_sentiments = len(train_set.sentiment_class_counts)

    print("Number of emotions:", num_emotions)
    print("Number of sentiments:", num_sentiments)

    for i in range(7):
        print(f"Emotion {train_set.emotion_to_str(i)} count:", train_set.emotion_class_counts[i])
    for i in range(3):
        print(f"Sentiment {train_set.sentiment_to_str(i)} count:", train_set.sentiment_class_counts[i])

    print("\n")


    class_counts = train_set.emotion_class_counts
    total_samples = 0
    for key in class_counts:
        total_samples += class_counts[key]

    print("Total samples:", total_samples)

    class_weights = torch.zeros(len(class_counts))
    for i in range(len(class_counts)):
        class_weights[i] = class_counts[i] / total_samples

    class_weights = 1 / class_weights

    class_weights = class_weights / class_weights.sum()

    class_weights = class_weights.to(device)

    print("Class weights:", class_weights)

    sentiment_class_counts = train_set.sentiment_class_counts
    sentiment_class_weights = torch.zeros(len(sentiment_class_counts))
    for key in sentiment_class_counts:
        sentiment_class_weights[key] = sentiment_class_counts[key] / total_samples

    sentiment_class_weights = 1 / sentiment_class_weights
    sentiment_class_weights = sentiment_class_weights / sentiment_class_weights.sum()
    sentiment_class_weights = sentiment_class_weights.to(device)

    print("Sentiment class weights:", sentiment_class_weights)

    lr = args.lr

    criterions = {
        'emotion': nn.CrossEntropyLoss(weight=class_weights),
        'sentiment': nn.CrossEntropyLoss(weight=sentiment_class_weights)
    }

    num_epochs = args.epochs

    max_utt = max(train_set.max_utterance_size, dev_set.max_utterance_size, test_set.max_utterance_size)

    print("Max utterance size:", max_utt)

    model = MultimodalClassifierWithLSTM(
        fusion_lstm=FusionLSTM(input_dim=1280, hidden_dim=256, num_layers=1, bidirectional=True, dropout=0.2),
        audio_lstm=FusionLSTM(input_dim=512, hidden_dim=256, num_layers=1, bidirectional=True, dropout=0.2),
        text_lstm=FusionLSTM(input_dim=768, hidden_dim=256, num_layers=1, bidirectional=True, dropout=0.2),
        audio_encoder=resnet18(modality='audio'),
        text_encoder=TextEncoder(fine_tune=True, unfreeze_last_n_layers=1),
        hidden_dim=256,
        num_emotions=num_emotions,
        num_sentiments=num_sentiments,
        max_utt=max_utt,
        alternating=True
    ).to(device)

    optimizers = []

    optimizer_t = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)    
    optimizer_a = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.0001, momentum=0.99)
    optimizers = [optimizer_t, optimizer_a]

    experiment_name = 'alternating_biLSTM_resnet18_roberta_AdamW_0001_SGD_0001_momentum_099'

    print("Training model...")

    best_model, results = train_and_validate(args,
        model, train_loader, dev_loader, 
        optimizers, criterions, num_epochs, 
        experiment_name=experiment_name, 
        device='cuda', save_dir='./saved_results'
    )

    print("Training complete.")

    true_and_pred_labels = test_inference(args, best_model, test_loader, criterions, experiment_name)

    save_path = os.path.join(f"images/{experiment_name}")

    unique_emotion_classes = []
    unique_sentiment_classes = []

    for key in train_set.emotion_class_counts:
        unique_emotion_classes.append(train_set.emotion_to_str(key))

    for key in train_set.sentiment_class_counts:
        unique_sentiment_classes.append(train_set.sentiment_to_str(key))


    for mode in ['confusion_matrix', 'classification_report']:
        analyze_results_per_class(
            true_and_pred_labels['emotion']['true'], 
            true_and_pred_labels['emotion']['pred'], 
            unique_emotion_classes,
            task_name="Emotions",
            mode=mode,
            save_path=save_path
        )
        analyze_results_per_class(
            true_and_pred_labels['sentiment']['true'], 
            true_and_pred_labels['sentiment']['pred'], 
            unique_sentiment_classes,
            task_name="Sentiments",
            mode=mode,
            save_path=save_path
        )

if __name__ == "__main__":
    main()

