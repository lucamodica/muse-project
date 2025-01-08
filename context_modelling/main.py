


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

    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, criterions, emotion_reg=0.6, sentiment_reg=0.4, device='cuda'):
    model.train()

    losses = {'emotion': 0.0, 'sentiment': 0.0}
    metrics = {
        'emotion': {'fused': [], 'labels': []},
        'sentiment': {'fused': [], 'labels': []}
    }


    # 1 batch in dataloader = 4 dialogs (batch size 4)
    # process each dialog independently (1 forward for each)
    # 1 dialog = 14 utterances (new batch size 14)



    loop = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in loop:
        # batch is a dict with fields: 
        #  "dialog_ids": [conv1_id, conv2_id, ...]
        #  "fbank_lists": [[utt1_fbank, utt2_fbank, ...], [utt1_fbank, ...], ...]
        #  "text_lists": [[str_utt1, str_utt2, ...], [str_utt1, ...], ...]
        #  "emotion_lists": [[e1, e2, ...], [e1, ...], ...]
        #  "sentiment_lists": [[s1, s2, ...], [s1, ...], ...]
        
        # We might have multiple conversations in this batch => batch_size = len(batch["dialog_ids"])
        
        batch_emotion_loss = 0.0
        batch_sentiment_loss = 0.0

        # We'll store predictions/labels for metrics across all convs in this batch
        batch_emotion_preds = []
        batch_emotion_labels = []
        batch_sentiment_preds = []
        batch_sentiment_labels = []

        for b_idx in range(len(batch["dialog_ids"])):
            fbank_list      = batch["fbank_lists"][b_idx]       
            text_list       = batch["text_lists"][b_idx]        
            emotion_list    = batch["emotion_lists"][b_idx]
            sentiment_list  = batch["sentiment_lists"][b_idx]

            emotion_tensor = torch.tensor(emotion_list, dtype=torch.long, device=device)
            sentiment_tensor = torch.tensor(sentiment_list, dtype=torch.long, device=device)

            print("Number of utterances in this conversation:", len(fbank_list))

            # Make a batch of audio utterances
            audio_array = torch.tensor(fbank_list).to(device)
            audio_array = audio_array.unsqueeze(1)


            emotion_logits, sentiment_logits = model(audio_array, text_list)


            if len(emotion_logits.shape) == 3:
                # (1, S_b, num_classes)
                emotion_logits = emotion_logits.squeeze(0) 
                sentiment_logits = sentiment_logits.squeeze(0)

            # Compute Loss
            e_loss = criterions['emotion'](emotion_logits, emotion_tensor)
            s_loss = criterions['sentiment'](sentiment_logits, sentiment_tensor)

            # We can sum up or average losses across convs
            batch_emotion_loss += e_loss
            batch_sentiment_loss += s_loss

            # Predictions
            e_preds = torch.argmax(emotion_logits, dim=-1).detach().cpu().numpy()
            s_preds = torch.argmax(sentiment_logits, dim=-1).detach().cpu().numpy()

            # Collect predictions and labels
            batch_emotion_preds.extend(e_preds)
            batch_emotion_labels.extend(emotion_list)

            batch_sentiment_preds.extend(s_preds)
            batch_sentiment_labels.extend(sentiment_list)

        # --- Now we have losses for each conversation in the batch. 
        #     We can average them or sum them, then do backprop. ---
        # Let's average:
        batch_emotion_loss = batch_emotion_loss / len(batch["dialog_ids"])
        batch_sentiment_loss = batch_sentiment_loss / len(batch["dialog_ids"])

        combined_loss = emotion_reg * batch_emotion_loss + sentiment_reg * batch_sentiment_loss
        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()

        # Accumulate for the entire epoch
        losses['emotion'] += batch_emotion_loss.item()
        losses['sentiment'] += batch_sentiment_loss.item()

        # Add these batch predictions to the "epoch-level" metrics
        metrics['emotion']['fused'].extend(batch_emotion_preds)
        metrics['emotion']['labels'].extend(batch_emotion_labels)

        metrics['sentiment']['fused'].extend(batch_sentiment_preds)
        metrics['sentiment']['labels'].extend(batch_sentiment_labels)

    # Average the losses across all steps in the DataLoader
    losses['emotion'] /= len(dataloader)
    losses['sentiment'] /= len(dataloader)

    # Compute epoch-level metrics
    emotion_metrics = compute_metrics(
        metrics['emotion']['labels'],
        metrics['emotion']['fused']
    )
    sentiment_metrics = compute_metrics(
        metrics['sentiment']['labels'],
        metrics['sentiment']['fused']
    )

    # Return same structure as before
    # If you also want "audio vs. text" alone, you'd need separate calls or separate heads.
    # For now, let's keep it simpler (fused). 
    return losses, {
        'emotion': {'fused': emotion_metrics},
        'sentiment': {'fused': sentiment_metrics}
    }


def validate_one_epoch(model, dataloader, criterions, device='cuda'):
    model.eval()

    losses = {'emotion': 0.0, 'sentiment': 0.0}
    metrics = {
        'emotion': {'fused': [], 'labels': []},
        'sentiment': {'fused': [], 'labels': []}
    }

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validation", leave=False)
        
        for batch in loop:
            batch_emotion_loss = 0.0
            batch_sentiment_loss = 0.0
            batch_emotion_preds = []
            batch_emotion_labels = []
            batch_sentiment_preds = []
            batch_sentiment_labels = []

            for b_idx in range(len(batch["dialog_ids"])):
                fbank_list      = batch["fbank_lists"][b_idx]
                text_list       = batch["text_lists"][b_idx]
                emotion_list    = batch["emotion_lists"][b_idx]
                sentiment_list  = batch["sentiment_lists"][b_idx]

                emotion_tensor = torch.tensor(emotion_list, dtype=torch.long, device=device)
                sentiment_tensor = torch.tensor(sentiment_list, dtype=torch.long, device=device)

                # Forward pass
                emotion_logits, sentiment_logits = model(fbank_list, text_list)
                
                if len(emotion_logits.shape) == 3:
                    emotion_logits = emotion_logits.squeeze(0)
                    sentiment_logits = sentiment_logits.squeeze(0)

                e_loss = criterions['emotion'](emotion_logits, emotion_tensor)
                s_loss = criterions['sentiment'](sentiment_logits, sentiment_tensor)

                batch_emotion_loss += e_loss
                batch_sentiment_loss += s_loss

                e_preds = torch.argmax(emotion_logits, dim=-1).detach().cpu().numpy()
                s_preds = torch.argmax(sentiment_logits, dim=-1).detach().cpu().numpy()

                batch_emotion_preds.extend(e_preds)
                batch_emotion_labels.extend(emotion_list)
                batch_sentiment_preds.extend(s_preds)
                batch_sentiment_labels.extend(sentiment_list)

            # Average across convs in this batch
            batch_emotion_loss /= len(batch["dialog_ids"])
            batch_sentiment_loss /= len(batch["dialog_ids"])

            losses['emotion'] += batch_emotion_loss.item()
            losses['sentiment'] += batch_sentiment_loss.item()

            metrics['emotion']['fused'].extend(batch_emotion_preds)
            metrics['emotion']['labels'].extend(batch_emotion_labels)
            metrics['sentiment']['fused'].extend(batch_sentiment_preds)
            metrics['sentiment']['labels'].extend(batch_sentiment_labels)

        losses['emotion'] /= len(dataloader)
        losses['sentiment'] /= len(dataloader)

        emotion_metrics = compute_metrics(
            metrics['emotion']['labels'],
            metrics['emotion']['fused']
        )
        sentiment_metrics = compute_metrics(
            metrics['sentiment']['labels'],
            metrics['sentiment']['fused']
        )

    return losses, {
        'emotion': {'fused': emotion_metrics},
        'sentiment': {'fused': sentiment_metrics}
    }

def compute_metrics(true_labels, predictions):
    """
    Compute classification metrics including accuracy, per-class F1 scores, and weighted average F1 score.
    """
    # Compute overall accuracy
    overall_accuracy = balanced_accuracy_score(true_labels, predictions)

    # Compute F1 scores
    report = classification_report(
        true_labels, predictions, output_dict=True, zero_division=0
    )
    per_class_f1 = {label: values["f1-score"] for label, values in report.items() if label.isdigit()}
    macro_f1 = report["macro avg"]["f1-score"]

    # Compile metrics into a dictionary
    metrics = {
        "balanced_acc": overall_accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
    }

    return metrics

def train_and_validate(args, model, train_loader, val_loader, optimizer, criterions, num_epochs, experiment_name, device='cuda', save_dir='./results'):
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
        for task in ['emotion', 'sentiment']:
            print(f"\t{task.capitalize()} Train Loss: {train_losses[task]:.4f}")
            print(f"\t{task.capitalize()} Val Loss:   {val_losses[task]:.4f}")
            for modality in ['fused', 'audio', 'text']:
                train_acc = train_metrics[task][modality]['balanced_acc'] * 100
                val_acc = val_metrics[task][modality]['balanced_acc'] * 100
                train_f1 = train_metrics[task][modality]['macro_f1'] * 100
                val_f1 = val_metrics[task][modality]['macro_f1'] * 100
                print(f"\t\t{modality.capitalize()} Train Balanced Acc: {train_acc:.2f}%, Train F1 (macro): {train_f1:.2f}%")
                print(f"\t\t{modality.capitalize()} Val Balanced Acc:   {val_acc:.2f}%, Val F1 (macro):   {val_f1:.2f}%")
                print('\n')
        print("---------------------------------------------------------------\n")

        # Save results for both tasks
        results['results_emotions'].append({
            'epoch': epoch + 1,
            'train_loss': train_losses['emotion'],
            'val_loss': val_losses['emotion'],
            'train_metrics': train_metrics['emotion'],
            'val_metrics': val_metrics['emotion']
        })
        results['results_sentiments'].append({
            'epoch': epoch + 1,
            'train_loss': train_losses['sentiment'],
            'val_loss': val_losses['sentiment'],
            'train_metrics': train_metrics['sentiment'],
            'val_metrics': val_metrics['sentiment']
        })

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

def main():
    args = get_arguments()

    device = 'cuda'

    print("Loading data...")

    train_set = MELDConversationDataset(csv_file="train_sent_emo.csv", root_dir="../meld-train-muse", mode="train")
    dev_set = MELDConversationDataset(csv_file="dev_sent_emo.csv", root_dir="../meld-dev-muse", mode="dev")
    test_set = MELDConversationDataset(csv_file='test_sent_emo.csv', root_dir='../meld-test-muse', mode='test')
    
    print("Data loaded.")

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=16, collate_fn=meld_collate_fn)
    dev_loader = DataLoader(dev_set, batch_size=4, shuffle=True, num_workers=16, collate_fn=meld_collate_fn)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=True, num_workers=16, collate_fn=meld_collate_fn)

    print("Data loaders created.")

    num_emotions = len(train_set.emotion_class_counts)
    num_sentiments = len(train_set.sentiment_class_counts)

    print("Number of emotions:", num_emotions)
    print("Number of sentiments:", num_sentiments)

    for i in range(6):
        print(f"Emotion {train_set.emotion_to_str(i)} count:", train_set.emotion_class_counts[i])
    for i in range(3):
        print(f"Sentiment {train_set.sentiment_to_str(i)} count:", train_set.sentiment_class_counts[i])

    # emotion_class_weights = compute_emotion_class_weights(train_set, device)
    # print("Computed class weights on device:", emotion_class_weights)

    lr = args.lr

    criterions = {
        #'emotion': nn.CrossEntropyLoss(weight=emotion_class_weights),
        'emotion': nn.CrossEntropyLoss(),
        'sentiment': nn.CrossEntropyLoss()
    }

    num_epochs = args.epochs

    model = MultimodalClassifierWithLSTM(
        fusion_lstm= FusionLSTM(input_dim=768, hidden_dim=256, num_layers=1, bidirectional=True, dropout=0.1),
        audio_encoder=resnet18(modality='audio'),
        text_encoder=TextEncoder(),
        hidden_dim=256,
        num_emotions=num_emotions,
        num_sentiments=num_sentiments
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    experiment_name = 'TEST'

    print("Training model...")

    model, results = train_and_validate(args,
        model, train_loader, dev_loader, 
        optimizer, criterions, num_epochs, 
        experiment_name=experiment_name, 
        device='cuda', save_dir='./saved_results'
    )
    print("Training complete.")


    #plot_metrics(loaded_results, metric='macro_f1', modality='fused', task='emotions')
    #true_and_pred_labels = test_inference(model, test_emb_loader, test_set, criterions, save_path='/kaggle/working')


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

