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
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoProcessor
import joblib
from pprint import pprint
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import warnings
from tabulate import tabulate
from pprint import pprint
from backbone import resnet18
from utils import plot_metrics, analyze_results_per_class, compute_metrics, task_result_to_table, flatten_metrics, compute_emotion_class_weights

from datasetv2 import MELDConversationDataset, meld_collate_fn, AudioTransformPipeline
from modelv2 import MultimodalClassifierWithLSTM, TextEncoder, AudioEncoder, FusionLSTM

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


def train_one_epoch(args, model, dataloader, optimizers, criterions, emotion_reg=0.5, sentiment_reg=0.5, device='cuda'):
    model.train()

    softmax = nn.Softmax(dim=-1)

    losses = {'fused_emotion': 0.0, 'fused_sentiment': 0.0}
    metrics = {
        'emotion': {'fused': [], 'labels': []},
        'sentiment': {'fused': [], 'labels': []}
    }

    optimizer = optimizers[0]

    loop = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in loop:

        batch_fused_emotion_loss = 0.0
        batch_fused_sentiment_loss = 0.0
            
        batch_fused_emotion_preds = []
        batch_fused_sentiment_preds = []

        batch_emotion_labels = []
        batch_sentiment_labels = []

        
        optimizer.zero_grad()
        batch_combined_loss = 0.0 # what is optimized

        for b_idx in range(len(batch["dialog_ids"])):
            fbank_list      = batch["audio_list"][b_idx]       
            text_list       = batch["text_list"][b_idx]        
            emotion_list    = batch["emotion_list"][b_idx]
            sentiment_list  = batch["sentiment_list"][b_idx]

            batch_emotion_labels.extend(emotion_list)
            batch_sentiment_labels.extend(sentiment_list)

            actual_len = len(emotion_list)

            audio_array = torch.as_tensor(fbank_list, dtype=torch.float, device=device)
            audio_array = audio_array.unsqueeze(1)

            emotion_tensor = torch.as_tensor(emotion_list, dtype=torch.long, device=device)
            sentiment_tensor = torch.as_tensor(sentiment_list, dtype=torch.long, device=device)

            emotion_logits, sentiment_logits = model(audio_array, text_list, alternating=False)

            if len(emotion_logits.shape) == 3:
                emotion_logits = emotion_logits.squeeze(0) 
                sentiment_logits = sentiment_logits.squeeze(0)

            # Unpad the logits
            emotion_logits = emotion_logits[:actual_len]
            sentiment_logits = sentiment_logits[:actual_len]

            # Unpad the tensors
            emotion_tensor = emotion_tensor[:actual_len]
            sentiment_tensor = sentiment_tensor[:actual_len]

            # Compute Loss
            e_loss = criterions['emotion'](emotion_logits, emotion_tensor)
            s_loss = criterions['sentiment'](sentiment_logits, sentiment_tensor)

            # Add to batch losses
            batch_fused_emotion_loss += e_loss 
            batch_fused_sentiment_loss += s_loss
            batch_combined_loss += emotion_reg * e_loss + sentiment_reg * s_loss

            # Predictions for non-padded time steps
            e_preds = torch.argmax(softmax(emotion_logits), dim=-1).detach().cpu().numpy()
            s_preds = torch.argmax(softmax(sentiment_logits), dim=-1).detach().cpu().numpy()

            # Add to batch predictions
            batch_fused_emotion_preds.extend(e_preds)
            batch_fused_sentiment_preds.extend(s_preds)

        losses['fused_emotion'] += (batch_fused_emotion_loss.item() / len(batch["dialog_ids"]))
        losses['fused_sentiment'] += (batch_fused_sentiment_loss.item() / len(batch["dialog_ids"]))
        metrics['emotion']['fused'].extend(batch_fused_emotion_preds)
        metrics['sentiment']['fused'].extend(batch_fused_sentiment_preds)
        metrics['emotion']['labels'].extend(batch_emotion_labels)
        metrics['sentiment']['labels'].extend(batch_sentiment_labels)
        
        # ------------------ FUSED BRANCH ------------------

        optimizer.zero_grad()
        batch_combined_loss.backward()
        optimizer.step()

    losses['fused_emotion'] /= len(dataloader)
    losses['fused_sentiment'] /= len(dataloader)

    fused_emotion_metrics = compute_metrics(metrics['emotion']['labels'], metrics['emotion']['fused'])
    fused_sentiment_metrics = compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['fused'])

    return losses, {
        'emotion': {'fused': fused_emotion_metrics},
        'sentiment': {'fused': fused_sentiment_metrics}
    }

def validate_one_epoch(args, model, dataloader, criterions, emotion_reg=0.5, sentiment_reg=0.5, device='cuda'):
    """
    Validate for one epoch. Mirrors the structure of train_one_epoch without gradient 
    computations, backward passes, or optimizer steps.
    """
    model.eval()
    softmax = nn.Softmax(dim=-1)

    # Initialize loss and metrics structures similar to training
    losses = {'fused_emotion': 0.0, 'fused_sentiment': 0.0}
    metrics = {
        'emotion': {'fused': [], 'labels': []},
        'sentiment': {'fused': [], 'labels': []}
    }

    loop = tqdm(dataloader, desc="Validating", leave=False)

    with torch.no_grad():
        for batch in loop:
            # Initialize batch-specific accumulators
            batch_fused_emotion_loss = 0.0
            batch_fused_sentiment_loss = 0.0

            batch_fused_emotion_preds = []
            batch_fused_sentiment_preds = []

            # Use local lists for labels; extend once per sample
            batch_emotion_labels = []
            batch_sentiment_labels = []

            # Process each conversation in the batch
            for b_idx in range(len(batch["dialog_ids"])):
                fbank_list      = batch["audio_list"][b_idx]
                text_list       = batch["text_list"][b_idx]
                emotion_list    = batch["emotion_list"][b_idx]
                sentiment_list  = batch["sentiment_list"][b_idx]

                # Extend local label lists with current sample's labels
                batch_emotion_labels.extend(emotion_list)
                batch_sentiment_labels.extend(sentiment_list)

                actual_len = len(emotion_list)

                audio_array = torch.as_tensor(fbank_list, dtype=torch.float, device=device).unsqueeze(1)

                # Non-alternating branch
                emotion_tensor = torch.as_tensor(emotion_list, dtype=torch.long, device=device)[:actual_len]
                sentiment_tensor = torch.as_tensor(sentiment_list, dtype=torch.long, device=device)[:actual_len]

                emotion_logits, sentiment_logits = model(audio_array, text_list, alternating=False)

                if len(emotion_logits.shape) == 3:
                    emotion_logits = emotion_logits.squeeze(0)
                    sentiment_logits = sentiment_logits.squeeze(0)

                emotion_logits = emotion_logits[:actual_len]
                sentiment_logits = sentiment_logits[:actual_len]

                e_loss = criterions['emotion'](emotion_logits, emotion_tensor)
                s_loss = criterions['sentiment'](sentiment_logits, sentiment_tensor)

                batch_fused_emotion_loss += e_loss
                batch_fused_sentiment_loss += s_loss

                e_preds = torch.argmax(softmax(emotion_logits), dim=-1).cpu().numpy()
                s_preds = torch.argmax(softmax(sentiment_logits), dim=-1).cpu().numpy()

                batch_fused_emotion_preds.extend(e_preds)
                batch_fused_sentiment_preds.extend(s_preds)

            # End of inner loop over conversations

            losses['fused_emotion'] += (batch_fused_emotion_loss.item() / len(batch["dialog_ids"]))
            losses['fused_sentiment'] += (batch_fused_sentiment_loss.item() / len(batch["dialog_ids"]))

            metrics['emotion']['fused'].extend(batch_fused_emotion_preds)
            metrics['sentiment']['fused'].extend(batch_fused_sentiment_preds)

            # Extend global labels once per batch after processing all conversations
            metrics['emotion']['labels'].extend(batch_emotion_labels)
            metrics['sentiment']['labels'].extend(batch_sentiment_labels)

    # Average the overall losses across the entire dataloader
    losses['fused_emotion'] /= len(dataloader)
    losses['fused_sentiment'] /= len(dataloader)

    # Compute fused metrics
    fused_emotion_metrics = compute_metrics(metrics['emotion']['labels'], metrics['emotion']['fused'])
    fused_sentiment_metrics = compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['fused'])
    
    return losses, {
        'emotion': {'fused': fused_emotion_metrics},
        'sentiment': {'fused': fused_sentiment_metrics}
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


    modalities = ['fused']

    for epoch in range(num_epochs):

        train_losses, train_metrics = train_one_epoch(args, model, train_loader, optimizers, criterions, device=device)
        val_losses, val_metrics = validate_one_epoch(args, model, val_loader, criterions, device=device)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        for task in ['emotion', 'sentiment']:
            print(f"\t{task.capitalize()} Train Loss: {train_losses[task]:.4f}")
            print(f"\t{task.capitalize()} Val Loss:   {val_losses[task]:.4f}")
            for modality in modalities:
                train_acc = train_metrics[task][modality]['balanced_acc'] * 100
                val_acc = val_metrics[task][modality]['balanced_acc'] * 100
                train_f1 = train_metrics[task][modality]['weighted_f1'] * 100
                val_f1 = val_metrics[task][modality]['weighted_f1'] * 100
                print(f"\t\t{modality.capitalize()} Train Balanced Acc: {train_acc:.2f}%, Train F1 (weighted): {train_f1:.2f}%")
                print(f"\t\t{modality.capitalize()} Val Balanced Acc:   {val_acc:.2f}%, Val F1 (weighted):   {val_f1:.2f}%")
                print('\n')

            if task == 'emotion' and modality == 'fused' and val_metrics[task]['fused']['weighted_f1'] > best_emotion_f1:
                best_emotion_f1 = val_metrics[task]['fused']['weighted_f1']
                best_emotion_weighted_acc = val_metrics[task]['fused']['balanced_acc']
                best_sentiment_f1 = val_metrics['sentiment']['fused']['weighted_f1']
                best_sentiment_weighted_acc = val_metrics['sentiment']['fused']['balanced_acc']
                best_epoch = epoch
                #save model copy in best model
                best_model_state = model.state_dict()

        print("\n\nBest epoch so far:", best_epoch)
        print("\nEMOTION:")
        print("\t- Best emotion F1:", best_emotion_f1)
        print("\t- Best emotion weighted acc:", best_emotion_weighted_acc)
        print("\nSENTIMENT:")
        print("\t- Best sentiment F1:", best_sentiment_f1)
        print("\t- Best sentiment weighted acc:", best_sentiment_weighted_acc)
        print("\n---------------------------------------------------------------\n\n")

        #load best model
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
                fbank_list      = batch["audio_list"][b_idx]
                text_list       = batch["text_list"][b_idx]
                emotion_list    = batch["emotion_list"][b_idx]
                sentiment_list  = batch["sentiment_list"][b_idx]

                actual_len = len(emotion_list)

                audio_array = torch.as_tensor(fbank_list, dtype=torch.float, device=device).unsqueeze(1)

                texts = text_list

                emotion_labels = torch.as_tensor(emotion_list, dtype=torch.long, device=device)
                sentiment_labels = torch.as_tensor(sentiment_list, dtype=torch.long, device=device)

                emotion_logits, sentiment_logits = model(audio_array, texts, alternating=False)

                emotion_logits = emotion_logits.squeeze(0)
                sentiment_logits = sentiment_logits.squeeze(0)

                emotion_logits = emotion_logits[:actual_len]
                sentiment_logits = sentiment_logits[:actual_len]

                e_preds = torch.argmax(softmax(emotion_logits), dim=-1).cpu().numpy()
                s_preds = torch.argmax(softmax(sentiment_logits), dim=-1).cpu().numpy()

                true_and_pred_labels['emotion']['true'].extend(emotion_list)
                true_and_pred_labels['emotion']['pred'].extend(e_preds)

                true_and_pred_labels['sentiment']['true'].extend(sentiment_list)
                true_and_pred_labels['sentiment']['pred'].extend(s_preds)
    
    return true_and_pred_labels

def main():
    args = get_arguments()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text_encoder_name = 'roberta-base'
    audio_encoder_name = 'openai/whisper-medium'
    
    audio_processor = AutoProcessor.from_pretrained(audio_encoder_name)
    audio_transform = AudioTransformPipeline()

    print("Loading data...")

    train_set = MELDConversationDataset(
      csv_file="train_sent_emo.csv", 
      root_dir="../meld-train-muse",
      audio_transform=audio_transform,
      audio_processor=audio_processor
    )
    dev_set = MELDConversationDataset(
      csv_file="dev_sent_emo.csv", 
      root_dir="../meld-dev-muse",
      audio_transform=audio_transform,
      audio_processor=audio_processor
    )
    test_set = MELDConversationDataset(
      csv_file='test_sent_emo.csv', 
      root_dir='../meld-test-muse',
      audio_transform=audio_transform,
      audio_processor=audio_processor
    )
    
    print("Data loaded.")

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4, collate_fn=meld_collate_fn)
    dev_loader = DataLoader(dev_set, batch_size=4, shuffle=True, num_workers=4, collate_fn=meld_collate_fn)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=True, num_workers=4, collate_fn=meld_collate_fn)

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

    #get weights for balancing classes

    class_counts = train_set.emotion_class_counts
    total_samples = 0
    for key in class_counts:
        total_samples += class_counts[key]

    print("Total samples:", total_samples)

    class_weights = torch.zeros(len(class_counts))
    for i in range(len(class_counts)):
        class_weights[i] = class_counts[i] / total_samples

    #invert the weights
    class_weights = 1 / class_weights

    #normalize the weights
    class_weights = class_weights / class_weights.sum()

    class_weights = class_weights.to(device)

    print("Class weights:", class_weights)


    lr = args.lr

    criterions = {
        'emotion': nn.CrossEntropyLoss(weight=class_weights),
        'sentiment': nn.CrossEntropyLoss()
    }

    num_epochs = args.epochs

    #set max utterance size to be the max of all train dev and test sets
    max_utt = max(train_set.max_utterance_size, dev_set.max_utterance_size, test_set.max_utterance_size)

    print("Max utterance size:", max_utt)

    audio_encoder = AudioEncoder(model_name=audio_encoder_name, preprocessing=False)
    text_encoder = TextEncoder(model_name=text_encoder_name, fine_tune=True, unfreeze_last_n_layers=1)

    model = MultimodalClassifierWithLSTM(
        fusion_lstm=FusionLSTM(input_dim=1280, hidden_dim=256, num_layers=1, bidirectional=True, dropout=0.2),
        audio_encoder=audio_encoder,
        text_encoder=text_encoder,
        hidden_dim=256,
        num_emotions=num_emotions,
        num_sentiments=num_sentiments,
        max_utt=max_utt,
        alternating=False
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    optimizers = [optimizer]


    experiment_name = 'pre-trained-shennanigans'

    print("Training model...")

    model, results = train_and_validate(args,
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

