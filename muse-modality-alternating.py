


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
import torch.optim as optim

from backbone import resnet18

from utils import plot_metrics, analyze_results_per_class, compute_metrics, task_result_to_table, flatten_metrics, compute_emotion_class_weights
from dataset import MELDDataset
from model import MultimodalClassifier

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

def initialize_losses_and_metrics(args):
    # tracked measures
    losses = {}
    metrics = {}
    if args.emotion_task:
        metrics['emotion'] = {'labels': []}
    if args.sentiment_task:
        metrics['sentiment'] = {'labels': []}

    if args.use_text:
        if args.emotion_task:
            losses['text_emotion'] = 0.0
            metrics['emotion']['text'] = []
        if args.sentiment_task:
            losses['text_sentiment'] = 0.0
            metrics['sentiment']['text'] = []
        if args.emotion_task and args.sentiment_task:
            losses['combined_text'] = 0.0
    if args.use_audio:
        if args.emotion_task:
            losses['audio_emotion'] = 0.0
            metrics['emotion']['audio'] = []
        if args.sentiment_task:
            losses['audio_sentiment'] = 0.0
            metrics['sentiment']['audio'] = []
        if args.emotion_task and args.sentiment_task:
            losses['combined_audio'] = 0.0
    if args.use_text and args.use_audio:
        if args.emotion_task and args.sentiment_task:
            losses['fused_combined'] = 0.0
        if args.emotion_task:
            losses['fused_emotion'] = 0.0
            metrics['emotion']['fused'] = []
        if args.sentiment_task:
            losses['fused_sentiment'] = 0.0    
            metrics['sentiment']['fused'] = []

    return losses, metrics

def process_task(args, model, m, e_labels, s_labels, e_reg, s_reg, e_criterion, s_criterion, device='cuda'):

    softmax = nn.Softmax(dim=1)

    combined_loss, e_loss, s_loss = None, None, None
    preds_e, preds_s = None, None
    e_logits, s_logits = None, None

    if args.emotion_task:
        e_logits = model.emotion_classifier(m)
        e_loss = e_criterion(e_logits, e_labels)
        preds_e = torch.argmax(softmax(e_logits), dim=1).detach().cpu().numpy()

    if args.sentiment_task:
        s_logits = model.sentiment_classifier(m)
        s_loss = s_criterion(s_logits, s_labels)
        preds_s = torch.argmax(softmax(s_logits), dim=1).detach().cpu().numpy()

    if args.emotion_task and args.sentiment_task:
        combined_loss = e_reg * e_loss + s_reg * s_loss if args.emotion_task and args.sentiment_task else None

    return combined_loss, e_loss, s_loss, preds_e, preds_s, e_logits, s_logits

def train_one_epoch(args, epoch, model, dataloader, optimizers, criterions,
                    emotion_reg=0.5, sentiment_reg=0.5, device='cuda'):
    model.train()
    softmax = nn.Softmax(dim=1)

    losses, metrics = initialize_losses_and_metrics(args)

    loop = tqdm(dataloader, desc="Training", leave=False)
    
    for audio_arrays, texts, emotion_labels, sentiment_labels in loop:
        audio_arrays = audio_arrays.to(device)
        emotion_labels = emotion_labels.to(device)
        sentiment_labels = sentiment_labels.to(device)

        optimizers[0].zero_grad()
        optimizers[1].zero_grad()

        if args.use_audio and args.use_text:
            a, t = model(audio_arrays, texts)
        elif args.use_audio:
            a = model.audio_encoder(audio_arrays.unsqueeze(1))
            a = F.adaptive_avg_pool2d(a, 1)
            a = torch.flatten(a, 1)
        elif args.use_text:
            t = model.text_encoder(texts)

        # ---------------------------- AUDIO MODALITY TRAINING ----------------------------

        if args.use_audio:

            combined_audio_loss, e_audio_loss, s_audio_loss, audio_e_pred, audio_s_pred, audio_e_logits, audio_s_logits = process_task(
                args,
                model=model, 
                m=a, 
                e_labels=emotion_labels,
                s_labels=sentiment_labels,
                e_reg=emotion_reg,
                s_reg=sentiment_reg,
                e_criterion=criterions['emotion'],
                s_criterion=criterions['sentiment'],
            )

            optimizers[0].zero_grad()

            if args.emotion_task and args.sentiment_task:
                combined_audio_loss.backward()
            elif args.emotion_task:
                e_audio_loss.backward()
            elif args.sentiment_task:
                s_audio_loss.backward()

            optimizers[0].step()

            if args.emotion_task and args.sentiment_task:
                losses['combined_audio'] += combined_audio_loss.item() 
            if args.emotion_task:
                losses['audio_emotion'] += e_audio_loss.item()
                metrics['emotion']['audio'].extend(audio_e_pred) 
            if args.sentiment_task:
                losses['audio_sentiment'] += s_audio_loss.item()
                metrics['sentiment']['audio'].extend(audio_s_pred)

        # ---------------------------- TEXT MODALITY TRAINING ----------------------------

        if args.use_text:

            combined_text_loss, e_text_loss, s_text_loss, text_e_pred, text_s_pred, text_e_logits, text_s_logits = process_task(
                args,
                model=model, 
                m=t, 
                e_labels=emotion_labels,
                s_labels=sentiment_labels,
                e_reg=emotion_reg,
                s_reg=sentiment_reg,
                e_criterion=criterions['emotion'],
                s_criterion=criterions['sentiment'],
            )

            optimizers[1].zero_grad()

            if epoch % 1 == 0:

                if args.emotion_task and args.sentiment_task:
                    combined_text_loss.backward()
                elif args.emotion_task:
                    e_text_loss.backward()
                elif args.sentiment_task:
                    s_text_loss.backward()

                optimizers[1].step()

            if args.emotion_task and args.sentiment_task:
                losses['combined_text'] += combined_text_loss.item()
            if args.emotion_task:
                losses['text_emotion'] += e_text_loss.item()
                metrics['emotion']['text'].extend(text_e_pred)
            if args.sentiment_task:
                losses['text_sentiment'] += s_text_loss.item()
                metrics['sentiment']['text'].extend(text_s_pred)

        # ---------------------------- MODALITY FUSION ----------------------------

        if args.use_audio and args.use_text:
            if args.emotion_task:
                e_logits = audio_e_logits * 0.5 + text_e_logits * 0.5
                e_loss = criterions['emotion'](e_logits, emotion_labels)
                predictions_e = torch.argmax(softmax(e_logits), dim=1).detach().cpu().numpy()
                losses['fused_emotion'] += e_loss.item()
                metrics['emotion']['fused'].extend(predictions_e)

            if args.sentiment_task:
                s_logits = audio_s_logits * 0.5 + text_s_logits * 0.5
                s_loss = criterions['sentiment'](s_logits, sentiment_labels)
                predictions_s = torch.argmax(softmax(s_logits), dim=1).detach().cpu().numpy()
                losses['fused_sentiment'] += s_loss.item()
                metrics['sentiment']['fused'].extend(predictions_s)

            if args.emotion_task and args.sentiment_task:
                combined_loss = emotion_reg * e_loss + sentiment_reg * s_loss
                losses['fused_combined'] += combined_loss.item()

        # ---------------------------- END ----------------------------

        if args.emotion_task:
            metrics['emotion']['labels'].extend(emotion_labels.cpu().numpy())
        if args.sentiment_task:
            metrics['sentiment']['labels'].extend(sentiment_labels.cpu().numpy())
    
    # Average losses
    for key in losses.keys():
        losses[key] /= len(dataloader)

    if args.emotion_task:
        emotion_metrics = {}
        if args.use_audio and args.use_text:
            emotion_metrics['fused'] = compute_metrics(metrics['emotion']['labels'], metrics['emotion']['fused'])
        if args.use_audio:
            emotion_metrics['audio'] = compute_metrics(metrics['emotion']['labels'], metrics['emotion']['audio'])
        if args.use_text:
            emotion_metrics['text'] = compute_metrics(metrics['emotion']['labels'], metrics['emotion']['text'])
    if args.sentiment_task:
        sentiment_metrics = {}
        if args.use_audio and args.use_text:
            sentiment_metrics['fused'] = compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['fused'])
        if args.use_audio:
            sentiment_metrics['audio'] = compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['audio'])
        if args.use_text:
            sentiment_metrics['text'] = compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['text'])
    

    if args.emotion_task and args.sentiment_task:
        return losses, {'emotion': emotion_metrics, 'sentiment': sentiment_metrics}
    elif args.emotion_task:
        return losses, {'emotion': emotion_metrics}
    elif args.sentiment_task:
        return losses, {'sentiment': sentiment_metrics}


def validate_one_epoch(args, model, dataloader, criterions, emotion_reg=0.5, sentiment_reg=0.5, device='cuda'):
    model.eval()
    softmax = nn.Softmax(dim=1)

    losses, metrics = initialize_losses_and_metrics(args)

    loop = tqdm(dataloader, desc="Validation", leave=False)

    for audio_arrays, texts, emotion_labels, sentiment_labels in loop:
        audio_arrays = audio_arrays.to(device)
        emotion_labels = emotion_labels.to(device)
        sentiment_labels = sentiment_labels.to(device)

        if args.use_audio and args.use_text:
            a, t = model(audio_arrays, texts)
        elif args.use_audio:
            a = model.audio_encoder(audio_arrays.unsqueeze(1))
            a = F.adaptive_avg_pool2d(a, 1)
            a = torch.flatten(a, 1)
        elif args.use_text:
            t = model.text_encoder(texts)

        # ---------------------------- AUDIO MODALITY VALIDATION ----------------------------

        if args.use_audio:

            combined_audio_loss, e_audio_loss, s_audio_loss, audio_e_pred, audio_s_pred, audio_e_logits, audio_s_logits = process_task(
                args,
                model=model, 
                m=a, 
                e_labels=emotion_labels,
                s_labels=sentiment_labels,
                e_reg=emotion_reg,
                s_reg=sentiment_reg,
                e_criterion=criterions['emotion'],
                s_criterion=criterions['sentiment']
            )

            if args.emotion_task and args.sentiment_task:
                losses['combined_audio'] += combined_audio_loss.item()
            if args.emotion_task:
                losses['audio_emotion'] += e_audio_loss.item()
                metrics['emotion']['audio'].extend(audio_e_pred) 
            if args.sentiment_task:
                losses['audio_sentiment'] += s_audio_loss.item()
                metrics['sentiment']['audio'].extend(audio_s_pred)

        # ---------------------------- TEXT MODALITY VALIDATION ----------------------------

        if args.use_text:

            combined_text_loss, e_text_loss, s_text_loss, text_e_pred, text_s_pred, text_e_logits, text_s_logits = process_task(
                args,
                model=model, 
                m=t, 
                e_labels=emotion_labels,
                s_labels=sentiment_labels,
                e_reg=emotion_reg,
                s_reg=sentiment_reg,
                e_criterion=criterions['emotion'],
                s_criterion=criterions['sentiment']
            )

            if args.emotion_task and args.sentiment_task:
                losses['combined_text'] += combined_text_loss.item()
            if args.emotion_task:
                losses['text_emotion'] += e_text_loss.item()
                metrics['emotion']['text'].extend(text_e_pred)
            if args.sentiment_task:
                losses['text_sentiment'] += s_text_loss.item()
                metrics['sentiment']['text'].extend(text_s_pred)

        # ---------------------------- MODALITY FUSION ----------------------------

        if args.use_audio and args.use_text:
            if args.emotion_task:
                e_logits = audio_e_logits * 0.5 + text_e_logits * 0.5
                e_loss = criterions['emotion'](e_logits, emotion_labels)
                predictions_e = torch.argmax(softmax(e_logits), dim=1).detach().cpu().numpy()
                losses['fused_emotion'] += e_loss.item()
                metrics['emotion']['fused'].extend(predictions_e)

            if args.sentiment_task:
                s_logits = audio_s_logits * 0.5 + text_s_logits * 0.5
                s_loss = criterions['sentiment'](s_logits, sentiment_labels)
                predictions_s = torch.argmax(softmax(s_logits), dim=1).detach().cpu().numpy()
                losses['fused_sentiment'] += s_loss.item()
                metrics['sentiment']['fused'].extend(predictions_s)

            if args.emotion_task and args.sentiment_task:
                combined_loss = emotion_reg * e_loss + sentiment_reg * s_loss
                losses['fused_combined'] += combined_loss.item()

        # ---------------------------- END ----------------------------

        if args.emotion_task:
            metrics['emotion']['labels'].extend(emotion_labels.cpu().numpy())
        if args.sentiment_task:
            metrics['sentiment']['labels'].extend(sentiment_labels.cpu().numpy())

    # Average losses
    for key in losses.keys():
        losses[key] /= len(dataloader)

    if args.emotion_task:
        emotion_metrics = {}
        if args.use_audio and args.use_text:
            emotion_metrics['fused'] = compute_metrics(metrics['emotion']['labels'], metrics['emotion']['fused'])
        if args.use_audio:
            emotion_metrics['audio'] = compute_metrics(metrics['emotion']['labels'], metrics['emotion']['audio'])
        if args.use_text:
            emotion_metrics['text'] = compute_metrics(metrics['emotion']['labels'], metrics['emotion']['text'])
    if args.sentiment_task:
        sentiment_metrics = {}
        if args.use_audio and args.use_text:
            sentiment_metrics['fused'] = compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['fused'])
        if args.use_audio:
            sentiment_metrics['audio'] = compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['audio'])
        if args.use_text:
            sentiment_metrics['text'] = compute_metrics(metrics['sentiment']['labels'], metrics['sentiment']['text'])

    if args.emotion_task and args.sentiment_task:
        return losses, {'emotion': emotion_metrics, 'sentiment': sentiment_metrics}
    elif args.emotion_task:
        return losses, {'emotion': emotion_metrics}
    elif args.sentiment_task:
        return losses, {'sentiment': sentiment_metrics}

def train_and_validate(args, model, train_loader, val_loader, optimizers, criterions, num_epochs, experiment_name, device='cuda', save_dir='./results'):
   
    os.makedirs(save_dir, exist_ok=True)
    
    results = {
        'experiment_name': experiment_name,
        'model_state_dict': None,
    }

    if args.emotion_task:
        results['emotion'] = {}
    if args.sentiment_task:
        results['sentiment'] = {}

    modalities = []
    if args.use_audio:
        modalities.append('audio')
    if args.use_text:
        modalities.append('text')
    if args.use_audio and args.use_text:
        modalities.append('fused')

    tasks = []
    if args.emotion_task:
        tasks.append('emotion')
    if args.sentiment_task:
        tasks.append('sentiment')

    emotion_reg = 0.6
    sentiment_reg = 0.4

    best_f1_emotion = 0.0
    best_weighted_acc_emotion = 0.0

    best_f1_sentiment = 0.0
    best_weighted_acc_sentiment = 0.0

    best_model_state_dict = None

    best_epoch = 0

    for epoch in range(num_epochs):
        train_losses, train_metrics = train_one_epoch(args, epoch, model, train_loader, optimizers, criterions, emotion_reg=emotion_reg, sentiment_reg=sentiment_reg, device=device)
        val_losses, val_metrics = validate_one_epoch(args, model, val_loader, criterions, emotion_reg=emotion_reg, sentiment_reg=sentiment_reg, device=device)

        print(f"Epoch [{epoch+1}/{num_epochs}]")

        for modality in modalities:
            print(f"\n\n ----- {modality.upper()} -----")
            for task in tasks:
                print(f"\n\txxxxx {task.upper()} xxxxx:")
                print(f"\tTrain Loss: {train_losses[f'{modality}_{task}']:.4f} | Val Loss: {val_losses[f'{modality}_{task}']:.4f}\n")
                print(f"\tTrain Balanced Acc: {train_metrics[task][modality]['balanced_acc']:.4f} | Val Balanced Acc: {val_metrics[task][modality]['balanced_acc']:.4f}")
                print(f"\tTrain Weighted F1: {train_metrics[task][modality]['weighted_f1']:.4f} | Val Weighted F1: {val_metrics[task][modality]['weighted_f1']:.4f}")
                if args.use_audio and args.use_text:
                    mod = 'fused'
                elif args.use_audio:
                    mod = 'audio'
                elif args.use_text:
                    mod = 'text'
                if modality == mod:
                    if task == 'emotion':
                        if val_metrics['emotion'][mod]['weighted_f1'] > best_f1_emotion:
                            best_f1_emotion = val_metrics['emotion'][mod]['weighted_f1']
                            best_weighted_acc_sentiment = val_metrics['sentiment'][mod]['balanced_acc']
                            results['model_state_dict'] = model.state_dict()
                            best_weighted_acc_emotion = val_metrics['emotion'][mod]['balanced_acc']
                            best_f1_sentiment = val_metrics['sentiment'][mod]['weighted_f1']
                            best_epoch = epoch
                            best_model_state_dict = model.state_dict()


        model.load_state_dict(best_model_state_dict)

        print("\n\nBest Epoch:", best_epoch)
        print("Best F1 Emotion:", best_f1_emotion)
        print("Best Weighted Acc Emotion:", best_weighted_acc_emotion)
        print("Best F1 Sentiment:", best_f1_sentiment)
        print("Best Weighted Acc Sentiment:", best_weighted_acc_sentiment)

        print("\n---------------------------------------------------------------\n\n")

    return model, results

def test_inference(args, model, test_loader, criterions, experiment_name, device='cuda', emotion_reg=0.6, sentiment_reg=0.4):
    """
    Perform inference on the test set and save the true and predicted labels to a file.

    :param model: The trained model.
    :param test_loader: DataLoader for the test set (the embedding one)
    :param criterions: Dictionary of loss functions for each task.
    :param save_path: Path to save the true and predicted labels.
    :param device: Device to use for inference ('cuda' or 'cpu').
    """
    model.eval()

    softmax = nn.Softmax(dim=1)

    true_and_pred_labels = {
        'emotion': {'true': [], 'pred': []},
        'sentiment': {'true': [], 'pred': []},
    }

    with torch.no_grad():
        for audio_arrays, texts, emotion_labels, sentiment_labels in test_loader:

            audio_arrays = audio_arrays.to(device)
            emotion_labels = emotion_labels.to(device)
            sentiment_labels = sentiment_labels.to(device)

            true_and_pred_labels['emotion']['true'].extend(emotion_labels.cpu().numpy())
            true_and_pred_labels['sentiment']['true'].extend(sentiment_labels.cpu().numpy())

            if args.use_audio and args.use_text:
                a, t = model(audio_arrays, texts)
            elif args.use_audio:
                a = model.audio_encoder(audio_arrays.unsqueeze(1))
                a = F.adaptive_avg_pool2d(a, 1)
                a = torch.flatten(a, 1)
            elif args.use_text:
                t = model.text_encoder(texts)

        # ---------------------------- AUDIO MODALITY VALIDATION ----------------------------

            if args.use_audio:

                combined_audio_loss, e_audio_loss, s_audio_loss, audio_e_pred, audio_s_pred, audio_e_logits, audio_s_logits = process_task(
                    args,
                    model=model, 
                    m=a, 
                    e_labels=emotion_labels,
                    s_labels=sentiment_labels,
                    e_reg=emotion_reg,
                    s_reg=sentiment_reg,
                    e_criterion=criterions['emotion'],
                    s_criterion=criterions['sentiment']
                )

        # ---------------------------- TEXT MODALITY VALIDATION ----------------------------

            if args.use_text:

                combined_text_loss, e_text_loss, s_text_loss, text_e_pred, text_s_pred, text_e_logits, text_s_logits = process_task(
                    args,
                    model=model, 
                    m=t, 
                    e_labels=emotion_labels,
                    s_labels=sentiment_labels,
                    e_reg=emotion_reg,
                    s_reg=sentiment_reg,
                    e_criterion=criterions['emotion'],
                    s_criterion=criterions['sentiment']
                )

        # ---------------------------- MODALITY FUSION ----------------------------

            if args.use_audio and args.use_text:
                if args.emotion_task:
                    e_logits = audio_e_logits * 0.5 + text_e_logits * 0.5
                    e_loss = criterions['emotion'](e_logits, emotion_labels)
                    predictions_e = torch.argmax(softmax(e_logits), dim=1).detach().cpu().numpy()
                    true_and_pred_labels['emotion']['pred'].extend(predictions_e)

                if args.sentiment_task:
                    s_logits = audio_s_logits * 0.5 + text_s_logits * 0.5
                    s_loss = criterions['sentiment'](s_logits, sentiment_labels)
                    predictions_s = torch.argmax(softmax(s_logits), dim=1).detach().cpu().numpy()
                    true_and_pred_labels['sentiment']['pred'].extend(predictions_s)
            elif args.use_audio:
                if args.emotion_task:
                    predictions_e = torch.argmax(softmax(audio_e_logits), dim=1).detach().cpu().numpy()
                    true_and_pred_labels['emotion']['pred'].extend(predictions_e)
                if args.sentiment_task:
                    predictions_s = torch.argmax(softmax(audio_s_logits), dim=1).detach().cpu().numpy()
                    true_and_pred_labels['sentiment']['pred'].extend(predictions_s)

    return true_and_pred_labels

def main():

    args = get_arguments()

    device = 'cuda'

    print("Loading data...")

    train_set = MELDDataset(csv_file="train_sent_emo.csv", root_dir="./meld-train-muse", mode="train")

    dev_set = MELDDataset(csv_file="dev_sent_emo.csv", root_dir="./meld-dev-muse", mode="dev")

    test_set = MELDDataset(csv_file='test_sent_emo.csv', root_dir='./meld-test-muse', mode='test')
    
    print("Data loaded.")

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=16)

    dev_loader = DataLoader(dev_set, batch_size=64, shuffle=True, num_workers=16)

    test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=16)

    print("Data loaders created.")

    audio_model_name = "facebook/wav2vec2-base"
    text_model_name = "roberta-base"

    #get weights for balancing classes

    class_counts = train_set.emotion_class_counts
    total_samples = 0
    for key in class_counts:
        total_samples += class_counts[key]

    print("Total samples:", total_samples)

    emotion_class_weights = torch.zeros(len(class_counts))
    for i in range(len(class_counts)):
        emotion_class_weights[i] = class_counts[i] / total_samples

    #invert the weights
    emotion_class_weights = 1 / emotion_class_weights

    #normalize the weights
    emotion_class_weights = emotion_class_weights / emotion_class_weights.sum()

    emotion_class_weights = emotion_class_weights.to(device)

    print("Class weights:", emotion_class_weights)

    lr = args.lr

    criterions = {
        'emotion': nn.CrossEntropyLoss(weight=emotion_class_weights),
        'sentiment': nn.CrossEntropyLoss()
    }

    num_epochs = args.epochs
    num_emotions = len(train_set.emotion_class_counts)
    num_sentiments = len(train_set.sentiment_class_counts)

    model = MultimodalClassifier(
        fusion_dim=768,
        alternating=True,
        text_fine_tune=True,
        #audio_fine_tune=True,
        #unfreeze_last_n_audio=2,
        unfreeze_last_n_text=1,
        audio_encoder=resnet18(modality='audio'),
        num_emotions=num_emotions,
        num_sentiments=num_sentiments
    ).to(device)

    lr1 = 0.0001
    lr2 = 0.00001
    optimizer1 = torch.optim.SGD(model.parameters(), lr=lr1, weight_decay=0.0001, momentum=0.99)
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=lr2, weight_decay=0.0001)
    
    optimizers = [optimizer1, optimizer2]

    experiment_name = 'only_text'

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

