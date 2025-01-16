


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, balanced_accuracy_score
import joblib
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
from backbone import resnet18
from model import MultimodalClassifier
from dataset import MELDDataset
from utils import plot_metrics, analyze_results_per_class, compute_metrics

def process_task(model, epoch, a, t, classifier, weight_size, labels, criterion, mode="train", device="cuda"):

    softmax = nn.Softmax(dim=1)

    fused = torch.cat([a, t], dim=-1)

    # --- Compute logits ---
    logits_fused = classifier(fused)

    # 2) Audio-only
    logits_a = (
        torch.mm(a, torch.transpose(classifier.weight[:, :weight_size // 2], 0, 1))
        + classifier.bias / 2
    )

    # 3) Text-only
    logits_t = (
        torch.mm(t, torch.transpose(classifier.weight[:, weight_size // 2:], 0, 1))
        + classifier.bias / 2
    )

    # --- Compute loss ---
    loss = criterion(logits_fused, labels)


    # --- Predictions ---
    preds_fused = torch.argmax(softmax(logits_fused), dim=1).detach().cpu().numpy()
    preds_audio = torch.argmax(softmax(logits_a),      dim=1).detach().cpu().numpy()
    preds_text  = torch.argmax(softmax(logits_t),      dim=1).detach().cpu().numpy()

    return loss, preds_fused, preds_audio, preds_text

def train_one_epoch(model, epoch, dataloader, optimizer, criterions,
                    emotion_reg=0.6, sentiment_reg=0.4, device='cuda'):
    model.train()

    # tracked measures
    losses = {'emotion': 0.0, 'sentiment': 0.0}
    metrics = {'emotion': {'fused': [], 'audio': [], 'text': [], 'labels': []},
               'sentiment': {'fused': [], 'audio': [], 'text': [], 'labels': []}}

    # Wrap dataloader with tqdm
    loop = tqdm(dataloader, desc="Training", leave=False)
    
    for audio_arrays, texts, emotion_labels, sentiment_labels in loop:
        # Move data to device
        audio_arrays = audio_arrays.to(device)
        emotion_labels = emotion_labels.to(device)
        sentiment_labels = sentiment_labels.to(device)

        optimizer.zero_grad()


        # Forward pass for audio and text
        a, t = model(audio_arrays, texts)

        # EMOTION TASK
        emotion_loss, e_fused, e_audio, e_text = process_task(
            model=model, 
            epoch=epoch,
            a=a, 
            t=t, 
            classifier=model.emotion_classifier,
            weight_size=model.emotion_classifier.weight.size(1),
            labels=emotion_labels, 
            criterion=criterions['emotion'],
            mode="train"
        )
        losses['emotion'] += emotion_loss.item()
        metrics['emotion']['fused'].extend(e_fused)
        metrics['emotion']['audio'].extend(e_audio)
        metrics['emotion']['text'].extend(e_text)
        metrics['emotion']['labels'].extend(emotion_labels.cpu().numpy())

        # SENTIMENT TASK
        sentiment_loss, s_fused, s_audio, s_text = process_task(
            model=model, 
            epoch=epoch,
            a=a, 
            t=t, 
            classifier=model.sentiment_classifier,
            weight_size=model.sentiment_classifier.weight.size(1),
            labels=sentiment_labels, 
            criterion=criterions['sentiment'],
            mode="train"
        )
        losses['sentiment'] += sentiment_loss.item()
        metrics['sentiment']['fused'].extend(s_fused)
        metrics['sentiment']['audio'].extend(s_audio)
        metrics['sentiment']['text'].extend(s_text)
        metrics['sentiment']['labels'].extend(sentiment_labels.cpu().numpy())

        # Backprop on weighted loss
        combined_loss = emotion_reg * emotion_loss + sentiment_reg * sentiment_loss
        combined_loss.backward()
        optimizer.step()
    
    # Average losses
    losses['emotion'] /= len(dataloader)
    losses['sentiment'] /= len(dataloader)

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

def validate_one_epoch(model, epoch, dataloader, criterions, emotion_reg=0.6, sentiment_reg=0.4, device='cuda'):
    model.eval()

    losses = {'emotion': 0.0, 'sentiment': 0.0}
    metrics = {'emotion': {'fused': [], 'audio': [], 'text': [], 'labels': []},
               'sentiment': {'fused': [], 'audio': [], 'text': [], 'labels': []}}

    with torch.no_grad():
        # Also wrap validation dataloader with tqdm
        loop = tqdm(dataloader, desc="Validation", leave=False)
        
        for audio_arrays, texts, emotion_labels, sentiment_labels in loop:
            audio_arrays = audio_arrays.to(device)
            emotion_labels = emotion_labels.to(device)
            sentiment_labels = sentiment_labels.to(device)

            a, t = model(audio_arrays, texts)

            emotion_loss, e_fused, e_audio, e_text = process_task(
                model=model, 
                epoch=epoch,
                a=a, 
                t=t, 
                classifier=model.emotion_classifier,
                weight_size=model.emotion_classifier.weight.size(1),
                labels=emotion_labels, 
                criterion=criterions['emotion'],
                mode="val"
            )
            losses['emotion'] += emotion_loss.item()
            metrics['emotion']['fused'].extend(e_fused)
            metrics['emotion']['audio'].extend(e_audio)
            metrics['emotion']['text'].extend(e_text)
            metrics['emotion']['labels'].extend(emotion_labels.cpu().numpy())

            sentiment_loss, s_fused, s_audio, s_text = process_task(
                model=model, 
                epoch=epoch,
                a=a, 
                t=t, 
                classifier=model.sentiment_classifier,
                weight_size=model.sentiment_classifier.weight.size(1),
                labels=sentiment_labels, 
                criterion=criterions['sentiment'],
                mode="val"
            )
            losses['sentiment'] += sentiment_loss.item()
            metrics['sentiment']['fused'].extend(s_fused)
            metrics['sentiment']['audio'].extend(s_audio)
            metrics['sentiment']['text'].extend(s_text)
            metrics['sentiment']['labels'].extend(sentiment_labels.cpu().numpy())
            
        losses['emotion'] /= len(dataloader)
        losses['sentiment'] /= len(dataloader)
    
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

    best_epoch = 0
    best_f1_emotion = 0.0
    best_f1_sentiment = 0.0
    best_weighted_acc_emotion = 0.0
    best_weighted_acc_sentiment = 0.0
    best_model_state = None

    emotion_reg = 0.6
    sentiment_reg = 0.4

    for epoch in range(num_epochs):
        train_losses, train_metrics = train_one_epoch(model, epoch, train_loader, optimizer, criterions, emotion_reg=emotion_reg, sentiment_reg=sentiment_reg, device=device)
        val_losses, val_metrics = validate_one_epoch(model, epoch, val_loader, criterions, emotion_reg=emotion_reg, sentiment_reg=sentiment_reg, device=device)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        for task in ['emotion', 'sentiment']:
            print(f"\t{task.capitalize()} Train Loss: {train_losses[task]:.4f}")
            print(f"\t{task.capitalize()} Val Loss:   {val_losses[task]:.4f}")
            for modality in ['fused', 'audio', 'text']:
                train_acc = train_metrics[task][modality]['balanced_acc'] * 100
                val_acc = val_metrics[task][modality]['balanced_acc'] * 100
                train_f1 = train_metrics[task][modality]['weighted_f1'] * 100
                val_f1 = val_metrics[task][modality]['weighted_f1'] * 100
                print(f"\t\t{modality.capitalize()} Train Balanced Acc: {train_acc:.2f}%, Train F1 (weighted): {train_f1:.2f}%")
                print(f"\t\t{modality.capitalize()} Val Balanced Acc:   {val_acc:.2f}%, Val F1 (weighted):   {val_f1:.2f}%")
                print('\n')

                if task == 'emotion' and modality == 'fused':
                    if val_f1 > best_f1_emotion:
                        best_f1_emotion = val_f1
                        best_weighted_acc_emotion = val_acc
                        best_epoch = epoch
                        best_weighted_acc_sentiment = val_metrics['sentiment']['fused']['balanced_acc'] * 100
                        best_f1_sentiment = val_metrics['sentiment']['fused']['weighted_f1'] * 100
                        best_model_state = model.state_dict()

        model.load_state_dict(best_model_state)

        print("\n\nBest epoch so far: ", best_epoch)
        print("EMOTION: ")
        print("\t- Best emotion F1: ", best_f1_emotion)
        print("\t- Best emotion weighted acc: ", best_weighted_acc_emotion)
        print("SENTIMENT: ")
        print("\t- Best sentiment F1: ", best_f1_sentiment)
        print("\t- Best sentiment weighted acc: ", best_weighted_acc_sentiment)
        
        print("\n---------------------------------------------------------------\n\n")

    return model, results

def test_inference(model, test_loader, criterions, experiment_name, device='cuda'):
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

    with torch.no_grad():
        for audio_arrays, texts, emotion_labels, sentiment_labels in test_loader:

            audio_arrays = audio_arrays.to(device)
            emotion_labels = emotion_labels.to(device)
            sentiment_labels = sentiment_labels.to(device)

            a, t = model(audio_arrays, texts)

            # ------------------- EMOTION TASK -------------------
            _, e_fused, _, _ = process_task(
                model=model, epoch=0, a=a, t=t, classifier=model.emotion_classifier,
                weight_size=model.emotion_classifier.weight.size(1),
                labels=emotion_labels, criterion=criterions['emotion'], mode="test"
            )
            true_and_pred_labels['emotion']['true'].extend(emotion_labels.cpu().numpy())
            true_and_pred_labels['emotion']['pred'].extend(e_fused)

            # ------------------ SENTIMENT TASK ------------------
            _, s_fused, _, _ = process_task(
                model=model, epoch=0, a=a, t=t, classifier=model.sentiment_classifier,
                weight_size=model.sentiment_classifier.weight.size(1),
                labels=sentiment_labels, criterion=criterions['sentiment'], mode="test"
            )
            true_and_pred_labels['sentiment']['true'].extend(sentiment_labels.cpu().numpy())
            true_and_pred_labels['sentiment']['pred'].extend(s_fused)

    # save all the results as pkl file
    results_path = os.path.join(
        'test_inference_results', 
        f"{experiment_name}_test_inference_results.pkl"
    )
    joblib.dump(true_and_pred_labels, results_path)
    return true_and_pred_labels

def main():

    torch.manual_seed(42)

    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

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

    lr = 0.0001

    criterions = {
        'emotion': nn.CrossEntropyLoss(weight=emotion_class_weights),
        'sentiment': nn.CrossEntropyLoss()
    }

    num_epochs = 40
    num_emotions = len(train_set.emotion_class_counts)
    num_sentiments = len(train_set.sentiment_class_counts)

    model = MultimodalClassifier(
        fusion_dim=1536,
        text_model_name=text_model_name,
        text_fine_tune=True,
        unfreeze_last_n_text=1,
        audio_encoder=resnet18(modality='audio'),
        num_emotions=num_emotions,
        num_sentiments=num_sentiments
        
    ).to(device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99, weight_decay=0.0001)
    experiment_name = 'joint_training_SGD_mom_099_wd_0001_lr_0001_epochs_40_roberta_resnet18'

    print("Training model...")

    best_model, results = train_and_validate(
        model, train_loader, dev_loader, 
        optimizer, criterions, num_epochs, 
        experiment_name=experiment_name, 
        device='cuda', save_dir='./saved_results'
    )
    
    print("Training complete.")

    true_and_pred_labels = test_inference(best_model, test_loader, criterions, experiment_name)

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

