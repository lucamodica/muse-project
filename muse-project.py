


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
from utils import plot_metrics, analyze_results_per_class

def process_task(model, a, t, classifier, weight_size, labels, criterion, device='cuda'):

    softmax = nn.Softmax(dim=1)

    fused = torch.cat([a, t], dim=-1)

    # --- Compute logits ---
    logits_fused = classifier(fused)

    # 2) Audio-only
    logits_a = (
        torch.mm(a, torch.transpose(classifier.weight[:, :weight_size / 2], 0, 1))
        + classifier.bias / 2
    )

    # 3) Text-only
    logits_t = (
        torch.mm(t, torch.transpose(classifier.weight[:, weight_size / 2 :], 0, 1))
        + classifier.bias / 2
    )

    # --- Compute loss ---
    loss = criterion(logits_fused, labels)

    # --- Predictions ---
    preds_fused = torch.argmax(softmax(logits_fused), dim=1).detach().cpu().numpy()
    preds_audio = torch.argmax(softmax(logits_a),      dim=1).detach().cpu().numpy()
    preds_text  = torch.argmax(softmax(logits_t),      dim=1).detach().cpu().numpy()

    return loss, preds_fused, preds_audio, preds_text

def train_one_epoch(model, dataloader, optimizer, criterions, emotion_reg=0.6, sentiment_reg=0.4, device='cuda'):
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
            a=a, 
            t=t, 
            classifier=model.emotion_classifier,
            weight_size=model.emotion_classifier.weight.size(1),
            labels=emotion_labels, 
            criterion=criterions['emotion']
        )
        losses['emotion'] += emotion_loss.item()
        metrics['emotion']['fused'].extend(e_fused)
        metrics['emotion']['audio'].extend(e_audio)
        metrics['emotion']['text'].extend(e_text)
        metrics['emotion']['labels'].extend(emotion_labels.cpu().numpy())

        # SENTIMENT TASK
        sentiment_loss, s_fused, s_audio, s_text = process_task(
            model=model, 
            a=a, 
            t=t, 
            classifier=model.sentiment_classifier,
            weight_size=model.sentiment_classifier.weight.size(1),
            labels=sentiment_labels, 
            criterion=criterions['sentiment']
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

def validate_one_epoch(model, dataloader, criterions, device='cuda'):
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
                a=a, 
                t=t, 
                classifier=model.emotion_classifier,
                weight_size=model.emotion_classifier.weight.size(1),
                labels=emotion_labels, 
                criterion=criterions['emotion']
            )
            losses['emotion'] += emotion_loss.item()
            metrics['emotion']['fused'].extend(e_fused)
            metrics['emotion']['audio'].extend(e_audio)
            metrics['emotion']['text'].extend(e_text)
            metrics['emotion']['labels'].extend(emotion_labels.cpu().numpy())

            sentiment_loss, s_fused, s_audio, s_text = process_task(
                model=model, 
                a=a, 
                t=t, 
                classifier=model.sentiment_classifier,
                weight_size=model.sentiment_classifier.weight.size(1),
                labels=sentiment_labels, 
                criterion=criterions['sentiment']
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

    num_epochs = 5
    num_emotions = len(train_set.get_emotions_dicts()[0].values())
    num_sentiments = len(train_set.get_sentiments_dicts()[0].values())

    model = MultimodalClassifier(
        text_model_name=text_model_name,
        text_fine_tune=True,
        unfreeze_last_n_text=6,
        audio_encoder=resnet18(modality='audio'),
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

    #load from saved_results
    loaded_results = joblib.load('./saved_results/TEST_results.pkl')


    plot_metrics(loaded_results, metric='macro_f1', modality='fused', task='emotions')
    true_and_pred_labels = test_inference(model, test_loader, criterions, experiment_name)


    for mode in ['confusion_matrix', 'classification_report', 'roc_curve']:
        analyze_results_per_class(
            true_and_pred_labels['emotion']['true'], 
            true_and_pred_labels['emotion']['pred'], 
            unique_classes,
            task_name="Emotions",
            mode=mode
        )

if __name__ == "__main__":
    main()

