import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

def compute_metrics(true_labels, predictions):
    """
    Compute classification metrics including accuracy, per-class F1 scores, and weighted average F1 score.
    """
    
    b_accuracy = round(balanced_accuracy_score(true_labels, predictions), 3)   
    report = classification_report(
        true_labels, predictions, output_dict=True, zero_division=0
    )
    
    per_class_f1 = {label: round(values["f1-score"], 3) for label, values in report.items() if label.isdigit()}
    weighted_f1 = round(report["weighted avg"]["f1-score"], 3)

    
    metrics = {
        "balanced_acc": round(b_accuracy, 3),
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1
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

def analyze_results_per_class(true_labels, predicted_labels, class_names, task_name="Sentiment", mode="confusion_matrix", save_path="./images"):
    """
    Analyze results per class with confusion matrix, classification report, or ROC curves.

    Args:
        true_labels (list or np.ndarray): True labels for the task.
        predicted_labels (list or np.ndarray): Predicted labels from the model.
        class_names (list): List of class names.
        task_name (str): Name of the task (e.g., "Sentiment" or "Emotion").
        mode (str): The type of analysis. Options: "confusion_matrix", "classification_report", "roc_curve".
    """
    os.makedirs('images', exist_ok=True)

    save_path = os.path.join(save_path, task_name)
    os.makedirs(save_path, exist_ok=True)
    
    if mode == "confusion_matrix":
        
        cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(class_names)))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Confusion Matrix for {task_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)

        plt.savefig(f'{save_path}/confusion_matrix.png')
        plt.close()

    elif mode == "classification_report":
        
        report = classification_report(true_labels, predicted_labels, target_names=class_names, zero_division=0)
        open_path = os.path.join(save_path, "classification_report.txt")
        with open(open_path, 'w') as f:
            f.write(f"Classification Report for {task_name}:\n\n")
            f.write(report)

    elif mode == "roc_curve":
        
        true_binarized = label_binarize(true_labels, classes=range(len(class_names)))
        predicted_binarized = label_binarize(predicted_labels, classes=range(len(class_names)))
        plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(true_binarized[:, i], predicted_binarized[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")  
        plt.title(f"ROC Curve for {task_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.savefig(f'images/alternating/{task_name}_roc_curve.png')
        plt.close()

def compute_emotion_class_weights(train_set, device, normalize=True):
    """
    Computes normalized inverse-frequency class weights for emotion labels 
    from the given training set and moves the result to the specified device.

    Args:
        train_set: Dataset object that contains samples and a method `get_emotions_dicts()`.
        device: The target device (e.g., 'cpu' or 'cuda') to move the tensor.
        normalize (bool): Whether to normalize the class weights to sum to 1. Default is True.

    Returns:
        torch.Tensor: Tensor containing class weights on the specified device.
    """
    
    emotion_labels = [sample[2] for sample in train_set.samples]

    
    unique_classes, counts = np.unique(emotion_labels, return_counts=True)
    print("Class labels:", unique_classes)
    print("Class counts:", counts)

    
    _, str_to_int = train_set.get_emotions_dicts()

    num_classes = len(str_to_int)
    ordered_counts = [0] * num_classes

    
    for class_label, count in zip(unique_classes, counts):
        class_idx = str_to_int[class_label]   
        ordered_counts[class_idx] = count

    ordered_counts = np.array(ordered_counts)
    print("Ordered counts:", ordered_counts)

    
    inverse_freq = 1.0 / np.maximum(ordered_counts, 1)

    
    emotions_class_weights = torch.tensor(inverse_freq, dtype=torch.float32)

    
    if normalize:
        emotions_class_weights = emotions_class_weights / emotions_class_weights.sum()

    
    emotions_class_weights = emotions_class_weights.to(device)

    return emotions_class_weights
