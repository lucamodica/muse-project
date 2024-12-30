import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader

############################
# 1) Define the train/val functions
############################

def train_one_epoch(model, dataloader, optimizer, criterion, device='cuda'):
    model.train()
    total_loss = 0.0

    for waveforms, sample_rate, texts, labels in dataloader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(waveforms, sample_rate, texts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_one_epoch(model, dataloader, criterion, device='cuda'):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for waveforms, sample_rate, texts, labels in dataloader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            logits = model(waveforms, sample_rate, texts)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Compute accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy


############################
# 2) Main function
############################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train CSV file.")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to validation CSV file.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--audio_fine_tune", action="store_true", help="Fine-tune the audio encoder?")
    parser.add_argument("--text_fine_tune", action="store_true", help="Fine-tune the text encoder?")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of emotion classes.")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    
    # 2.1) Build dataset & dataloaders
    # (Assuming you have the IemocapDataset from earlier)
    train_dataset = IemocapDataset(csv_file=args.train_csv)
    val_dataset   = IemocapDataset(csv_file=args.val_csv)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 2.2) Build model
    model = MultimodalClassifier(
        audio_model_name="facebook/wav2vec2-base",
        text_model_name="bert-base-uncased",
        audio_fine_tune=args.audio_fine_tune,
        text_fine_tune=args.text_fine_tune,
        num_classes=args.num_classes
    ).to(device)

    # 2.3) Define optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 2.4) Training Loop
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device=device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device=device)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    print("Training complete.")

############################
# 3) Entrypoint
############################

if __name__ == "__main__":
    main()
