"""
train.py
Fine-tunes a ResNet18 on breast histopathology image patches (IDC vs non-IDC).

Dataset: "Breast Histopathology Images" (Kaggle, paultimothymooney/breast-histopathology-images)
Expected folder layout after you download + reorganize (see prepare_data.py):

    data/
      train/
        class_0/   (non-IDC patches)
        class_1/   (IDC patches)
      val/
        class_0/
        class_1/

Run on Colab or locally with a GPU. CPU works too, just slower (try 2-3 epochs first
to confirm the pipeline runs end-to-end before committing to a long training run).

Usage:
    python train.py --data_dir ./data --epochs 8 --batch_size 64
"""

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def build_model(num_classes=2, freeze_backbone=False):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    # Swap the final layer for our binary classification head.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_dataloaders(data_dir, batch_size, img_size=224):
    # ImageNet normalization stats since we're using ImageNet-pretrained weights.
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_tf)
    val_ds = datasets.ImageFolder(f"{data_dir}/val", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Classes: {train_ds.classes}  (index 0 -> {train_ds.classes[0]}, index 1 -> {train_ds.classes[1]})")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    return train_loader, val_loader, train_ds.classes


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, classes = get_dataloaders(args.data_dir, args.batch_size)

    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        start = time.time()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - start

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss: {train_loss:.4f} | val_acc: {val_acc:.4f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
            }, args.output)
            print(f"  -> saved new best checkpoint ({val_acc:.4f}) to {args.output}")

    print(f"Done. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="model.pth")
    args = parser.parse_args()
    train(args)
