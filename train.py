import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from torch.cuda.amp import autocast, GradScaler
import argparse


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a deep learning model")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to the dataset directory")
    parser.add_argument("--model_architecture", type=str, choices=["efficientnet", "resnet", "vgg16"], default="resnet",
                        help="Model architecture to use for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units in the classifier")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train the model on")
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }

    # Load datasets
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(args.data_dir, 'train'), data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(args.data_dir, 'valid'), data_transforms['val']),
    }

    # Data loaders
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=2)
        for x in ['train', 'val']
    }

    # Get class-to-index mapping
    class_to_idx = image_datasets['train'].class_to_idx

    # Select model architecture
    if args.model_architecture == "efficientnet":
        model = models.efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_to_idx))
    elif args.model_architecture == "resnet":
        model = models.resnet50(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
    elif args.model_architecture == "vgg16":
        model = models.vgg16(weights="DEFAULT")
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(class_to_idx))

    model = model.to(args.device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Mixed Precision Training
    scaler = GradScaler() if args.device == "cuda" else None

    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()

            if scaler:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

        train_acc = correct_preds / total_preds
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {running_loss:.4f}, Accuracy: {train_acc * 100:.2f}%")

        # Validation phase
        model.eval()
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct_preds += (preds == labels).sum().item()
                total_preds += labels.size(0)

        val_acc = correct_preds / total_preds
        print(f"Validation Accuracy: {val_acc * 100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_to_idx': class_to_idx,
            }, 'best_model.pth')

        scheduler.step()

    print(f"Training complete. Best validation accuracy: {best_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
