import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import os


# Function to load a checkpoint
def load_checkpoint(filepath, device="cpu"):
    try:
        checkpoint = torch.load(filepath, map_location=device)
        architecture = checkpoint['architecture']
        class_to_idx = checkpoint['class_to_idx']

        # Load the correct model architecture
        if architecture == "efficientnet":
            model = models.efficientnet_b0(weights="DEFAULT")
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_to_idx))
        elif architecture == "resnet":
            model = models.resnet50(weights="DEFAULT")
            model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))
        elif architecture == "vgg16":
            model = models.vgg16(weights="DEFAULT")
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, len(class_to_idx))
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Load the state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.class_to_idx = class_to_idx
        model.idx_to_class = {v: k for k, v in class_to_idx.items()}  # Reverse mapping
        model.eval()
        return model

    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    except KeyError as e:
        raise ValueError(f"Invalid checkpoint file. Missing key: {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the checkpoint: {e}")


# Function to process the image
def process_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        raise RuntimeError(f"An error occurred while processing the image: {e}")


# Function to make predictions
def predict(image_path, model, topk=5, device="cpu"):
    # Validate topk
    if topk <= 0:
        raise ValueError(f"Invalid value for topk: {topk}. It should be greater than 0.")

    image = process_image(image_path).to(device)
    model = model.to(device)

    try:
        with torch.no_grad():
            outputs = model(image)
            probs, indices = torch.topk(torch.softmax(outputs, dim=1), topk)

        probs = probs.squeeze().cpu().numpy()
        indices = indices.squeeze().cpu().numpy()

        # Map indices to class labels
        class_indices = [model.idx_to_class[idx] for idx in indices]

        return probs, class_indices

    except KeyError as e:
        raise RuntimeError(f"An error occurred during prediction. Missing key in model: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during prediction: {e}")

