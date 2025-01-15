import torch
from torchvision import transforms, models
from PIL import Image
import argparse
import json

# Function to load the checkpoint
def load_checkpoint(filepath, device):
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
    model.idx_to_class = {v: k for k, v in class_to_idx.items()}  # Reverse the class_to_idx
    model.eval()
    return model

# Function to process the image
def process_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to predict the class of an image
def predict(image_path, model, topk=5, device="cpu"):
    image = process_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs, indices = torch.topk(torch.softmax(outputs, dim=1), topk)

    # Convert indices to class names
    probs = probs.squeeze().cpu().numpy()
    indices = indices.squeeze().cpu().numpy()
    class_names = [model.idx_to_class[idx] for idx in indices]

    return probs, class_names

# Main function to parse arguments and make predictions
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict the class of an input image using a trained model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to return")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run inference on")
    args = parser.parse_args()

    # Load the model
    model = load_checkpoint(args.checkpoint, args.device)
    model = model.to(args.device)

    # Predict the class
    probs, class_names = predict(args.image_path, model, args.topk, device=args.device)

    # Print the results
    print(f"Top {args.topk} Predictions:")
    for i, (prob, class_name) in enumerate(zip(probs, class_names)):
        print(f"{i + 1}: {class_name} with probability {prob:.4f}")

if __name__ == "__main__":
    main()

