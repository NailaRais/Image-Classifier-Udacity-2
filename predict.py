import argparse
import json
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

# Function to load a checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location="cuda" if torch.cuda.is_available() else "cpu")
    
    # Manually specify architecture as 'efficientnet' for this case
    architecture = 'efficientnet'  # Assuming EfficientNet was used for training
    
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

# Function to process the image
def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Function to make predictions
def predict(image_path, model, topk=5, device="cpu"):
    image = process_image(image_path).to(device)
    model = model.to(device)

    with torch.no_grad():
        outputs = model(image)
        probs, indices = torch.topk(torch.softmax(outputs, dim=1), topk)

    probs = probs.squeeze().cpu().numpy()
    indices = indices.squeeze().cpu().numpy()
    class_indices = [model.idx_to_class[idx] for idx in indices]

    return probs, class_indices

# Argument parser
def get_input_args():
    parser = argparse.ArgumentParser(description="Predict the class of an image using a trained model")
    parser.add_argument('--image_path', type=str, default='./assets/Flowers.png', help="Path to input image")
    parser.add_argument('--checkpoint', type=str, default='./checkpoint.pth', help="Path to model checkpoint")
    parser.add_argument('--top_k', type=int, default=5, help="Number of top predictions to return")
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help="Path to JSON file mapping categories to names")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use for inference ('cpu' or 'gpu')")
    parser.add_argument('--output_file', type=str, help="File to save prediction results")

    return parser.parse_args()

# Main function
def main():
    # Get the input arguments
    args = get_input_args()

    # Load the model
    model = load_checkpoint(args.checkpoint)

    # Load category names
    with open(args.category_names, 'r') as json_file:
        category_names = json.load(json_file)

    # Set device
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"

    # Predict
    probs, class_indices = predict(args.image_path, model, topk=args.top_k, device=device)

    # Map class indices to labels
    labels = [category_names.get(str(index), "Unknown") for index in class_indices]

    # Save predictions to output file
    if args.output_file:
        with open(args.output_file, 'w') as output_file:
            for i, (label, prob) in enumerate(zip(labels, probs)):
                output_file.write(f"{i + 1}: {label} with probability {prob:.4f}\n")

    # Display predictions
    for i, (label, prob) in enumerate(zip(labels, probs)):
        print(f"{i + 1}: {label} with probability {prob:.4f}")

    print("Finished Predicting!")

if __name__ == "__main__":
    main()

