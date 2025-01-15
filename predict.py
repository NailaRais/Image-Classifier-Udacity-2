import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import json
import gdown
import os

def download_model():
    url = 'https://drive.google.com/uc?id=145gmljz5SYogBe9e-UmEC8KKtWk7hQ5q'
    output = 'checkpoint.pth'
    gdown.download(url, output, quiet=False)

def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def load_model():
    model = models.efficientnet_b0(weights="DEFAULT")
    model.eval()
    return model

def load_class_names(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def predict(image_path, model, topk=5, device="cpu"):
    image = process_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs, indices = torch.topk(torch.softmax(outputs, dim=1), topk)
        
        class_names = load_class_names("cat_to_name.json")
        named_classes = [
            class_names.get(str(idx), f"Unknown class {idx}") for idx in indices.squeeze().cpu().numpy()
        ]
        return probs.squeeze().cpu().numpy(), named_classes

def display_prediction(image_path, model, output_file="results/prediction_results.txt", device="cpu"):
    probs, named_classes = predict(image_path, model, device=device)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    result = f"Top 5 predictions: {named_classes} with probabilities: {probs}\n"
    with open(output_file, "w") as file:
        file.write(result)
    print(f"Results saved to {output_file}")
    
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def main():
    download_model()
    model = load_model()

    parser = argparse.ArgumentParser(description="Predict the class of an input image using a pretrained model")
    parser.add_argument("--image_path", type=str, default="assets/Flowers.png", help="Path to the input image")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run inference on")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to return")
    parser.add_argument("--output_file", type=str, default="results/prediction_results.txt", help="Path to save prediction results")
    args = parser.parse_args()

    print(f"Image path: {args.image_path}")
    print(f"Device: {args.device}")
    print(f"Top-K: {args.topk}")
    print(f"Output file: {args.output_file}")

    model = model.to(args.device)
    display_prediction(args.image_path, model, output_file=args.output_file, device=args.device)

if __name__ == "__main__":
    main()
