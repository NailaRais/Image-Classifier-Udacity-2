import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import json
import gdown

# Function to download the model from Google Drive
def download_model():
    url = 'https://drive.google.com/uc?id=145gmljz5SYogBe9e-UmEC8KKtWk7hQ5q'
    output = 'checkpoint.pth'
    gdown.download(url, output, quiet=False)

# Function to process the image
def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image

# Function to load the pretrained EfficientNet model
def load_model():
    model = models.efficientnet_b0(weights="DEFAULT")
    model.eval()
    return model

# Function to load class names from JSON
def load_class_names(json_file):
    with open(json_file, 'r') as f:
        class_names = json.load(f)
    return class_names

# Function to predict the class of an image
def predict(image_path, model, topk=5, device="cpu"):
    image = process_image(image_path)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        probs, indices = torch.topk(torch.softmax(outputs, dim=1), topk)
        return probs.squeeze().cpu().numpy(), indices.squeeze().cpu().numpy()

# Function to display an image along with the top 5 classes and save the results to a text file
def display_prediction(image_path, model, output_file="prediction_results.txt", device="cpu"):
    probs, indices = predict(image_path, model, device=device)
    class_names = load_class_names("cat_to_name.json")  # Load class names
    named_classes = [class_names[str(idx)] for idx in indices]  # Map to class names
    result = f"Top 5 predictions: {named_classes} with probabilities: {probs}\n"
    
    # Save the results to a text file
    with open(output_file, "w") as file:
        file.write(result)
    print(f"Results saved to {output_file}")
    
    # Display the image
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def main():
    # Download the model
    download_model()

    # Load the model
    model = load_model()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict the class of an input image using a pretrained model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run inference on")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to return")
    parser.add_argument("--output_file", type=str, default="results/prediction_results.txt", help="Path to save prediction results")
    args = parser.parse_args()

    print(f"Image path: {args.image_path}")
    print(f"Device: {args.device}")
    print(f"Top-K: {args.topk}")
    print(f"Output file: {args.output_file}")

    model = model.to(args.device)

    # Display the prediction and save results to a file
    display_prediction(args.image_path, model, output_file=args.output_file, device=args.device)

if __name__ == "__main__":
    main()
