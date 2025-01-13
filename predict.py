import argparse
import torch
from torchvision import transforms
from PIL import Image
from load_checkpoint import load_checkpoint
import matplotlib.pyplot as plt

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

# Function to predict the class of an image
def predict(image_path, model, topk=5, device="cpu"):
    image = process_image(image_path)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        probs, indices = torch.topk(torch.softmax(outputs, dim=1), topk)
        return probs.squeeze().cpu().numpy(), indices.squeeze().cpu().numpy()

# Function to display an image along with the top 5 classes
def display_prediction(image_path, model, device="cpu"):
    probs, classes = predict(image_path, model, device=device)
    print(f"Top 5 predictions: {classes} with probabilities: {probs}")
    
    # Display the image
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict the class of an input image using a trained model")
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to the model checkpoint")
    parser.add_argument("--image_path", type=str, default="assets/Flowers.png", help="Path to the input image")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run inference on")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to return")
    args = parser.parse_args()

    print(f"Model path: {args.model_path}")
    print(f"Image path: {args.image_path}")
    print(f"Device: {args.device}")
    print(f"Top-K: {args.topk}")

    # Load the model
    try:
        model = load_checkpoint(args.model_path)
        model = model.to(args.device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Display the prediction
    display_prediction(args.image_path, model, device=args.device)

if __name__ == "__main__":
    main()
