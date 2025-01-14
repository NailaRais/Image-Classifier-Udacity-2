import torch
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import argparse

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

# Function to load the model checkpoint from the Hugging Face Hub
def load_checkpoint(repo_id, model_filename="pytorch_model.bin"):
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Create the model architecture and load the state dict
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(checkpoint['class_to_idx']))
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")
    return model, checkpoint['class_to_idx']

# Function to predict the class of an image
def predict(image_path, model, class_to_idx, topk=5, device="cpu"):
    image = process_image(image_path)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        probs, indices = torch.topk(torch.softmax(outputs, dim=1), topk)
        classes = [list(class_to_idx.keys())[list(class_to_idx.values()).index(idx)] for idx in indices.squeeze().tolist()]
        return probs.squeeze().cpu().numpy(), classes

# Function to display an image along with the top 5 classes and save the results to a text file
def display_prediction(image_path, model, class_to_idx, output_file="prediction_results.txt", device="cpu"):
    probs, classes = predict(image_path, model, class_to_idx, device=device)
    result = f"Top 5 predictions: {classes} with probabilities: {probs}\n"
    
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run inference and save prediction results to a file")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repository ID (e.g., 'nailarais1/image-classifier-efficientnet')")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run inference on")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to return")
    parser.add_argument("--output_file", type=str, default="results/inference_results.txt", help="Path to save prediction results")
    args = parser.parse_args()

    print(f"Repository ID: {args.repo_id}")
    print(f"Image path: {args.image_path}")
    print(f"Device: {args.device}")
    print(f"Top-K: {args.topk}")
    print(f"Output file: {args.output_file}")

    # Load the model
    try:
        model, class_to_idx = load_checkpoint(args.repo_id)
        model = model.to(args.device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Display the prediction and save results to a file
    display_prediction(args.image_path, model, class_to_idx, output_file=args.output_file, device=args.device)

if __name__ == "__main__":
    main()
