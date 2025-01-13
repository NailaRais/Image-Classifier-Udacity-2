import torch
from torchvision import transforms
from PIL import Image
from load_checkpoint import load_checkpoint
import matplotlib.pyplot as plt
from torch import nn  # Importing nn for model definition

# Set the device (CPU or GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to process the image
def process_image(image_path):
    """
    Preprocesses the image for the model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Processed image tensor.
    """
    image = Image.open(image_path).convert("RGB")  # Ensure RGB image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to predict the class of an image
def predict(image_path, model, topk=5):
    image = process_image(image_path)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        probs, indices = torch.topk(torch.softmax(outputs, dim=1), topk)
        return probs.squeeze().cpu().numpy(), indices.squeeze().cpu().numpy()

# Function to display an image along with the top 5 classes
def display_prediction(image_path, model):
    probs, classes = predict(image_path, model)
    print(f'Top 5 predictions: {classes} with probabilities: {probs}')
    
    # Display the image
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Example usage
model = load_checkpoint('best_model_compressed.pth')
image_path = 'assets/Flowers.png'
display_prediction(image_path, model)
