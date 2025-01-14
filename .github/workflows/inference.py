import torch
from torchvision import transforms
from PIL import Image
from load_checkpoint import load_checkpoint
import matplotlib.pyplot as plt
from torch import nn  # Importing nn for model definition

# Function to process the image
def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Function to display the image
def display_image(image, top_k_classes, top_k_probs):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.barh(top_k_classes, top_k_probs)
    plt.xlabel("Probability")
    plt.title("Top K Predictions")
    plt.show()

# Main function
if __name__ == '__main__':
    # Load the model
    model = load_checkpoint('best_model.pth')
    model.eval()

    # Process the image
    image_path = 'assets/Flowers.png'  # Update this path to the correct image
    image = process_image(image_path)

    # Make predictions
    with torch.no_grad():
        outputs = model(image)
        probs, indices = torch.topk(torch.softmax(outputs, dim=1), 5)
        probs = probs.squeeze().cpu().numpy()
        indices = indices.squeeze().cpu().numpy()

    # Display the results
    display_image(Image.open(image_path), indices, probs)
