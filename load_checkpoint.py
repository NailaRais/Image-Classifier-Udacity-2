import torch
from transformers import AutoModelForImageClassification

# Function to load the model from Hugging Face
def load_checkpoint(model_path):
    """
    Load a pre-trained model from the Hugging Face Hub.

    Args:
        model_path (str): The Hugging Face model repository ID or local directory.

    Returns:
        model: The loaded PyTorch model.
    """
    # Load model directly from Hugging Face repository or local path
    model = AutoModelForImageClassification.from_pretrained(model_path)
    return model

# Example usage
if __name__ == "__main__":
    # Replace with your Hugging Face model repository ID or local path
    model_path = "nailarais1/image-classifier-efficientnet"  # model path
    try:
        model = load_checkpoint(model_path)
        model.eval()  # Set the model to evaluation mode
        print("Model successfully loaded from Hugging Face!")
    except Exception as e:
        print(f"Error loading model: {e}")

