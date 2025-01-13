import torch
from torchvision import models

import torch
from torchvision import models
from torch import nn  # Importing nn for model definition

# Function to load the checkpoint
def load_checkpoint(filepath):
    # Initialize the EfficientNet model (use the same architecture as during training)
    model = models.efficientnet_b0(weights=None)  # No pre-trained weights since we load custom weights
    
    # Load the checkpoint
    checkpoint = torch.load(filepath, map_location=torch.device("cpu"))  # Adjust for device
    
    # Check if checkpoint contains a 'state_dict'
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    return model

# Example usage
model = load_checkpoint("best_model_compressed.pth")
model.eval()  # Set the model to evaluation mode
print("EfficientNet model successfully loaded!")
