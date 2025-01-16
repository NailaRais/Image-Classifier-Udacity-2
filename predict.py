import argparse
import json
import torch
import numpy as np
import fmodel

# Argument parser
parser = argparse.ArgumentParser(description="Predict the class of an image using a trained model")
parser.add_argument('--image_path', default='./flowers/test/1/image_06752.jpg', type=str, help="Path to input image")
parser.add_argument('--checkpoint', default='./checkpoint.pth', type=str, help="Path to model checkpoint")
parser.add_argument('--top_k', default=5, type=int, help="Number of top predictions to return")
parser.add_argument('--category_names', default='cat_to_name.json', type=str, help="Path to JSON file mapping categories to names")
parser.add_argument('--device', default="gpu", type=str, help="Device to use for inference ('cpu' or 'gpu')")
parser.add_argument('--output_file', default='results/prediction_results.txt', type=str, help="File to save prediction results")

args = parser.parse_args()

# Main function
def main():
    # Load the model
    try:
        device = "cuda" if args.gpu == "gpu" and torch.cuda.is_available() else "cpu"
        model = fmodel.load_checkpoint(args.checkpoint, device=device)
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return

    # Load category names
    try:
        with open(args.category_names, 'r') as json_file:
            category_names = json.load(json_file)
    except Exception as e:
        print(f"Error loading category names: {e}")
        return

    # Predict
    try:
        probs, class_indices = fmodel.predict(args.input, model, topk=args.top_k, device=device)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # Map class indices to labels
    labels = [category_names.get(str(index), "Unknown") for index in class_indices]

    # Display predictions
    print("\nPredictions:")
    for i, (label, prob) in enumerate(zip(labels, probs)):
        print(f"{i + 1}: {label} with probability {prob:.4f}")

    print("\nFinished Predicting!")


if __name__ == "__main__":
    main()
