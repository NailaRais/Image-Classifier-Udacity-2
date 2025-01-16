import argparse
import json
import numpy as np
import fmodel

# Argument parser
parser.add_argument('input', default='./flower_data/test/1/image_06752.jpg', nargs='?', type=str, help="Path to input image")
parser.add_argument('--dir', default="./flower_data/", type=str, help="Path to data directory")
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', type=str, help="Path to model checkpoint")
parser.add_argument('--top_k', default=5, type=int, help="Number of top predictions to return")
parser.add_argument('--category_names', default='cat_to_name.json', type=str, help="Path to JSON file mapping categories to names")
parser.add_argument('--gpu', default="gpu", type=str, help="Device to use for inference ('cpu' or 'gpu')")

args = parser.parse_args()

# Main function
def main():
    # Load the model
    model = fmodel.load_checkpoint(args.checkpoint)

    # Load category names
    with open(args.category_names, 'r') as json_file:
        category_names = json.load(json_file)

    # Set device
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"

    # Predict
    probs, class_indices = fmodel.predict(args.image_path, model, topk=args.top_k, device=device)

    # Map class indices to labels
    labels = [category_names.get(str(index), "Unknown") for index in class_indices]

    # Display predictions
    for i, (label, prob) in enumerate(zip(labels, probs)):
        print(f"{i + 1}: {label} with probability {prob:.4f}")

    # Save predictions to output file
    if args.output_file:
        with open(args.output_file, 'w') as f:
            for i, (label, prob) in enumerate(zip(labels, probs)):
                f.write(f"{i + 1}: {label} with probability {prob:.4f}\n")
        print(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    main()

