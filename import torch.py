import torch
import gzip
import shutil

# Path to the original .pth file
input_path = "best_model.pth"

# Path to the compressed .pth file (will still have the .pth extension)
output_path = "best_model_compressed.pth"

# Compress the .pth file using gzip
with open(input_path, 'rb') as f_in:
    with gzip.open(output_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print(f"Compressed file saved as: {output_path}")
