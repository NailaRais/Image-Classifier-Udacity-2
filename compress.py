import gzip
import shutil

# Compress the .pth file using gzip
with open('best_model.pth', 'rb') as f_in:
    with gzip.open('best_model.pth.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
