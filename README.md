[![Predict](https://github.com/NailaRais/Image-Classifier-Udacity/actions/workflows/predict.yml/badge.svg)](https://github.com/NailaRais/Image-Classifier-Udacity/actions/workflows/predict.yml)
[![Train Multiple Models](https://github.com/NailaRais/Image-Classifier-Udacity/actions/workflows/train_model.yml/badge.svg)](https://github.com/NailaRais/Image-Classifier-Udacity/actions/workflows/train_model.yml)
[![Run Jupyter Notebook with Papermill](https://github.com/NailaRais/Image-Classifier-Udacity/actions/workflows/notebook.yml/badge.svg)](https://github.com/NailaRais/Image-Classifier-Udacity/actions/workflows/notebook.yml)

# **Image Classifier Project**

This project implements an image classifier using PyTorch. It allows users to train a model on a dataset of images and make predictions on new images with ease.

---

## **Requirements**
- Python 3.8 or later
- PyTorch
- torchvision
- Pillow
- numpy
- tqdm
- matplotlib

---

## **Installation**

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/NailaRais/Image-Classifier-Udacity.git
   cd your-repo-name
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

### **1. Training the Model**
Train the model with your dataset by running:
   ```bash
   python train.py
   ```
During execution, you will be prompted to specify:
- **Model Architecture:** Choose the base model (e.g., ResNet, VGG).
- **Learning Rate:** Set the learning rate for training.
- **Hidden Units:** Define the number of hidden units in the classifier.
- **Number of Epochs:** Specify how many epochs the model should train.

The trained model is saved as `best_model.pth` in the current directory.

---

### **2. Making Predictions**
To predict the class of a new image, use:
   ```bash
   python predict.py
   ```
You will be prompted to provide:
- **Model Architecture:** Specify the trained model's architecture.
- **Top Classes to Display:** Define how many top predictions should be shown.

The output will display the predicted class and the associated probabilities in the console.

---

## **GitHub Actions Workflow**
A GitHub Actions workflow is included to automatically train the model whenever changes are pushed to the `main` branch.  
You can find the workflow file here:
```
.github/workflows/train_model.yml
```

---

## **License**
This project is licensed under the MIT License.  
For more details, refer to the [LICENSE](LICENSE) file.
```



