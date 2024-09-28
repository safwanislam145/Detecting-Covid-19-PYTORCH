# COVID-19 Detection from Chest X-ray Images

This repository contains a Jupyter Notebook for training a deep learning model to detect COVID-19 from chest X-ray images using PyTorch and torchvision.

## Tech Stack

- **Python**: The programming language used.
- **Jupyter Notebook**: An interactive environment for running Python code.
- **PyTorch**: A deep learning framework for building and training neural networks.
- **torchvision**: A package that provides image transformations and pre-trained models.
- **PIL (Python Imaging Library)**: For image processing.
- **Matplotlib**: For plotting and visualizing images.

## Dataset

The dataset used is the COVID-19 Radiography Dataset, which contains chest X-ray images categorized into three classes:
- Normal
- Viral Pneumonia
- COVID-19

## Notebook Overview

1. **Importing Libraries**
   - Imports necessary libraries and sets a random seed for reproducibility.
   - Prints the version of PyTorch being used.

2. **Preparing Training and Test Sets**
   - Lists the contents of the dataset directory.
   - Renames directories to standardize class names and creates a test directory if it doesn't exist.
   - Moves a subset of images from each class to the test directory for validation purposes.

3. **Creating Custom Dataset**
   - Defines a custom dataset class `ChestXRayDataset` that inherits from `torch.utils.data.Dataset`.
   - Initializes by loading image paths and applying transformations.
   - Defines methods to get the length of the dataset and to get an item by index.

4. **Image Transformations**
   - Defines transformations for training and test datasets using `torchvision.transforms.Compose`.
   - Training transformations include resizing, random horizontal flip, converting to tensor, and normalization.
   - Test transformations include resizing, converting to tensor, and normalization.

5. **Prepare DataLoader**
   - Creates training and test datasets using the custom dataset class and the defined transformations.
   - Initializes DataLoader objects for training and test datasets with a specified batch size.

6. **Data Visualization**
   - Defines a function `show_images` to visualize a batch of images along with their labels and predictions.
   - Displays a batch of images from the training and test datasets.

7. **Creating the Model**
   - Loads a pre-trained ResNet-18 model from torchvision.
   - Modifies the final fully connected layer to output three classes (normal, viral, covid).
   - Defines the loss function as cross-entropy loss and the optimizer as Adam.

8. **Training Model**
   - Defines a `train` function to train the model for a specified number of epochs.
   - Includes training and validation phases, with periodic evaluation and early stopping based on validation accuracy.
   - Prints training and validation losses and accuracy.

9. **Running the Training**
   - Measures the time taken to train the model for one epoch using the `%%time` magic command.
   - Calls the `train` function to start training.

10. **Displaying Predictions**
    - Calls the `show_preds` function to display predictions on a batch of test images.

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>