# CNN Classifier for MNIST Dataset

## Introduction
This repository contains code for implementing and comparing two different convolutional neural network (CNN) models for classifying the MNIST dataset. The models include a custom CNN architecture based on the PyTorch library, and the Faster R-CNN model. The comparison is based on various metrics such as accuracy, F1 score, loss, and training time.

## Dataset
The MNIST dataset is used for training and testing the models. You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

## CNN Architecture
### Custom CNN
The custom CNN architecture is implemented using PyTorch. It includes convolutional layers, pooling layers, fully connected layers, and various hyperparameters such as kernels, padding, stride, optimizers, and regularization.

### Faster R-CNN
The Faster R-CNN model is implemented using a pre-trained ResNet-50 backbone. The model is fine-tuned for the MNIST dataset.

## Data Loading
The dataset is loaded using a custom data loader that reads images and labels from the MNIST dataset files.

## Visualization
The code includes functions to visualize random images from the dataset.

## Training and Evaluation
The custom CNN is trained using the MNIST dataset, and its performance is evaluated based on accuracy. Similarly, the Faster R-CNN model is fine-tuned and evaluated.

## Comparison
The two models are compared using metrics such as accuracy, F1 score, loss, and training time. Additionally, the custom CNN is compared to models retrained on VGG16 and AlexNet architectures.

## Conclusion
The results and conclusions drawn from the experiments are discussed in the code.

## Usage
1. Download the MNIST dataset from the provided Kaggle link.
2. Update the file paths in the code to point to the dataset files.
3. Run the code to train and evaluate the custom CNN and Faster R-CNN models.

Feel free to explore and modify the code for further experiments and analysis.
