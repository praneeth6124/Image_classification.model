# Image Classification with ANN and CNN using EarlyStopping, Dropout, and MaxPooling

## Project Overview
This project involves building and comparing two image classification models using the CIFAR-10 dataset. The dataset consists of 60,000 32x32 color images in 10 different classes. The goal was to evaluate and optimize the performance of two models: a simple Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN), with EarlyStopping, Dropout, and MaxPooling regularization techniques applied to prevent overfitting and enhance model performance.

## Key Features
Dataset: CIFAR-10 dataset, containing 10 classes (e.g., airplane, automobile, bird, cat, etc.).

Preprocessing: Data scaling by dividing pixel values by 255 to normalize the images.

## Models Used:

ANN (Artificial Neural Network)

CNN (Convolutional Neural Network)

## Regularization Techniques:

EarlyStopping: To stop training early when validation performance stops improving, avoiding overfitting.

Dropout: Used to prevent overfitting by randomly deactivating a fraction of neurons during training.

MaxPooling: Applied after convolutional layers to reduce spatial dimensions, help with model efficiency, and prevent overfitting.

## Results
ANN Model:

Training Accuracy: 49.57%

Training Loss: 1.4366

CNN Model:

Training Accuracy: 79.08%

Training Loss: 0.5840

Validation Accuracy: 70.07%

Validation Loss: 0.9144

Both models were evaluated using accuracy, loss, validation accuracy, and validation loss metrics. The CNN model significantly outperformed the ANN model in terms of both training and validation accuracy, demonstrating the power of convolutional layers in image classification tasks.

## Classification Report Comparison
A comparison of the classification reports from both models shows the effectiveness of CNN in classifying images from the CIFAR-10 dataset with higher precision, recall, and F1-score across most classes.

## Technologies Used
Python

TensorFlow / Keras

CIFAR-10 dataset

Libraries: numpy, matplotlib, scikit-learn, tensorflow

## Model Architecture
### ANN:

A simple neural network consisting of several dense layers.

The model was trained and evaluated without convolutional layers, resulting in limited performance.

### CNN:

The CNN architecture includes convolutional layers followed by MaxPooling and dense layers for final classification.

MaxPooling was applied after convolutional layers to reduce spatial dimensions, helping with computational efficiency and improving model performance.

Regularization techniques like Dropout and EarlyStopping were applied to improve generalization and prevent overfitting.