
# Predicting Side Effects from Clinical Trial Data

This project is a simple neural network model designed to predict whether a patient will experience side effects from a new drug based on their age. The model is created using Keras, a high-level neural networks API, and TensorFlow, an open-source machine learning framework. This project is part of a learning exercise to understand the basics of neural network modeling and data analysis.

## Project Overview

### Scientific Data Analysis

The dataset used in this project is based on a simulated clinical trial for a new drug, with the following characteristics:

- **Participants:** 2500 individuals
- **Age Range:** 13 to 100 years old
- **Age Groups:**
  - Under 65: 50% of participants
  - Over 65: 50% of participants
- **Side Effects:**
  - Under 65: 95% did not experience side effects
  - Over 65: 95% did experience side effects

The goal is to build a model that predicts whether a patient experiences side effects based on their age.

## Model Architecture

The model is a simple sequential neural network consisting of:

1. **Input Layer:** Accepts the patient's age as the input feature.
2. **Hidden Layers:** Several dense layers with ReLU activation functions to capture complex patterns in the data.
3. **Output Layer:** A single neuron with a sigmoid activation function to predict the likelihood of experiencing side effects (output of 0 or 1).

## Installation

To run this project, you need to have Python installed along with the following libraries:

- TensorFlow
- Keras (included with TensorFlow)

You can install the necessary packages using pip:

```bash
pip install tensorflow
```

## Usage

1. **Data Preparation:** Ensure that the data is preprocessed correctly. The age feature should be scaled appropriately (e.g., using MinMaxScaler or StandardScaler).
2. **Model Training:** Train the model using the provided dataset. The model can be compiled and trained using the Adam optimizer and binary crossentropy loss function.
3. **Prediction:** After training, the model can be used to predict side effects in new patients based on their age.


## Contributing

This project is for learning purposes, and contributions are welcome to improve the model or enhance the dataset.
