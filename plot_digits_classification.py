"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Import necessary libraries
from sklearn import metrics
from utils import read_data, preprocess_data, split_train_dev_test, train_model, predict_and_eval, visualize_confusion_matrix

# Read images and labels from the dataset
images, labels = read_data()

# Preprocess the image data
data = preprocess_data(images)

test_size = 0.2  # Proportion of data to be used for testing
dev_size = 0.2  # Proportion of data to be used for development

# Split the data and labels into training, development, and testing sets
X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(data, labels, test_size, dev_size)

model_parameters = {"gamma": 0.001} # Hyperparameters
model_type = "svm" # Model to train

# Train the model
model = train_model(X_train, y_train, model_parameters, model_type)

# Predict and evaluate the performance
predicted, report = predict_and_eval(model, X_test, y_test)

print(
    f"Classification report for classifier {model}:\n"
    f"{report}\n"
)

# Visualize the confusion matrix
visualize_confusion_matrix(y_test, predicted)
