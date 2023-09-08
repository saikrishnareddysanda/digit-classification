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
from itertools import product
from utils import (
    read_data,
    preprocess_data,
    split_train_dev_test,
    get_acc,
    create_combination_dictionaries_from_lists,
    tune_hparams,
)

# Read images and labels from the dataset
images, labels = read_data()

gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]
hyperparameter_names = ["gamma", "C"]
hyperparameter_lists = [gamma_ranges, C_ranges]

list_of_all_param_combination = create_combination_dictionaries_from_lists(
    hyperparameter_names, hyperparameter_lists
)

test_size_list = [0.1, 0.2, 0.3]
dev_size_list = [0.1, 0.2, 0.3]

for test_size, dev_size in product(test_size_list, dev_size_list):
    train_size = round((1 - (test_size + dev_size)), 2)

    # Split the data and labels into training, development, and testing sets
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(
        images, labels, test_size, dev_size
    )

    # Preprocess the image data
    X_train = preprocess_data(X_train)
    X_dev = preprocess_data(X_dev)
    X_test = preprocess_data(X_test)

    best_hparams, best_model, best_dev_accuracy = tune_hparams(
        X_train, y_train, X_dev, y_dev, list_of_all_param_combination
    )

    train_acc = get_acc(best_model, X_train, y_train)
    test_acc = get_acc(best_model, X_test, y_test)
    print(
        "test_size = ", test_size,
        "dev_size = ", dev_size,
        "train_size = ", train_size,
        "train_acc = ", train_acc,
        "dev_acc = ", best_dev_accuracy,
        "test_acc = ", test_acc
    )
    print("Best Hyperparamters:", best_hparams)
