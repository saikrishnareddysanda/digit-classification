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
import pandas as pd

# Read images and labels from the dataset
images, labels = read_data()

num_runs = 5

results_list = []

for run in range(num_runs):
    run_dict_metrics = {}
    for model_type in ['svm', 'tree']:

        if model_type == 'svm':
            gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
            C_ranges = [0.1, 1, 2, 5, 10]
            hyperparameter_names = ["gamma", "C"]
            hyperparameter_lists = [gamma_ranges, C_ranges]
        elif model_type == 'tree':
            criterion = ['gini', 'entropy', 'log_loss']
            max_depth = [None, 100, 200]
            hyperparameter_names = ["criterion", "max_depth"]
            hyperparameter_lists = [criterion, max_depth]   

        list_of_all_param_combination = create_combination_dictionaries_from_lists(
            hyperparameter_names, hyperparameter_lists
        )

        test_size_list = [0.2]
        dev_size_list = [0.2]

        # print("Number of total samples in the datset:" , images.shape[0])
        # print("Height of the image is:", images.shape[1])
        # print("Width of the image is:", images.shape[2])

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
                X_train, y_train, X_dev, y_dev, list_of_all_param_combination, model_type
            )

            train_acc = get_acc(best_model, X_train, y_train)
            test_acc = get_acc(best_model, X_test, y_test)
            
            metrics = {
                "run":run,
                "model_type": model_type,
                "train_acc": train_acc,
                "dev_acc": best_dev_accuracy,
                "test_acc": test_acc
            }
            print(metrics)
            results_list.append(metrics)

print(results_list)
print(pd.DataFrame(results_list))
print(pd.DataFrame(results_list).groupby('model_type').describe().T)