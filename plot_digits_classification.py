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
import argparse
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


parser = argparse.ArgumentParser()
parser.add_argument('--production_model', type=str, default='tree')
parser.add_argument('--candidate_model', type=str, default='svm')
parser.add_argument('--num_runs', type=int, default=1)
parser.add_argument('--test_size', type=float, default=0.1)
parser.add_argument('--dev_size', type=float, default=0.1)

args = parser.parse_args()

# Read images and labels from the dataset
images, labels = read_data()


test_size_list = [args.test_size]
dev_size_list = [args.dev_size]


def train_model(model_type, X_train, y_train, X_dev, y_dev):
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

    best_hparams, best_model, best_dev_accuracy = tune_hparams(
        X_train, y_train, X_dev, y_dev, list_of_all_param_combination, model_type
    )

    return best_model

def compare_models(prod_model, cand_model):
    for run in range(args.num_runs):
        # Split the data and labels into training, development, and testing sets
        X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(
            images, labels, test_size_list[0], dev_size_list[0]
        )

        # Preprocess the image data
        X_train = preprocess_data(X_train)
        X_dev = preprocess_data(X_dev)
        X_test = preprocess_data(X_test)

        prod_best_model = train_model(prod_model, X_train, y_train, X_dev, y_dev)
        cand_best_model = train_model(cand_model, X_train, y_train, X_dev, y_dev)

        prod_test_acc = get_acc(prod_best_model, X_test, y_test)
        cand_test_acc = get_acc(cand_best_model, X_test, y_test)

        print("Production model accuaracy: ", prod_test_acc)
        print("Candidate model accuaracy: ", cand_test_acc)

        print("Confusion Matrix of prod and cand (10*10)")
        print(confusion_matrix(prod_best_model.predict(X_test), cand_best_model.predict(X_test)))

        prod_pred = prod_best_model.predict(X_test)
        cand_pred = cand_best_model.predict(X_test)

        print("Confusion matrix of prod and cand (2*2)")
        print(confusion_matrix(prod_pred==y_test, cand_pred==y_test))

        _, _, prod_f1, _ = precision_recall_fscore_support(y_test, prod_pred, average='macro')
        _, _, cand_f1, _ = precision_recall_fscore_support(y_test, cand_pred, average='macro')

        print("Production model F1 macro: ", prod_f1)
        print("Candidate model F1 macro: ", cand_f1)       



compare_models(args.production_model, args.candidate_model)
# print(results_list)
# print(pd.DataFrame(results_list))
# print(pd.DataFrame(results_list).groupby('model_type').describe().T)
