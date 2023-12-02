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
import os
from joblib import dump, load
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
import json
import argparse
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


parser = argparse.ArgumentParser()
parser.add_argument('--num_runs', type=int, default=1)
parser.add_argument('--test_size', type=float, default=0.1)
parser.add_argument('--dev_size', type=float, default=0.1)

args = parser.parse_args()

# Read images and labels from the dataset
images, labels = read_data()


test_size_list = [args.test_size]
dev_size_list = [args.dev_size]


config_path = './config.json'

with open(config_path, 'r') as f:
    config = json.load(f)

    if 'svm' in config:
        gamma_ranges = config['svm']['gamma']
        C_ranges = config['svm']['C']
    
    if 'tree' in config:
        criterion = config['tree']['criterion']
        max_depth = config['tree']['max_depth']
    if 'lr' in config:
        solver = config['lr']['solver']

def train_model(model_type, X_train, y_train, X_dev, y_dev):
    if model_type == 'svm':
        hyperparameter_names = ["gamma", "C"]
        hyperparameter_lists = [gamma_ranges, C_ranges]

    elif model_type == 'tree':
        hyperparameter_names = ["criterion", "max_depth"]
        hyperparameter_lists = [criterion, max_depth]
    elif model_type =='lr':
        hyperparameter_names = ["solver"]
        hyperparameter_lists = [solver]        

    list_of_all_param_combination = create_combination_dictionaries_from_lists(
        hyperparameter_names, hyperparameter_lists
    )

    best_hparams, best_model, best_dev_accuracy = tune_hparams(
        X_train, y_train, X_dev, y_dev, list_of_all_param_combination, model_type
    )

    return best_model


for run in range(args.num_runs):
    # Split the data and labels into training, development, and testing sets
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(
        images, labels, test_size_list[0], dev_size_list[0]
    )

    # Preprocess the image data
    X_train = preprocess_data(X_train)
    X_dev = preprocess_data(X_dev)
    X_test = preprocess_data(X_test)

    for model_type in config:
        best_model = train_model(model_type, X_train, y_train, X_dev, y_dev)

        if model_type == 'lr':
            for i in os.listdir('./models'):
                if i.startswith('M22AIE225'):
                    model = load('./models/'+i)
                    print(model_type, "Model accuracy with solver", i.split('_')[2], "is: " , get_acc(model, X_test, y_test))
        
        test_acc = get_acc(best_model, X_test, y_test)

        # print("model accuaracy: ", test_acc)



