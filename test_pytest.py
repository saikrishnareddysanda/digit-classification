import os
from joblib import load
from sklearn.linear_model import LogisticRegression
from utils import create_combination_dictionaries_from_lists, split_train_dev_test, read_data


def test_create_combination_dictionaries_from_lists_count():
    gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
    C_ranges = [0.1, 1, 2, 5, 10]
    hyperparameter_names = ["gamma", "C"]
    hyperparameter_lists = [gamma_ranges, C_ranges]

    list_of_all_param_combination = create_combination_dictionaries_from_lists(
    hyperparameter_names, hyperparameter_lists)

    assert len(list_of_all_param_combination) == len(gamma_ranges) * len(C_ranges)

def test_create_combination_dictionaries_from_lists_value():
    gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
    C_ranges = [0.1, 1, 2, 5, 10]
    hyperparameter_names = ["gamma", "C"]
    hyperparameter_lists = [gamma_ranges, C_ranges]

    list_of_all_param_combination = create_combination_dictionaries_from_lists(
    hyperparameter_names, hyperparameter_lists)

    expected_comb1 = {"gamma": 0.001, "C": 0.1}
    expected_comb2 = {"gamma": 0.01, "C": 5}

    assert (expected_comb1 in list_of_all_param_combination) and (expected_comb2 in list_of_all_param_combination)


def test_lr_loading():
    for i in os.listdir('./models'):
        if i.startswith('M22AIE225'):
            loaded_model = load('./models/'+i)

            assert isinstance(loaded_model, LogisticRegression)


def test_lr_correct_solver_loading():
    for i in os.listdir('./models'):
        if i.startswith('M22AIE225'):
            loaded_model = load('./models/'+i)

            solver_name_from_file = i.split('_')[2].split('.')[0]
            assert loaded_model.get_params()['solver'] == solver_name_from_file


# def test_data_splitting():
#     X, y = read_data()

#     X = X[:100,:,:]
#     y = y[:100]

#     test_size = .2
#     dev_size = .6
#     train_size = 1 - (test_size + dev_size)

#     X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(
#         X, y, test_size, dev_size
#     )
#     # import pdb; pdb.set_trace()
#     # assert (len(X_train) == int(train_size * len(X))) and \
#     # (len(X_dev) == int(dev_size * len(X))) and \
#     # (len(X_test) == int(test_size * len(X)))