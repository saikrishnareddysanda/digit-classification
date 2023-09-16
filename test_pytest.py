from utils import create_combination_dictionaries_from_lists


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
