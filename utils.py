import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import  metrics, svm
from skimage.transform import  resize
from sklearn.model_selection import train_test_split
from itertools import product

def read_data():
    # Load the digits dataset
    digits = datasets.load_digits()
    return digits.images, digits.target

def preprocess_data(images, shape):
    n_samples = len(images)
    # Reshape the images into 2D array
    image_resize = resize(images, (images.shape[0],shape[0],shape[1]))
    data = image_resize.reshape((n_samples, -1))

    return data

def split_train_dev_test(X, y, test_size, dev_size, random_state=1):
    # Split the data to train and (test + dev) sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + dev_size, random_state=random_state)
    # Split the (test + dev) data into test and dev sets
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + dev_size), random_state=random_state)
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def train_model(X_train, y_train, model_parameters, model_type="svm"):
    if model_type == "svm":
        model = svm.SVC(**model_parameters)
    # Fit the model 
    model.fit(X_train, y_train)
    return model

def predict_and_eval(model, X_test, y_test):
    # Make predictions
    predicted = model.predict(X_test)
    # Generate classification report using the predictions and ground truth labels
    report = metrics.classification_report(y_test, predicted)
    return predicted, report

def get_acc(model, X, y):
    predicted = model.predict(X)
    return round(metrics.accuracy_score(y, predicted), 4)

def visualize_confusion_matrix(y_test, predicted):
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    # Display the confusion matrix
    plt.show()

def create_combination_dictionaries_from_lists(dict_keys, value_lists):
    result = [{name: value for name, value in zip(dict_keys, sublist)} for sublist in product(*value_lists)]
    return result

def tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination):
    best_accuracy = -1

    for model_params in list_of_all_param_combination:
        model = train_model(X_train, y_train, model_params)
        accuracy = get_acc(model, X_dev, y_dev)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_hyperparameters = model_params

    return best_hyperparameters, best_model, best_accuracy