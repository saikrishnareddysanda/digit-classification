import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import  metrics, svm
from sklearn.model_selection import train_test_split

def read_data():
    # Load the digits dataset
    digits = datasets.load_digits()
    return digits.images, digits.target

def preprocess_data(images):
    n_samples = len(images)
    # Reshape the images into 2D array
    data = images.reshape((n_samples, -1))
    return data

def split_train_dev_test(X, y, test_size, dev_size, random_state=1):
    # Split the data to train and (test + dev) sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + dev_size, random_state=random_state)
    # Split the (test + dev) data into test and dev sets
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=dev_size / (test_size + dev_size), random_state=random_state)
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

def visualize_confusion_matrix(y_test, predicted):
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    # Display the confusion matrix
    plt.show()