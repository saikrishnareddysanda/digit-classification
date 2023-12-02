from api.app import app
import numpy as np
import json


def test_predict_svm():
    response = app.test_client().get('/predict/svm')
    print(response.data)
    assert response.status_code == 200
    assert json.loads(response.data) == {"message": "Model (svm) loaded successfully"}

def test_predict_tree():
    response = app.test_client().get('/predict/tree')
    assert response.status_code, 200
    assert json.loads(response.data), {"message": "Model (tree) loaded successfully"}

def test_predict_lr():
    response = app.test_client().get('/predict/lr')
    assert response.status_code == 200
    assert json.loads(response.data) == {"message": "Model (lr) loaded successfully"}