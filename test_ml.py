import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import compute_model_metrics, inference, train_model


def test_train_model_type():
    """
    Test that train_model returns a RandomForestClassifier object.
    """
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)


def test_inference_length():
    """
    Test that inference returns the same number of predictions as input rows.
    """
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])

    model = train_model(X_train, y_train)
    preds = inference(model, X_train)

    assert len(preds) == len(X_train)


def test_compute_model_metrics_range():
    """
    Test that compute_model_metrics returns values between 0 and 1.
    """
    y = np.array([1, 0, 1, 1])
    preds = np.array([1, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
