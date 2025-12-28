import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

def directional_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred))

def information_coefficient(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    return {
        "mse": mean_squared_error(y_test, preds),
        "directional_accuracy": directional_accuracy(y_test, preds),
        "ic": information_coefficient(y_test, preds),
    }
