from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import spearmanr

def directional_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred))

def information_coefficient(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation

def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return model, {
        "mse": mean_squared_error(y_test, preds),
        "directional_accuracy": directional_accuracy(y_test, preds),
        "ic": information_coefficient(y_test, preds),
    }
