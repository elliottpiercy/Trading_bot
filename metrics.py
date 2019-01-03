import numpy as np

# Mean squared error
def mse(truth,pred):
    return np.mean((truth-pred)**2)

# Mean absolute error
def mae(truth,pred):
    return np.mean(np.abs(truth-pred))

# Mean absolute percentage error
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
