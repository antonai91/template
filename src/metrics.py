import numpy as np

def mdape(y_true, y_pred):
    return np.median(np.abs(y_true - y_pred) / y_true)*100