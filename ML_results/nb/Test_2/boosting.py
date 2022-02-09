import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import pandas as pd
import timeit
import time
import tensorflow as tf
from joblib import dump, load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import pandas as pd
import timeit
import time

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


def poly_svm(train_data):
    degrees = [1,2,3,4]
    times = []
    regs = []
    for degree in degrees:
        t0 = time.time()
        reg = SVR(kernel="poly", degree = degree)
        reg.fit(train_data[:,:-1], train_data[:,-1])
        t1 = time.time()
        times.append(t1-t0)
        regs.append(reg)
        print("Done degree {}".format(degree))
    return times, regs

def rbf_svm(train_data):
    times = []
    regs = []
    t0 = time.time()
    reg = SVR(kernel="rbf")
    reg.fit(train_data[:,:-1], train_data[:,-1])
    t1 = time.time()
    times.append(t1-t0)
    regs.append(reg)
    return times, regs

def NN(data):
    X_train = data[:,:-1]
    y_train = data[:,-1]
    #create model
    model = Sequential()

    #get number of columns in training data
    n_cols = X_train.shape[1]

    #add model layers
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='relu'))

    model.compile(optimizer='adam', loss='mse')
    early_stopping_monitor = EarlyStopping(patience=10)
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stopping_monitor])

    return model

def boost(d_le1, d_g1):
    data = []
    for row in d_le1:
        if (row[2] > 1.5 and row[2] < 1.65):
            data.append(np.delete(row, 2))
    data = np.array(data)


    return data

# Loading datasets
d1_g1 = np.load("data/1_RN_data_ac_g1.npy")
d1_le1 = np.load("data/1_RN_data_ac_le1.npy")

data = boost(d1_le1, d1_g1)
print(data)
