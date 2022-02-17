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

# Loading datasets
d1_g1 = np.load("data/1_RN_data_ac_g1.npy")
d1_le1 = np.load("data/1_RN_data_ac_le1.npy")

d2_g1 = np.load("data/2_RN_eqn_ac_g1.npy")
d2_le1 = np.load("/home/raavan/dev/ML_for_SIF_Paper/ML_results/nb/Test_2/data/2_RN_eqn_ac_le1.npy")

d3a_g1 = np.load("/home/raavan/dev/ML_for_SIF_Paper/ML_results/nb/Test_2/data/3a_FULL_data_ac_g1.npy")
d3a_le1 = np.load("/home/raavan/dev/ML_for_SIF_Paper/ML_results/nb/Test_2/data/3a_FULL_data_ac_g1.npy")

d3b_g1 = np.load("/home/raavan/dev/ML_for_SIF_Paper/ML_results/nb/Test_2/data/3b_PHI_data_ac_g1.npy")
d3b_le1 = np.load("/home/raavan/dev/ML_for_SIF_Paper/ML_results/nb/Test_2/data/3b_PHI_data_ac_le1.npy")

############################################################################################
################################ Polynomial SVM ############################################
############################################################################################
'''
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

# DATASET 1
times, regs = poly_svm(d1_g1)
# saving models
i = 1
for reg in regs:
    dump(reg, 'results/no_boosting/models/d1g1_poly_degree_{}.joblib'.format(i))
    i += 1

np.save("results/no_boosting/times/d1g1_times.npy", times)

times, regs = poly_svm(d1_le1)
# saving models
i = 1
for reg in regs:
    dump(reg, 'results/no_boosting/models/d1le1_poly_degree_{}.joblib'.format(i))
    i += 1

np.save("results/no_boosting/times/d1le1_times.npy", times)

# DATASET 2
times, regs = poly_svm(d2_g1)
# saving models
i = 1
for reg in regs:
    dump(reg, 'results/no_boosting/models/d2g1_poly_degree_{}.joblib'.format(i))
    i += 1

np.save("results/no_boosting/times/d2g1_times.npy", times)

times, regs = poly_svm(d2_le1)
# saving models
i = 1
for reg in regs:
    dump(reg, 'results/no_boosting/models/d2le1_poly_degree_{}.joblib'.format(i))
    i += 1

np.save("results/no_boosting/times/d2le1_times.npy", times)

# DATASET 3a
times, regs = poly_svm(d3a_g1)
# saving models
i = 1
for reg in regs:
    dump(reg, 'results/no_boosting/models/d3ag1_poly_degree_{}.joblib'.format(i))
    i += 1

np.save("results/no_boosting/times/d3ag1_times.npy", times)

times, regs = poly_svm(d3a_le1)
# saving models
i = 1
for reg in regs:
    dump(reg, 'results/no_boosting/models/d3ale1_poly_degree_{}.joblib'.format(i))
    i += 1

np.save("results/no_boosting/times/d3ale1_times.npy", times)

# DATASET 3b
times, regs = poly_svm(d3b_g1)
# saving models
i = 1
for reg in regs:
    dump(reg, 'results/no_boosting/models/d3bg1_poly_degree_{}.joblib'.format(i))
    i += 1

np.save("results/no_boosting/times/d3bg1_times.npy", times)

times, regs = poly_svm(d3b_le1)
# saving models
i = 1
for reg in regs:
    dump(reg, 'results/no_boosting/models/d3ble1_poly_degree_{}.joblib'.format(i))
    i += 1

np.save("results/no_boosting/times/d3ble1_times.npy", times)
'''
##########################################################################################
#################################### RFB SVM #############################################
##########################################################################################
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

# DATASET 1
times, regs = rbf_svm(d1_g1)
# saving models
i = 1
for reg in regs:
    dump(reg, 'results/no_boosting/models/d1g1_rbf_degree_{}.joblib'.format(i))
    i += 1

np.save("results/no_boosting/times/d1g1_rbf_times.npy", times)

times, regs = rbf_svm(d1_le1)
# saving models
i = 1
for reg in regs:
    dump(reg, 'results/no_boosting/models/d1le1_rbf_degree_{}.joblib'.format(i))
    i += 1

np.save("results/no_boosting/times/d1le1_rbf_times.npy", times)

# DATASET 2
times, regs = rbf_svm(d2_g1)
# saving models
i = 1
for reg in regs:
    dump(reg, 'results/no_boosting/models/d2g1_rbf_degree_{}.joblib'.format(i))
    i += 1

np.save("results/no_boosting/times/d2g1_rbf_times.npy", times)

times, regs = rbf_svm(d2_le1)
# saving models
i = 1
for reg in regs:
    dump(reg, 'results/no_boosting/models/d2le1_rbf_degree_{}.joblib'.format(i))
    i += 1

np.save("results/no_boosting/times/d2le1_rbf_times.npy", times)

# DATASET 3a
times, regs = rbf_svm(d3a_g1)
# saving models
i = 1
for reg in regs:
    dump(reg, 'results/no_boosting/models/d3ag1_rbf_degree_{}.joblib'.format(i))
    i += 1

np.save("results/no_boosting/times/d3ag1_rbf_times.npy", times)

times, regs = rbf_svm(d3a_le1)
# saving models
i = 1
for reg in regs:
    dump(reg, 'results/no_boosting/models/d3ale1_rbf_degree_{}.joblib'.format(i))
    i += 1

np.save("results/no_boosting/times/d3ale1_rbf_times.npy", times)

# DATASET 3b
times, regs = rbf_svm(d3b_g1)
# saving models
i = 1
for reg in regs:
    dump(reg, 'results/no_boosting/models/d3bg1_rbf_degree_{}.joblib'.format(i))
    i += 1

np.save("results/no_boosting/times/d3bg1_rbf_times.npy", times)

times, regs = rbf_svm(d3b_le1)
# saving models
i = 1
for reg in regs:
    dump(reg, 'results/no_boosting/models/d3ble1_rbf_degree_{}.joblib'.format(i))
    i += 1

np.save("results/no_boosting/times/d3ble1_rbf_times.npy", times)

##########################################################################################
########################################## NN ############################################
##########################################################################################
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

# DATASET 1
model = NN(d1_g1)
# saving models
model.save("results/no_boosting/models/d1g1.h5")

model = NN(d1_le1)
# saving models
model.save("results/no_boosting/models/d1le1.h5")

# DATASET 2
model = NN(d2_g1)
# saving models
model.save("results/no_boosting/models/d2g1.h5")

model = NN(d2_le1)
# saving models
model.save("results/no_boosting/models/d2le1.h5")

# DATASET 3a
model = NN(d3a_g1)
# saving models
model.save("results/no_boosting/models/d3ag1.h5")

model = NN(d3a_le1)
# saving models
model.save("results/no_boosting/models/d3ale1.h5")

# DATASET 3b
model = NN(d3b_g1)
# saving models
model.save("results/no_boosting/models/d3bg1.h5")

times, regs = rbf_svm(d3b_le1)
# saving models
model.save("results/no_boosting/models/d3ble1.h5")
