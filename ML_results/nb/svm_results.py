import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import pandas as pd
import timeit
import time
import tensorflow as tf
from joblib import dump, load
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

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

def poly_svm_boost(train_data, M_train_data, data_type):
    degrees = [1,2,3,4]
    times = []
    M_regs = []
    g_regs = []
    for degree in degrees:
        t0 = time.time()
        reg = SVR(kernel="poly", degree = degree)
        reg.fit(M_train_data[:,:-1], M_train_data[:,-1])
        M = reg.predict(train_data[:,:2])
        train_data[:,-1] = train_data[:,-1]/M
        reg_g = SVR(kernel="poly", degree = degree)
        if data_type == 'le':
            reg_g.fit(train_data[:,1:-1], train_data[:,-1])
        elif data_type == 'g':
            reg_g.fit(train_data[:,:-1], train_data[:,-1])
        else:
            raise Exception("Invalid data type")
        t1 = time.time()
        times.append(t1-t0)
        M_regs.append(reg)
        g_regs.append(reg_g)
        print("Done degree {}".format(degree))

    return times, M_regs, g_regs

def rbf_svm(train_data):
    t0 = time.time()
    reg = SVR(kernel="rbf")
    reg.fit(train_data[:,:-1], train_data[:,-1])
    t1 = time.time()
    return t1-t0, reg

def rbf_svm_boost(train_data, M_train_data, data_type):
    t0 = time.time()
    reg = SVR(kernel="rbf")
    reg.fit(M_train_data[:,:-1], M_train_data[:,-1])
    M = reg.predict(train_data[:,:2])
    train_data[:,-1] = train_data[:,-1]/M
    reg_g = SVR(kernel="rbf")
    if data_type == 'le':
        reg_g.fit(train_data[:,1:-1], train_data[:,-1])
    elif data_type == 'g':
        reg_g.fit(train_data[:,:-1], train_data[:,-1])
    else:
        raise Exception("Invalid data type")
    t1 = time.time()
    return t1-t0, reg, reg_g

def no_boosting(data, data_name):
    ###################################
    ########## No boosting ############
    ###################################
    # Poly SVM
    times, regs = poly_svm(data)
    # saving models
    i = 1
    for reg in regs:
        dump(reg, 'results/no_boosting/run_2/models/poly/{}_poly_degree_{}.joblib'.format(data_name,
                                                                                     i))
        i += 1
    np.save("results/no_boosting/run_2/times/{}_poly_times.npy".format(data_name), times)

    # RBF SVM
    rbf_time, rbf_reg = rbf_svm(data)
    # saving models
    dump(rbf_reg, 'results/no_boosting/run_2/models/rbf/{}_rbf.joblib'.format(data_name))
    np.save("results/no_boosting/run_2/times/{}_rbf_times.npy".format(data_name), rbf_time)
    return times, regs, rbf_time, rbf_reg

def boosting(full_data, M_data, data_name, data_type):
    ###################################
    ############ Boosting #############
    ###################################
    # Poly SVM
    times, M_regs, g_regs = poly_svm_boost(full_data, M_data, data_type)

    # saving models
    for i in range(4):
        dump(M_regs[i], 'results/boosting/models/poly/M_{}_poly_degree_{}.joblib'.format(data_name,i))
        dump(g_regs[i], 'results/boosting/models/poly/g_{}_poly_degree_{}.joblib'.format(data_name,i))

    np.save("results/boosting/times/{}_poly_times.npy".format(data_name), times)

    # RBF SVM
    rbf_times, rbf_M_reg, rbf_g_reg = rbf_svm_boost(full_data, M_data, data_type)

    # saving models
    dump(rbf_M_reg, 'results/boosting/models/rbf/M_{}_rbf.joblib'.format(data_name))
    dump(rbf_g_reg, 'results/boosting/models/rbf/g_{}_rbf.joblib'.format(data_name))

    np.save("results/boosting/times/{}_rbf_times.npy".format(data_name), rbf_times)
    return times, M_regs, g_regs, rbf_times, rbf_M_reg, rbf_g_reg

def evaluate(reg, X, Y):
    results = reg.predict(X)
    mse = np.sum((results - Y)**2)/len(data)
    return mse, results

def testing(d, d_le1, d_g1, d_test, data_name):
    ################# a/c <= 1###########################
    # No Boosting
    times, regs, rbf_time, rbf_reg = no_boosting(d_le1, data_name+"_le1")
    # Boosting
    data = []
    for row in d_le1:
        if (row[2] > 1.5 and row[2] < 1.65):
            data.append(np.delete(row, 2))
    M_data = np.array(data)
    boost_times, M_regs, g_regs, rbf_boost_times, rbf_M_regs, rbf_g_regs = boosting(d_le1,M_data,data_name+"_le1","le")
    # Testing
    d_test_le = d_test[d_test[:,0]<=1]
    no_boost_mse = []
    boost_mse = []
    # Poly
    for i in range(4):
        # NO BOOST
        mse, _ = evaluate(regs[i], d_test_le[:,:-1], d_test_le[:,-1])
        no_boost_mse.append(mse)

        # BOOST
        _, M = evaluate(M_regs[i], d_test_le[:,:2], d_test_le[:,-1])
        _, g = evaluate(g_regs[i], d_test_le[:,1:-1], d_test_le[:,-1])
        boost_mse.append(np.sum((M*g - d_test_le[:,-1])**2)/len(d_test_le[:,-1]))

    # RBF
    # NO BOOST
    mse, _ = evaluate(rbf_reg, d_test_le[:,:-1], d_test_le[:,-1])
    no_boost_mse.append(mse)

    # BOOST
    _, M = evaluate(rbf_M_regs, d_test_le[:,:2], d_test_le[:,-1])
    _, g = evaluate(rbf_g_regs, d_test_le[:,1:-1], d_test_le[:,-1])
    boost_mse.append(np.sum((M*g - d_test_le[:,-1])**2)/len(d_test_le[:,-1]))

    ################# a/c > 1###########################
    # No Boosting
    times, regs, rbf_time, rbf_reg = no_boosting(d_g1, data_name+"_g1")
    # Boosting
    data = []
    for row in d_g1:
        if (row[2] > 1.5 and row[2] < 1.65):
            data.append(np.delete(row, 2))
    M_data = np.array(data)
    boost_times, M_regs, g_regs, rbf_boost_times, rbf_M_regs, rbf_g_regs = boosting(d_g1,M_data,data_name+"_g1","g")
    # Testing
    d_test_g = d_test[d_test[:,0]>1]
    no_boost_mse_g = []
    boost_mse_g = []
    # Poly
    for i in range(4):
        # NO BOOST
        mse, _ = evaluate(regs[i], d_test_g[:,:-1], d_test_g[:,-1])
        no_boost_mse_g.append(mse)

        # BOOST
        _, M = evaluate(M_regs[i], d_test_g[:,:2], d_test_g[:,-1])
        _, g = evaluate(g_regs[i], d_test_g[:,1:-1], d_test_g[:,-1])
        boost_mse_g.append(np.sum((M*g - d_test_g[:,-1])**2)/len(d_test_g[:,-1]))

    # RBF
    # NO BOOST
    mse, _ = evaluate(rbf_reg, d_test_g[:,:-1], d_test_g[:,-1])
    no_boost_mse_g.append(mse)

    # BOOST
    _, M = evaluate(rbf_M_regs, d_test_g[:,:2], d_test_g[:,-1])
    _, g = evaluate(rbf_g_regs, d_test_g[:,1:-1], d_test_g[:,-1])
    boost_mse_g.append(np.sum((M*g - d_test_g[:,-1])**2)/len(d_test_g[:,-1]))

    return boost_mse, no_boost_mse, boost_mse_g, no_boost_mse_g
