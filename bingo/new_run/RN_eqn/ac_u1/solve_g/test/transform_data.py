# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import sys
import time
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


ac_greater_1 = 0
solve_g = 1
fname = '2_RN_eqn.csv'
test_vals = 1

# what phi values to filter
# if solving for g
phis = np.linspace(0,np.pi,10)

# if solving for M
#phis = [np.pi/2]

def sort(data):
    models = []
    model = np.unique(data[:,[0,1,2]], axis=0)
    
    for i in model:
        models.append(data[np.where((data[:,[0,1,2]] == i).all(axis=1))])

    return models

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def transform_data(fname):
    # read in training data
    if not solve_g:
        df = pd.read_csv(fname)
        data = df[['a/c','a/t','c/b','phi','F','Mg']].values
    elif solve_g:
        # g_data [a/t, a/t, c/b, phi, F, Mg, M, g]
        data = np.load('g_data.npy')

    # sort data by each FE model
    models = sort(data)

    data = np.zeros(data.shape[1])
    # loop through each FE model and grab values at correct phi location
    if ac_greater_1: 
        for m in models:
            # skip values of a/c <= 1
            if m[0,0] <= 1:
                continue
            for ph in phis:
                data = np.row_stack((data,m[np.where(m[:,3] == find_nearest(m[:,3],ph))[0][0]]))

    elif not ac_greater_1:
        for m in models:
            # skip values of a/c > 1
            if m[0,0] > 1:
                continue
            for ph in phis:
                data = np.row_stack((data,m[np.where(m[:,3] == find_nearest(m[:,3],ph))[0][0]]))

    data = data[1::]
    if ac_greater_1:
        # change a/c to c/a for a/c > 1
        data[:, 0] = 1/data[:,0]

    # choose inputs
    if not solve_g:
        # x [a/c, a/t]
        x = data[:,[0,1]]
        # y [M*g]
        y = data[:,-1]
    elif solve_g:
        # x [a/c, a,t, phi]
        x = data[:, [0, 1, 3]]
        # y [g]
        y = data[:, -1]

        print(data[:,-2].flatten()*data[:,-1].flatten())
        print(data[:,-3])
        if not np.isclose(data[:,-3].flatten(), data[:,-2].flatten()*data[:,-1].flatten()).all():
            raise ValueError('M and g not matching with Mg')

    return x, y
