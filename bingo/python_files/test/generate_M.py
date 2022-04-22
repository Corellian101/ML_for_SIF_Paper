import numpy as np
from sort_data import *
import sys
import pandas as pd

fname = sys.argv[-2]
ac_greater_1 = int(sys.argv[-1])

print(fname)
print(ac_greater_1)

def create_M_input(array, ac_greater_1):
    array = split_ac(array, ac_greater_1)
    models = sort(array)
    data = np.zeros(array.shape[1])
    for m in models:
        ph = np.pi/2
        data = np.row_stack((data,m[np.where(m[:,3] == find_nearest(m[:,3],ph))[0][0]]))
    data = data[1::]
    return data

data = pd.read_csv(fname)[['a/c','a/t','c/b','phi','F','Mg']].values

if fname == '3_FRANC3D_FULL_TRAIN.csv':
    data = remove_cb(data)

data_m = create_M_input(data, ac_greater_1)
if ac_greater_1:
    data_m[:,0] = 1/data_m[:,0]
print(data_m)
np.save('M_data.npy', data_m)

