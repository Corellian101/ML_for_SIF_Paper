from bingo.evolutionary_optimizers.parallel_archipelago import load_parallel_archipelago_from_file
import numpy as np
import pandas as pd
import sys
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def split_ac(data, ac_greater_1):
    if ac_greater_1 == 1:
        data = data[np.where(data[:,0] > 1)[0]]
        data[:,0] = 1/data[:,0]
    else:
        data = data[np.where(data[:,0] <= 1)[0]]   
    return data

def sort(data):
    models = []
    model = np.unique(data[:,[0,1,2]], axis=0)
    
    for i in model:
        models.append(data[np.where((data[:,[0,1,2]] == i).all(axis=1))])

    return models 

def create_g_input(array, ac_greater_1):
    array = split_ac(array, ac_greater_1)
    models = sort(array)
    data = np.zeros(array.shape[1])
    for m in models:
        phis = np.linspace(0, np.pi, 20)
        for ph in phis:
            data = np.row_stack((data,m[np.where(m[:,3] == find_nearest(m[:,3],ph))[0][0]]))
    data = data[1::]
    return data

chkpt = sys.argv[-4]
model = sys.argv[-3]
fname = sys.argv[-2]
ac_greater_1 = int(sys.argv[-1])

print(chkpt)
print(model)
print(fname)
print(ac_greater_1)
archipelago = load_parallel_archipelago_from_file(chkpt)
member = archipelago.hall_of_fame[int(model)]

print('%.3e    ' % member.fitness, member.get_complexity(),'   f(X_0) =', member)
df = pd.read_csv(fname)

data = df[['a/c','a/t','c/b','phi','F','Mg']].values
data = create_g_input(data, ac_greater_1)
print(data)
M = member.evaluate_equation_at(data[:,[0,1]])
g = data[:,-1].flatten()/M.flatten()
print(M)
print(data[:,-1])
print(g)

data = np.column_stack((data,M, g))
print(data[:,[0,1,3,-3,-2,-1]])

params = ['checkpoint = '+str(chkpt), 'model = '+str(model), 'data file = '+str(fname), 'a/c greater than 1 = '+str(ac_greater_1)]
print(params)
with open('g_params.txt', 'w') as f:
    for item in params:
        f.write("%s\n" % item)

np.save('g_data.npy', data)

