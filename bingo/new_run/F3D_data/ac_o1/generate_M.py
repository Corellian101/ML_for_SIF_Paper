from bingo.evolutionary_optimizers.parallel_archipelago import load_parallel_archipelago_from_file
import numpy as np
import pandas as pd
import sys

chkpt = sys.argv[-4]
model = sys.argv[-3]
fname = sys.argv[-2]
ac_greater_1 = sys.argv[-1]

print(chkpt)
print(model)
print(fname)
archipelago = load_parallel_archipelago_from_file(chkpt)
member = archipelago.hall_of_fame[int(model)]

print('%.3e    ' % member.fitness, member.get_complexity(),'   f(X_0) =', member)
df = pd.read_csv(fname)
if ac_greater_1:
    data[:,0] = 1/data[:,0]

data = df[['a/c','a/t','c/b','phi','F','Mg']].values
M = member.evaluate_equation_at(data[:,[0,1]])
g = data[:,-1].flatten()/M.flatten()
print(M)
print(data[:,-1])
print(g)

data = np.column_stack((data,M, g))
if ac_greater_1:
    data[:,0] = 1/data[:,0]
print(data)

params = ['checkpoint = '+str(chkpt), 'model = '+str(model), 'data file = '+str(fname), 'a/c greater than 1 = '+str(ac_greater_1)]
print(params)
with open('g_params.txt', 'w') as f:
    for item in params:
        f.write("%s\n" % item)

np.save('g_data.npy', data)

