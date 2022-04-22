from bingo.evolutionary_optimizers.parallel_archipelago import load_parallel_archipelago_from_file
import numpy as np
import pandas as pd
import sys

chkpt = sys.argv[-3]
model = sys.argv[-2]
fname = sys.argv[-1]

print(chkpt)
print(model)
print(fname)
archipelago = load_parallel_archipelago_from_file(chkpt)
member = archipelago.hall_of_fame[int(model)]

print('%.3e    ' % member.fitness, member.get_complexity(),'   f(X_0) =', member)
df = pd.read_csv(fname)
data = df[['a/c','a/t','c/b','phi','F','Mg']].values
data[:,0] = 1/data[:,0]
M = member.evaluate_equation_at(data[:,[0,1]])
g = data[:,-1].flatten()/M.flatten()
print(M)
print(data[:,-1])
print(g)

data = np.column_stack((data,M, g))
print(data)
np.save('g_data.npy', data)

