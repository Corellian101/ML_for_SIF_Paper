import numpy as np


g = np.load('g_data.npy')
print(['a/c', 'a/t', 'phi', 'Mg', 'M', 'g'])
print(g[:,[0,1,3,-3,-2,-1]])
