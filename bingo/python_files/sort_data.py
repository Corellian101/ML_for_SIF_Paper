import numpy as np
import pandas as pd
#df = pd.read_csv('data_for_bingo1.csv')
#data = np.genfromtxt('data_for_bingo1.csv',delimiter=',',skip_header=1)
#df['sin_phi'] = np.sin(df.phi)
#df['cos_phi'] = np.cos(df.phi)

#y = df.F.values
#x = df[['a_c', 'a_t', 'sin_phi', 'cos_phi', 'c_b']].values
#%%

def sort(data):
    models = []
    model = np.unique(data[:,[0,1,2]], axis=0)
    
    for i in model:
        models.append(data[np.where((data[:,[0,1,2]] == i).all(axis=1))])

    return models 

def sort2(data):
    models = []
    model = np.unique(data[:,[0,1]], axis=0)
    
    for i in model:
        models.append(data[np.where((data[:,[0,1]] == i).all(axis=1))])

    return models 

def remove_cb(data):
    return data[np.where(data[:,2] == 0.2)[0],:]
        

#data = df[['a_c', 'a_t', 'c_b','phi','F']].values
#models = sort(data)
    
#%%

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#phis = np.linspace(0,np.pi,10)
    
#d1 = np.zeros(5)
#for model in models:
#    if model[0,0] > 1:
#        continue
#    for ph in phis:
#        d1 = np.row_stack((d1,model[np.where(model[:,3] == find_nearest(model[:,3],ph))[0][0]]))    
        
#phis = np.linspace(0,np.pi,10)
#phis = [np.pi/2]  
#d1 = np.zeros(5)
#for model in models:
#    if model[0,0] > 1:
#        continue
#    for ph in phis:
#        d1 = np.row_stack((d1,model[np.where(model[:,3] == find_nearest(model[:,3],ph))[0][0]]))      
    
#phis = np.linspace(0,np.pi,10)
    
#d1 = np.zeros(5)
#for model in models:
#    if model[0,0] > 1:
#        continue
#    d1 = np.row_stack((d1,model[np.argmax(model[:,4])]))    
