import numpy as np
import Raju_Newman_new as RN
import os

file_path = '../all_bingo_data/'
all_data = np.zeros([2,8])
for i in os.listdir(file_path)[0:-2]:
    fname = i.split('.')[0]
    if i.split('.')[1] != 'csv':
        continue
    
    a_c = float(fname.split('_')[0][2::])
    if a_c > 10 and a_c != 20:
        a_c = a_c/10
    a_c = a_c*10**-(fname.split('_')[0][2::].count('0'))
    
    
    a_t = float(fname.split('_')[1][2::])
    if a_t > 10 and a_c != 20:
        a_t = a_t/10
    a_t = a_t*10**-(fname.split('_')[1][2::].count('0'))
    try:
        c_b = float(fname.split('_')[2][2::])
        if c_b > 10 and a_c != 20:
            c_b = c_b/10
        c_b = c_b*10**-(fname.split('_')[2][2::].count('0'))
    except:
        c_b = 0.2
 #%%       
    t = 0.75
    a = a_t*t
    c = a/a_c
   
    Fs = np.genfromtxt(file_path+i)
    F = Fs[:,1]
    Nphi = np.arccos((Fs[:,6])/c)*2/np.pi
    phi = Nphi*(np.pi/2)
    RN_eqn = RN.F_s(a_c,a_t,c_b,phi)
    error = abs(F-RN_eqn)/RN_eqn
    data = np.column_stack(([a_c]*len(F),[a_t]*len(F),Nphi,phi,[c_b]*len(F),F,RN_eqn,error))
    all_data = np.concatenate((all_data,data),axis=0)
    all_data = all_data[~np.isnan(all_data).any(axis=1)]
    all_data_bingo = all_data[2::,:]
    
np.savetxt("data_for_bingo4.csv", all_data_bingo, delimiter=",",fmt='%10.5f')
