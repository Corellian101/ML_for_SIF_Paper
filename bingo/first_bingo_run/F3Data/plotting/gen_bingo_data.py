import numpy as np
from Plotting import *
from Raju_Newman_new import *
from sort_data import *

def create_M_input(array, ac_greater_1):
    array = split_ac(array, ac_greater_1)
    models = sort(array)
    data = np.zeros(array.shape[1])
    for m in models:
        ph = np.pi/2
        data = np.row_stack((data,m[np.where(m[:,3] == find_nearest(m[:,3],ph))[0][0]]))
    data = data[1::]
    return data

def create_g_input(array, ac_greater_1):
    array = split_ac(array, ac_greater_1)
    models = sort(array)
    data = np.zeros(array.shape[1])
    for m in models:
        ph = np.linspace(0, np.pi, 20)
        for ph in phis:
            data = np.row_stack((data,m[np.where(m[:,3] == find_nearest(m[:,3],ph))[0][0]]))
    data = data[1::]
    return data
RN_data = pd.read_csv('1_RN_data.csv')[['a/c','a/t','c/b','phi','F','Mg']].values
RN_eqn = pd.read_csv('2_RN_eqn_TRAIN.csv')[['a/c','a/t','c/b','phi','F','Mg']].values
F3d_data = pd.read_csv('3_FRANC3D_FULL_TRAIN.csv')[['a/c','a/t','c/b','phi','F', 'Mg']].values

F3d_data = remove_cb(F3d_data)
RN_eqn = remove_cb(RN_eqn)

# np.savetxt('./bingo_data_sets/F3d_data.csv', F3d_data, delimiter=',',header='a/c,a/t,c/b,phi,F,Mg',fmt='%10.5f', comments='')
# np.savetxt('./bingo_data_sets/RN_eqn.csv', RN_eqn, delimiter=',',header='a/c,a/t,c/b,phi,F,Mg',fmt='%10.5f', comments='')
# np.savetxt('./bingo_data_sets/RN_data.csv', RN_data, delimiter=',',header='a/c,a/t,c/b,phi,F,Mg',fmt='%10.5f', comments='')


RN_data_M_u1 = create_M_input(RN_data, ac_greater_1=False)
RN_data_M_o1 = create_M_input(RN_data, ac_greater_1=True)
RN_data_M_o1[:,0] = 1/RN_data_M_o1[:,0]
# np.savetxt('./bingo_data_sets/RN_data_M_u1.csv', RN_data_M_u1, delimiter=',',header='a/c,a/t,c/b,phi,F,Mg',fmt='%10.5f', comments='')
# np.savetxt('./bingo_data_sets/RN_data_M_o1.csv', RN_data_M_o1, delimiter=',',header='a/c,a/t,c/b,phi,F,Mg',fmt='%10.5f', comments='')

RN_eqn_M_u1 = create_M_input(RN_eqn, ac_greater_1=False)
RN_eqn_M_o1 = create_M_input(RN_eqn, ac_greater_1=True)
RN_eqn_M_o1[:,0] = 1/RN_eqn_M_o1[:,0]
# np.savetxt('./bingo_data_sets/RN_eqn_M_u1.csv', RN_eqn_M_u1, delimiter=',',header='a/c,a/t,c/b,phi,F,Mg',fmt='%10.5f', comments='')
# np.savetxt('./bingo_data_sets/RN_eqn_M_o1.csv', RN_eqn_M_o1, delimiter=',',header='a/c,a/t,c/b,phi,F,Mg',fmt='%10.5f', comments='')

F3d_data_M_u1 = create_M_input(F3d_data, ac_greater_1=False)
F3d_data_M_o1 = create_M_input(F3d_data, ac_greater_1=True)
F3d_data_M_o1[:,0] = 1/F3d_data_M_o1[:,0]
# np.savetxt('./bingo_data_sets/F3d_data_M_u1.csv', F3d_data_M_u1, delimiter=',',header='a/c,a/t,c/b,phi,F,Mg',fmt='%10.5f', comments='')
# np.savetxt('./bingo_data_sets/F3d_data_M_o1.csv', F3d_data_M_o1, delimiter=',',header='a/c,a/t,c/b,phi,F,Mg',fmt='%10.5f', comments='')


