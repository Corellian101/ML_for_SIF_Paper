from sympy import Symbol, integrate, init_printing, diff, simplify, Matrix, print_latex, sqrt, cos, sin
import matplotlib.pyplot as plt
import numpy as np
from Raju_Newman_new import F_s, F_s_bingo
from sort_data import sort
import pandas as pd

def plot_eqns(n, models):
    plt.figure()
    plt.plot(models[n][:,3],models[n][:,4],'ro', label='FEA')
    plt.plot(models[n][:,3],F_s(models[n][:,0],models[n][:,1],models[n][:,2],models[n][:,3]),label='RN')
    plt.plot(models[n][:,3],F_s_bingo(models[n][:,0],models[n][:,1],models[n][:,2],models[n][:,3]),label='BINGO')
    #plt.ylim([1.25,1.5])
    plt.legend()
    plt.title(f"a/c={models[n][0,0]}, a/t={models[n][0,1]}, c/b={models[n][0,2]}, model={n}")
    plt.xlabel('phi')
    plt.ylabel('F')
    return

X_0 = Symbol('a/c')
X_1 = Symbol('a/t')
X_2 = Symbol('c/b')
X_3 = Symbol('phi')
g_u1 = simplify((1.0263496149498648 + sin(6373.023412385456 + 11071.131423790328 + -11350.186594670391 - (6373.023412385456 - (sin(X_3)))))*(X_1) + cos(11071.131423790328))
g_o1 = simplify(1 + (0.1+0.35*((X_0**(-1)))*(X_1)**2)*(1-(sin(X_3)))**2)
M_o1 = simplify((X_0)*(0.2965415493436768) - (-0.02115042312837423) + (sqrt(X_1))*((sqrt(1/X_0))*(0.2502798692342405 + 0.2502798692342405) - (0.3628181028391626)) + (sqrt(sqrt(1/X_0)))*(0.6637021883924064))
M_u1 = simplify(1.0052643174132263 - ((X_1)*((X_0 + -5.897146261824969)*(X_0) - ((-8.41374265030078)*(sqrt(X_0)) + 3.65713751815375))))

df = pd.read_csv('data_for_bingo4.csv')
F3d_data = df[['a_c','a_t','c_b','phi','F']].values
models = sort(F3d_data)

# for i in np.arange(0,900,10):
#     plot_eqns(i,models)
    
bingo_results = F_s_bingo(F3d_data[:,0],F3d_data[:,1],F3d_data[:,2],F3d_data[:,3])
RN_results = F_s(F3d_data[:,0],F3d_data[:,1],F3d_data[:,2],F3d_data[:,3])
FE_results = F3d_data[:,-1]

error_RN = abs(RN_results-FE_results)/FE_results
error_bingo = abs(bingo_results-FE_results)/FE_results


plt.hist(error_bingo)
plt.hist(error_RN)


