from sympy import Symbol, integrate, init_printing, diff, simplify, Matrix, print_latex, sqrt, cos, sin, expand
from sympy.parsing.sympy_parser import parse_expr
import matplotlib.pyplot as plt
import numpy as np
from Raju_Newman_new import F_s, F_s_bingo, Bingo_cust, M
import Raju_Newman_new as RN
from sort_data import sort
import pandas as pd

def plot_eqns(F_RN, F_bingo, input_vars, F_input, n):
        
    data = np.column_stack((input_vars, F_input, F_RN, F_bingo))
    models = sort(data)
    error_RN = abs(models[n][:,-3]-models[n][:,-2])/models[n][:,-3]
    error_bingo = abs(models[n][:,-3]-models[n][:,-1])/models[n][:,-3]
    
    fig, ax1 = plt.subplots()
    lns1 = ax1.plot(models[n][:,3],models[n][:,-3],'r.', label='Input Data')
    lns2 = ax1.plot(models[n][:,3],models[n][:,-2],'b-', label='RN')
    lns3 = ax1.plot(models[n][:,3],models[n][:,-1],'g-', label='Bingo')
    plt.ylabel('Boundry Correction Factor')
    plt.xlabel('phi [radians]')

    ax2 = ax1.twinx()
    lns4 = ax2.plot(models[n][:,3], error_bingo, 'g--', label='Bingo Error')
    lns5 = ax2.plot(models[n][:,3], error_RN, 'b--', label='RN Error')
    plt.ylabel('Error')
    
    lns = lns1+lns2+lns3+lns4+lns5
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    #plt.ylim([1.25,1.5])
    plt.title(f"a/c={models[n][0,0]}, a/t={models[n][0,1]}, c/b={models[n][0,2]}, model={n}")
    plt.xlabel('phi')
    plt.figure()
    return

def split_ac(data, ac_greater_1):
    if ac_greater_1:
        data = data[np.where(data[:,0] > 1)[0]]
        #data[:,0] = 1/data[:,0]
    else:
        data = data[np.where(data[:,0] <= 1)[0]]   
    return data

def num_models(dataset):
    models = sort(dataset)
    return len(models)


def error_plotting(F_input, F_RN, F_bingo, input_vars=None, plot_both=False, plot_only2=False):
    
    error_RN = abs(F_input-F_RN)/F_input
    error_bingo = abs(F_input-F_bingo)/F_input
    if not plot_only2:
        plt.figure()
        plt.hist(error_bingo, color='g', label='bingo Error')
        plt.hist(error_RN, color='b', label='RN Error')
        plt.title(f"RN mean error = {np.round(np.mean(error_RN), 4)}, bingo mean error = {np.round(np.mean(error_bingo), 4)} \n RN max error = {np.round(np.max(error_RN), 4)}, bingo max error = {np.round(np.max(error_bingo), 4)}")
        plt.legend()
        
    if plot_both:
        plt.figure()
        plt.hist(error_RN, color='b', label='RN Error')
        plt.hist(error_bingo, color='g', label='bingo Error')
        plt.title(f"RN mean error = {np.round(np.mean(error_RN), 4)}, bingo mean error = {np.round(np.mean(error_bingo), 4)} \n RN max error = {np.round(np.max(error_RN), 4)}, bingo max error = {np.round(np.max(error_bingo), 4)}")
        plt.legend()
    if plot_only2:
        plt.figure()
        plt.hist(error_RN, color='b', label='RN Error')
        plt.hist(error_bingo, color='g', label='bingo Error')
        plt.title(f"RN mean error = {np.round(np.mean(error_RN), 4)}, bingo mean error = {np.round(np.mean(error_bingo), 4)} \n RN max error = {np.round(np.max(error_RN), 4)}, bingo max error = {np.round(np.max(error_bingo), 4)}")
        plt.legend()
    return

def get_data_sets():
    RN_data = pd.read_csv('1_RN_data.csv')[['a/c','a/t','c/b','phi','Mg','F']].values
    RN_eqn = pd.read_csv('2_RN_eqn.csv')[['a/c','a/t','c/b','phi','Mg','F']].values
    F3d_data = pd.read_csv('3_FRANC3D_FULL.csv')[['a/c','a/t','c/b','phi','Mg', 'F']].values
    
    return RN_data, RN_eqn, F3d_data
    


#%% Simplify


'''
# Current best
#
g_u1 = simplify((1.0263496149498648 + sin(6373.023412385456 + 11071.131423790328 + -11350.186594670391 - (6373.023412385456 - (sin(X_3)))))*(X_1) + cos(11071.131423790328))
g_o1 = simplify(1 + (0.1+0.35*((X_0**(-1)))*(X_1)**2)*(1-(sin(X_3)))**2)
M_o1 = simplify((X_0)*(0.2965415493436768) - (-0.02115042312837423) + (sqrt(X_1))*((sqrt(1/X_0))*(0.2502798692342405 + 0.2502798692342405) - (0.3628181028391626)) + (sqrt(sqrt(1/X_0)))*(0.6637021883924064))
M_u1 = simplify(1.0052643174132263 - ((X_1)*((X_0 + -5.897146261824969)*(X_0) - ((-8.41374265030078)*(sqrt(X_0)) + 3.65713751815375))))
#
'''
#
#%%
# RN eqn



# RN Data

class RN_data_funs:
    def Mu1(X_0, X_1, X_2=0, X_3=0):
        from numpy import sqrt, cos, sin
        M = (33622907.281902 - (33622908.186748 - (sqrt(X_0))))*((sqrt(X_0) - (-0.055040))*(0.480859) - ((2.129986)*(X_1))) - (-1.083060)
        
        ### TRIAL 2 ###
        M = ((-4.862855 - (X_0) - (-14.496016 + X_1) - (7.994456))*(((-4.862855 - (X_0))*(-4.862855 - (X_0) - (-14.496016 + X_1)) + 48.333960)*(X_1) - (-0.091848)))*(0.251563) + 1.055752
        #M = (-0.092176 + X_1)*((-2.404498 + X_0)*(X_0) + 0.484722) - (1905.566261) + 1906.231764 - ((-1.084407)*(sqrt(X_1)))
        return M
    
    def gu1(X_0, X_1, X_3, X_2):
        from numpy import sqrt, cos, sin
        g = (((sin(X_2) - (1.055261))*(X_1))*(-0.000110))*(4741.340369) + (0.114983)*(X_0) + 0.89453125
        #g =-53602933519417.929688 + (-0.054486 + ((0.896710)*(sin(X_2 + 3722.785157)) - (-0.984231))*(X_0))*(X_1 + -0.213224) - (-53602933519418.929688)
        return g
    
    def Mo1(X_0, X_1, X_2=0, X_3=0):
        from numpy import sqrt, cos, sin
        X_0 = 1/X_0
        M = sqrt(sqrt(X_0)) - (0.145056)
        return M
    
    def go1(X_0, X_1, X_3, X_2):
        from numpy import sqrt, cos, sin
        g = 1 + (0.1+0.35*((X_0**(-1)))*(X_1)**2)*(1-(sin(X_2)))**2
        return g


# RN Eqn

class RN_eqn_funs:
    def Mu1(X_0, X_1, X_2=0, X_3=0):
        from numpy import sqrt, cos, sin
        # sqrt3 = np.emath.sqrt
        # def sqrt(x):
        #     return abs(sqrt3(x))
        M = (((13.804167)*(X_0 - (3.565979)))*(sqrt(X_0)) + 36.977445)*((0.031819)*((X_1)*(X_1 - (-1.319777))) - (0.001833)) + 1.051169
        M = (sqrt(X_0)*(13.804167*X_0 - 49.225369634493) + 36.977445)*(0.031819*X_1*(X_1 + 1.319777) - 0.001833) + 1.051169
        
        ### TRIAL 2 ###
        #M = ((-0.124278 + 1.057496 + (1.405610 - (sqrt(X_0)))*((1.405610 - (sqrt(X_0)))*(X_1)))*(0.730550))*((1.405610 - (sqrt(X_0)))*((1.405610 - (sqrt(X_0)))*(X_1))) + 86.239802 - (43.129421) + -42.079126
        M = (((1.180636)*(sqrt(X_0)) + -1.438596)*(-0.217433 - (X_1)))*(((1.180636)*(sqrt(X_0)) + -1.438596)*(-0.217433 - (X_1))) + 62.414228 - (20.460517) - (20.229128) + -20.673957
        #M = ((-0.480718)*((2.798314)*(sqrt(0.202493 - (X_0))) + -2.634924))*(X_1 + -0.103196) - (4.054467) + 5.101003
        #M = 1.357489 + ((1.357489 - (sqrt(X_0)))*((1.357489 - (sqrt(X_0)))*(X_1)))*(1.228277) - (-38.163851) - (19.314149) + -19.210837
        #M = 1.34698*X_1*(1 - 0.711434893035764*sqrt(X_0))**2 + 1.03125499999999
        return M
    
    def gu1(X_0, X_1, X_3, X_2):
        from numpy import sqrt, cos, sin
        g = 0.9993 + (0.4085*X_1 + 0.2309)**2 * (1.0128 - sin(X_2))**2
        #g = 1 + (0.1 + 0.35*(X_1)**2)*(1-sin(X_2))**2
        return g
    
    def Mo1(X_0, X_1, X_2=0, X_3=0):
        from numpy import sqrt, cos, sin
        X_0 = 1/X_0
        M = ((X_0)*(0.274624) - (59882531.611787) + 59882531.514921)*((-0.352740 + X_0)*(X_1 + -1.793803 + -0.352740 + X_0) - (-3.116885)) - (-0.604788)
        #M = 0.274624*X_0**3 + 0.274624*X_1*X_0**2 - 0.783229*X_0**2 - 0.193737*X_1*X_0 + 1.306*X_0 + 0.0341685*X_1 + 0.229524
        
        ### TRIAL 2 ###
        M = ((((sqrt(X_0) + -0.993416)*(X_0))*(-0.746112) - (0.119041))*(0.079181 - (X_1)) - (-1.058784))*(sqrt(X_0)) - (11811.573801) + 11811.545610
        #M = ((X_0 - (0.241857))*(-0.245796 + ((-0.241857 + X_0)*(X_1))*(0.215774)) - (0.149835 + -1.056510))*(X_0) - (-0.299141)
        #M = 0.612554 - (((X_0)*(0.985023) - (0.346824))*((((X_0)*(0.985023) - (0.346824))*(X_1 + -0.767345))*(-0.281420) + -0.842511 - (-0.062843)))
        return M
    
    def go1(X_0, X_1, X_3, X_2):
        from numpy import sqrt, cos, sin
        g = 1 + (0.1+0.35*((X_0**(-1)))*(X_1)**2)*(1-(sin(X_2)))**2
        X_0 = 1/X_0
        #g = 0.999991 - (((8833227.041394 + (sin(X_2))*(1.999006) - (-1261496.774534) - ((sin(X_2))*(sin(X_2))) + -10094724.815343)*(X_1 + 0.222239))*(0.232156))
        return g

# F3d data

class F3d_data_funs:
    def Mu1(X_0, X_1, X_2=0, X_3=0):
        from numpy import sqrt, cos, sin
        sqrt3 = np.emath.sqrt
        def sqrt(x):
            return abs(sqrt3(x))
        # orginial equation
        #M = 1.0052643174132263 - ((X_1)*((X_0 + -5.897146261824969)*(X_0) - ((-8.41374265030078)*(sqrt(X_0)) + 3.65713751815375)))
        
        M = (((sqrt(-0.341410 + sqrt(X_0)))*((-1.957341 + sqrt(-0.341410 + sqrt(X_0)))*(-0.139728)) - (0.132972))*(0.108615 - (X_1)))*(20.159387) + 1.051741
        #M = 1.05174 - 5.5135 *(sqrt(sqrt(X_0) - 0.34141) - 0.510897 *sqrt(X_0) - 0.311769) *(X_1 - 0.108615)
        
        
        ### TRIAL 2 ###
        #M = (-660.396388 + (-89.045028 - (-90.064271) - (X_0))*((0.008540 - (X_1))*(-89.045028 - (-90.064271) - (X_0)) + 0.106868))*(-1.517551) + -1001.092859
        M = (1.114310)*(-63.472139 + (1.122792 - (sqrt(X_0 + -0.208461)))*((1.122792 - (sqrt(X_0 + -0.208461)))*(-0.111355 + X_1))) - (-36.533956 + -35.246715)
        return M
    
    def gu1(X_0, X_1, X_2, X_3):
        X_2 = X_3
        from numpy import sqrt, cos, sin
        g = (1.0263496149498648 + sin(6373.023412385456 + 11071.131423790328 + -11350.186594670391 - (6373.023412385456 - (sin(X_3)))))*(X_1) + cos(11071.131423790328)
        g = -(-1.0263496149499*X_1 + X_1*sin(279.05517088006 - sin(X_3)) - 0.9873999518025)
        #g = 1 + (0.1 + 0.35*(X_1)**2)*(1-sin(X_2))**2
        return g
    
    def Mo1(X_0, X_1, X_2=0, X_3=0):
        from numpy import sqrt, cos, sin
        # original equation
        # sqrt3 = np.emath.sqrt
        # def sqrt(x):
        #     return abs(sqrt3(x))
        X_0 = 1/X_0
        #M = (X_0)*(0.2965415493436768) - (-0.02115042312837423) + (sqrt(X_1))*((sqrt(X_0))*(0.2502798692342405 + 0.2502798692342405) - (0.3628181028391626)) + (sqrt(sqrt(X_0)))*(0.6637021883924064)
        M = (X_0 + -4.120056 + (X_1)*(-2.311179))*(((sqrt(X_1) + -3.330640)*(X_0))*(0.066869) + -217034054.103521 - (-217034054.146461)) - (-0.504788)
        
        ### TRIAL 2 ###
        #M = sqrt(((-0.764500 + X_0)*(-0.852586 - (X_1)) + 0.186688)*((X_1)*(-0.764500) - (-2.200765)) + -1.997414 - (1.365838)) - (0.819053)
        M = 0.790029 - ((-0.585648 + X_0)*((0.156573)*(54911.261124 + X_0 - (28145.860335) + -26770.085597 + (-1.440581)*(X_1))))
        return M
    
    def Mo12(X_0, X_1, X_2=0, X_3=0):
        from numpy import sqrt, cos, sin
        # original equation
        X_0 = 1/X_0
        M = (X_0)*(0.2965415493436768) - (-0.02115042312837423) + (sqrt(X_1))*((sqrt(X_0))*(0.2502798692342405 + 0.2502798692342405) - (0.3628181028391626)) + (sqrt(sqrt(X_0)))*(0.6637021883924064)
        #M = (X_0 + -4.120056 + (X_1)*(-2.311179))*(((sqrt(X_1) + -3.330640)*(X_0))*(0.066869) + -217034054.103521 - (-217034054.146461)) - (-0.504788)
        return M
    
    def go1(X_0, X_1, X_2, X_3):
        from numpy import sqrt, cos, sin
        # opriginal equation used RN g
        X_2 = X_3
        X_0 = 1/X_0
        g = 1 + (0.1+0.35*((X_0))*(X_1)**2)*(1-(sin(X_3)))**2
        return g







#%%

if __name__ == "__main__":
    X_0 = Symbol('\\left(\\frac{a}{c}\\right)')
    X_1 = Symbol('\\left(\\frac{a}{t}\\right)')
    X_2 = Symbol('\\phi')
    X_3 = Symbol('\\phi')

    X_0 = Symbol('X_0')
    X_1 = Symbol('X_1')
    X_2 = Symbol('X_2')
    X_3 = Symbol('X_3')
    
    RN_data, RN_eqn, F3d_data = get_data_sets()
    dataset = RN_data
    
    ac_over_1 = 0
    dataset = split_ac(dataset, ac_over_1)
    
    input_vars = dataset[:,0:4]
    F_input = dataset[:,-1]
    n = 1
    Mu1 = RN_data_funs.Mu1
    gu1 = RN_data_funs.gu1
    Mo1 = RN_data_funs.Mo1
    go1 = RN_data_funs.go1
    
    F_bingo = Bingo_cust(input_vars, Mu1 = Mu1, gu1 = gu1,Mo1 = Mo1, go1 = go1)
    
    F_RN = F_s(*input_vars.T, only_mg=0)
    error_plotting(F_input, F_RN, F_bingo, plot_both=True)
    plot_eqns(F_RN, F_bingo, input_vars, F_input, n)

    # plt.figure()
    # plt.plot(F_input,F_input, 'r.')
    # plt.plot(F_input,F_RN, 'b.')
    # plt.plot(F_input,F_bingo, 'g.')



