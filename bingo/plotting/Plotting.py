from sympy import Symbol, integrate, init_printing, diff, simplify, Matrix, print_latex, sqrt, cos, sin, expand
from sympy.parsing.sympy_parser import parse_expr
import matplotlib.pyplot as plt
import numpy as np
from Raju_Newman_new import *
from sort_data import sort, find_nearest_idx, remove_cb, find_nearest
import pandas as pd
from sklearn.metrics import mean_absolute_error

class Plotting:
    def __init__(self, case, ac_greater_1):
        self.case = case
        self.ac_greater_1 = ac_greater_1
        self.input_vars = None
        self.F_input = None
        self.F_bingo = None
        RN_data, RN_eqn, F3d_data = self.get_data_sets()
        RN_data_test, RN_eqn_test, F3d_data_test = self.get_data_sets(test=1)
        if self.case == "RN_data":
            dataset = RN_data
            dataset_test = RN_data_test
            self.bingo_funs = RN_data_funs
        elif self.case == "RN_eqn":
            dataset = RN_eqn
            dataset_test = RN_eqn_test
            self.bingo_funs = RN_eqn_funs
        elif self.case == "F3d_data":
            dataset = F3d_data
            dataset_test = F3d_data_test
            self.bingo_funs = F3d_data_funs
        else:
            raise ValueError("not a valid case")

        dataset = self.split_ac(dataset, ac_greater_1)
        self.n_models = self.num_models(dataset)
        dataset_test = self.split_ac(dataset_test, ac_greater_1)
        self.input_vars = dataset[:,0:4]
        self.F_input = dataset[:,-1]
        self.F_RN = F_s(*self.input_vars.T, only_mg=0)
        self.F_bingo = Bingo_cust(self.input_vars, Mu1 = self.bingo_funs.Mu1, gu1 = self.bingo_funs.gu1, 
                Mo1 = self.bingo_funs.Mo1, go1 = self.bingo_funs.go1)

        self.input_vars_test = dataset_test[:,0:4]
        self.F_input_test = dataset_test[:,-1]
        self.F_RN_test = F_s(*self.input_vars_test.T, only_mg=0)
        self.F_bingo_test = Bingo_cust(self.input_vars_test, Mu1 = self.bingo_funs.Mu1, gu1 = self.bingo_funs.gu1,
                Mo1 = self.bingo_funs.Mo1, go1 = self.bingo_funs.go1)

    @staticmethod
    def a_c_to_n(models, a_c, a_t, c_b):
        for i in range(len(models)):
            if (models[i][0,[0,1,2]] == [a_c, a_t, c_b]).all():
                return i  
        return None

    def plot_eqns(self, n=None, a_c=None, a_t=None, c_b = None):
        F_RN = self.F_RN
        F_bingo = self.F_bingo
        input_vars = self.input_vars
        F_input = self.F_input
        print

        data = np.column_stack((input_vars, F_input, F_RN, F_bingo))
        models = sort(data)
        
        data = np.zeros(data.shape[1])
        for m in models:
            phis = np.linspace(0, np.pi, 20)
            for ph in phis:
                data = np.row_stack((data,m[np.where(m[:,3] == find_nearest(m[:,3],ph))[0][0]]))
        data = data[1::]
        models = sort(data)
        
        if n == None:
            n = self.a_c_to_n(models, a_c, a_t, c_b)
        if n == None:
            plt.plot([1,1],[1,1])
            return
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

    @staticmethod 
    def split_ac(data, ac_greater_1):
        if ac_greater_1:
            data = data[np.where(data[:,0] > 1)[0]]
            #data[:,0] = 1/data[:,0]
        else:
            data = data[np.where(data[:,0] <= 1)[0]]   
        return data

    @staticmethod
    def num_models(dataset):
        models = sort(dataset)
        return len(models)

    @staticmethod
    def pull_from_data(data, a_c, a_t, phi, c_b=0.2):
        locs = np.where((data[:,[0,1,2]] == [a_c, a_t, c_b]).all(axis=1))
        data = data[locs]
        idx = find_nearest_idx(data[:,3], phi)
        return data[idx]

    def error_plotting(self, plot_both=False, plot_only2=False):
        F_input = self.F_input_test
        F_RN = self.F_RN_test
        F_bingo = self.F_bingo_test
        
        # data = np.column_stack((self.input_vars_test, F_input, F_RN, F_bingo))
        # data = remove_cb(data)
        # F_input = data[:,-3]
        # F_RN = data[:, -2]
        # F_bingo = data[:, -1]
 
        
        error_RN = abs(F_input-F_RN)/F_input
        error_bingo = abs(F_input-F_bingo)/F_input
        mae_RN = np.round(mean_absolute_error(F_input, F_RN), 4)
        mae_bingo = np.round(mean_absolute_error(F_input, F_bingo), 4)
        if not plot_only2:
            plt.figure()
            plt.hist(error_bingo, color='g', alpha=0.5, label='bingo Error')
            plt.hist(error_RN, color='b', alpha=0.5, label='RN Error')
            plt.title(f"RN mean error = {np.round(np.mean(error_RN), 4)}, bingo mean error = {np.round(np.mean(error_bingo), 4)} \n RN max error = {np.round(np.max(error_RN), 4)}, bingo max error = {np.round(np.max(error_bingo), 4)} \n RN mae = {mae_RN}, bingo mae = {mae_bingo}")
            plt.legend()
            
        if plot_both:
            plt.figure()
            plt.hist(error_RN, color='b', label='RN Error')
            plt.hist(error_bingo, color='g', label='bingo Error')
            plt.title(f"RN mean error = {np.round(np.mean(error_RN), 4)}, bingo mean error = {np.round(np.mean(error_bingo), 4)} \n RN max error = {np.round(np.max(error_RN), 4)}, bingo max error = {np.round(np.max(error_bingo), 4)} \n RN mae = {mae_RN}, bingo mae = {mae_bingo}")
            plt.legend()
        if plot_only2:
            plt.figure()
            plt.hist(error_RN, color='b', label='RN Error')
            plt.hist(error_bingo, color='g', label='bingo Error')
            plt.title(f"RN mean error = {np.round(np.mean(error_RN), 4)}, bingo mean error = {np.round(np.mean(error_bingo), 4)} \n RN max error = {np.round(np.max(error_RN), 4)}, bingo max error = {np.round(np.max(error_bingo), 4)} \n RN mae = {mae_RN}, bingo mae = {mae_bingo}")
            plt.legend()
        return

    @staticmethod
    def get_data_sets(test=0):
        if not test:
            RN_data = pd.read_csv('1_RN_data.csv')[['a/c','a/t','c/b','phi','Mg','F']].values
            #RN_eqn = pd.read_csv('2_RN_eqn.csv')[['a/c','a/t','c/b','phi','Mg','F']].values
            RN_eqn = pd.read_csv('Sorted_RN_eqn_data.csv')[['a/c','a/t','c/b','phi','Mg','F']].values
            F3d_data = pd.read_csv('3_FRANC3D_FULL.csv')[['a/c','a/t','c/b','phi','Mg', 'F']].values
            
            #F3d_data = pd.read_csv('bingo_data_sets/F3d_data_M_o1_cb.csv')[['a/c','a/t','c/b','phi','Mg', 'F']].values
        elif test:
            RN_data = pd.read_csv('1_RN_data_TEST.csv')[['a/c','a/t','c/b','phi','Mg','F']].values
            RN_eqn = pd.read_csv('2_RN_eqn_TEST.csv')[[' a/c','a/t','c/b','phi','Mg','F']].values
            F3d_data = pd.read_csv('3_FRANC3D_FULL_TEST.csv')[['a/c','a/t','c/b','phi','Mg', 'F']].values
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
        
        ### TRIAL 2 ###
        g = (X_1 + 0.645087)*((X_1 + 0.645087)*(0.14846353627*sin(X_2) - 0.297424128459432)*sin(X_2) + 0.347009) + 0.803747
        return g
    
    def Mo1(X_0, X_1, X_2=0, X_3=0):
        from numpy import sqrt, cos, sin
        X_0 = 1/X_0
        M = ((X_0)*(0.274624) - (59882531.611787) + 59882531.514921)*((-0.352740 + X_0)*(X_1 + -1.793803 + -0.352740 + X_0) - (-3.116885)) - (-0.604788)
        
        ### TRIAL 2 ###
        M = sqrt(X_0)*((X_1 - 0.079181)*(0.746112*X_0*(sqrt(X_0) - 0.993416) + 0.119041) + 1.058784) - 0.0281910000012431
        ### STACK SIZE 30 ###
        M = (0.10565*(X_1 + 0.360041)*(X_0**2*(0.720527*X_1 - 0.352391021998) + 0.105133) - 0.2631525974)*((X_0 - (X_1 + 0.360041)*(-0.720527*X_0**2*(X_1 - 0.489074) - 0.105133) - 2.490796)*(-X_0**2 + (X_1 + 0.360041)*(-0.720527*X_0**2*(X_1 - 0.489074) - 0.105133) - 4.581786) - 16.295681) - 1.036403
        return M
    
    def go1(X_0, X_1, X_3, X_2):
        from numpy import sqrt, cos, sin
        g = 1 + (0.1+0.35*((X_0**(-1)))*(X_1)**2)*(1-(sin(X_2)))**2
        X_0 = 1/X_0
        #g = 0.999991 - (((8833227.041394 + (sin(X_2))*(1.999006) - (-1261496.774534) - ((sin(X_2))*(sin(X_2))) + -10094724.815343)*(X_1 + 0.222239))*(0.232156))
        
        ### TRIAL 2 ###
        g =  ((((-0.973016)*(sin(X_2)) - (-0.973278))*(-0.608237))*(((-0.973016)*(sin(X_2)) - (-0.973278))*(-0.608237)))*((X_0)*((X_1)*(X_1)) + 0.425402 + -0.140293) - (-0.999990)
        g = 0.35*(X_0*X_1**2 + 0.285109)*(sin(X_2) - 1)**2 + 1
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
        #g = 1 + (0.1 + 0.35*(X_1)**2)*(1-sin(X_2))**2
        g = (1.0263496149498648 + sin(6373.023412385456 + 11071.131423790328 + -11350.186594670391 - (6373.023412385456 - (sin(X_3)))))*(X_1) + cos(11071.131423790328)
        g = -(-1.0263496149499*X_1 + X_1*sin(279.05517088006 - sin(X_3)) - 0.9873999518025)
        
        
        ### TRIAL 2 ###
        g = (((1.015248 - (sin(X_2)))*(-0.061607))*(-7.069833) + -5.222969 + (X_1)*(X_1) + 2.775947 - (-2.266952))*(((1.015248 - (sin(X_2)))*(-0.061607))*(-7.069833)) + sqrt(1.015248)
        
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
    
    def go1(X_0, X_1, X_2, X_3):
        from numpy import sqrt, cos, sin
        # opriginal equation used RN g
        X_2 = X_3
        X_0 = 1/X_0
        g = 1 + (0.1+0.35*((X_0))*(X_1)**2)*(1-(sin(X_3)))**2
        #g = 1.313275 + (cos(((0.770511)*(X_1) - ((0.336310)*(0.336310)))*(X_0) + -1.674651 - (sin(X_2)) + -0.577035))*(0.313262)
        
        ### TRAIL 2 ###
        g = (118.696317 + sin((cos(X_1) + 1.306766 + sin(X_2))*(0.476230)) - (119.687454))*(-0.505565 - (X_0)) + 1.008365
        #g = (sin(sin(X_2) + ((X_1)*(X_0 + 0.257186))*(-0.564576) + 0.758759))*(-54.537231 - (-54.224772)) + 1.312375
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
    #RN_data = Plotting("RN_data", 0)
    
    model = Plotting("F3d_data", 1)
    model.plot_eqns(a_c=2, a_t=0.8, c_b = 0.2)
    model.error_plotting()
    
    phi = np.linspace(0, np.pi, 100)
    plt.figure()
    plt.plot(phi, )
    RN_data, RN_eqn, F3d_data = Plotting.get_data_sets()
    # a_c = 0.2
    # test_smith_f3d = F3d_data[np.where((F3d_data[:,[0, 1, 2]] == [a_c, 0.2, 0.2]).all(axis=1))[0]]
    # test_smith_RN = RN_data[np.where((RN_data[:,[0, 1, 2]] == [a_c, 0.2, 0.2]).all(axis=1))[0]]
    # F_paris = Paris(test_smith_f3d[:,0], 0.2,test_smith_f3d[:,3])
    # smith_data = smith(test_smith_f3d[:,3],test_smith_f3d[:,0])
    # smith_RN_eqn = F_s(*test_smith_f3d[:,0:4].T)
    # plt.figure()
    # plt.title('a/c = '+str(a_c))
    # plt.plot(test_smith_f3d[:,3], test_smith_f3d[:,-1], 'ro', label='FRANC3D')
    # plt.plot(test_smith_f3d[:,3], internal_ellipse(test_smith_f3d[:,0], test_smith_f3d[:,3]), 'k-', label='embedded ellipse')
    # plt.plot(test_smith_f3d[:,3], smith_data, 'b-', label='Smith eqn a/t=c/b = 0')
    # plt.plot(test_smith_f3d[:,3], smith_RN_eqn, 'g-', label='RN eqn')
    # plt.plot(test_smith_RN[:,3], test_smith_RN[:,-1], 'k^', label='RN data')
    # plt.plot(test_smith_f3d[:,3], F_paris, 'mo', label='Paris estimate')
    # plt.legend()
    
#%%
    test = sort(RN_eqn)
    def plot(n):
        plt.figure()
        data = abs(np.insert(test[n][:,3], 0, 0) - np.append(test[n][:,3], 0))
        plt.plot(test[n][:,3], data[1::])
        plt.ylabel('spacing')
        plt.xlabel('phi')
