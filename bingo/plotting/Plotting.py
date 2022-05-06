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
        
        # data = np.zeros(data.shape[1])
        # for m in models:
        #     phis = np.linspace(0, np.pi, 20)
        #     for ph in phis:
        #         data = np.row_stack((data,m[np.where(m[:,3] == find_nearest(m[:,3],ph))[0][0]]))
        # data = data[1::]
        # models = sort(data)
        
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
    
    model = Plotting("RN_data", 0)
    #model.plot_eqns(a_c=2, a_t=0.8, c_b = 0.2)
    model.error_plotting()
    
    # phi = np.linspace(0, np.pi, 100)
    # plt.figure()
    # plt.plot(phi, )
    # RN_data, RN_eqn, F3d_data = Plotting.get_data_sets()
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
