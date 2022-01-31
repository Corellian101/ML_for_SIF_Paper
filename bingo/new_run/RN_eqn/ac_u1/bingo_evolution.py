# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import sys
import time
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo.symbolic_regression.agraph.component_generator import ComponentGenerator
from bingo.symbolic_regression.explicit_regression import ExplicitRegression, ExplicitTrainingData

from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.parallel_archipelago import ParallelArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.fitness_predictor_island import FitnessPredictorIsland
from bingo.local_optimizers.continuous_local_opt import ContinuousLocalOptimization
from bingo.stats.pareto_front import ParetoFront
from bingo.evolutionary_optimizers.parallel_archipelago import load_parallel_archipelago_from_file
from bingo.evolutionary_algorithms.deterministic_crowding import DeterministicCrowdingEA
from bingo.evolutionary_optimizers.island import Island

POP_SIZE = 150
STACK_SIZE = 64
MAX_GENERATIONS = 800000
FITNESS_THRESHOLD = 1e-10
CHECK_FREQUENCY = 1000
MIN_GENERATIONS = 1000
CROSSOVER_PROBABILITY = 0.6
MUTATION_PROBABILITY = 0.4
STAGNATION = 100000

def sort(data):
    models = []
    model = np.unique(data[:,[0,1,2]], axis=0)
    
    for i in model:
        models.append(data[np.where((data[:,[0,1,2]] == i).all(axis=1))])

    return models

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def agraph_similarity(ag_1, ag_2):
    return ag_1.fitness == ag_2.fitness and ag_1.get_complexity() == ag_2.get_complexity()

def print_pareto_front(hall_of_fame):
    print("  FITNESS    COMPLEXITY    EQUATION")

    for member in hall_of_fame:
        print('%.3e    ' % member.fitness, member.get_complexity(),
                '   f(X_0) =', member)

def plot_pareto_front(hall_of_fame):
    fitness_vals = []
    complexity_vals = []
    for member in hall_of_fame:
        fitness_vals.append(member.fitness)
        complexity_vals.append(member.get_complexity())
    plt.figure()
    plt.step(complexity_vals, fitness_vals, 'k', where='post')
    plt.plot(complexity_vals, fitness_vals, 'or')
    plt.xlabel('Complexity')
    plt.ylabel('Fitness')
    plt.savefig('pareto_front')

def execute_generational_steps(model):
    communicator = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()

    x = None
    y = None

    if rank == 0:

        df = pd.read_csv('2_RN_eqn.csv')
        #df['sin_phi'] = np.sin(df.phi)
        #df['cos_phi'] = np.cos(df.phi)

        #y = df.F.values

        #x = df[['a_c', 'a_t', 'sin_phi', 'cos_phi', 'c_b']].values
        data = df[['a/c','a/t','c/b','phi','F','Mg']].values
        models = sort(data)

        data = np.zeros(6)
        #phis = np.linspace(0,np.pi,10)
        phis = [np.pi/2]
        for m in models:
            if m[0,0] > 1:
                continue
            for ph in phis:
                data = np.row_stack((data,m[np.where(m[:,3] == find_nearest(m[:,3],ph))[0][0]]))

        data = data[1::]
        x = data[:,[0,1]]
        y = data[:,-1]

        print(y)
        print()
        print(x)
        print()
        print(np.shape(x))
        print(np.shape(y))
        print(np.mean(y))
        print(max(y))
        #stop
        x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        # if only doing gradient boosting
        if model is not None:
            y_model = 1
            for i in model:
                y_model *= i.evaluate_equation_at(x[:,[0,1,2,3]])
            y = y.flatten()/y_model.flatten()


    #print(data[:,3])
    #print(x[0,1000])
    x = MPI.COMM_WORLD.bcast(x, root=0)
    y = MPI.COMM_WORLD.bcast(y, root=0)

    training_data = ExplicitTrainingData(x, y)

    component_generator = ComponentGenerator(x.shape[1])
    #component_generator = ComponentGenerator(1)
    component_generator.add_operator('+') # +
    component_generator.add_operator('-') # -
    component_generator.add_operator('*') # *
    component_generator.add_operator('sqrt') # sqrt
    #component_generator.add_operator('sin') # sqrt
    #component_generator.add_operator('cos')

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)


    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator, use_simplification=False)

    fitness = ExplicitRegression(training_data=training_data, metric='rmse', relative=True)

    lo_options={'disp':False,
             'xatol':1e-6,
             'fatol':1e-6,
             'maxfev':1000,
             'maxiter':1000,
             'adaptive':True}
    local_opt_fitness = ContinuousLocalOptimization(
             fitness, algorithm='Nelder-Mead', options=lo_options,
             tolerance=1e-10)

    evaluator = Evaluation(local_opt_fitness)


    ea = DeterministicCrowdingEA(evaluator, crossover, mutation, 0.4, 0.6)
    island = Island(ea, agraph_generator, POP_SIZE)

    pareto_front = ParetoFront(secondary_key = lambda ag: ag.get_complexity(),
            similarity_function=agraph_similarity)

    archipelago = ParallelArchipelago(island, hall_of_fame=pareto_front)
    #archipelago = load_parallel_archipelago_from_file('checkpoint#.pkl')

    optim_result = archipelago.evolve_until_convergence(MAX_GENERATIONS, FITNESS_THRESHOLD,
            convergence_check_frequency=CHECK_FREQUENCY, min_generations=MIN_GENERATIONS,
            checkpoint_base_name='checkpoint_', num_checkpoints=5,stagnation_generations=STAGNATION)

    if optim_result.success:
        if rank == 0:
            print("best: ", archipelago.get_best_individual())

    if rank == 0:
        print(optim_result)
        print("Generation: ", archipelago.generational_age)
        print_pareto_front(pareto_front)
        plot_pareto_front(pareto_front)

def main(model):
    execute_generational_steps(model)

if __name__ == '__main__':
    print('starting')

    model = None
    pickle_file = None
    model_num = []
    if pickle_file is not None: 
        model = [] 
        for i in range(len(pickle_file)):
            pickle = load_parallel_archipelago_from_file(pickle_file[i])
            print(pickle)
            model_number = model_num[i]
            print(model_number)
            model.append(pickle.hall_of_fame[model_number])

    print('done')
    main(model)

