import numpy as np
import sys
from sympy import *

from bingo.evolutionary_optimizers.parallel_archipelago import load_parallel_archipelago_from_file

checkpoint_num = 2000#sys.argv[1]
pickle_file = "".join(['checkpoint_', str(checkpoint_num), '.pkl'])
pickle = load_parallel_archipelago_from_file(pickle_file)


X_0, X_1, X_2, X_3, X_4, X_5 = symbols('X_0 X_1 X_2 X_3 X_4 X_5')

def simplify_equation(equation):
    equation = equation.replace(')(',')*(')
    equation = equation.replace('^','**')
    equation = equation.replace('  ','')
    equation = equation.replace(u'\xa0', ' ')
    simplified = simplify(equation)
    simplified = N(simplified, 5)
    simplified = str(simplified).replace('X_0', 'a_c')
    simplified = str(simplified).replace('X_1', 'a_t')
    simplified = str(simplified).replace('X_2', 'y')
    simplified = str(simplified).replace('X_3', 'z')
    simplified = str(simplified).replace('X_4', 'y2')
    simplified = str(simplified).replace('X_5', 'z2')
    return simplified

for individual in pickle.hall_of_fame:
    equation = individual.get_console_string()
    simple_equation = simplify_equation(equation)
    print(individual.get_complexity(), ': ', simple_equation)
    print('Fitness: ', individual.fitness)

