import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from bingo.evolutionary_optimizers.parallel_archipelago import load_parallel_archipelago_from_file

pickles = [load_parallel_archipelago_from_file(checkpoint) for checkpoint in glob.glob('checkpoint*.pkl')]

def get_digit_from_filename(filename):
    number = [int(s) for s in re.split('_|\.', filename) if s.isdigit()]
    return number

def get_population_fitness(pickle):
    fitnesses = [individual.fitness for individual in pickle._island.get_population() if (individual.get_complexity() > 4) & (individual.fitness < 8)]
    return fitnesses

checkpoint_nums = [get_digit_from_filename(filename) for filename in glob.glob('checkpoint*.pkl')]
fitnesses = [get_population_fitness(pickle) for pickle in pickles]

#checkpoint_nums, fitnesses = zip(*sorted(zip(checkpoint_nums, fitnesses)))
pop = pickles[0]._island._population_size
stack = pickles[0]._island._generator().command_array.shape[0]
generations = 2000
fig, ax = plt.subplots()
ax.set_title(str(generations)+' generations, '+str(stack)+' stack size, '+str(pop)+' pop size')
ax.boxplot(fitnesses)
ax.set_ylim(0,8)
plt.xticks(list(range(1, len(checkpoint_nums) + 1)), checkpoint_nums)
plt.ylabel('Fitness (Mean Absolute Error)')
plt.xlabel('Generations')
plt.savefig('pop'+str(pop)+'_stck'+str(stack)+'_gen'+str(generations), dpi=300)
