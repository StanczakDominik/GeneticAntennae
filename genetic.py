"""genetic algorithm optimization of radio antenna coverage"""
import matplotlib.pyplot as plt
import numpy as np

from calculations import antenna_coverage_population, utility_function, selection, crossover_cutoff, mutation
from plotting import plot_fitness, plot_population, plot_single

XMIN = 0.
XMAX = 1.
YMIN = 0.
YMAX = 1.

# grid resolution
NX = 100
NY = 100

# prepare 2D grid
x, DX = np.linspace(XMIN, XMAX, NX, retstep=True, endpoint=False)
y, DY = np.linspace(YMIN, YMAX, NY, retstep=True, endpoint=False)
X, Y = np.meshgrid(x, y)
R = np.stack((X, Y), axis=0)

NPOPULATION = 25
NGENERATIONS = 20
NANTENNAE = 5
# N pi r^2 = 1
# r = (1/ N pi)**0.5
DEFAULT_POWER = 0.2
P_CROSSOVER = 0.5
P_MUTATION = 1
MUTATION_STD = 0.1

# # population as weights, for now let's focus on the uniform population case
DISTANCES = ((R - np.array([(XMAX-XMIN)/2, (YMAX-YMIN)/2], ndmin=3).T)**2).sum(axis=0)
# WEIGHTS = np.exp(-DISTANCES*10)
UNIFORM_WEIGHTS = np.ones_like(DISTANCES)
WEIGHTS = UNIFORM_WEIGHTS

WEIGHTS_NORM = np.sum(WEIGHTS)
# WEIGHTS /= WEIGHTS_NORM

DEBUG_MESSAGES = False
PLOT_AT_RUNTIME = False
SAVE_AT_RUNTIME = False
SHOW_FINAL_CONFIGURATION = True
np.random.seed(0)

#=======CALCULATIONS======

TEMP_ARRAY = np.empty((NPOPULATION, NANTENNAE, 2))

#=====MAIN===========
def main_loop(n_generations, weights=WEIGHTS):
    """
    TODO: czym właściwie jest nasza generacja?
    * zbiorem N anten (macierz Nx2 floatów)?
    * chyba to - M zestawów po N anten (macierz MxNx2), i każdy pojedynczy
    reprezentant to macierz Nx2?
    * jeśli to drugie to będę pewnie musiał przerobić coverage() żeby to się
    sensownie wektorowo liczyło

    na razie zróbmy wolno i na forach :)
    """

    # r_antennae_population = np.random.random((NPOPULATION, NANTENNAE, 2))
    r_antennae_population = np.ones((NPOPULATION, NANTENNAE, 2))/2
    print("Generation {}/{}, {:.0f}% done".format(0, n_generations, 0), end='')

    max_fitness_history = np.zeros(n_generations)
    mean_fitness_history = np.zeros(n_generations)
    std_fitness_history = np.zeros(n_generations)
    for n in range(n_generations): # TODO: ew. inny warunek, np. mała różnica kolejnych wartości
        print("\rGeneration {}/{}, {:.0f}% done".format(n, n_generations, n / n_generations * 100), end='')
        plot_population(r_antennae_population, n, R, DEFAULT_POWER, weights,
                        filename="data/{:02d}.png".format(n_generations),
                        show=PLOT_AT_RUNTIME, save=SAVE_AT_RUNTIME)

        #nonessential
        coverage_population = antenna_coverage_population(r_antennae_population, R, zasieg=DEFAULT_POWER)
        if DEBUG_MESSAGES:
            print(utility_function(coverage_population, weights))

        if DEBUG_MESSAGES:
            print(r_antennae_population)
            print("Before selection")
        max_fit, mean_fit, std_fit = selection(r_antennae_population, R, weights, zasieg=DEFAULT_POWER, TEMP_ARRAY=TEMP_ARRAY)
        max_fitness_history[n] = max_fit
        mean_fitness_history[n] = mean_fit
        std_fitness_history[n] = std_fit
        if DEBUG_MESSAGES:
            print("After selection")
            print(r_antennae_population)

        crossover_cutoff(r_antennae_population, P_CROSSOVER, TEMP_ARRAY=TEMP_ARRAY)
        if DEBUG_MESSAGES:
            print("After crossover")
            print(r_antennae_population)

        mutation(r_antennae_population, MUTATION_STD, P_MUTATION)
        if DEBUG_MESSAGES:
            print("After mutation")
            print(r_antennae_population)
    print("\rJob's finished!")
    plot_population(r_antennae_population, n_generations, R, DEFAULT_POWER, weights,
                    filename="data/{:02d}.png".format(n_generations),
                    show=SHOW_FINAL_CONFIGURATION, save=True)
    #znalezienie optymalnego
    values = antenna_coverage_population(r_antennae_population, R, DEFAULT_POWER)
    utility_function_values = utility_function(values, WEIGHTS)
    best_candidate = np.argmax(utility_function_values)
    r_best = r_antennae_population[best_candidate]
    print(r_best)

    plot_single(r_best, n_generations, R, DEFAULT_POWER, weights,
                filename="data/{:02d}_max.png".format(n_generations),
                show=SHOW_FINAL_CONFIGURATION, save=True)

    print(mean_fitness_history)
    print(max_fitness_history)
    plot_fitness(mean_fitness_history, max_fitness_history, std_fitness_history, filename="data/fitness.png", save=True)
    plt.show()
    
if __name__=="__main__":
    main_loop(NGENERATIONS)