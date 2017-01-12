"""genetic algorithm optimization of radio antenna coverage"""
import numpy as np
import matplotlib.pyplot as plt

# grid dimensions
# DO NOT CHANGE THESE
# FUCKS UP MATH
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
NGENERATIONS = 50
NANTENNAE = 20
# N pi r^2 = 1

# r = (1/ N pi)**0.5
DEFAULT_POWER = (np.pi * NANTENNAE)**-0.5
P_CROSSOVER = 0.01
P_MUTATION = 1e-2
MUTATION_STD = 0.01

# # population as weights, for now let's focus on the uniform population case
DISTANCES = ((R - np.array([(XMAX-XMIN)/2, (YMAX-YMIN)/2], ndmin=3).T)**2).sum(axis=0)
WEIGHTS = np.exp(-DISTANCES*10)
UNIFORM_WEIGHTS = np.ones_like(DISTANCES)
WEIGHTS = UNIFORM_WEIGHTS

WEIGHTS_NORM = np.sum(WEIGHTS)
# WEIGHTS /= WEIGHTS_NORM

DEBUG_MESSAGES = False
PLOT_AT_RUNTIME = False
SAVE_AT_RUNTIME = False
np.random.seed(0)

#=======CALCULATIONS======
def antenna_coverage_population(R_antenna, r_grid, zasieg=DEFAULT_POWER):
    result_array = np.empty((NPOPULATION, NX, NY), dtype=bool)
    for i, population_member in enumerate(R_antenna):
        result_array[i] = antenna_coverage(population_member, r_grid, zasieg)
    return result_array

def antenna_coverage(r_antenna, r_grid, zasieg=DEFAULT_POWER):
    """
    compute coverage of grid by single antenna
    assumes coverage is zasieg/distance^2

    array of distances squared from each antenna
    uses numpy broadcasting
    cast (N, 2) - (2, NX, NY) to (N, 2, 1, 1) - (1, 2, NX, NY)
    both are then broadcasted to (N, 2, NX, NY)
    """
    distance_squared = ((r_antenna[..., np.newaxis, np.newaxis] -\
                         r_grid[np.newaxis, ...])**2).sum(axis=1)

    # TODO: decide if we want to go for 1/r^2 antenna coverage
    # result = (zasieg*ANTENNA_RADIUS**2/distance_squared).sum(axis=0)
    # # cover case where antenna is located in grid point
    # result[np.isinf(result)] = 0 # TODO: find better solution

    # binary coverage case
    result = (distance_squared < zasieg**2) # is grid entry covered by any
    result = result.sum(axis=0) > 0        # logical or
    result = result > 0

    # TODO: ujemne wagi przez maksymalną możliwą

    return result

def utility_function(coverage_population, weights = WEIGHTS):
    """returns total coverage as fraction of grid size
    for use in the following genetic operators
    this way we're optimizing a bounded function (values from 0 to 1)"""
    return (weights.reshape(1,NX,NY)*coverage_population).sum(axis=(1,2))/NX/NY

TEMP_ARRAY = np.empty((NPOPULATION, NANTENNAE, 2))

def selection(r_antennae_population, weights = WEIGHTS):
    coverage_population = antenna_coverage_population(r_antennae_population, R)
    utility_function_values = utility_function(coverage_population, weights)
    utility_function_total = utility_function_values.sum()
    utility_function_normalized = utility_function_values / utility_function_total
    dystrybuanta = utility_function_normalized.cumsum()
    random_x = np.random.random(NPOPULATION).reshape(1, NPOPULATION)

    new_r_antennae_population = TEMP_ARRAY
    # for i, x in enumerate(random_x.T):
    #     indeks = (x > dystrybuanta).sum()
    #     print(indeks)
    #     new_r_antennae_population[i] = r_antennae_population[indeks]
    selected_targets = (random_x > dystrybuanta.reshape(NPOPULATION, 1)).sum(axis=0)
    # print(selected_targets, utility_function_values, sep='\n')
    print(utility_function_values.max(), utility_function_values.mean(), utility_function_values.std())

    new_r_antennae_population[...] = r_antennae_population[selected_targets]
    r_antennae_population[...] = new_r_antennae_population[...]
    return utility_function_values.max(), utility_function_values.mean(), utility_function_values.std()

def crossover_cutoff(r_antennae_population, probability_crossover = P_CROSSOVER):
    TEMP_ARRAY[...] = r_antennae_population[...]
    for i in range(0, NPOPULATION, 2):
        if i + 1 < NPOPULATION and np.random.random() < probability_crossover:
            cutoff = np.random.randint(0, NANTENNAE)
            a = r_antennae_population[i+1]
            b = r_antennae_population[i]
            if DEBUG_MESSAGES:
                print("Exchanging these two at k = {}".format(cutoff))
                print(a)
                print(b)
            TEMP_ARRAY[i, cutoff:] = a[cutoff:]
            TEMP_ARRAY[i+1, cutoff:] = b[cutoff:]
            if DEBUG_MESSAGES:
                print("They are now", TEMP_ARRAY[i], TEMP_ARRAY[i+1], sep="\n")
    r_antennae_population[...] = TEMP_ARRAY[...]

def mutation(r_antennae_population, gaussian_std = MUTATION_STD, p_mutation=P_MUTATION):
    """
    https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)

    acts in place!

    podoba mi się pomysł który mają na wiki, żeby mutacja
    1. przesuwała wszystkie anteny o jakiś losowy wektor z gaussowskiej
    dystrybucji
    2. (konieczność u nas:) renormalizowała położenia anten do pudełka
    (xmin, xmax), (ymin, ymax)

    pytanie - czy wystarczy zrobić okresowe warunki brzegowe (modulo), czy skoki
    tym spowodowane będą za duże (bo nie mamy okresowości na pokryciu)?
    w tym momencie - może lepiej zamiast tego robić coś typu max(xmax, x+dx)?
    """
    # don't want to move population P's antenna A's X without moving its Y
    which_to_move = (np.random.random((NPOPULATION, NANTENNAE)) < p_mutation)
    # which_to_move = np.ones((NPOPULATION, NANTENNAE))
    how_much_to_move = np.random.normal(scale=gaussian_std,
                                        size=(NPOPULATION, NANTENNAE, 2))
    if DEBUG_MESSAGES:
        print(which_to_move)
    r_antennae_population += which_to_move[..., np.newaxis] * how_much_to_move

    # TODO: zamienić okresowe warunki brzegowe na wymuszanie 0 w ujemnych fitnessach
    # TODO: bądź negatywne premiowanie jeśli kółka się nie spełniają
    r_antennae_population[:, :, 0] %= XMAX        # does this need xmin somehow?
    r_antennae_population[:, :, 1] %= YMAX        # likewise?

#======PLOTTING======
def plot_fitness(mean_fitness_history, max_fitness_history, std_fitness_history, filename=False, show=True, save=True):
    if show or save:
        plt.plot(mean_fitness_history, "o-", label="Average fitness")
        plt.plot(max_fitness_history, "o-", label="Max fitness")
        plt.fill_between(np.arange(std_fitness_history.size),
                        mean_fitness_history+std_fitness_history,
                        mean_fitness_history-std_fitness_history,
                        alpha=0.5,
                        facecolor='orange')
        plt.xlabel("Generation #")
        plt.ylabel("Fitness")
        plt.ylim(0,1)
        plt.legend()
        plt.grid()
        if filename and save:
            plt.savefig(filename)
        if show:
            plt.show()
        else:
            pass

def plot_population(r_antennae_population, generation_number, filename=None, show=True, save=True):
    """
    plot grid values (coverage (weighted optionally) and antenna locations)
    """
    if show or save:
        fig, axis = plt.subplots()
        values = antenna_coverage_population(r_antennae_population, r_grid=R)
        utility_function_values = utility_function(values, WEIGHTS)
        best_candidate = np.argmax(utility_function_values)

        for i, antenna_locations in enumerate(r_antennae_population):
            x_a, y_a = antenna_locations.T
            marker_size = 10
            alpha = 0.6
            if i == best_candidate:
                marker_size *= 2
                alpha = 1
            axis.plot(x_a, y_a, "*", label="#{}".format(i), ms=marker_size, alpha=alpha)

        axis.contourf(X, Y, WEIGHTS, 100, cmap='viridis', label="Coverage")
        configurations = axis.contourf(X, Y, values.sum(axis=0), 100, cmap='viridis', alpha=0.5)
        fig.colorbar(configurations)

        axis.set_title(r"Generation {}, $<f>$ {:.2f} $\pm$ {:.2f}, max {:.2f}".format(
            generation_number,
            utility_function_values.mean(),
            utility_function_values.std(),
            utility_function_values.max(),
            ))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_xlim(XMIN, XMAX)
        axis.set_ylim(YMIN, YMAX)
        # axis.legend(loc='best')
        if save and filename:
            fig.savefig(filename)
        if show:
            plt.show()
        else:
            return fig

def plot_single(r_antennae, generation_number, filename=None, show=True, save=True):
    """
    plot grid values (coverage (weighted optionally) and antenna locations)
    """
    if show or save:
        fig, axis = plt.subplots()
        values = antenna_coverage(r_antennae, r_grid=R)
        utility_function_value = (WEIGHTS*values).sum()/NX/NY

        x_a, y_a = r_antennae.T
        marker_size = 20
        alpha = 1
        axis.plot(x_a, y_a, "*", ms=marker_size, alpha=alpha)

        axis.contourf(X, Y, WEIGHTS, 100, cmap='viridis', label="Coverage")
        configurations = axis.contourf(X, Y, values, 100, cmap='viridis', alpha=0.5)
        fig.colorbar(configurations)

        axis.set_title(r"Generation {}, optimal candidate, f {:.2f}".format(
            generation_number,
            utility_function_value,
            ))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_xlim(XMIN, XMAX)
        axis.set_ylim(YMIN, YMAX)
        # axis.legend(loc='best')
        if filename and save:
            fig.savefig(filename)
        if show:
            plt.show()
        else:
            return fig

#=====MAIN===========
def main_loop(NGENERATIONS):
    """
    TODO: czym właściwie jest nasza generacja?
    * zbiorem N anten (macierz Nx2 floatów)?
    * chyba to - M zestawów po N anten (macierz MxNx2), i każdy pojedynczy
    reprezentant to macierz Nx2?
    * jeśli to drugie to będę pewnie musiał przerobić coverage() żeby to się
    sensownie wektorowo liczyło

    na razie zróbmy wolno i na forach :)
    """

    r_antennae_population = np.random.random((NPOPULATION, NANTENNAE, 2))

    print("Generation {}/{}, {:.0f}% done".format(0, NGENERATIONS, 0),end='')

    max_fitness_history = np.zeros(NGENERATIONS)
    mean_fitness_history = np.zeros(NGENERATIONS)
    std_fitness_history = np.zeros(NGENERATIONS)
    for n in range(NGENERATIONS): # TODO: ew. inny warunek, np. mała różnica kolejnych wartości
        print("\rGeneration {}/{}, {:.0f}% done".format(n, NGENERATIONS, n/NGENERATIONS*100),end='')
        plot_population(r_antennae_population, n,
                        filename="{:02d}.png".format(NGENERATIONS),
                        show=PLOT_AT_RUNTIME, save=SAVE_AT_RUNTIME)

        #nonessential
        coverage_population = antenna_coverage_population(r_antennae_population, R)
        if DEBUG_MESSAGES:
            print(utility_function(coverage_population))

        if DEBUG_MESSAGES:
            print(r_antennae_population)
            print("Before selection")
        max_fit, mean_fit, std_fit = selection(r_antennae_population)
        max_fitness_history[n] = max_fit
        mean_fitness_history[n] = mean_fit
        std_fitness_history[n] = std_fit
        if DEBUG_MESSAGES:
            print("After selection")
            print(r_antennae_population)

        crossover_cutoff(r_antennae_population)
        if DEBUG_MESSAGES:
            print("After crossover")
            print(r_antennae_population)

        mutation(r_antennae_population)
        if DEBUG_MESSAGES:
            print("After mutation")
            print(r_antennae_population)
    print("\rJob's finished!")
    plot_population(r_antennae_population, NGENERATIONS,
                    filename="{:02d}.png".format(NGENERATIONS),
                    show=PLOT_AT_RUNTIME, save=True)
    #znalezienie optymalnego
    values = antenna_coverage_population(r_antennae_population, r_grid=R)
    utility_function_values = utility_function(values, WEIGHTS)
    best_candidate = np.argmax(utility_function_values)
    r_best = r_antennae_population[best_candidate]
    print(r_best)

    plot_single(r_best, NGENERATIONS,
                filename="{:02d}_max.png".format(NGENERATIONS),
                show=True, save=True)

    np.savetxt("meanfit.dat", mean_fitness_history)
    np.savetxt("maxfit.dat", max_fitness_history)
    print(mean_fitness_history)
    print(max_fitness_history)
    plot_fitness(mean_fitness_history, max_fitness_history, std_fitness_history)
if __name__=="__main__":
    main_loop(NGENERATIONS)
    # plot_fitness(np.loadtxt("meanfit.dat"), np.loadtxt("maxfit.dat"))
