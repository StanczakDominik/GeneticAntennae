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

N_POPULATION = 20
N_GENERATIONS = 20
N_ANTENNAE = 3
DEFAULT_POWER = 0.5
P_CROSSOVER = 0.8
P_MUTATION = 1e-3
MUTATION_STD = 0.6

# # population as weights, for now let's focus on the uniform population case
DISTANCES = ((R - np.array([(XMAX-XMIN)/2, (YMAX-YMIN)/2], ndmin=3).T)**2).sum(axis=0)
WEIGHTS = np.exp(-DISTANCES*10)
# UNIFORM_WEIGHTS = np.ones_like(DISTANCES)
# WEIGHTS = UNIFORM_WEIGHTS

DEBUG_MESSAGES = False

np.random.seed(0)

def antenna_coverage_population(R_antenna, r_grid, power=DEFAULT_POWER):
    result_array = np.empty((N_POPULATION, NX, NY), dtype=bool)
    for i, population_member in enumerate(R_antenna):
        result_array[i] = antenna_coverage(population_member, r_grid, power)
    return result_array

def antenna_coverage(r_antenna, r_grid, power=DEFAULT_POWER):
    """compute coverage of grid by single antenna
    assumes coverage is power/distance^2

    array of distances squared from each antenna
    uses numpy broadcasting
    cast (N, 2) - (2, NX, NY) to (N, 2, 1, 1) - (1, 2, NX, NY)
    both are then broadcasted to (N, 2, NX, NY)
    """
    distance_squared = ((r_antenna[..., np.newaxis, np.newaxis] -\
                         r_grid[np.newaxis, ...])**2).sum(axis=1)

    # TODO: decide if we want to go for 1/r^2 antenna coverage
    # result = (power*ANTENNA_RADIUS**2/distance_squared).sum(axis=0)
    # # cover case where antenna is located in grid point
    # result[np.isinf(result)] = 0 # TODO: find better solution

    # binary coverage case
    result = (distance_squared < power**2) # is grid entry covered by any
    result = result.sum(axis=0) > 0        # logical or
    result = result > 0

    return result

def plot(values, antenna_locations):
    """
    plot grid values (coverage (weighted optionally) and antenna locations)
    """
    fig, axis = plt.subplots()
    x_a, y_a = antenna_locations.T
    axis.plot(x_a, y_a, "k*", label="Antennae locations")

    contours = axis.contour(X, Y, values, 100, cmap='viridis', label="Coverage")
    colors = axis.contourf(X, Y, values, 100, cmap='viridis')
    fig.colorbar(colors)


    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.legend(loc='best')

    return fig

def plot_population(r_antennae_population, generation_number):
    """
    plot grid values (coverage (weighted optionally) and antenna locations)
    """
    fig, axis = plt.subplots()
    for i, antenna_locations in enumerate(r_antennae_population):
        x_a, y_a = antenna_locations.T
        axis.plot(x_a, y_a, "*", label="#{}".format(i), ms=10)

    values = antenna_coverage_population(r_antennae_population, r_grid=R).sum(axis=0)
    import ipdb; ipdb.set_trace()
    contours = axis.contour(X, Y, WEIGHTS, 100, cmap='viridis', label="Coverage")
    colors = axis.contourf(X, Y, values, 100, cmap='viridis')
    fig.colorbar(colors)

    axis.set_title("Generation {}".format(generation_number))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_xlim(XMIN, XMAX)
    axis.set_ylim(YMIN, YMAX)
    # axis.legend(loc='best')

    return fig



def utility_function(coverage_population, weights = WEIGHTS):
    """returns total coverage as fraction of grid size
    for use in the following genetic operators
    this way we're optimizing a bounded function (values from 0 to 1)"""
    return (weights*coverage_population).sum(axis=(1,2))/NX/NY

temp_array = np.empty((N_POPULATION, N_ANTENNAE, 2))
def selection(r_antennae_population, tmp_array = temp_array):
    coverage_population = antenna_coverage_population(r_antennae_population, R)
    utility_function_values = utility_function(coverage_population)
    utility_function_total = utility_function_values.sum()
    utility_function_normalized = utility_function_values / utility_function_total
    dystrybuanta = utility_function_normalized.cumsum()
    random_x = np.random.random(N_POPULATION).reshape(1, N_POPULATION)

    new_r_antennae_population = tmp_array
    # for i, x in enumerate(random_x.T):
    #     indeks = (x > dystrybuanta).sum()
    #     print(indeks)
    #     new_r_antennae_population[i] = r_antennae_population[indeks]
    new_r_antennae_population[...] = r_antennae_population[(random_x >\
        dystrybuanta.reshape(N_POPULATION, 1)).sum(axis=0)]

    r_antennae_population[...] = new_r_antennae_population[...]


def crossover_vector(r_antennae_population, probability_crossover = P_CROSSOVER):
    """
    can't take average here because average is symmetric
    a, b -> c, c
    but who can say c is better? and now it fills the population

    instead I'm doing
    a, b -> a/3 + 2b/3, b/3 + 2/a3
    and because a, b are vectors in (0,1)^2, it's going to stay in there
    """

    for i in range(0, N_POPULATION, 2):
        if i + 1 < N_POPULATION and np.random.random() < probability_crossover:
            a, b = r_antennae_population[i], r_antennae_population[i+1]
            if DEBUG_MESSAGES:
                print("Exchanging these two", a, b, sep="\n")
            aprime = a/3 + 2*b/3
            bprime = b/3 + 2*a/3
            r_antennae_population[i] = aprime
            r_antennae_population[i] = bprime

def crossover_cutoff(r_antennae_population, probability_crossover = P_CROSSOVER, tmp_array = temp_array):
    """
    -    MAYBE instead, take two populations
-    xy  XY
-    xy  XY
-    xy  XY
-    xy  XY
-    xy  XY
-
-    select a random cutoff and swap:
-    xy  XY
-    xy  XY
-    XY  xy
-    XY  xy
-    XY  xy
-
-    this is asymmetric and awesome
    """
    tmp_array[...] = r_antennae_population[...]
    for i in range(0, N_POPULATION, 2):
        if i + 1 < N_POPULATION and np.random.random() < probability_crossover:
            cutoff = np.random.randint(0, N_ANTENNAE)
            a = r_antennae_population[i+1]
            b = r_antennae_population[i]
            if DEBUG_MESSAGES:
                print("Exchanging these two at k = {}".format(cutoff), a, b, sep="\n")
            tmp_array[i, cutoff:] = a[cutoff:]
            tmp_array[i+1, cutoff:] = b[cutoff:]
            if DEBUG_MESSAGES:
                print("They are now", tmp_array[i], tmp_array[i+1], sep="\n")
    r_antennae_population[...] = tmp_array[...]


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
    which_to_move = (np.random.random((N_POPULATION, N_ANTENNAE)) < p_mutation)
    how_much_to_move = np.random.normal(scale=gaussian_std, size=(N_POPULATION, N_ANTENNAE, 2))
    if DEBUG_MESSAGES:
        print(which_to_move)
    r_antennae_population += which_to_move[..., np.newaxis] * how_much_to_move
    r_antennae_population[:, :, 0] %= XMAX        # does this need xmin somehow?
    r_antennae_population[:, :, 1] %= YMAX        # likewise?


def main_loop(N_generations):
    """
    TODO: czym właściwie jest nasza generacja?
    * zbiorem N anten (macierz Nx2 floatów)?
    * chyba to - M zestawów po N anten (macierz MxNx2), i każdy pojedynczy
    reprezentant to macierz Nx2?
    * jeśli to drugie to będę pewnie musiał przerobić coverage() żeby to się
    sensownie wektorowo liczyło

    na razie zróbmy wolno i na forach :)
    """

    # init antenna locations
    r_antennae_population = np.random.random((N_POPULATION, N_ANTENNAE, 2))
    # cov = antenna_coverage(r_antennae_population[0], R)
    # plot(cov, r_antennae_population[0])
    # plt.show()
    print("Generation {}/{}, {:.0f}% done".format(0, N_generations, 0))
    # r_antennae_population[:, :, 0] = (XMAX - XMIN)*r_antennae_population[:,:,0] + XMIN
    # r_antennae_population[:, :, 1] = (YMAX - YMIN)*r_antennae_population[:,:,1] + YMIN
    for n in range(N_generations): #ew. inny warunek, np. mała różnica kolejnych wartości
        print("\rGeneration {}/{}, {:.0f}% done".format(n, N_generations, n/N_generations*100),end='')
        fig = plot_population(r_antennae_population, n)
        fig.savefig("{}.png".format(n))
        plt.close(fig)

        #nonessential
        coverage_population = antenna_coverage_population(r_antennae_population, R)
        if DEBUG_MESSAGES:
            print(utility_function(coverage_population))

        if DEBUG_MESSAGES:
            print(r_antennae_population)
            print("Before selection")
        selection(r_antennae_population)
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
    print("Job's finished!")
    plot_population(r_antennae_population, N_generations).savefig("{}.png".format(N_GENERATIONS))
    # jakoś wybrać maksymalny zestaw
    # printnąć położenia
    # plotnąć jaki jest wspaniały

if __name__=="__main__":
    main_loop(N_GENERATIONS)
