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
NX = 10
NY = 10

# prepare 2D grid
x, DX = np.linspace(XMIN, XMAX, NX, retstep=True, endpoint=False)
y, DY = np.linspace(YMIN, YMAX, NY, retstep=True, endpoint=False)
X, Y = np.meshgrid(x, y)
R = np.stack((X, Y), axis=0)

N_POPULATION = 4
N_ANTENNAE = 3
np.random.seed(0)
r_antennae_population = np.random.random((N_POPULATION, N_ANTENNAE, 2))
r_antennae_population[:, :, 0] = (XMAX - XMIN)*r_antennae_population[:,:,0] + XMIN
r_antennae_population[:, :, 1] = (YMAX - YMIN)*r_antennae_population[:,:,1] + YMIN

def antenna_coverage_population(R_antenna, r_grid, power=0.1):
    result_array = np.empty((N_POPULATION, NX, NY), dtype=bool)
    for i, population_member in enumerate(R_antenna):
        result_array[i] = antenna_coverage(population_member, r_grid, power)
    return result_array

def antenna_coverage(r_antenna, r_grid, power=0.1):
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

coverage_population = antenna_coverage_population(r_antennae_population, R)
# for i in range(N_POPULATION):
#     plot(coverage_population[i], antenna_r[i])
#     plt.show()

# # population as weights, for now let's focus on the uniform population case
# DISTANCES = ((R - np.array([(XMAX-XMIN)/2, (YMAX-YMIN)/2], ndmin=3).T)**2).sum(axis=0)
# population = np.exp(-DISTANCES*10)
#
# plot(coverage*population, antenna_r)
# plt.title("Weighted by population")
# plt.show()

def utility_function(coverage_population):
    """returns total coverage as fraction of grid size
    for use in the following genetic operators
    this way we're optimizing a bounded function (values from 0 to 1)"""
    return coverage_population.sum(axis=(1,2))/NX/NY

print(utility_function(coverage_population))
temp_array = np.empty((N_POPULATION, N_ANTENNAE, 2))
def selection(r_antennae_population, tmp_array = temp_array):
    """
    https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)
    1. The fitness function is evaluated for each individual, providing fitness
    values, which are then normalized. Normalization means dividing the fitness
    value of each individual by the sum of all fitness values, so that the sum
    of all resulting fitness values equals 1.

    2. The population is sorted by descending fitness values.

    3. Accumulated normalized fitness values are computed (the accumulated
    fitness value of an individual is the sum of its own fitness value plus the
    fitness values of all the previous individuals). The accumulated fitness of
    the last individual should be 1 (otherwise something went wrong in the
    normalization step).

    4. A random number R between 0 and 1 is chosen.

    5. The selected individual is the first one whose accumulated normalized
    value is greater than R.
    """
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
print("Before selection")
print(r_antennae_population)
selection(r_antennae_population)
print("After selection")
print(r_antennae_population)

def crossover_vector(r_antennae_population, probability_crossover = 0.5):
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
            print("Exchanging these two", a, b, sep="\n")
            aprime = a/3 + 2*b/3
            bprime = b/3 + 2*a/3
            r_antennae_population[i] = aprime
            r_antennae_population[i] = bprime

def crossover_cutoff(r_antennae_population, probability_crossover = 0.5, tmp_array = temp_array):
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
            print("Exchanging these two at k = {}".format(cutoff), a, b, sep="\n")
            tmp_array[i, cutoff:] = a[cutoff:]
            tmp_array[i+1, cutoff:] = b[cutoff:]
            print("They are now", tmp_array[i], tmp_array[i+1], sep="\n")
    r_antennae_population[...] = tmp_array[...]

crossover_cutoff(r_antennae_population)
print("After crossover")
print(r_antennae_population)

def mutation(r_antennae_population, gaussian_std = 0.01, p_mutation=0.4):
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
    print(which_to_move)
    r_antennae_population += which_to_move[..., np.newaxis] * how_much_to_move
    r_antennae_population[:, :, 0] %= XMAX        # does this need xmin somehow?
    r_antennae_population[:, :, 1] %= YMAX        # likewise?

mutation(r_antennae_population)
print("After mutation")
print(r_antennae_population)

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
    r_antennae = np.random.random((N_ANTENNAE, 2))
    for n in N_generations: #ew. inny warunek, np. mała różnica kolejnych wartości
        selection(r_antennae)

        #  losowo: liczba losowa dla każdej anteny oddzielnie?
        mutation(r_antenna)
        crossover(r_antennae)

    # jakoś wybrać maksymalny zestaw
    # printnąć położenia
    # plotnąć jaki jest wspaniały
