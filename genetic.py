"""genetic algorithm optimization of radio antenna coverage"""
import numpy as np
import matplotlib.pyplot as plt

# grid dimensions
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

N_ANTENNAE = 3
np.random.seed(0)
antenna_r = np.random.random((N_ANTENNAE, 2))

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

    return result.astype(float)

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

coverage = antenna_coverage(antenna_r, R)

plot(coverage, antenna_r)
plt.show()

# population as weights, for now let's focus on the uniform population case
DISTANCES = ((R - np.array([(XMAX-XMIN)/2, (YMAX-YMIN)/2], ndmin=3).T)**2).sum(axis=0)
population = np.exp(-DISTANCES*10)

plot(coverage*population, antenna_r)
plt.title("Weighted by population")
plt.show()

def utility_function(coverage):
    """returns total coverage as fraction of grid size
    for use in the following genetic operators
    this way we're optimizing a bounded function (values from 0 to 1)"""
    return coverage.sum()/NX/NY

def selection(r_antennae):
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

def crossover(r_antennae):
    """
    https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)

    how do we do this?
    * take average? introduces giant changes

    na razie wiem tyle że bardziej mi się podoba nazwa recombination
    jest bardziej plazmowa
    """

def mutation(r_antenna, gaussian_std = 0.01):
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
    r_antenna += np.random.normal(loc=r_antenna, scale=gaussian_std)
    r_antenna[:, 0] %= XMAX        # does this need xmin somehow?
    r_antenna[:, 1] %= YMAX        # likewise?


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
