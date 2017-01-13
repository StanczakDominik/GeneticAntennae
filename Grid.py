import numpy as np
import scipy.spatial


class Grid():
    """spatial grid with human population on it"""

    def __init__(self, nx=100, ny=100, xmax=1, ymax=1):
        self.NX = nx
        self.NY = ny
        self.xmax = xmax
        self.ymax = ymax
        self.x = np.linspace(0, xmax, nx)
        self.y = np.linspace(0, ymax, ny)
        self.x, self.y = np.meshgrid(self.x, self.y)
        self.grid = np.stack((self.x, self.y), axis=0)
        self.weights = np.ones_like(self.x)
        self.points_for_tree = list(zip(self.x.ravel(), self.y.ravel()))
        self.tree = scipy.spatial.cKDTree(self.points_for_tree)

    def query(self, population, locations):
        return self.tree.query(locations, population.number_expected_neighbors,
                               distance_upper_bound=population.DEFAULT_POWER)

    def set_radial_exp_weights(self, exponent):
        distances = ((self.grid - np.array([self.NX / 2, self.NY / 2], ndmin=3).T) ** 2).sum(axis=0)
        self.weights = np.exp(-exponent * distances)

    def renormalize_weights(self):
        self.weights /= self.weights.sum()

    def antenna_coverage_population(self, population, i=-1):
        if i > -1:
            dataset = population.position_history[i]
        else:
            dataset = population.r_antennae_population

        result_array = np.empty((population.NPOPULATION, self.NX, self.NY), dtype=bool)
        for j, population_member in enumerate(dataset):
            self.antenna_coverage_tree(population, population_member)
            antena_coverage_value = antenna_coverage(population_member, self.grid, population.DEFAULT_POWER)
            result_array[j] = antena_coverage_value
        return result_array

    def utility_function(self, coverage_population):
        """returns total coverage as fraction of grid size
        for use in the following genetic operators
        this way we're optimizing a bounded function (values from 0 to 1)"""
        return (self.weights.reshape(1, self.NX, self.NY) * coverage_population).sum(axis=(1, 2)) / self.NX / self.NY

    def antenna_coverage_tree(self, population, antenna_set):
        nb_distances, nb_indices = self.query(population, antenna_set)
        picked_neighbors = self.points_for_tree[nb_indices]
        print(7)

def antenna_coverage(r_antenna, r_grid, zasieg):
    """
    compute coverage of grid by single antenna
    assumes coverage is zasieg/distance^2

    array of distances squared from each antenna
    uses numpy broadcasting
    cast (N, 2) - (2, NX, NY) to (N, 2, 1, 1) - (1, 2, NX, NY)
    both are then broadcasted to (N, 2, NX, NY)
    """
    distance_squared = ((r_antenna[..., np.newaxis, np.newaxis] -
                         r_grid[np.newaxis, ...]) ** 2).sum(axis=1)

    # TODO: decide if we want to go for 1/r^2 antenna coverage
    # result = (zasieg*ANTENNA_RADIUS**2/distance_squared).sum(axis=0)
    # # cover case where antenna is located in grid point
    # result[np.isinf(result)] = 0 # TODO: find better solution

    # binary coverage case
    result = (distance_squared < zasieg ** 2)  # is grid entry covered by any
    result = result.sum(axis=0) > 0  # logical or
    result = result > 0

    # TODO: ujemne wagi przez maksymalną możliwą

    return result
