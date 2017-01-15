import numpy as np
import pandas as pd
import scipy.spatial


class GeoGrid():
    """spatial grid with human population on it"""

    def __init__(self):
        df = pd.read_csv('fixed_data.csv', index_col=0)
        self.N = df['N'].values
        self.E = df['E'].values
        self.populations = df['populations'].values
        self.countries = df['countries'].values
        self.populations[np.logical_not(self.countries == "PL")] = 0#-self.populations[np.logical_not(self.countries == "PL")]
        self.populations = self.populations / np.abs(self.populations).sum()
        # self.populations = self.populations / 38000
        self.points_for_tree = list(zip(self.E, self.N))
        self.tree = scipy.spatial.cKDTree(self.points_for_tree)

    def query(self, population, antenna_set):
        return self.tree.query(antenna_set,
                               population.number_expected_neighbors)

    def utility_function(self, population, i = -1):
        if i == -1:
            dataset = population.r_antennae_population
        else:
            dataset = population.position_history[i]

        utility = np.zeros(population.NPOPULATION)
        for j, antenna_set in enumerate(dataset):
            distances, indices = self.query(population, antenna_set)
            # print((distances < population.DEFAULT_POWER).sum())
            # print(indices.size, distances.max(), distances.min(), distances.std(), distances.mean())
            covered_antennae = np.unique(indices[distances < population.DEFAULT_POWER])
            total_population_reached = np.sum(self.populations[covered_antennae])
            utility[j] = total_population_reached
        utility[utility<=0] = 0
        return utility

    def utility_function_general(self, population, dataset):
        utility = np.zeros(population.NPOPULATION)
        for j, antenna_set in enumerate(dataset):
            distances, indices = self.query(population, antenna_set)
            covered_antennae = np.unique(indices[distances < population.DEFAULT_POWER])
            total_population_reached = np.sum(self.populations[covered_antennae])
            utility[j] = total_population_reached
        utility[utility <= 0] = 0
        return utility


if __name__ == '__main__':
    from Population import Population
    import matplotlib.pyplot as plt
    g = GeoGrid()
    pop = Population(g)
    # map = Basemap(projection='cyl', llcrnrlon=12, urcrnrlon=25, llcrnrlat=48, urcrnrlat=56, resolution='l')
    # map.drawcoastlines()
    # map.fillcontinents()
    # map.drawcountries()
    # map.drawparallels(np.arange(48, 56, 1))
    polish_indices = g.countries == "PL"
    plt.scatter(g.E[polish_indices], g.N[polish_indices], g.populations[polish_indices], color="blue")
    x, y = pop.r_antennae_population.T
    plt.scatter(x, y, color="red")
    plt.show()
