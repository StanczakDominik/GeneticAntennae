import numpy as np
import pandas as pd
import scipy.spatial

class GeoGrid():
    """GEOSTAT population data"""

    def __init__(self, country_code):
        df = pd.read_csv('fixed_data.csv', index_col=0)
        self.country_code = country_code
        self.N = df['N'].values
        self.E = df['E'].values
        self.populations = df['populations'].values
        self.countries = df['countries'].values
        self.populations[np.logical_not(self.countries == country_code)] = 0
        self.populations = self.populations / np.abs(self.populations).sum()
        self.points_for_tree = list(zip(self.E, self.N))
        self.tree = scipy.spatial.cKDTree(self.points_for_tree)

    def query(self, population, antenna_set):
        return self.tree.query(antenna_set,
                               population.number_expected_neighbors)

    def utility_function_general(self, population, dataset):
        npopulation, nantennae, dim2 = dataset.shape
        utility = np.zeros((npopulation, nantennae))
        for j, antenna_set in enumerate(dataset):
            distances, indices = self.query(population, antenna_set)
            cleaned_indices = distances < population.DEFAULT_POWER
            occurences = np.bincount(indices[cleaned_indices], minlength=self.N.size).astype(float)
            occurences[occurences == 0] = np.inf

            for k, antenna_location in enumerate(antenna_set):
                covered_here = indices[k]
                cleaned_indices_here = cleaned_indices[k]
                populations_here = self.populations[covered_here]
                occurences_here = occurences[covered_here]
                reached_normed_population = np.sum(cleaned_indices_here * populations_here / occurences_here)
                utility[j, k] = reached_normed_population

        utility[utility <= 0] = 0
        return utility