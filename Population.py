import matplotlib.pyplot as plt
import numpy as np

from patches import circles


class Population():
    """population of sets of antennae"""

    def __init__(self, grid,
                n_pop=50,
                n_antennae=10,
                default_power=0.3,
                p_cross=0.5, p_mutation=1,
                std_mutation=0.3,
                n_generations=50):
        self.grid = grid
        # TODO: rename these
        self.NPOPULATION = n_pop
        self.NANTENNAE = n_antennae
        self.DEFAULT_POWER = default_power

        self.P_CROSSOVER = p_cross
        self.P_MUTATION = p_mutation
        self.MUTATION_STD = std_mutation

        self.number_expected_neighbors = int(((111*default_power)**2 * np.pi))


        self.r_antennae_population = np.ones((self.NPOPULATION, self.NANTENNAE, 2)) * \
                                     np.array([[[(42+56)/2, (29+35)/2]]])
        self.mutation_std_array = np.ones(self.NPOPULATION) * std_mutation
        # TODO: set initial position as parameter of population
        self.utility_values = np.zeros((self.NPOPULATION))

        self.TEMP_ARRAY = np.zeros_like(self.r_antennae_population)

        self.n_generations = n_generations
        self.utility_values_history = np.zeros((self.n_generations, self.NPOPULATION))
        self.mutation_std_history = np.zeros((self.n_generations, self.NPOPULATION))
        self.max_fitness_history = np.zeros(self.n_generations)
        self.mean_fitness_history = np.zeros(self.n_generations)
        self.std_fitness_history = np.zeros(self.n_generations)
        self.crossovers_history = np.zeros(self.n_generations)
        self.indices_to_swap_history = np.zeros((self.n_generations, self.NPOPULATION))
        self.position_history = np.zeros(((self.n_generations, self.NPOPULATION, self.NANTENNAE, 2)))
        self.iteration = 0
        print("Generation {}/{}, {:.0f}% done".format(0, self.n_generations, 0),end='')

    def calculate_utility(self, plotting=False, i=-1):
        coverage_population_values = self.grid.antenna_coverage_population(self)
        utility_function_values = self.grid.utility_function(coverage_population_values)
        if plotting:
            return utility_function_values, coverage_population_values
        else:
            return utility_function_values
    """ genetic operators """

    def selection(self):
        utility_function_values = self.grid.utility_function(self)
        utility_function_total = utility_function_values.sum()

        utility_function_normalized = utility_function_values / utility_function_total
        dystrybuanta = utility_function_normalized.cumsum()
        random_x = np.random.random(self.NPOPULATION).reshape(1, self.NPOPULATION)

        new_r_antennae_population = self.TEMP_ARRAY
        # for i, x in enumerate(random_x.T):
        #     indeks = (x > dystrybuanta).sum()
        #     print(indeks)
        #     new_r_antennae_population[i] = r_antennae_population[indeks]
        selected_targets = (random_x > dystrybuanta.reshape(self.NPOPULATION, 1)).sum(axis=0)
        # print(selected_targets, utility_function_values, sep='\n')
        # print(utility_function_values.max(), utility_function_values.mean(), utility_function_values.std())

        new_r_antennae_population[...] = self.r_antennae_population[selected_targets]
        self.r_antennae_population[...] = new_r_antennae_population[...]
        self.utility_values = utility_function_values[selected_targets]

        self.max_fitness_history[self.iteration] = utility_function_values.max()
        self.mean_fitness_history[self.iteration] = utility_function_values.mean()
        self.std_fitness_history[self.iteration] = utility_function_values.std()

    def crossover_cutoff(self):
        self.TEMP_ARRAY[...] = self.r_antennae_population[...]
        number_crossovers_occurred = 0
        for i in range(0, self.NPOPULATION, 2):
            if i + 1 < self.NPOPULATION and np.random.random() < self.P_CROSSOVER:
                number_crossovers_occurred += 1
                cutoff = np.random.randint(0, self.NANTENNAE)
                a = self.r_antennae_population[i + 1]
                b = self.r_antennae_population[i]
                self.TEMP_ARRAY[i, cutoff:] = a[cutoff:]
                self.TEMP_ARRAY[i + 1, cutoff:] = b[cutoff:]
        self.r_antennae_population[...] = self.TEMP_ARRAY[...]
        self.crossovers_history[self.iteration] = number_crossovers_occurred

    def mutation(self):
        # TODO: implement 1/5 success rule or some other way
        self.mutation_std_history[self.iteration] = self.mutation_std_array
        which_to_move = (np.random.random((self.NPOPULATION, self.NANTENNAE)) < self.P_MUTATION)
        how_much_to_move = np.random.normal(scale=self.mutation_std_array[:, np.newaxis, np.newaxis],
                                            size=(self.NPOPULATION, self.NANTENNAE, 2))
        self.TEMP_ARRAY = self.r_antennae_population + which_to_move[..., np.newaxis] * how_much_to_move
        utility_function_values = self.grid.utility_function_general(self, self.TEMP_ARRAY)

        # print(self.TEMP_ARRAY - self.r_antennae_population)
        utility_change = utility_function_values - self.utility_values
        indices_to_swap = utility_change > 0
        self.indices_to_swap_history[self.iteration] = indices_to_swap
        self.r_antennae_population[indices_to_swap] = self.TEMP_ARRAY[indices_to_swap]
        # print(utility_function_values - self.utility_values)

    def mutation_onefifth(self, run_every_n_mutations=5, c_decrease=0.82, c_increase=1.22):
        if self.iteration > run_every_n_mutations and self.iteration % run_every_n_mutations == 0:
            running_sum = self.indices_to_swap_history[
                          self.iteration - run_every_n_mutations:self.iteration + 1].cumsum(axis=0)
            percentage_successful = (running_sum[-1] - running_sum[0]) / run_every_n_mutations
            increase_these = percentage_successful > 0.2
            self.mutation_std_array[increase_these] *= c_increase
            self.mutation_std_array[np.logical_not(increase_these)] *= c_decrease
            print(f"Increased {increase_these.sum()}/{self.NPOPULATION} standard deviations.")

    def generation_cycle(self):
        self.position_history[self.iteration] = self.r_antennae_population
        self.selection()
        # self.crossover_cutoff()
        self.mutation()
        self.mutation_onefifth()
        print(f"\rGeneration {self.iteration}/{self.n_generations}, {self.iteration/self.n_generations*100:.0f}% done, , fitness is {self.mean_fitness_history[self.iteration]:.2f} +- {self.std_fitness_history[self.iteration]:.2f}",end='')
        self.iteration += 1

    """ plotting routines"""

    def plot_fitness(self, savefilename=False, show=True,
                     save=True):
        fig, ax = plt.subplots()
        ax.plot(self.mean_fitness_history, "o-", label="Average fitness")
        ax.plot(self.max_fitness_history, "o-", label="Max fitness")
        ax.fill_between(np.arange(self.n_generations),
                        self.mean_fitness_history + self.std_fitness_history,
                        self.mean_fitness_history - self.std_fitness_history,
                        alpha=0.5,
                        facecolor='orange',
                        label="1 std")
        ax.set_xlabel("Generation #")
        ax.set_ylabel("Fitness")
        # ax.set_ylim(0, 1)
        ax.legend()
        ax.grid()
        if savefilename:
            fig.savefig("data/" + savefilename + ".png")
        if show:
            return fig
        plt.close(fig)

    def plot_population(self, generation_number=-1, only_winner=False, savefilename=None, show=True):
        """
        plot grid values (coverage (weighted optionally) and antenna locations)
        """
        # TODO: only_winner implementation

        if generation_number == -1:
            generation_number = self.n_generations - 1

        utility_function_values = self.grid.utility_function(self)
        best_candidate = np.argmax(utility_function_values)

        fig, axis = plt.subplots()

        polish_indices = self.grid.countries == "PL"
        axis.scatter(self.grid.E[polish_indices],
            self.grid.N[polish_indices],
            self.grid.populations[polish_indices]*1000,
         color="red")

        for i, antenna_locations in enumerate(self.position_history[generation_number]):
            x_a, y_a = antenna_locations.T
            marker_size = 10
            alpha = 0.2
            if i == best_candidate:
                marker_size *= 2
                alpha = 1
                axis.plot(x_a, y_a, "*", label="#{}".format(i), ms=marker_size, alpha=alpha)
            circles(x_a, y_a, self.DEFAULT_POWER, alpha=alpha/4)

        axis.set_title(r"Generation {}, $<f>$ {:.2f} $\pm$ {:.2f}, max {:.2f}".format(
            generation_number,
            self.mean_fitness_history[generation_number],
            self.std_fitness_history[generation_number],
            self.max_fitness_history[generation_number],
        ))
        axis.set_xlabel("Longitude [deg]")
        axis.set_ylabel("Latitude [deg]")
        axis.legend(loc='best')
        if savefilename:
            fig.savefig("data/" + str(generation_number) + savefilename + ".png")
        if show:
            return fig
        plt.close(fig)


if __name__ == '__main__':
    from GeoData import GeoGrid

    g = GeoGrid()
    pop = Population(g)
    pop.generation_cycle()
