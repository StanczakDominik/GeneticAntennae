import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from GeoData import GeoGrid
from patches import circles


class Population():
    """population of sets of antennae"""

    def __init__(self, grid,
                 n_pop,
                 n_trial,
                 n_antennae,
                 default_power,
                 std_mutation,
                 n_generations,
                 initial_E,
                 initial_N,
                 ):
        self.grid = grid
        self.NPOPULATION = n_pop
        self.TRIAL_POPULATION = n_trial
        self.NANTENNAE = n_antennae
        self.DEFAULT_POWER = default_power
        self.MUTATION_STD = std_mutation
        self.number_expected_neighbors = int(((111*default_power)**2 * np.pi))

        self.initial_E = initial_E
        self.initial_N = initial_N
        self.r_antennae_population = np.ones((self.NPOPULATION, self.NANTENNAE, 2)) * \
                                     np.array([[[(initial_E), (initial_N)]]])
        self.mutation_std_array = np.ones((self.NPOPULATION, self.NANTENNAE)) * std_mutation
        # TODO: set initial position as parameter of population
        self.utility_values = self.grid.utility_function_general(self, dataset=self.r_antennae_population)

        self.TEMP_ARRAY = np.zeros_like(self.r_antennae_population)

        self.n_generations = n_generations
        self.utility_values_history = np.zeros((self.n_generations, self.NPOPULATION, self.NANTENNAE))
        self.mutation_std_history = np.zeros((self.n_generations, self.NPOPULATION, self.NANTENNAE))
        self.max_fitness_history = np.zeros(self.n_generations)
        self.mean_fitness_history = np.zeros(self.n_generations)
        self.std_fitness_history = np.zeros(self.n_generations)
        self.crossovers_history = np.zeros(self.n_generations)
        self.indices_to_swap_history = np.zeros((self.n_generations, self.NPOPULATION, self.NANTENNAE))
        self.position_history = np.zeros(((self.n_generations, self.NPOPULATION, self.NANTENNAE, 2)))
        self.iteration = 0

    def save(self, filename):
        with h5py.File(f"data/{filename}.hdf5", "w") as f:
            f.create_dataset("utility_values_history", data=self.utility_values_history)
            f.create_dataset("mutation_std_history", data=self.mutation_std_history)
            f.create_dataset("max_fitness_history", data=self.max_fitness_history)
            f.create_dataset("mean_fitness_history", data=self.mean_fitness_history)
            f.create_dataset("std_fitness_history", data=self.std_fitness_history)
            f.create_dataset("position_history", data=self.position_history)
            f.attrs["n_pop"] = self.NPOPULATION
            f.attrs["n_trial"] = self.TRIAL_POPULATION
            f.attrs["n_antennae"] = self.NANTENNAE
            f.attrs["default_power"] = self.DEFAULT_POWER
            f.attrs["n_generations"] = self.n_generations
            f.attrs["country_code"] = self.grid.country_code
            f.attrs["std_mutation"] = self.MUTATION_STD
            f.attrs["initial_E"] = self.initial_E
            f.attrs["initial_N"] = self.initial_N


    """ genetic operators """

    def selection_mu_plus_lambda(self):
        random_indices = np.random.randint(0, self.NPOPULATION, size=self.TRIAL_POPULATION)
        random_children = self.r_antennae_population[random_indices]
        random_stds = self.mutation_std_array[random_indices]
        random_utilities = self.utility_values[random_indices]

        random_children, random_swapped_indices = self.mutation_mulambda(random_children, random_utilities, random_stds)

        total_pop = np.concatenate((self.r_antennae_population, random_children), axis=0)
        total_stds = np.concatenate((self.mutation_std_array, random_stds), axis=0)
        total_indices = np.concatenate((np.zeros((self.NPOPULATION, self.NANTENNAE)), random_swapped_indices), axis=0)

        utility_function_values = self.grid.utility_function_general(self, dataset=total_pop)
        summed_utilities = utility_function_values.sum(axis=1)
        selected_targets = np.argsort(summed_utilities)[-self.NPOPULATION:]

        self.r_antennae_population[...] = total_pop[selected_targets]
        self.mutation_std_array[...] = total_stds[selected_targets]
        self.utility_values = utility_function_values[selected_targets]
        self.indices_to_swap_history[self.iteration] = total_indices[selected_targets]

        self.max_fitness_history[self.iteration] = summed_utilities.max()
        self.mean_fitness_history[self.iteration] = summed_utilities.mean()
        self.std_fitness_history[self.iteration] = summed_utilities.std()

    def mutation_mulambda(self, dataset, utilities, stds):
        # TODO: implement 1/5 success rule or some other way
        npopulation, nantennae, ndimensions = dataset.shape
        utilities = self.grid.utility_function_general(self, dataset)
        self.mutation_std_history[self.iteration] = self.mutation_std_array
        how_much_to_move = np.random.normal(scale=stds[..., np.newaxis],
                                            size=(npopulation, nantennae, 2))
        TEMP = dataset + how_much_to_move
        utility_function_values = self.grid.utility_function_general(self, TEMP)

        utility_change = utility_function_values - utilities
        indices_to_swap = utility_change > 0
        dataset[indices_to_swap] = TEMP[indices_to_swap]
        return dataset, indices_to_swap
        # print(utility_function_values - self.utility_values)

    def mutation_onefifth(self, run_every_n_mutations=5, c_decrease=0.82, c_increase=1.22):
        if self.iteration >= run_every_n_mutations and self.iteration % run_every_n_mutations == 0:
            running_sum = self.indices_to_swap_history[
                          self.iteration - run_every_n_mutations:self.iteration + 1].cumsum(axis=0)
            percentage_successful = (running_sum[-1] - running_sum[0]) / run_every_n_mutations
            increase_these = percentage_successful > 0.2
            self.mutation_std_array[increase_these] *= c_increase
            self.mutation_std_array[np.logical_not(increase_these)] *= c_decrease
            print(f"\nIncreased {increase_these.sum()}/{self.NPOPULATION * self.NANTENNAE} standard deviations.")
        self.mutation_std_history[self.iteration] = self.mutation_std_array

    def generation_cycle(self):
        self.position_history[self.iteration] = self.r_antennae_population
        self.selection_mu_plus_lambda()
        self.mutation_onefifth()
        print(
            f"\rGeneration {self.iteration}/{self.n_generations}, {self.iteration/self.n_generations*100:.0f}% done, fitness is {self.mean_fitness_history[self.iteration]:.2f} +- {self.std_fitness_history[self.iteration]:.2f}",
            end='')
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
        ax.legend(loc='best')
        ax.grid()
        if savefilename:
            fig.savefig("data/" + savefilename + ".png")
        if show:
            return fig
        plt.close(fig)

    def plot_std(self, savefilename=False, show=True,
                 save=True):
        fig, ax = plt.subplots()
        for std in self.mutation_std_history:
            mean_std = std.mean(axis=1)
            std_std = std.std(axis=1)
            ax.plot(range(self.n_generations), mean_std, "o-", label="Mutation STD")
            ax.fill_between(np.arange(self.n_generations),
                            mean_std + std_std,
                            mean_std - std_std,
                            alpha=0.5,
                            facecolor='orange',
                            label="1 std")
        ax.set_xlabel("Generation #")
        ax.set_ylabel("STD")
        # ax.set_ylim(0, 1)
        # ax.legend(loc='best')
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

        utility_function_values = self.grid.utility_function_general(self,
                                                                     self.position_history[generation_number]).sum(
            axis=1)
        best_candidate = np.argmax(utility_function_values)

        fig, axis = plt.subplots()

        polish_indices = self.grid.countries == self.grid.country_code
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
        # axis.set_xlabel("Longitude [deg]")
        # axis.set_ylabel("Latitude [deg]")
        # axis.legend(loc='best')
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        if savefilename:
            fig.savefig("data/" + savefilename + str(generation_number) + ".png")
        if show:
            return fig
        plt.close(fig)

    def plot_animation(self, savefilename="animation"):
        """
        plot grid values (coverage (weighted optionally) and antenna locations)
        """

        fig, axis = plt.subplots()

        polish_indices = self.grid.countries == self.grid.country_code
        axis.set_ylabel("Latitude [deg]")
        axis.set_xlabel("Longitude [deg]")

        axis.scatter(self.grid.E[polish_indices],
                     self.grid.N[polish_indices],
                     self.grid.populations[polish_indices] * 1000,
                     color="red")

        title = axis.set_title(r"Generation {}, $<f>$ {:.2f} $\pm$ {:.2f}, max {:.2f}".format(
            0,
            self.mean_fitness_history[0],
            self.std_fitness_history[0],
            self.max_fitness_history[0],
        ))

        marker_size = 20
        alpha = 1
        stars, = axis.plot([], [], "*", ms=marker_size, alpha=alpha)
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])

        def animate(generation_number):
            print(f"Plotting frame {generation_number}")

            utility_function_values = self.grid.utility_function_general(self,
                                                                         self.position_history[generation_number]).sum(
                axis=1)
            i = np.argmax(utility_function_values)

            x_a, y_a = self.position_history[generation_number, i].T
            stars.set_data(x_a, y_a)
            title.set_text(r"Generation {}, $<f>$ {:.2f} $\pm$ {:.2f}, max {:.2f}".format(
                generation_number,
                self.mean_fitness_history[generation_number],
                self.std_fitness_history[generation_number],
                self.max_fitness_history[generation_number],
            ))
            return [stars, title]

        anim = animation.FuncAnimation(fig, animate, frames=range(self.n_generations), interval=500)
        anim.save(f"data/{savefilename}.mp4")


def load(filename):
    with h5py.File(f"data/{filename}.hdf5", "r") as f:
        grid = GeoGrid(country_code=f.attrs['country_code'])
        pop = Population(grid,
                         n_pop=f.attrs['n_pop'],
                         n_trial=f.attrs['n_trial'],
                         n_antennae=f.attrs['n_antennae'],
                         n_generations=f.attrs['n_generations'],
                         default_power=f.attrs['default_power'],
                         initial_E=f.attrs['initial_E'],
                         initial_N=f.attrs['initial_N'],
                         std_mutation=f.attrs['std_mutation'],
                         )
        pop.utility_values_history = f["utility_values_history"][...]
        pop.mutation_std_history = f["mutation_std_history"][...]
        pop.max_fitness_history = f["max_fitness_history"][...]
        pop.mean_fitness_history = f["mean_fitness_history"][...]
        pop.std_fitness_history = f["std_fitness_history"][...]
        pop.position_history = f["position_history"][...]
    return pop
