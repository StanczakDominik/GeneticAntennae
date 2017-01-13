import numpy as np
from matplotlib import pyplot as plt

from calculations import antenna_coverage_population, antenna_coverage, utility_function


def plot_fitness(mean_fitness_history, max_fitness_history, std_fitness_history, filename=False, show=True, save=True):
    if show or save:
        fig, ax = plt.subplots()
        ax.plot(mean_fitness_history, "o-", label="Average fitness")
        ax.plot(max_fitness_history, "o-", label="Max fitness")
        ax.fill_between(np.arange(std_fitness_history.size),
                        mean_fitness_history+std_fitness_history,
                        mean_fitness_history-std_fitness_history,
                        alpha=0.5,
                        facecolor='orange',
                        label = "1 std")
        ax.set_xlabel("Generation #")
        ax.set_ylabel("Fitness")
        ax.set_ylim(0,1)
        ax.legend()
        ax.grid()
        if filename and save:
            fig.savefig(filename)
        if show:
            return fig
        else:
            plt.close(fig)


def plot_population(r_antennae_population, generation_number, r_grid, zasieg, weights, filename=None, show=True, save=True):
    """
    plot grid values (coverage (weighted optionally) and antenna locations)
    """
    if show or save:
        X, Y = r_grid
        fig, axis = plt.subplots()
        values = antenna_coverage_population(r_antennae_population, r_grid, zasieg)
        utility_function_values = utility_function(values, weights)
        best_candidate = np.argmax(utility_function_values)

        for i, antenna_locations in enumerate(r_antennae_population):
            x_a, y_a = antenna_locations.T
            marker_size = 10
            alpha = 0.6
            if i == best_candidate:
                marker_size *= 2
                alpha = 1
            axis.plot(x_a, y_a, "*", label="#{}".format(i), ms=marker_size, alpha=alpha)

        axis.contourf(X, Y, weights, 100, cmap='viridis', label="Coverage")
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
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        # axis.legend(loc='best')
        if save and filename:
            fig.savefig(filename)
        if show:
            return fig
        else:
            plt.close(fig)


def plot_single(r_antennae, generation_number, r_grid, zasieg, weights, filename=None, show=True, save=True):
    """
    plot grid values (coverage (weighted optionally) and antenna locations)
    """
    if show or save:
        X, Y = r_grid
        NX, NY = X.shape
        fig, axis = plt.subplots()
        values = antenna_coverage(r_antennae, r_grid, zasieg)
        utility_function_value = (weights*values).sum()/NX/NY

        x_a, y_a = r_antennae.T
        marker_size = 20
        alpha = 1
        axis.plot(x_a, y_a, "*", ms=marker_size, alpha=alpha)

        axis.contourf(X, Y, weights, 100, cmap='viridis', label="Coverage")
        configurations = axis.contourf(X, Y, values, 100, cmap='viridis', alpha=0.5)
        fig.colorbar(configurations)

        axis.set_title(r"Generation {}, optimal candidate, f {:.2f}".format(
            generation_number,
            utility_function_value,
            ))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        # axis.legend(loc='best')
        if filename and save:
            fig.savefig(filename)
        if show:
            return fig
        else:
            plt.close(fig)

