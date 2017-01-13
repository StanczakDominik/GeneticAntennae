import matplotlib.pyplot as plt

from Grid import Grid
from Population import Population

if __name__ == '__main__':
    grid = Grid()
    pop = Population(grid)
    for n in range(pop.n_generations):
        pop.generation_cycle()
    pop.plot_fitness()
    pop.plot_population(5)
    plt.show()
