import matplotlib.pyplot as plt

from Grid import Grid
from Population import Population

if __name__ == '__main__':
    grid = Grid()
    pop = Population(grid)
    for n in range(pop.n_generations):
        pop.generation_cycle()
    pop.plot_fitness()
    for i in [0, 5, 10, -1]:
        pop.plot_population(i)
    plt.show()
