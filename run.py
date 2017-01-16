import os

import numpy as np

from GeoData import GeoGrid
from Population import Population, load
import time

def run(E, N, country_code,
        n_pop,
        n_trial,
        n_antennae,
        n_generations,
        prefix=""):
    """

    float E
    float N
    :string country_code: to be selected from
        AL        AT        BE        BG
        CH        CZ        DE        DK
        EE        EL        ES        FI
        FR        HR        HU        IE
        IT        LI        LT        LV
        MT        NL        NO        PL
        PT        RO        SE        SI
        SK        UK        XK*

    :return
    """
    np.random.seed(0)
    grid = GeoGrid(country_code)

    pop = Population(grid,
                     n_pop=n_pop,
                     n_trial=n_trial,
                     n_antennae=n_antennae,
                     n_generations=n_generations,
                     default_power=0.3,
                     std_mutation=1,
                     initial_E=E,
                     initial_N=N,
                     )

    print(
        f"Running {pop.NPOPULATION} populations of {pop.NANTENNAE} antennae each for {pop.n_generations} generations for {country_code }")
    start_time = time.time()
    for n in range(pop.n_generations):
        pop.generation_cycle()
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime: {runtime/1000} s")
    pop.save(prefix + country_code)
    return pop


def animate(pop, savename):
    pop.plot_animation(f"{savename}animation")
    print("Plotted animation")

    pop.plot_fitness(savefilename=savename + "fitness", show=False)
    print("Plotted fitness")

    for i in np.linspace(0, pop.n_generations, 5, endpoint=False, dtype=int):
        print(f"Plotted generation {i}")
        pop.plot_population(i, savefilename=savename + "snapshot", show=False)

    pop.plot_population(savefilename=savename + "snapshot_final", show=False)
    print("Plotted final snapshot")
    print(f"Finished plotting for {savename}")


if __name__ == '__main__':
    countries = [
        [31, 20, "ES"],
        [50, 33, "PL"],
        [37, 26, "FR"],
        [43, 30, "DE"],
        [44, 22, "IT"],
        [35, 34, "UK"],
    ]
    for parameters in countries:
        if os.path.isfile("data/" + parameters[2] + ".hdf5"):
            pop = load(parameters[2])
        else:
            pop = run(*parameters, n_pop=16, n_trial=48, n_antennae=90, n_generations=50, )

    for parameters in countries:
        if os.path.isfile("data/" + parameters[2] + ".hdf5"):
            pop = load(parameters[2])
            animate(pop, parameters[2])
