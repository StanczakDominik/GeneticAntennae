import os

import Population
import run

if __name__ == '__main__':
    countries = [
        [35, 34, "UK"],
        [31, 20, "ES"],
        [37, 26, "FR"],
        [43, 30, "DE"],
        [44, 22, "IT"],
        [50, 33, "PL"],
    ]
    for parameters in countries:
        if os.path.isfile("data/debug" + parameters[2] + ".hdf5"):
            pop = Population.load("debug" + parameters[2])
        else:
            pop = run.run(*parameters, n_pop=4, n_trial=12, n_antennae=10, n_generations=10, prefix="debug")
        run.animate(pop, "debug" + parameters[2])
