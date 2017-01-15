import numpy as np

from GeoData import GeoGrid
from Population import Population


def run(E=50, N=33, country_code='PL',
        n_pop=16,
        n_trial=48,
        n_antennae=90,
        n_generations=50,
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
                     p_cross=0.8,
                     p_mutation=1,
                     std_mutation=1,
                     initial_E=E,
                     initial_N=N,
                     )

    print(
        f"Running {pop.NPOPULATION} populations of {pop.NANTENNAE} antennae each for {pop.n_generations} generations for {country_code }")
    for n in range(pop.n_generations):
        pop.generation_cycle()

    pop.save(prefix + country_code)
    pop.plot_animation(f"{prefix}{country_code}animation")
    print("Plotted animation")

    pop.plot_fitness(savefilename=prefix + country_code + "fitness", show=False)
    print("Plotted fitness")

    for i in np.linspace(0, pop.n_generations, 5, endpoint=False, dtype=int):
        print(f"Plotted generation {i}")
        pop.plot_population(i, savefilename=prefix + country_code + "snapshot", show=False)

    pop.plot_population(savefilename=prefix + country_code + "snapshot_final", show=False)
    print("Plotted final snapshot")
    print(f"Finished plotting for {country_code}")


if __name__ == '__main__':
    countries = [
        [-2, 54, "UK"],
        [-3, 42, "ES"],
        [3, 47, "FR"],
        [10, 51, "DE"],
        [12, 43, "IT"],
        [50, 33, "PL"],
    ]
    for parameters in countries:
        run(*parameters)
