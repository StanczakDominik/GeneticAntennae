import numpy as np

XMAX = 1
YMAX = 1


def antenna_coverage_population(R_antenna, r_grid, zasieg):
    NPOPULATION = R_antenna.shape[0]
    NX, NY = r_grid[0].shape
    result_array = np.empty((NPOPULATION, NX, NY), dtype=bool)
    for i, population_member in enumerate(R_antenna):
        result_array[i] = antenna_coverage(population_member, r_grid, zasieg)
    return result_array


def antenna_coverage(r_antenna, r_grid, zasieg):
    """
    compute coverage of grid by single antenna
    assumes coverage is zasieg/distance^2

    array of distances squared from each antenna
    uses numpy broadcasting
    cast (N, 2) - (2, NX, NY) to (N, 2, 1, 1) - (1, 2, NX, NY)
    both are then broadcasted to (N, 2, NX, NY)
    """
    distance_squared = ((r_antenna[..., np.newaxis, np.newaxis] -
                         r_grid[np.newaxis, ...]) ** 2).sum(axis=1)

    # TODO: decide if we want to go for 1/r^2 antenna coverage
    # result = (zasieg*ANTENNA_RADIUS**2/distance_squared).sum(axis=0)
    # # cover case where antenna is located in grid point
    # result[np.isinf(result)] = 0 # TODO: find better solution

    # binary coverage case
    result = (distance_squared < zasieg ** 2)  # is grid entry covered by any
    result = result.sum(axis=0) > 0  # logical or
    result = result > 0

    # TODO: ujemne wagi przez maksymalną możliwą

    return result


def utility_function(coverage_population, weights):
    """returns total coverage as fraction of grid size
    for use in the following genetic operators
    this way we're optimizing a bounded function (values from 0 to 1)"""
    NX, NY = weights.shape
    return (weights.reshape(1, NX, NY) * coverage_population).sum(axis=(1, 2)) / NX / NY


def selection(r_antennae_population, R, weights, zasieg, TEMP_ARRAY):
    NPOPULATION = r_antennae_population.shape[0]
    coverage_population = antenna_coverage_population(r_antennae_population, R, zasieg)
    utility_function_values = utility_function(coverage_population, weights)
    utility_function_total = utility_function_values.sum()
    utility_function_normalized = utility_function_values / utility_function_total
    dystrybuanta = utility_function_normalized.cumsum()
    random_x = np.random.random(NPOPULATION).reshape(1, NPOPULATION)

    new_r_antennae_population = TEMP_ARRAY
    # for i, x in enumerate(random_x.T):
    #     indeks = (x > dystrybuanta).sum()
    #     print(indeks)
    #     new_r_antennae_population[i] = r_antennae_population[indeks]
    selected_targets = (random_x > dystrybuanta.reshape(NPOPULATION, 1)).sum(axis=0)
    # print(selected_targets, utility_function_values, sep='\n')
    # print(utility_function_values.max(), utility_function_values.mean(), utility_function_values.std())

    new_r_antennae_population[...] = r_antennae_population[selected_targets]
    r_antennae_population[...] = new_r_antennae_population[...]
    return utility_function_values.max(), utility_function_values.mean(), utility_function_values.std()


def crossover_cutoff(r_antennae_population, probability_crossover, TEMP_ARRAY):
    NPOPULATION = r_antennae_population.shape[0]
    NANTENNAE = r_antennae_population.shape[1]
    TEMP_ARRAY[...] = r_antennae_population[...]
    for i in range(0, NPOPULATION, 2):
        if i + 1 < NPOPULATION and np.random.random() < probability_crossover:
            cutoff = np.random.randint(0, NANTENNAE)
            a = r_antennae_population[i + 1]
            b = r_antennae_population[i]
            # if DEBUG_MESSAGES:
            #     print("Exchanging these two at k = {}".format(cutoff))
            #     print(a)
            #     print(b)
            TEMP_ARRAY[i, cutoff:] = a[cutoff:]
            TEMP_ARRAY[i + 1, cutoff:] = b[cutoff:]
            # if DEBUG_MESSAGES:
            #     print("They are now", TEMP_ARRAY[i], TEMP_ARRAY[i+1], sep="\n")
    r_antennae_population[...] = TEMP_ARRAY[...]


def mutation(r_antennae_population, gaussian_std, p_mutation):
    """
    https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)

    acts in place!

    podoba mi się pomysł który mają na wiki, żeby mutacja
    1. przesuwała wszystkie anteny o jakiś losowy wektor z gaussowskiej
    dystrybucji
    2. (konieczność u nas:) renormalizowała położenia anten do pudełka
    (xmin, xmax), (ymin, ymax)

    pytanie - czy wystarczy zrobić okresowe warunki brzegowe (modulo), czy skoki
    tym spowodowane będą za duże (bo nie mamy okresowości na pokryciu)?
    w tym momencie - może lepiej zamiast tego robić coś typu max(xmax, x+dx)?
    """
    NPOPULATION = r_antennae_population.shape[0]
    NANTENNAE = r_antennae_population.shape[1]
    # don't want to move population P's antenna A's X without moving its Y
    which_to_move = (np.random.random((NPOPULATION, NANTENNAE)) < p_mutation)
    # which_to_move = np.ones((NPOPULATION, NANTENNAE))
    how_much_to_move = np.random.normal(scale=gaussian_std,
                                        size=(NPOPULATION, NANTENNAE, 2))
    # if DEBUG_MESSAGES:
    #     print(which_to_move)
    r_antennae_population += which_to_move[..., np.newaxis] * how_much_to_move

    # TODO: zamienić okresowe warunki brzegowe na wymuszanie 0 w ujemnych fitnessach
    # TODO: bądź negatywne premiowanie jeśli kółka się nie spełniają
    r_antennae_population[:, :, 0] %= XMAX  # does this need xmin somehow?
    r_antennae_population[:, :, 1] %= YMAX  # likewise?
