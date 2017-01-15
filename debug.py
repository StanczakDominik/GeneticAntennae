from run import run

if __name__ == '__main__':
    countries = [
        [50, 33, "PL"],
    ]
    for parameters in countries:
        run(*parameters, n_pop=4, n_trial=12, n_antennae=10, n_generations=10, prefix="debug")
