"""genetic optimization of radio antenna coverage"""
import numpy as np
import matplotlib.pyplot as plt

# grid dimensions
XMIN = 0.
XMAX = 1.
YMIN = 0.
YMAX = 1.

# grid resolution
NX = 100
NY = 100

# prepare 2D grid
x, DX = np.linspace(XMIN, XMAX, NX, retstep=True, endpoint=False)
y, DY = np.linspace(YMIN, YMAX, NY, retstep=True, endpoint=False)
X, Y = np.meshgrid(x, y)
R = np.stack((X,Y), axis=0)

N_ANTENNAE = 3
np.random.seed(0)
antenna_r = np.random.random((N_ANTENNAE, 2))
ANTENNA_RADIUS = 2

def antenna_coverage(r_antenna, r_grid, power=0.1):
    """compute coverage of grid by single antenna
    assumes coverage is power/distance^2

    TODO: vectorise this to cover all antennae in array simultaneously
    """

    # array of distances squared from each antenna
    # uses numpy broadcasting
    # (N, 2) - (2, NX, NY)
    distance_squared = ((r_antenna[..., np.newaxis, np.newaxis] - r_grid[np.newaxis, ...])**2).sum(axis=1)

    # if we want to go for 1/r^2 antenna coverage
    # result = (power*ANTENNA_RADIUS**2/distance_squared).sum(axis=0)
    # # cover case where antenna is located in grid point
    # result[np.isinf(result)] = 0 # TODO: find better solution

    # binary coverage case
    result = (distance_squared<power**2) # is grid entry covered by any
    result = result.sum(axis=0) >0        # logical or
    result = result > 0

    return result.astype(float)

def plot(coverage, r):
    fig, ax = plt.subplots()
    x_a, y_a = r.T
    ax.plot(x_a, y_a, "k*", label="Antennae locations")

    contours = ax.contour(X, Y, coverage, 100, cmap='viridis', label="Coverage")
    colors = ax.contourf(X, Y, coverage, 100, cmap='viridis') #contourf: contour FILLED
    fig.colorbar(colors)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc='best')

    return fig

coverage = antenna_coverage(antenna_r, R)

# population as weights
DISTANCES = ((R-np.array([(XMAX-XMIN)/2, (YMAX-YMIN)/2], ndmin=3).T)**2).sum(axis=0)
population = np.exp(-DISTANCES*10)


plot(coverage*population, antenna_r)
plt.show()
