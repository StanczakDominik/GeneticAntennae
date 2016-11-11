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
R = np.dstack((X,Y)) # shape: (NX, NY, 2)


N_ANTENNAE = 1
antenna_r = np.zeros((N_ANTENNAE, 2), dtype=float)
ANTENNA_RADIUS = 2

def antenna_coverage(r_antenna, X_grid, Y_grid, power=1):
    """compute coverage of grid by single antenna
    assumes coverage is power/distance^2

    TODO: vectorise this to cover all antennae in array simultaneously
    """
    X_a, Y_a = r_antenna.T #transpose to make array.shape (2, N)
    distance_squared = (X_a-X_grid)**2 + (Y_a - Y_grid)**2
    result = power/distance_squared

    # cover case where antenna is located in grid point
    result[np.isinf(result)] = 0 # TODO: find better solution
    return result

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


r = np.random.random((N_ANTENNAE, 2))
coverage = antenna_coverage(r, X, Y)
plot(coverage, r)
plt.show()
