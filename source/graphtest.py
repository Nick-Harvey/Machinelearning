import numpy as np
from matplotlib import pyplot as plt

def get_radius(T, params):
	m, n_1, n_2, n_3 = params
	U = (m * T) / 4

	return (np.fabs(np.cos(U)) ** n_2 + np.fabs(np.sin(U)) ** n_3) ** (-1. / n_1)

grid_size = (3, 4)
T = np.linspace(0, 2 * np.pi, 1024)

for i in range(grid_size[0]):
	for j in range(grid_size[1]):
		params = np.random.random_integers(1, 20, size = 4)
		R = get_radius(T, params)

		axes = plt.subplot2grid(grid_size, (i, j), rowspan=1, colspan=1)
		axes.get_xaxis().set_visible(False)
		axes.get_yaxis().set_visible(False)

		plt.plot(R * np.cos(T), R * np.sin(T), c = 'k')
		plt.title('%d, %d, %d, %d' % tuple(params), fontsize = 'small')

plt.tight_layout()
plt.show()