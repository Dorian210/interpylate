# %%
import numpy as np
import matplotlib.pyplot as plt
from interpylate import NLinearRegularGridInterpolator

# Create a 1D dataset
x = np.linspace(0, 2*np.pi, 20)
y = np.sin(x)
dataset = y

# Create an instance of the interpolator for one dimension
interpolator = NLinearRegularGridInterpolator(1)

# Compute the interpolation and its gradient at specified continuous indices
continuous_inds = np.linspace(0, dataset.shape[0] - 1, 50, endpoint=False)
evaluated = interpolator.evaluate(dataset, continuous_inds)
gradient, = interpolator.grad(dataset, continuous_inds)

# Plot the results
plt.plot(dataset, 'o', label='Original Data')
plt.plot(continuous_inds, evaluated, label='Interpolation')
plt.plot(continuous_inds, gradient, label='Gradient')
plt.xlabel('continuous indices')
plt.ylabel('data')
plt.title('Linear Interpolation in One Dimension')
plt.legend()
plt.show()
# %%
