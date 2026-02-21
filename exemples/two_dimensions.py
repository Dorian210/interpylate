# %%
import numpy as np
import matplotlib.pyplot as plt
from interpylate import NLinearRegularGridInterpolator

# Create a 2D dataset
x = np.linspace(0, 2*np.pi, 20)
y = np.linspace(0, 2*np.pi, 20)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2
dataset = Z

# Create an instance of the interpolator for two dimensions
interpolator = NLinearRegularGridInterpolator(2)

# Define the continuous indices for interpolation
x_new = np.linspace(0, dataset.shape[0] - 1, 500, endpoint=False)
y_new = np.linspace(0, dataset.shape[1] - 1, 500, endpoint=False)
X_new, Y_new = np.meshgrid(x_new, y_new)
continuous_inds = np.stack((X_new.flat, Y_new.flat))

# Compute the interpolation, its gradient, and its hessian at specified continuous indices
evaluated = interpolator.evaluate(dataset, continuous_inds)
gradx, grady = interpolator.grad(dataset, continuous_inds)
hessian, = interpolator.hess(dataset, continuous_inds)

# Plot the results
fig, axs = plt.subplots(4, 2, figsize=(10, 12))

# Original Dataset
im1 = axs[0, 0].imshow(dataset)
plt.colorbar(im1, ax=axs[0, 0])
axs[0, 0].axis('off')
axs[0, 0].set_title('Original Dataset')

# Interpolated Dataset
im2 = axs[1, 0].imshow(evaluated.reshape(X_new.shape))
plt.colorbar(im2, ax=axs[1, 0])
axs[1, 0].axis('off')
axs[1, 0].set_title('Interpolated Dataset')

# Gradient in x-direction
im3 = axs[2, 0].imshow(gradx.reshape(X_new.shape))
plt.colorbar(im3, ax=axs[2, 0])
axs[2, 0].axis('off')
axs[2, 0].set_title('Gradient (x-direction)')

# Gradient in y-direction
im4 = axs[2, 1].imshow(grady.reshape(X_new.shape))
plt.colorbar(im4, ax=axs[2, 1])
axs[2, 1].axis('off')
axs[2, 1].set_title('Gradient (y-direction)')

# Hessian
im5 = axs[3, 0].imshow(hessian.reshape(X_new.shape))
plt.colorbar(im5, ax=axs[3, 0])
axs[3, 0].axis('off')
axs[3, 0].set_title('Hessian')

axs[0, 1].remove()
axs[1, 1].remove()
axs[3, 1].remove()

plt.tight_layout()
plt.show()
# %%
