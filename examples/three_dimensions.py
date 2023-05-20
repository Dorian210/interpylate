# %%
import numpy as np
import matplotlib.pyplot as plt
from interpylate import NLinearRegularGridInterpolator

# Create a 3D dataset
x = np.linspace(0, 2*np.pi, 20)
y = np.linspace(0, 2*np.pi, 20)
z = np.linspace(0, 2*np.pi, 20)
X, Y, Z = np.meshgrid(x, y, z)
W = X**2 + Y**2 + Z**2 + X*Y*Z
dataset = W

# Create an instance of the interpolator for three dimensions
interpolator = NLinearRegularGridInterpolator(3)

# Define the continuous indices for interpolation
x_new = np.linspace(0, dataset.shape[0] - 1, 50, endpoint=False)
y_new = np.linspace(0, dataset.shape[1] - 1, 50, endpoint=False)
z_new = np.linspace(0, dataset.shape[2] - 1, 50, endpoint=False)
X_new, Y_new, Z_new = np.meshgrid(x_new, y_new, z_new)
continuous_inds = np.array([X_new.flatten(), Y_new.flatten(), Z_new.flatten()])

# Compute the interpolation at specified continuous indices
evaluated = interpolator.evaluate(dataset, continuous_inds)
gradx, grady, gradz = interpolator.grad(dataset, continuous_inds)
hessxy, hessxz, hessyz = interpolator.hess(dataset, continuous_inds)

# Plot the results
fig, axs = plt.subplots(4, 3, figsize=(10, 12), subplot_kw={'projection': '3d'})

# Original Dataset
im1 = axs[0, 0].scatter(X, Y, Z, c=dataset)
plt.colorbar(im1, ax=axs[0, 0])
axs[0, 0].axis('off')
axs[0, 0].set_title('Original Dataset')

# Interpolated Dataset
im2 = axs[1, 0].scatter(X_new, Y_new, Z_new, c=evaluated.reshape(X_new.shape))
plt.colorbar(im2, ax=axs[1, 0])
axs[1, 0].axis('off')
axs[1, 0].set_title('Interpolated Dataset')

# Gradient in x-direction
im3 = axs[2, 0].scatter(X_new, Y_new, Z_new, c=gradx.reshape(X_new.shape))
plt.colorbar(im3, ax=axs[2, 0])
axs[2, 0].axis('off')
axs[2, 0].set_title('Gradient (x-direction)')

# Gradient in y-direction
im4 = axs[2, 1].scatter(X_new, Y_new, Z_new, c=grady.reshape(X_new.shape))
plt.colorbar(im4, ax=axs[2, 1])
axs[2, 1].axis('off')
axs[2, 1].set_title('Gradient (y-direction)')

# Gradient in z-direction
im5 = axs[2, 2].scatter(X_new, Y_new, Z_new, c=gradz.reshape(X_new.shape))
plt.colorbar(im5, ax=axs[2, 2])
axs[2, 2].axis('off')
axs[2, 2].set_title('Gradient (z-direction)')

# Hessian in xy-direction
im6 = axs[3, 0].scatter(X_new, Y_new, Z_new, c=hessxy.reshape(X_new.shape))
plt.colorbar(im6, ax=axs[3, 0])
axs[3, 0].axis('off')
axs[3, 0].set_title('Hessian (xy-direction)')

# Hessian in xz-direction
im7 = axs[3, 1].scatter(X_new, Y_new, Z_new, c=hessxz.reshape(X_new.shape))
plt.colorbar(im7, ax=axs[3, 1])
axs[3, 1].axis('off')
axs[3, 1].set_title('Hessian (xz-direction)')

# Hessian in yz-direction
im8 = axs[3, 2].scatter(X_new, Y_new, Z_new, c=hessyz.reshape(X_new.shape))
plt.colorbar(im8, ax=axs[3, 2])
axs[3, 2].axis('off')
axs[3, 2].set_title('Hessian (yz-direction)')

axs[0, 1].remove()
axs[0, 2].remove()
axs[1, 1].remove()
axs[1, 2].remove()

plt.tight_layout()
plt.show()