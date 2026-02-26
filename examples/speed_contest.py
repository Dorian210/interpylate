# %%
import numpy as np
from interpylate import NLinearRegularGridInterpolator
from scipy.interpolate import RegularGridInterpolator
n = 1000
nb_evaluation = n**2
mode = 'float'
if mode=='int':
    pix = np.random.randint(256, size=(n, n, n))
elif mode=='float':
    pix = np.random.rand(n, n, n)

my_interp = NLinearRegularGridInterpolator(3)
my_inds = (n - 1)*np.random.rand(3, nb_evaluation)
%timeit my_interp.evaluate(pix, my_inds)  # type: ignore
del my_interp
del my_inds

x = np.arange(pix.shape[0])  # type: ignore
y = np.arange(pix.shape[1])  # type: ignore
z = np.arange(pix.shape[2])  # type: ignore
scipy_interp = RegularGridInterpolator((x, y, z), pix, method='linear', bounds_error=True, fill_value=None)  # type: ignore
scipy_inds = (n - 1)*np.random.rand(nb_evaluation, 3)
%timeit scipy_interp(scipy_inds)  # type: ignore
del x
del y
del z
del scipy_interp
del scipy_inds
# %%
