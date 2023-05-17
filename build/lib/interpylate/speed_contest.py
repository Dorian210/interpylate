# %%
import numpy as np
from interpylate import NLinearRegularGridInterpolatorLarge
from scipy.interpolate import RegularGridInterpolator
n = 1000
nb_evaluation = 100*n
mode = 'float'
if mode=='int':
    pix = np.random.randint(256, size=(n, n, n))
elif mode=='float':
    pix = np.random.rand(n, n, n)

my_interp = NLinearRegularGridInterpolatorLarge(3)
my_inds = (n - 1)*np.random.rand(3, nb_evaluation)
%timeit my_interp.evaluate(pix, my_inds)
del my_interp
del my_inds

x = np.arange(pix.shape[0])
y = np.arange(pix.shape[1])
z = np.arange(pix.shape[2])
scipy_interp = RegularGridInterpolator((x, y, z), pix, method='linear', bounds_error=True, fill_value=None)
scipy_inds = (n - 1)*np.random.rand(nb_evaluation, 3)
%timeit scipy_interp(scipy_inds)
del x
del y
del z
del scipy_interp
del scipy_inds