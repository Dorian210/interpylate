---
interpylate
---

interpylate is a Python library for N-linear regular grid interpolation. It provides a flexible and efficient method to interpolate N-dimensional arrays using a N-linear approach.

It can be useful for image interpolation and its 3D (or more) equivalent.

# Installation

``` bash
pip install interpylate
```

# Usage

``` python
from interpylate import NLinearRegularGridInterpolator

# Create an instance of the interpolator
interpolator = NLinearRegularGridInterpolator(dim)

# Evaluate the interpolation at specified coordinates
interpolated_values = interpolator.evaluate(NDarray, continuous_inds)

# Compute the gradient of the interpolated array
gradient = interpolator.grad(NDarray, continuous_inds, evaluate_too=False)

# Compute the hessian of the interpolated array
hessian = interpolator.hess(NDarray, continuous_inds, grad_too=False, evaluate_too=False)
```

# Documentation

For detailed documentation, please refer to the interpylate [documentation](https://dorian210.github.io/interpylate/interpylate.html).

# Examples

You can find usage examples in the [examples](https://github.com/Dorian210/interpylate/tree/main/examples) directory of the interpylate GitHub repository.

# Contributing

Contributions to interpylate are welcome! If you have any bug reports, feature requests, or want to contribute code, please visit the [GitHub repository](https://github.com/Dorian210/interpylate).

# License

interpylate is licensed under the CeCILL License. See [LICENSE](https://github.com/Dorian210/interpylate/tree/main/LICENSE) for more information.