from .LinearRegularGridInterpolator import LinearRegularGridInterpolator
from .BiLinearRegularGridInterpolator import BiLinearRegularGridInterpolator
from .TriLinearRegularGridInterpolator import TriLinearRegularGridInterpolator
from .NLinearRegularGridInterpolatorLarge import NLinearRegularGridInterpolatorLarge

class NLinearRegularGridInterpolator:
    """
    N-linear grid interpolator : interpolate a ND array between the indices using a N-linear method.
    For exemple, for a 2D array, the interpolator finds a, b, c and d on each square of the grid such that 
    F(x, y) = a + bx + cy + dxy
    """
    def __init__(self, dim):
        """
        Create and initialize the interpolator.

        Parameters
        ----------
        dim : int
            The dimension of the array to be interpolated.
        """
        if dim==1:
            self.interpolator = LinearRegularGridInterpolator()
        elif dim==2:
            self.interpolator = BiLinearRegularGridInterpolator()
        elif dim==3:
            self.interpolator = TriLinearRegularGridInterpolator()
        else:
            self.interpolator = NLinearRegularGridInterpolatorLarge(dim)
        self.dim = dim
    
    
    def evaluate(self, NDarray, continuous_inds):
        """
        Evaluate the interpolation at the coordinates given as input.

        Parameters
        ----------
        NDarray : numpy.ndarray
            The array to interpolate.
        continuous_inds : numpy.ndarray of float
            Coordinates where the interpolation is computed.
            It's shape should be (dim, n) if dim is the number of dimensions of the array interpolated.

        Returns
        -------
        numpy.ndarray of float
            Interpolated values at the coordinates given.
            If ``continuous_inds`` is of shape (dim, n), the output will be of shape (n,).
        """
        return self.interpolator.evaluate(NDarray, continuous_inds)
    
    def grad(self, NDarray, continuous_inds, evaluate_too=False):
        """
        Compute the gradient of the interpolated array.

        Parameters
        ----------
        NDarray : numpy.ndarray
            The array to interpolate.
        continuous_inds : numpy.ndarray of float
            Coordinates where the gradient is computed.
            It's shape should be (dim, n) if dim is the number of dimensions of the array interpolated.
        evaluate_too : bool, optional
            Set to True to evaluate on these points too, by default False

        Returns
        -------
        list of numpy.ndarray of float
            The derivative of the interpolated array in each axis's direction.
            If ``continuous_inds`` is of shape (dim, n), each of the output will be of shape (n,).
        """
        return self.interpolator.grad(NDarray, continuous_inds, evaluate_too)
    
    def hess(self, NDarray, continuous_inds, grad_too=False, evaluate_too=False):
        """
        Compute the hessian of the interpolated array.

        Parameters
        ----------
        NDarray : numpy.ndarray
            The array to interpolate.
        continuous_inds : numpy.ndarray of float
            Coordinates where the hessian is computed.
            It's shape should be (dim, n) if dim is the number of dimensions of the array interpolated.
        grad_too : bool, optional
            Set to True to compute the gradient on these points too, by default False
        evaluate_too : bool, optional
            Set to True to evaluate on these points too, by default False

        Returns
        -------
        list of numpy.ndarray of float
            The second order derivatives of the interpolated array in each axis couple's direction.
            If ``continuous_inds`` is of shape (dim, n), each of the output will be of shape (n,).
        """
        return self.interpolator.hess(NDarray, continuous_inds, grad_too, evaluate_too)