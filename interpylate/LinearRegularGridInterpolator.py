import warnings

class LinearRegularGridInterpolator:
    """
    Linear grid interpolator : interpolate a 1D array between the indices using a linear (afine) method :
    F(x) = a + bx
    """
    def __init__(self):
        """
        Create the interpolator.
        """
        pass
    
    def _get_corners(self, vector, i):
        i0, i1 = i, i + 1
        l = vector[i0]
        m = vector[i1]
        return (l, m)

    def _make_coefs(self, vector, i):
        l, m = self._get_corners(vector, i)
        a = l
        b = m - a
        return [a, b]

    def _get_inds_coords(self, continuous_inds, size):
        inds = continuous_inds.astype('int')
        mask_negative = inds<0
        inds[mask_negative] = 0
        max_ind = size - 2
        mask_too_large = inds>max_ind
        inds[mask_too_large] = max_ind
        if mask_negative.any() or mask_too_large.any():
            warnings.warn("Interpolate outside of the array !")
        coords = continuous_inds - inds
        return (inds, coords)

    def evaluate(self, vector, continuous_inds):
        """
        Evaluate the interpolation at the coordinates given as input.

        Parameters
        ----------
        vector : numpy.ndarray
            The array to interpolate.
        continuous_inds : numpy.ndarray of float
            Coordinates where the interpolation is computed.
            It's shape should be (n,).

        Returns
        -------
        evaluated : numpy.ndarray of float
            Interpolated values at the coordinates given.
            If ``continuous_inds`` is of shape (n,), the output will be of shape (n,).
        """
        (i, x) = self._get_inds_coords(vector.size, continuous_inds)
        [a, b] = self._make_coefs(vector, i)
        evaluated = (a + b*x)
        return evaluated

    def grad(self, vector, continuous_inds, evaluate_too=False):
        """
        Compute the gradient of the interpolated array.

        Parameters
        ----------
        vector : numpy.ndarray
            The array to interpolate.
        continuous_inds : numpy.ndarray of float
            Coordinates where the gradient is computed.
            It's shape should be (n,).
        evaluate_too : bool, optional
            Set to True to evaluate on these points too, by default False

        Returns
        -------
        [grad_x] : list of numpy.ndarray of float
            The derivative of the interpolated array in each axis's direction.
            If ``continuous_inds`` is of shape (n,), each of the output will be of shape (n,).
        """
        (i, x) = self._get_inds_coords(vector.size, continuous_inds)
        [a, b] = self._make_coefs(vector, i)
        grad_x = (b)
        if evaluate_too:
            evaluated = (a + b*x)
            return [grad_x], evaluated
        return [grad_x]

    def hess(self, vector, continuous_inds, grad_too=False, evaluate_too=False):
        """
        Compute the hessian of the interpolated array.

        Parameters
        ----------
        vector : numpy.ndarray
            The array to interpolate.
        continuous_inds : numpy.ndarray of float
            Coordinates where the hessian is computed.
            It's shape should be (n,).
        grad_too : bool, optional
            Set to True to compute the gradient on these points too, by default False
        evaluate_too : bool, optional
            Set to True to evaluate on these points too, by default False

        Returns
        -------
        [] : list of numpy.ndarray of float
            The second order derivatives of the interpolated array in each axis couple's direction.
            If ``continuous_inds`` is of shape (n,), each of the output will be of shape (n,).
        """
        if grad_too:
            if evaluate_too:
                [grad_x], evaluated = self.grad(vector, continuous_inds, evaluate_too=True)
                return [], [grad_x], evaluated
            [grad_x] = self.grad(vector, continuous_inds)
            return [], [grad_x]
        return []
