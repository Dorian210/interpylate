import numpy as np
from numba import njit
import warnings

class NLinearRegularGridInterpolatorLarge:
    """
    N-linear grid interpolator : interpolate a ND array between the indices using a N-linear method.
    For N<4, consider using ``TriLinearRegularGridInterpolator``, ``BiLinearRegularGridInterpolator``, 
    or ``LinearRegularGridInterpolator``.
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
        self.dim = dim
        self.nb_coefs = 2**self.dim
        self.masks = self._make_masks()
        self.mat = self._make_mat()
    
    def _make_masks(self):
        masks = np.zeros((self.nb_coefs, self.dim), dtype='bool')
        for i in range(self.nb_coefs):
            tmp = np.array(list(bin(i)[2:]), dtype='int')
            masks[i, (self.dim - tmp.size):] = tmp
        return masks
    
    def _make_mat(self):
        M = np.logical_not(np.logical_and(self.masks[None, :, :], self.masks[:, None, :]).any(axis=-1)).astype('int')
        mat = np.linalg.inv(M).astype('int')
        return mat
    
    def _make_corners_inds(self, inds):
        inds_inds_plus_one = np.repeat(inds[None], 2, axis=0)
        inds_inds_plus_one[1] += 1
        arr = np.repeat(np.arange(self.dim)[:, None], self.nb_coefs, axis=1)
        corners_inds = inds_inds_plus_one[self.masks.T.astype('int'), arr]
        return corners_inds
    
    def _get_corners(self, NDarray, inds):
        corners_inds = self._make_corners_inds(inds)
        corners = NDarray[tuple(corners_inds)]
        return corners
    
    def _make_coefs(self, NDarray, inds):
        corners = self._get_corners(NDarray, inds)
        coefs = self.mat @ corners
        return coefs
    
    def _get_inds_coords(self, shape, continuous_inds):
        inds = continuous_inds.astype('int')
        mask_negative = inds<0
        inds[mask_negative] = 0
        outside = mask_negative.any()
        for axis in range(self.dim):
            max_size = shape[axis] - 2
            mask_too_large = inds[axis]>max_size
            inds[axis, mask_too_large] = max_size
            outside = outside or mask_too_large.any()
        if outside:
            warnings.warn("Interpolate outside of the array !")
        coords = continuous_inds - inds
        return inds, coords
    
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
        inds, coords = self._get_inds_coords(NDarray.shape, continuous_inds)
        coefs = self._make_coefs(NDarray, inds)
        evaluated = _evaluate(coords, self.masks, coefs)
        return evaluated
    
    def _change_masks_and_coef_inds_by_deriving(self, masks, coef_inds, k):
        to_keep = masks[:, k]
        new_masks = masks[to_keep]
        new_masks[:, k] = False
        new_coef_inds = coef_inds[to_keep]
        return new_masks, new_coef_inds
    
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
        inds, coords = self._get_inds_coords(NDarray.shape, continuous_inds)
        coefs = self._make_coefs(NDarray, inds)
        coef_inds = np.arange(self.nb_coefs)
        masks = self.masks
        grad = [None]*self.dim
        for axis in range(self.dim):
            new_masks, new_coef_inds = self._change_masks_and_coef_inds_by_deriving(masks, coef_inds, axis)
            grad[axis] = _evaluate(coords, new_masks, coefs[new_coef_inds])
        if evaluate_too:
            evaluated = _evaluate(coords, self.masks, coefs)
            return grad, evaluated
        return grad
    
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
        inds, coords = self._get_inds_coords(NDarray.shape, continuous_inds)
        coefs = self._make_coefs(NDarray, inds)
        coef_inds = np.arange(self.nb_coefs)
        masks = self.masks
        hess = []
        if grad_too:
            grad = [None]*self.dim
        for axis1 in range(self.dim):
            new1_masks, new1_coef_inds = self._change_masks_and_coef_inds_by_deriving(masks, coef_inds, axis1)
            for axis2 in range(axis1 + 1, self.dim):
                new2_masks, new2_coef_inds = self._change_masks_and_coef_inds_by_deriving(new1_masks, new1_coef_inds, axis2)
                hess.append(_evaluate(coords, new2_masks, coefs[new2_coef_inds]))
            if grad_too:
                grad[axis1] = _evaluate(coords, new1_masks, coefs[new1_coef_inds])
        if grad_too:
            if evaluate_too:
                evaluated = _evaluate(coords, self.masks, coefs)
                return hess, grad, evaluated
            return hess, grad
        return hess # (d2interp_dxy, d2interp_dxz, d2interp_dyz)

@njit(cache=True)
def _evaluate(coords, masks, coefs):
    nb_eval = coords.shape[1]
    evaluated = np.zeros(nb_eval, dtype='float')
    for i, mask in enumerate(masks):
        prod = np.ones(nb_eval, dtype='float')
        for j, boolean in enumerate(mask):
            if boolean:
                prod *= coords[j]
        evaluated += coefs[i]*prod
    return evaluated