import numpy as np
import numba as nb

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
        inds[inds<0] = 0
        max_ind = size - 2
        inds[inds>max_ind] = max_ind
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
        numpy.ndarray of float
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
        list of numpy.ndarray of float
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
        list of numpy.ndarray of float
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

class BiLinearRegularGridInterpolator:
    """
    Bi-linear grid interpolator : interpolate a 2D array between the indices using a bi-linear method :
    F(x, y) = a + bx + cy + dxy
    """
    def __init__(self):
        """
        Create the interpolator.
        """
        pass
    
    def _get_corners(self, image, i, j):
        i0, j0, i1, j1 = i, j, i + 1, j + 1
        l = image[i0, j0]
        m = image[i1, j0]
        n = image[i0, j1]
        o = image[i1, j1]
        return (l, m, n, o)

    def _make_coefs(self, image, i, j):
        l, m, n, o = self._get_corners(image, i, j)
        a = l
        b = m - a
        c = n - a
        d = o - a - b - c
        return [a, b, c, d]

    def _get_inds_coords_axis(self, continuous_inds_axis, axis_size):
        inds = continuous_inds_axis.astype('int')
        inds[inds<0] = 0
        max_ind = axis_size - 2
        inds[inds>max_ind] = max_ind
        coords = continuous_inds_axis - inds
        return (inds, coords)

    def _get_inds_coords(self, shape, continuous_inds):
        inds = [None]*2
        coords = [None]*2
        for axis in range(2):
            (inds[axis], coords[axis]) = self._get_inds_coords_axis(continuous_inds[axis], shape[axis])
        return (inds, coords)

    def evaluate(self, image, continuous_inds):
        """
        Evaluate the interpolation at the coordinates given as input.

        Parameters
        ----------
        image : numpy.ndarray
            The array to interpolate.
        continuous_inds : numpy.ndarray of float
            Coordinates where the interpolation is computed.
            It's shape should be (2, n).

        Returns
        -------
        numpy.ndarray of float
            Interpolated values at the coordinates given.
            If ``continuous_inds`` is of shape (2, n), the output will be of shape (n,).
        """
        ([i, j], [x, y]) = self._get_inds_coords(image.shape, continuous_inds)
        [a, b, c, d] = self._make_coefs(image, i, j)
        evaluated = (a + b*x +c*y + d*x*y)
        return evaluated

    def grad(self, image, continuous_inds, evaluate_too=False):
        """
        Compute the gradient of the interpolated array.

        Parameters
        ----------
        image : numpy.ndarray
            The array to interpolate.
        continuous_inds : numpy.ndarray of float
            Coordinates where the gradient is computed.
            It's shape should be (2, n).
        evaluate_too : bool, optional
            Set to True to evaluate on these points too, by default False

        Returns
        -------
        list of numpy.ndarray of float
            The derivative of the interpolated array in each axis's direction.
            If ``continuous_inds`` is of shape (2, n), each of the output will be of shape (n,).
        """
        ([i, j], [x, y]) = self._get_inds_coords(image.shape, continuous_inds)
        [a, b, c, d] = self._make_coefs(image, i, j)
        grad_x = (b + d*y)
        grad_y = (c + d*x)
        if evaluate_too:
            evaluated = (a + b*x +c*y + d*x*y)
            return [grad_x, grad_y], evaluated
        return [grad_x, grad_y]

    def hess(self, image, continuous_inds, grad_too=False, evaluate_too=False):
        """
        Compute the hessian of the interpolated array.

        Parameters
        ----------
        image : numpy.ndarray
            The array to interpolate.
        continuous_inds : numpy.ndarray of float
            Coordinates where the hessian is computed.
            It's shape should be (2, n).
        grad_too : bool, optional
            Set to True to compute the gradient on these points too, by default False
        evaluate_too : bool, optional
            Set to True to evaluate on these points too, by default False

        Returns
        -------
        list of numpy.ndarray of float
            The second order derivatives of the interpolated array in each axis couple's direction.
            If ``continuous_inds`` is of shape (2, n), each of the output will be of shape (n,).
        """
        ([i, j], [x, y]) = self._get_inds_coords(image.shape, continuous_inds)
        [a, b, c, d] = self._make_coefs(image, i, j)
        hess_xy = (d)
        if grad_too:
            grad_x = (b + d*y)
            grad_y = (c + d*x)
            if evaluate_too:
                evaluated = (a + b*x +c*y + d*x*y)
                return [hess_xy], [grad_x, grad_y], evaluated
            return [hess_xy], [grad_x, grad_y]
        return [hess_xy]

class TriLinearRegularGridInterpolator:
    """
    Tri-linear grid interpolator : interpolate a 3D array between the indices using a tri-linear method :
    F(x, y, z) = a + bx + cy + dz + exy + fxz + gyz + hxyz
    """
    def __init__(self):
        """
        Create the interpolator.
        """
        pass
    
    def _get_corners(self, volume, i, j, k):
        i0, j0, k0, i1, j1, k1 = i, j, k, i + 1, j + 1, k + 1
        l = volume[i0, j0, k0]
        m = volume[i1, j0, k0]
        n = volume[i0, j1, k0]
        o = volume[i0, j0, k1]
        p = volume[i1, j1, k0]
        q = volume[i1, j0, k1]
        r = volume[i0, j1, k1]
        s = volume[i1, j1, k1]
        return (l, m, n, o, p, q, r, s)

    def _make_coefs(self, volume, i, j, k):
        l, m, n, o, p, q, r, s = self._get_corners(volume, i, j, k)
        a = l
        b = m - a
        c = n - a
        d = o - a
        e = p - a - b - c
        f = q - a - b - d
        g = r - a - c - d
        h = s - a - b - c - d - e - f - g
        return [a, b, c, d, e, f, g, h]

    def _get_inds_coords_axis(self, continuous_inds_axis, axis_size):
        inds = continuous_inds_axis.astype('int')
        inds[inds<0] = 0
        max_ind = axis_size - 2
        inds[inds>max_ind] = max_ind
        coords = continuous_inds_axis - inds
        return (inds, coords)

    def _get_inds_coords(self, shape, continuous_inds):
        inds = [None]*3
        coords = [None]*3
        for axis in range(3):
            (inds[axis], coords[axis]) = self._get_inds_coords_axis(continuous_inds[axis], shape[axis])
        return (inds, coords)

    def evaluate(self, volume, continuous_inds):
        """
        Evaluate the interpolation at the coordinates given as input.

        Parameters
        ----------
        volume : numpy.ndarray
            The array to interpolate.
        continuous_inds : numpy.ndarray of float
            Coordinates where the interpolation is computed.
            It's shape should be (3, n).

        Returns
        -------
        numpy.ndarray of float
            Interpolated values at the coordinates given.
            If ``continuous_inds`` is of shape (3, n), the output will be of shape (n,).
        """
        ([i, j, k], [x, y, z]) = self._get_inds_coords(volume.shape, continuous_inds)
        [a, b, c, d, e, f, g, h] = self._make_coefs(volume, i, j, k)
        evaluated = (a + b*x +c*y + d*z + e*x*y + f*x*z + g*y*z + h*x*y*z)
        return evaluated

    def grad(self, volume, continuous_inds, evaluate_too=False):
        """
        Compute the gradient of the interpolated array.

        Parameters
        ----------
        volume : numpy.ndarray
            The array to interpolate.
        continuous_inds : numpy.ndarray of float
            Coordinates where the gradient is computed.
            It's shape should be (3, n).
        evaluate_too : bool, optional
            Set to True to evaluate on these points too, by default False

        Returns
        -------
        list of numpy.ndarray of float
            The derivative of the interpolated array in each axis's direction.
            If ``continuous_inds`` is of shape (3, n), each of the output will be of shape (n,).
        """
        ([i, j, k], [x, y, z]) = self._get_inds_coords(volume.shape, continuous_inds)
        [a, b, c, d, e, f, g, h] = self._make_coefs(volume, i, j, k)
        grad_x = (b + e*y + f*z + h*y*z)
        grad_y = (c + e*x + g*z + h*x*z)
        grad_z = (d + f*x + g*y + h*x*y)
        if evaluate_too:
            evaluated = (a + b*x +c*y + d*z + e*x*y + f*x*z + g*y*z + h*x*y*z)
            return [grad_x, grad_y, grad_z], evaluated
        return [grad_x, grad_y, grad_z]

    def hess(self, volume, continuous_inds, grad_too=False, evaluate_too=False):
        """
        Compute the hessian of the interpolated array.

        Parameters
        ----------
        volume : numpy.ndarray
            The array to interpolate.
        continuous_inds : numpy.ndarray of float
            Coordinates where the hessian is computed.
            It's shape should be (3, n).
        grad_too : bool, optional
            Set to True to compute the gradient on these points too, by default False
        evaluate_too : bool, optional
            Set to True to evaluate on these points too, by default False

        Returns
        -------
        list of numpy.ndarray of float
            The second order derivatives of the interpolated array in each axis couple's direction.
            If ``continuous_inds`` is of shape (3, n), each of the output will be of shape (n,).
        """
        ([i, j, k], [x, y, z]) =self._get_inds_coords(volume.shape, continuous_inds)
        [a, b, c, d, e, f, g, h] = self._make_coefs(volume, i, j, k)
        hess_xy = (e + h*z)
        hess_xz = (f + h*y)
        hess_yz = (g + h*x)
        if grad_too:
            grad_x = (b + e*y + f*z + h*y*z)
            grad_y = (c + e*x + g*z + h*x*z)
            grad_z = (d + f*x + g*y + h*x*y)
            if evaluate_too:
                evaluated = (a + b*x +c*y + d*z + e*x*y + f*x*z + g*y*z + h*x*y*z)
                return [hess_xy, hess_xz, hess_yz], [grad_x, grad_y, grad_z], evaluated
            return [hess_xy, hess_xz, hess_yz], [grad_x, grad_y, grad_z]
        return [hess_xy, hess_xz, hess_yz]

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
        inds[inds<0] = 0
        for axis in range(self.dim):
            max_size = shape[axis] - 2
            inds[axis, inds[axis]>max_size] = max_size
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

@nb.njit(cache=True)
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