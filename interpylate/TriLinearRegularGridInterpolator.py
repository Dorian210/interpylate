import warnings

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
        mask_negative = inds<0
        inds[mask_negative] = 0
        max_ind = axis_size - 2
        mask_too_large = inds>max_ind
        inds[mask_too_large] = max_ind
        outside_axis = mask_negative.any() or mask_too_large.any()
        coords = continuous_inds_axis - inds
        return (inds, coords, outside_axis)

    def _get_inds_coords(self, shape, continuous_inds):
        inds = [None]*3
        coords = [None]*3
        outside = False
        for axis in range(3):
            (inds[axis], coords[axis], outside_axis) = self._get_inds_coords_axis(continuous_inds[axis], shape[axis])
            outside = outside_axis or outside
        if outside:
            warnings.warn("Interpolate outside of the array !")
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