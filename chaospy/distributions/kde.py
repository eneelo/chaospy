"""Gaussian kernel density estimation."""
from __future__ import division

import numpy
from scipy.special import comb, ndtr, ndtri, factorial2
from scipy.stats import gaussian_kde

from .baseclass import Dist
from .approximation import approximate_inverse
from .collection.mv_normal import MvNormal


def batch_input(method):
    """
    Wrapper function ensuring that a KDE method never causes memory errors.
    """
    def wrapper(self, loc):
        out = numpy.zeros(loc.shape)
        for idx in range(0, loc.size, self.stride):
            out[:, idx:idx+self.stride] = method(
                self, loc[:, idx:idx+self.stride])
        return out
    return wrapper


class GaussianKDE(Dist):
    """
    Examples:
        >>> samples = [-1, 0, 1]
        >>> dist = GaussianKDE(samples, 0.4**2)
        >>> dist.pdf([-1, -0.5, 0, 0.5, 1]).round(4)
        array([0.3471, 0.3047, 0.3617, 0.3047, 0.3471])
        >>> dist.cdf([-1, -0.5, 0, 0.5, 1]).round(4)
        array([0.1687, 0.3334, 0.5   , 0.6666, 0.8313])
        >>> dist.inv([0, 0.25, 0.5, 0.75, 1]).round(4)
        array([-3.7109, -0.7645,  0.    ,  0.7645,  3.2256])
        >>> dist.mom([1, 2, 3]).round(4)
        array([0.    , 0.8267, 0.    ])
        >>> # Does dist normalize to one
        >>> t = numpy.linspace(-4, 4, 1000000)
        >>> abs(numpy.mean(dist.pdf(t))*(t[-1]-t[0]) - 1)  # err
        1.0000000212340154e-06

        >>> samples = [[-1, 0, 1], [0, 1, 2]]
        >>> dist = GaussianKDE(samples, 0.4)
        >>> dist.pdf([[0, 0, 1, 1], [0, 1, 0, 1]]).round(4)
        array([0.0482, 0.0977, 0.008 , 0.0482])
        >>> dist.inv([[0, 0, 1, 1], [0, 1, 0, 1]]).round(4)
        array([[-5.6258, -5.6258,  4.5322,  4.5322],
               [-4.7298,  5.6605, -1.9197,  6.2376]])
        >>> dist.mom([(0, 1, 1), (1, 0, 1)]).round(4)
        array([1.    , 0.    , 0.6667])

    """

    def __init__(self, samples, bandwidth2="scott", batch_size=1e7):
        """
        Args:
            samples (numpy.ndarray):
                The samples to generate density estimation
            bandwidth2 (str, float, numpy.ndarray):
                The 'bandwidth**2' of the density estimation. Corresponds to
                the variance/covariance of the underlying Gaussian mixture
                module. Note that this value is typically provided as standard
                deviation, i.e. ``bandwidth == sqrt(bandwidth2)``. If string is
                provided, then the bandwidth are calculated automatically.
            batch_size (int, float):
                The number of samples allowed to be processed at the same time.
                Prevents memory overflows for large sample sizes.

        """
        self.samples = numpy.atleast_2d(samples)
        assert self.samples.ndim == 2
        self.dim = len(self.samples)

        if batch_size < self.samples.size:
            batch_size = self.samples.size
        self.stride = int(batch_size//self.samples.size)

        # the scale is taken from Scott-92.
        # The Scott factor is taken from scipy docs.
        if bandwidth2 in ("scott", "silverman"):
            qrange = numpy.quantile(self.samples, [0.25, 0.75], axis=1).ptp(axis=0)
            scale = numpy.min([numpy.std(samples, axis=1), qrange/1.34], axis=0)
            if bandwidth2 == "scott":
                scott_factor = self.samples.shape[1]**(-1./(self.dim+4))
            else:
                scott_factor = (self.samples.shape[1]*(self.dim+2)/4.)**(-1./(self.dim+4))
            bandwidth2 = numpy.diag(scale*scott_factor)**2

        else:
            bandwidth2 = numpy.asfarray(bandwidth2)
            if bandwidth2.ndim in (0, 1):
                bandwidth2 = bandwidth2*numpy.eye(self.dim)
        assert bandwidth2.shape == (self.dim, self.dim)
        self.bandwidth2 = bandwidth2
        self.L = numpy.linalg.cholesky(bandwidth2)
        self.Li = numpy.linalg.inv(self.L)

        Dist.__init__(self)

    def __len__(self):
        """Number of dimensions."""
        return self.dim


    @staticmethod
    def _kernel(z_loc):
        """The underlying density kernel."""
        return numpy.prod(numpy.e**(-z_loc**2/2.)/numpy.sqrt(2*numpy.pi), axis=-1)

    @batch_input
    def _pdf(self, x_loc):
        """Kernel density function."""
        out = numpy.zeros(x_loc.shape)

        # grid up every location to evaluate against every sample
        s, t = numpy.mgrid[:x_loc.shape[-1], :self.samples.shape[-1]]

        # The first dimension
        x_loc_ = x_loc[0, s]
        samples = self.samples[0, t]
        z_loc = (x_loc_-samples)[:, :, numpy.newaxis]*self.Li[0, 0]
        kernel0 = self._kernel(z_loc)/self.L[0, 0]
        out[0] = numpy.mean(kernel0, axis=-1)

        # Dimensions after the first
        for idx in range(1, len(self)):

            # grid up and normalize new samples
            x_loc_ = numpy.dstack([x_loc_, x_loc[idx, s]])
            samples = numpy.dstack([samples, self.samples[idx, t]])
            z_loc = numpy.dstack([z_loc, numpy.dot((x_loc_-samples), self.Li[idx, :idx+1])])

            # evaluate kernel
            kernel = self._kernel(z_loc)
            kernel *= (numpy.linalg.det(self.L[:idx, :idx])/
                    numpy.linalg.det(self.L[:idx+1, :idx+1]))
            out[idx] = numpy.mean(kernel, axis=-1)/numpy.mean(kernel0, axis=-1)

            # store kernel for next iteration
            kernel0 = kernel

        return out

    def _ikernel(self, z_loc):
        """The integrand of the underlying density kernel."""
        kernel = 1
        if z_loc.shape[-1] > 1:
            kernel = self._kernel(z_loc[:, :, :-1])
        return kernel*ndtr(z_loc[:, :, -1])

    @batch_input
    def _cdf(self, x_loc):
        """Forward mapping."""
        out = numpy.zeros(x_loc.shape)

        s, t = numpy.mgrid[:x_loc.shape[-1], :self.samples.shape[-1]]

        # The first dimension
        x_loc_ = x_loc[0, s]
        samples = self.samples[0, t]
        z_loc = (x_loc_-samples)[:, :, numpy.newaxis]*self.Li[0, 0]
        kernel0 = self._ikernel(z_loc)
        out[0] = numpy.mean(self._ikernel(z_loc), axis=-1)

        # Dimensions after the first
        for idx in range(1, len(self)):

            # grid up and normalize new samples
            x_loc_ = numpy.dstack([x_loc_, x_loc[idx, s]])
            samples = numpy.dstack([samples, self.samples[idx, t]])
            z_loc = numpy.dstack([z_loc, numpy.dot((x_loc_-samples), self.Li[idx, :idx+1])])

            # evaluate kernel
            out[idx] = (numpy.mean(self._ikernel(z_loc), axis=-1)/
                        numpy.mean(self._kernel(z_loc[:, :, :-1]), axis=-1))

        return out

    @batch_input
    def _ppf(self, u_loc):
        """Inverse mapping."""
        # speed up convergence considerable, by giving very good initial position.
        x0 = numpy.quantile(self.samples, u_loc[0])[numpy.newaxis]
        return approximate_inverse(self, u_loc, x0=x0, tol=1e-8)

    def _mom(self, k_loc):
        """Raw statistical moments."""
        out = [MvNormal._mom(self, k_loc, self.L, self.Li, sample)
               for sample in self.samples.T]
        return numpy.mean(out)

    def _lower(self):
        """Lower bounds."""
        return self.samples.min(axis=-1)-7.5*numpy.sum(self.L, axis=0)

    def _upper(self):
        """Upper bounds."""
        return self.samples.max(axis=-1)+7.5*numpy.sum(self.L, axis=0)
