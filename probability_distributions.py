from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.special import gammaln
from scipy.stats._multivariate import multi_rv_generic, multi_rv_frozen


_LOG_2PI = np.log(2 * np.pi)
_LOG_2 = np.log(2)
_LOG_PI = np.log(np.pi)


_doc_random_state = """\
random_state : None or int or np.random.RandomState instance, optional
    If int or RandomState, use it for drawing the random variates.
    If None (or np.random), the global np.random state is used.
    Default is None.
"""


class normal_invgamma_gen(multi_rv_generic):
    r"""
    A normal inverse gamma random variable.
    Methods
    -------
    ``pdf(x, sig2, loc=0, variance_scale=1, shape=1, scale=1)``
        Probability density function.
    ``logpdf(x, sig2, loc=0, variance_scale=1, shape=1, scale=1)``
        Log of the probability density function.
    ``rvs(loc=0, variance_scale=1, shape=1, scale=1, size=1, random_state=None)``
        Draw random samples from normal and inverse gamma distribution.
    ``mean(loc=0, variance_scale=1, shape=1, scale=1)``
        Mean of the random variates.
    ``mode(loc=0, variance_scale=1, shape=1, scale=1)``
        Mode of the random variates.
    Parameters
    ----------
    x : array
            One-dimensional array.
    sig2 : array
            One-dimensional array.
    %(_nig_doc_default_callparams)s
    %(_doc_random_state)s
    Alternatively, the object may be called (as a function) to fix the
    loc, variance_scale, shape, scale parameters, returning a "frozen"
    normal inverse gamma random variable:
    rv = normal_invgamma(loc=0, variance_scale=1, shape=1, scale=1)
        - Frozen object with the same methods but holding the given
          loc, variance_scale, shape, scale.
    Notes
    -----
    %(_nig_doc_callparams_note)s
    The probability density function for `normal_invgamma` is
    .. math::
        f(x,\sigma^2) = \frac {1} {\sigma\sqrt{2\pi\nu} } \, \frac{\beta^\alpha}
                        {\Gamma(\alpha)} \, \left( \frac{1}{\sigma^2} \right)^
                        {\alpha + 1}\exp \left( \frac { -\beta}{\sigma^2} -
                        \frac{(x - \mu)^2} {2\sigma^2\nu}  \right),
    where :math:`\mu` is the loc, :math:`\nu` the variance scale,
    :math:`\alpha` the shape, :math:`\beta` the scale and :math:`x` and
    `\sigma^2` takes values.
    .. versionadded:: 0.19.0
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import axes3d
    >>> from probability_distributions import normal_invgamma
    >>> x = np.linspace(0, 5, 10, endpoint=False)
    >>> sig2 = np.linspace(5, 10, 10, endpoint=False)
    >>> z = normal_invgamma.pdf(x, sig2, loc=2, variance_scale=3, shape=5, scale=2); z
    array([5.15655226e-06, 3.07181064e-06, 1.87273568e-06, 1.16664399e-06,
           7.41424234e-07, 4.79916138e-07, 3.15923431e-07, 2.11211796e-07,
           1.43229349e-07, 9.84088858e-08])
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    """
    def __init__(self, seed=None):
        super(normal_invgamma_gen, self).__init__(seed)

    def __call__(self, loc=0, variance_scale=1, shape=1, scale=1, seed=None):
        """
        Create a frozen normal inverse gamma distribution.
        See `normal_invgamma_frozen` for more information.
        """
        return normal_invgamma_frozen(loc, variance_scale, shape, scale,
                                      seed=seed)

    def _check_parameters(self, loc, variance_scale, shape, scale):
        if not np.isscalar(loc):
            raise ValueError("""loc must be a scalar""")
        if not np.isscalar(variance_scale) or variance_scale < 0:
            raise ValueError("""variance_scale must be a positive scalar""")
        if not np.isscalar(shape) or shape < 0:
            raise ValueError("""shape must be a positive scalar""")
        if not np.isscalar(scale) or scale < 0:
            raise ValueError("""scale must be a positive scalar""")
        return loc, variance_scale, shape, scale

    def _check_input(self, x, sig2):
        if x.ndim != 1:
            raise ValueError("""array must be one dimensional""")
        if sig2.ndim != 1:
            raise ValueError("""array must be one dimensional""")
        if any(sig2 < 0):
            raise ValueError("""array must consist of positive values,
                                as it represents variance""")
        return x, sig2

    def logpdf(self, x, sig2, loc=0, variance_scale=1, shape=1, scale=1):
        """
        Log of the Normal inverse gamma probability density function.
        Parameters
        ----------
        x : array
            One-dimensional array.
        sig2 : array
            One-dimensional array.
        %(_nig_doc_default_callparams)s
        Returns
        -------
        pdf : ndarray
            Log of the probability density function evaluated at `(x, sig2)`.
        """
        loc, variance_scale, shape, scale = self._check_parameters(
            loc, variance_scale, shape, scale)
        x, sig2 = self._check_input(x, sig2)
        Zinv = shape * np.log(scale) - gammaln(shape) - 0.5 * (np.log(variance_scale) + _LOG_2PI)
        out = Zinv - 0.5 * np.log(sig2) - (shape + 1.) * np.log(sig2) - scale /sig2 - 0.5 /(sig2 * variance_scale) * ( x -loc )**2
        return out

    def pdf(self, x, sig2, loc=0, variance_scale=1, shape=1, scale=1):
        """
        Normal inverse gamma probability density function.
        Parameters
        ----------
        x : array
            One-dimensional array.
        sig2 : array
            One-dimensional array.
        %(_nig_doc_default_callparams)s
        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at `(x, sig2)`.
        Notes
        -----
        %(_nig_doc_callparams_note)s
        """
        out = np.exp(self.logpdf(x, sig2, loc, variance_scale, shape, scale))
        return out

    def rvs(self, loc=0, variance_scale=1, shape=1, scale=1, size=1, random_state=None):
        """
        Draw random samples from normal and inverse gamma distributions.
        Parameters
        ----------
        %(_nig_doc_default_callparams)s
        size : integer, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s
        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `2`).
        Notes
        -----
        %(_nig_doc_callparams_note)s
        """
        loc, variance_scale, shape, scale = self._check_parameters(
            loc, variance_scale, shape, scale)
        random_state = self._get_random_state(random_state)

        sig2_rv = 1/ random_state.gamma(shape, scale, size)
        x_rv = random_state.normal(loc, np.sqrt(sig2_rv * variance_scale), size)
        return np.array(list(zip(x_rv, sig2_rv)))

    def mean(self, loc=0, variance_scale=1, shape=1, scale=1):
        """
        Compute the mean for input random variates.
        Parameters
        ----------
        %(_nig_doc_default_callparams)s
        Returns
        -------
        (x, s) : tuple of scalar values
            Mean of the random variates
        Notes
        -----
        %(_nig_doc_callparams_note)s
        """
        loc, variance_scale, shape, scale = self._check_parameters(
            loc, variance_scale, shape, scale)
        x_mean = loc
        if shape > 1:
            sig2_mean = scale / (shape - 1)
        else:
            sig2_mean = np.inf
        return x_mean, sig2_mean

    def mode(self, loc=0, variance_scale=1, shape=1, scale=1):
        """
        Compute the mode for input random variates.
        Parameters
        ----------
        %(_nig_doc_default_callparams)s
        Returns
        -------
        (x, s) : tuple of scalar values
            Mode of the random variates
        Notes
        -----
        %(_nig_doc_callparams_note)s
        """
        loc, variance_scale, shape, scale = self._check_parameters(
            loc, variance_scale, shape, scale)
        x_mode = loc
        sig2_mode = scale / (shape + 1)
        return x_mode, sig2_mode


normal_invgamma = normal_invgamma_gen()


class normal_invgamma_frozen(multi_rv_frozen):
    """
    Create a frozen normal inverse gamma distribution.
    Parameters
    ----------
    loc : float, optional
        Mean of the distribution (default zero)
    variance_scale : positive float, optional
        Scale on normal distribution prior (default one)
    shape : positive float, optional
        Shape of the distribution (default one)
    scale : positive float, optional
        Scale on inverse gamma distribution prior (default one)
    seed : None or int or np.random.RandomState instance, optional
        This parameter defines the RandomState object to use for drawing
        random variates.
        If None (or np.random), the global np.random state is used.
        If integer, it is used to seed the local RandomState instance
        Default is None.
    Examples
    --------
    When called with the default parameters, this will create a 2D random
    variable with loc 0, variance_scale 1, shape 1, scale 1:
    >>> from probability_distributions import normal_invgamma
    >>> r = normal_invgamma()
    >>> r.loc
    0
    >>> r.variance_scale
    1
    >>> r.shape
    1
    >>> r.scale
    1
    """

    def __init__(self, loc=0, variance_scale=1, shape=1, scale=1, seed=None):
        self._dist = normal_invgamma_gen(seed)
        self.loc, self.variance_scale, self.shape, self.scale = self._dist._check_parameters(
            loc, variance_scale, shape, scale)

    def logpdf(self, x, sig2):
        x, sig2 = self._dist._check_input(x, sig2)
        out = self._dist.logpdf(x, sig2, self.loc, self.variance_scale, self.shape, self.scale)
        return out

    def pdf(self, x, sig2):
        return np.exp(self.logpdf(x, sig2))

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.loc, self.variance_scale,
                              self.shape, self.scale, size, random_state)

    def mean(self):
        return self._dist.mean(self.loc, self.variance_scale, self.shape, self.scale)

    def mode(self):
        return self._dist.mode(self.loc, self.variance_scale, self.shape, self.scale)



if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d

    mu0 = 0
    beta = 2
    a = 5
    b = 6

    xs = np.linspace(-2, 2, 20, endpoint=False)
    sigma2s = np.linspace(0, .5, endpoint=False)

    xmat, sigma2mat = np.meshgrid(xs, sigma2s)

    zmat = np.empty(xmat.shape)
    for i in range(xmat.shape[1]):
        z = normal_invgamma.pdf(xmat[i], sigma2mat[i], loc=mu0, variance_scale=beta, shape=a, scale=b)
        zmat[i] = z

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xmat, sigma2mat, z)
    plt.show()
