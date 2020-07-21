import matplotlib as mpl
import matplotlib.pyplot as plt
# matplotlib.use("Agg")
import numpy as np
import healpy as hp
import healcorr
from astropy.table import Table
import pathlib
import emcee
import re
from scipy.optimize import minimize
from matplotlib.ticker import MaxNLocator
import matplotlib.transforms as transforms
# import seaborn as sns
from functools import partial
import warnings
from chainconsumer import ChainConsumer
from scipy.stats import binned_statistic, median_absolute_deviation as mad
import treecorr
import twopoint
import itertools
import copy
try:
    import tqdm
except ImportError:
    tqdm.tqdm = lambda x, *args, **kwargs: return x
    tqdm.trange = lambda *args, **kwargs: return range(*args)
import pickle
import contextlib
import os
import sys
lsssys_path = "/home/wagoner47/lss_sys"
if lsssys_path not in sys.path:
    sys.path.insert(0, lsssys_path)
import lsssys
from lsssys import config

# Note: when using other systematics or systematics with different file names,
# the following definitions may need to change
band_systematics = [
    syst + f"{band}" for syst, sys_dict in 
    config.sysmap_shortnames_2col_format.items() if "band" in 
    sys_dict["filename"] for band in config.bandlist]
other_systematics = [
    syst for syst, sys_dict in config.sysmap_shortnames_2col_format.items() if 
    "band" not in sys_dict["filename"]] + list(config.sysmap_shortnames.keys())
all_systematics = band_systematics + other_systematics
bs_pattern = "_"
bs_replace = r"\_"
all_delta_systematics_plot_names = [
    (fr"$\delta \left(\mathrm{{"
     fr"{this_sys[:-1].replace(bs_pattern, bs_replace)}}}"
     fr"\right)_{this_sys[-1]}$") for this_sys in 
    band_systematics] + [
        (fr"$\delta \left(\mathrm{{"
         fr"{this_sys.replace(bs_pattern, bs_replace)}}}\right)$") for this_sys 
        in other_systematics]
# (End need to change)

#---------------------------- Low level utilities -----------------------------#
def pairwise(iterable):
    """
    Taken directly from the :mod:`python.itertools` documentation examples

    This takes any iterable and returns a generator over successive pairs of
    elements in the iterable, i.e. s -> (s0,s1), (s1,s2), (s2,s3), ...

    :param iterable: Any iterable object with at least 2 elements
    :type iterable: iterable
    :return: Generator of tuples of the successive pairs of elements
    :rtype: generator
    """
    now, later = itertools.tee(copy.deepcopy(iterable))
    next(later, None)
    return zip(now, later)

@contextlib.contextmanager
def update_env(*remove, **update):
    """
    This is a contextmanager to temporarily update the environment variables.
    
    The positional arguments (`remove`) are items that should be removed from
    the current environment variables.
    
    The keyword arguments (`update`) are items that should be added or changed
    in the current environment variables.
    """
    orig_env = copy.deepcopy(os.environ)
    try:
        [os.environ.pop(r) for r in remove]
        os.environ.update(update)
        yield
    finally:
        os.environ = copy.deepcopy(orig_env)

#------------------------------ Setup function(s) -----------------------------#
def read_sysmap(sys_shortname, nside, fracgood, minfrac=0.001):
    """
    A function for reading in the systematics map at a given resolution, given 
    that it may not be stored at that resolution.
    
    :param sys_shortname: One of the short names for a systematic map from 
        :mod:`lsssys.config`
    :type sys_shortname: ``str``
    :param nside: The desired resolution of the map. Will try to read the map at 
        this resolution first, and then try reading at the default resolution 
        and then up/degrading as needed.
    :type nside: ``int``
    :param fracgood: The pixel coverage map, which will be added to the 
        systematics map object to help with changing the resolution. This should 
        be given at the default resolution (nside=4096), as it can easily be 
        degraded but not upgraded
    :type fracgood: array-like ``float``
    :param minfrac: The minimum pixel coverage fraction to keep. Default 0.001 
        (to match the default in :func:`lsssys.Map.degrade`)
    :type minfrac: ``float``, optional
    :return: The systematics map object at the desired resolution
    :rtype: :class:`lsssys.SysMap`
    """
    assert hp.npix2nside(len(fracgood)) == 4096, ("Wrong resolution for "
                                                  "'fracgood'")
    fracdet = np.clip(fracgood, 0.0, None)
    assert hp.isnsideok(nside), "Invalid 'nside' parameter"
    if nside == 4096:
        # This is the default resolution, don't need to do anything else
        smap = lsssys.SysMap(sys_shortname)
        smap.addmask(fracdet < minfrac, fracdet)
        return smap
    band = None
    if sys_shortname in band_systematics:
        sys_dict = config.sysmap_shortnames_2col_format[sys_shortname[:-1]]
        band = sys_shortname[-1]
    elif sys_shortname in other_systematics:
        if sys_shortname in config.sysmap_shortnames_2col_format.keys():
            sys_dict = config.sysmap_shortnames_2col_format[sys_shortname]
        else:
            sys_dict = config.sysmap_shortnames[sys_shortname]
    else:
        raise ValueError("Unrecognized 'sys_shortname'")
    if "path" in sys_dict:
        fname = sys_dict["path"]
    else:
        fname = ""
    fname = fname + sys_dict["filename"].replace("4096", str(nside))
    fname = fname.format(band=band)
    try:
        smap = lsssys.SysMap(
            fname, nside, forcefileinput=True, fileinputnest=False)
    except (FileNotFoundError, IOError):
        smap = lsssys.SysMap(sys_shortname)
    smap.label = sys_shortname
    fracdet_final = hp.ud_grade(fracdet, nside)
    if smap.nside == 4096:
        # Add the fracgood without masking, change the resolution, and save for 
        # later
        smap.addmask(fracdet < 0.0, fracdet)
        smap.degrade(nside, True, weightedmean=True)
        smap.save(fname, pixcolname="PIXEL", valuecolname="SIGNAL")
    # Mask based on fracdet_final
    smap.addmask(fracdet_final < minfrac, fracdet_final)
    return smap

#---------------------------- Model and likelihood ----------------------------#
def delta_sys_linear_alpha(params, delta_sys):
    """
    Calculate the correction due to individual systematics maps with the linear
    model

    :param params: The model parameters to use. This may be of size
        ``len(delta_sys)`` with every entry being a map coefficient, or size
        ``len(delta_sys)+1`` with the 0th element being the variance of the true
        fluctuations, which is ignored
    :type params: array-like of ``float`` (``Nmaps``,) or (``Nmaps+1``,)
    :param delta_sys: The standardized systematics. This should be 2D, with the
        0th axis correpsonding to individual maps and the 1st axis to
        individual pixels
    :type delta_sys: 2D array-like of ``float`` (``Nmaps``,``Npix``)
    :return: The estimated correction to the density fluctuations due to each
        systematic on each pixel
    :rtype: :class:`numpy.ndarray` of ``float`` (``Nmaps``,``Npix``)
    :raises ValueError: If length of ``params`` and length of 0th axis of
        ``delta_sys`` do not match properly (as described above)
    """
    delta_sys_ = np.atleast_2d(delta_sys)
    if (delta_sys_.shape[0] != np.size(params)
          and delta_sys_.shape[0] != np.size(params) - 1):
        raise ValueError(
            "Mismatch between delta_sys ({}) and params ({})".format(
                delta_sys_.shape[0], np.size(params)))
    if delta_sys_.shape[0] == len(params) - 1:
        a_alpha = np.atleast_1d(params).flatten()[1:]
    else:
        a_alpha = np.atleast_1d(params).flatten()
    return a_alpha[:,None] * delta_sys_

def delta_sys_linear(params, delta_sys):
    """
    Calculate the correction due to systematics with the linear model using the
    given parameters

    :param params: The model parameters to use. This should be of size
        ``len(delta_sys)`` with all entries being the coefficients, or size
        ``len(delta_sys)+1`` with the 0th element being the variance of true
        fluctuations, which is ignored
    :type params: array-like of ``float`` (``Nmaps``,) or (``Nmaps+1``,)
    :param delta_sys: The standardized systematics. This should be 2D, with the
        0th axis correpsonding to individual maps and the 1st axis to
        individual pixels
    :type delta_sys: 2D array-like of ``float`` (``Nmaps``,``Npix``)
    :return: The estimated correction to the density fluctuations due to
        systematics on each pixel
    :rtype: :class:`numpy.ndarray` of ``float`` (``Npix``,)
    :raises ValueError: If length of ``params`` and length of 0th axis of
        ``delta_sys`` do not match properly (as described above)
    """
    return np.sum(delta_sys_linear_alpha(params, delta_sys), axis=0)

def linear_weights(params, delta_sys):
    """
    Calculate the systematics weights from the linear model with the given
    parameters. The weight is
    :math:`\frac{1}{1 + \delta_{\varepsilon, \mathrm{lin}}}`

    :param params: The model parameters to use. This should be of size
        ``len(delta_sys)`` with all entries being the coefficients, or size
        ``len(delta_sys)+1`` with the 0th element being the variance of true
        fluctuations, which is ignored
    :type params: array-like of ``float`` (``Nmaps``,) or (``Nmaps+1``,)
    :param delta_sys: The standardized systematics. This should be 2D, with the
        0th axis correpsonding to individual maps and the 1st axis to
        individual pixels (or galaxies, as these do not need to be unique)
    :type delta_sys: 2D array-like of ``float`` (``Nmaps``,``N``)
    :return: The weight for each set of systematics
    :rtype: :class:`numpy.ndarray` of ``float`` (``N``,)
    :raises ValueError: If length of ``params`` and length of 0th axis of
        ``delta_sys`` do not match properly (as described above)
    """
    return 1. / (1. + delta_sys_linear(params, delta_sys))

def lnlike(theta, delta_sys, delta_obs):
    """
    The log-likelihood function for my linear model.

    This does not include the prior. The data is expected to be masked to only
    include 'good' pixels

    :param theta: The likelihood parameters. The 0th element is the variance on
        the true density fluctuations, and the remaining ``Nmaps`` elements are
        the coefficients of the maps
    :type theta: array-like of ``float`` (``Nmaps+1``,)
    :param delta_sys: The standardized systematics. This should be 2D, with the
        0th axis correpsonding to individual maps and the 1st axis to
        individual pixels
    :type delta_sys: 2D array-like of ``float`` (``Nmaps``, ``Npix``)
    :param delta_obs: The observed density fluctuations by pixel
    :type delta_obs: array-like of ``float`` (``Npix``,)
    :return: The log-likelihood of the data given the parameters
    :rtype: ``float``
    :raises ValueError: If the shape of ``delta_sys`` is not
        (``len(theta)-1``, ``len(delta)``)
    """
    ln_sigmag2 = np.log(theta[0])
    one_over_sigmag2 = 1. / theta[0]
    if len(delta_sys) != len(theta) - 1:
        raise ValueError("Invalid length for 0th axis of delta_sys")
    if not all([len(delta_sysi) == len(delta_obs) for delta_sysi in delta_sys]):
        raise ValueError("Invalid length for 1st axis of delta_sys")
    delta_elin = delta_sys_linear(theta, delta_sys)
    n_pix = len(delta_obs)
    return -0.5 * (n_pix * ln_sigmag2 + one_over_sigmag2
                   * np.sum((np.asanyarray(delta_obs) - delta_elin)**2))

def lnprior(theta):
    """
    The log prior probability for the linear model

    Currently, the only thing the prior checks is that ``ln_sigmag2`` is not
    infinite

    :param theta: The likelihood parameters. The 0th element should be the log
        of the variance of the true density fluctuations, and the remaining
        elements should be the map coefficients
    :type theta: array-like of ``float``
    :return: Either 0 if the parameters are allowed or negative infinity
        otherwise
    :rtype: ``float``
    """
    if theta[0] > 0.0:
        return 0
    return -np.inf

def lnprob(theta, delta_sys, delta_obs):
    """
    The full log probability for the linear model

    This includes the log-likelihood as well as the prior

    :param theta: The likelihood parameters. The 0th element is the variance on
        the true density fluctuations, and the remaining ``Nmaps`` elements are
        the coefficients of the maps
    :type theta: array-like of ``float`` (``Nmaps+1``,)
    :param delta_sys: The standardized systematics. This should be 2D, with the
        0th axis correpsonding to individual maps and the 1st axis to
        individual pixels
    :type delta_sys: 2D array-like of ``float`` (``Nmaps``, ``Npix``)
    :param delta_obs: The observed density fluctuations by pixel
    :type delta_obs: array-like of ``float`` (``Npix``,)
    :return: The log-likelihood of the data given the parameters
    :rtype: ``float``
    :raises ValueError: If the shape of ``delta_sys`` is not
        (``len(theta)-1``, ``len(delta)``)
    """
    lnp = lnprior(theta)
    if not np.isfinite(lnp):
        return -np.inf
    log_like = lnlike(theta, delta_sys, delta_obs)
    return lnp + log_like

def m_matrix(delta_sys):
    """
    Builds the matrix needed for algebraic minimization, and also helps compute
    the gradient and Hessian matrix

    :param delta_sys: The standardized systematics. The 0th axis should
        correspond to individual maps, and the 1st axis to individual pixels.
        The systematics should only include good pixels
    :type delta_sys: 2D array-like of ``float`` (``Nmaps``, ``Npix``)
    :return: A matrix where element (alpha, beta) is the sum over pixels of
        the alpha systematic times the beta systematic. The shape is
        (``Nmaps``, ``Nmaps``)
    :rtype: 2D :class:`numpy.ndarray` of ``float``
    """
    return np.dot(delta_sys, np.transpose(delta_sys))

def b_vector(delta_sys, delta_obs):
    """
    Builds the vector needed for algebraic minimization, and also helps compute
    the gradient and Hessian matrix

    :param delta_sys: The standardized systematics. The 0th axis should
        correspond to individual maps, and the 1st axis to individual pixels.
        The systematics should only include good pixels
    :type delta_sys: 2D array-like of ``float`` (``Nmaps``, ``Npix``)
    :param delta_obs: The observed density fluctuations for each pixel, only
        including good pixels
    :type delta_obs: array-like of ``float`` (``Npix``,)
    :return: A vector where element alpha is the sum over pixels of the alpha
        systematic times the observed density fluctuations. The shape is
        (``Nmaps``,)
    :rtype: 1D :class:`numpy.ndarray` of ``float``
    """
    return np.dot(delta_sys, delta_obs)

def grad_lnprob(theta, delta_sys, delta_obs):
    """
    Compute the gradient of the log probability for the linear model

    This may be useful when calling :func:`scipy.optimize.minimize` for
    stabilizing the minimization result

    :param theta: The likelihood parameters. The 0th element is the variance on
        the true density fluctuations, and the remaining ``Nmaps`` elements are
        the coefficients of the maps
    :type theta: array-like of ``float`` (``Nmaps+1``,)
    :param delta_sys: The standardized systematics. This should be 2D, with the
        0th axis correpsonding to individual maps and the 1st axis to
        individual pixels
    :type delta_sys: 2D array-like of ``float`` (``Nmaps``, ``Npix``)
    :param delta_obs: The observed density fluctuations by pixel
    :type delta_obs: array-like of ``float`` (``Npix``,)
    :return: The gradient of the log probability
    :rtype: :class:`numpy.ndarray` of ``float`` (``Nmaps+1``,)
    :raises ValueError: If the shape of ``delta_sys`` is not
        (``len(theta)-1``, ``len(delta)``)
    """
    one_over_sigmag2 = 1. / np.exp(theta[0])
    if len(delta_sys) != len(theta) - 1:
        raise ValueError("Invalid length for 0th axis of delta_sys")
    if not all([len(delta_sysi) == len(delta_obs) for delta_sysi in delta_sys]):
        raise ValueError("Invalid length for 1st axis of delta_sys")
    delta_t = delta_obs - delta_sys_linear(theta, delta_sys)
    grad_as = one_over_sigmag2 * np.dot(delta_sys, delta_t)
    grad_sigmag2 = (0.5 * one_over_sigmag2
                    * (-len(delta_t) + one_over_sigmag2 * np.dot(
                        delta_t, delta_t)))
    return np.insert(grad_as, 0, grad_sigmag2)

def hess_lnprob(theta, delta_sys, delta_obs):
    """
    Compute the Hessian matrix (matrix of second derivatives) of the log
    probability for the linear model

    This may be useful when calling :func:`scipy.optimize.minimize` for
    stabilizing the minimization result

    :param theta: The likelihood parameters. The 0th element is the variance on
        the true density fluctuations, and the remaining ``Nmaps`` elements are
        the coefficients of the maps
    :type theta: array-like of ``float`` (``Nmaps+1``,)
    :param delta_sys: The standardized systematics. This should be 2D, with the
        0th axis correpsonding to individual maps and the 1st axis to
        individual pixels
    :type delta_sys: 2D array-like of ``float`` (``Nmaps``, ``Npix``)
    :param delta_obs: The observed density fluctuations by pixel
    :type delta_obs: array-like of ``float`` (``Npix``,)
    :return: The Hessian of the log probability
    :rtype: :class:`numpy.ndarray` of ``float`` (``Nmaps+1``,``Nmaps+1``)
    :raises ValueError: If the shape of ``delta_sys`` is not
        (``len(theta)-1``, ``len(delta)``)
    """
    one_over_sigmag2 = 1. / theta[0]
    if len(delta_sys) != len(theta) - 1:
        raise ValueError("Invalid length for 0th axis of delta_sys")
    if not all([len(delta_sysi) == len(delta_obs) for delta_sysi in delta_sys]):
        raise ValueError("Invalid length for 1st axis of delta_sys")
    delta_t = delta_obs - delta_sys_linear(theta, delta_sys)
    n_pix = delta_obs_m_elin.size
    hess_mat = np.zeros((len(theta), len(theta)))
    hess_mat[0,1:] = -one_over_sigmag2**2 * np.dot(delta_sys, delta_t)
    hess_mat += hess_mat.T
    hess_mat[0,0] = (one_over_sigmag2**2
                     * (-0.5 * len(delta_t) - one_over_sigmag2 * np.dot(
                         delta_t, delta_t)))
    hess_mat[1:,1:] = -one_over_sigmag2 * m_matrix(delta_sys)
    return hess_mat

def map_chisq(theta, delta_sys, delta_obs, c_inv):
    """
    Get the :math:`\chi^2` of the estimated true overdensity map assuming the
    given inverse covariance matrix

    :param theta: Parameters at which to calculate the chi-squared
    :type theta: 1D array-like ``float`` (``Nmaps``,)
    :param delta_sys: Systematics fluctuations, where the 0th axis correpsonds
        to maps and the 1st axis to pixels
    :type delta_sys: 2D array-like ``float`` (``Nmaps``, ``Npix``)
    :param delta_obs: Observed galaxy overdensity
    :type delta_obs: 1D array-like ``float`` (``Npix``,)
    :param c_inv: The inverse covariance matrix of the true overdensity map
    :type c_inv: 2D array-like ``float`` (``Npix``, ``Npix``)
    :return: The chi-squared of the maps assuming the given parameters
    :rtype: ``float``
    :raises ValueError: If the dimensions of ``theta``, ``delta_sys``,
        ``delta_obs``, and ``c_inv`` do not match
    """
    if len(delta_sys) != len(theta):
        raise ValueError("Invalid length for 0th axis of delta_sys")
    if not all([len(delta_sysi) == len(delta_obs) for delta_sysi in delta_sys]):
        raise ValueError("Invalid length for 1st axis of delta_sys")
    if np.shape(c_inv) != (len(delta_obs), len(delta_obs)):
        raise ValueError("Invalid shape for inverse covariance matrix")
    delta_t = np.asanyarray(delta_obs) - delta_sys_linear(theta, delta_sys)
    return np.dot(delta_t.T, np.dot(c_inv, delta_t))

def gauss_map_lnprior_const_cov(theta):
    """
    The log prior likelihood for the Gaussian true overdensity map assuming a
    constant covariance matrix

    This currently does not have any conditions, so the output is always 0

    :param theta: The parameters to check in the prior. It is not actually used
        in the current implementation
    :type theta: 1D array-like ``float``
    :return: The log prior likelihood of the parameters. Currently this is
        always zero
    :rtype: ``float``
    """
    return 0.0

def gauss_map_lnprob_const_cov(theta, delta_sys, delta_obs, c_inv):
    """
    The log-probability for the true overdensity map assuming a constant
    covariance matrix and a Gaussian distribution. This is not technically
    correct, but it works to prevent overcorrecting for the systematics.

    Note that this includes the prior already

    :param theta: The parameters at which to calculate
    :type theta: 1D array-like ``float`` (``Nmaps``,)
    :param delta_sys: The systematics fluctuations. The 0th axis should
        correspond to maps, and the 1st axis to pixels
    :type delta_sys: 2D array-like ``float`` (``Nmaps``, ``Npix``)
    :param delta_obs: The observed galaxy overdensity
    :type delta_obs: 1D array-like ``float`` (``Npix``,)
    :param c_inv: The inverse covariance matrix of the map
    :type c_inv: 2D array-like ``float`` (``Npix``, ``Npix``)
    :return: The log-probability of the data given the parameters
    :rtype: ``float``
    :raises ValueError: If the dimensions of ``theta``, ``delta_sys``,
        ``delta_obs``, and ``c_inv`` do not match
    """
    lnp = gauss_map_lnprior_const_cov(theta)
    if not np.isfinite(lnp):
        return -np.inf
    return -0.5 * map_chisq(theta, delta_sys, delta_obs, c_inv)

class GaussMapClass(object):
    """
    A class to encapsulate the Gaussian map log-probability function and the
    data that it needs in a way that is faster to use with multiprocessing.Pool
    """
    delta_sys = None
    delta_obs = None
    c_inv = None

    @classmethod
    def set_data(cls, sys_delta, obs_delta, inv_c):
        cls.delta_sys = sys_delta
        cls.delta_obs = obs_delta
        cls.c_inv = inv_c

    @classmethod
    def lnprob(cls, theta):
        return gauss_map_lnprob_const_cov(
            theta, cls.delta_sys, cls.delta_obs, cls.c_inv)

def const_plus_power_law(x, a, p, b):
    """
    This function is the power law to be fit to correlation functions for the
    covariance matrix. It is a constant for ``x`` near zero, and then a power
    law for non-zero ``x``

    :param x: The values at which to evaluate the function
    :type x: ``float`` or array-like ``float``
    :param a: The power law amplitude
    :type a: ``float``
    :param p: The power of the power law
    :type p: ``float``
    :param b: The constant to use for ``x`` near zero
    :type b: ``float``
    :return: The power law evaluated at ``x`` with the given parameters
    :rtype: ``float`` or :class:`numpy.ndarray` of ``float`` with same shape
        as ``x``
    """
    y = np.copy(x)
    shape = y.shape
    y = y.flatten()
    x_zero = np.isclose(y, 0.0)
    y[~x_zero] = a * y[~x_zero]**-p
    y[x_zero] = b
    if np.sum(shape, dtype=int) == 0:
        return y.item()
    return y.reshape(shape)

def corr_func_power_law(theta_eval, fit_params, theta_max):
    """
    This is the full power law function for building the covariance matrix from
    the correlation function. It takes the fit parameters as well as the maximum
    angular separation, beyond which the function evaluates to zero

    :param theta_eval: The angular separations at which to evaluate
    :type theta_eval: ``float`` or array-like ``float``
    :param fit_params: The best fit parameters for the power law. The elements
        should be ``a`` (the power law amplitude), ``p`` (the power of the power
        law), ``b`` (the constant for approximately zero angular separations)
    :type fit_params: 1D array-like ``float`` (3,)
    :param theta_max: The maximum angular separation. For angular separations
        larger than this value, the function evaluates to zero rather than using
        the power law fit
    :type theta_max: ``float``
    :return: The modified power law function evaluated at ``theta_eval``
    :rtype: ``float`` or :class:`numpy.ndarray` ``float`` with same shape as
        ``theta_eval``
    """
    shape = np.shape(theta_eval)
    x = np.ravel(theta_eval)
    y = np.zeros_like(x)
    y[x <= theta_max] = const_plus_power_law(x[x <= theta_max], *fit_params)
    if np.sum(shape, dtype=int) == 0:
        return y.item()
    return y.reshape(shape)

#------------------------------- Fitting+chains -------------------------------#
def fit_algebraically(delta_sys, delta_obs):
    """
    Perform an algebraic minimization

    :param delta_sys: The standardized systematics. The 0th axis should
        correspond to individual maps, and the 1st axis to individual pixels.
        The systematics should only include good pixels
    :type delta_sys: 2D array-like of ``float`` (``Nmaps``, ``Npix``)
    :param delta_obs: The observed density fluctuations for each pixel, only
        including good pixels
    :type delta_obs: array-like of ``float`` (``Npix``,)
    :return: The parameters that minimize the likelihood, with shape
        (``Nmaps+1``,)
    :rtype: 1D :class:`numpy.ndarray` of ``float``
    """
    m_mat = m_matrix(delta_sys)
    b_vec = b_vector(delta_sys, delta_obs)
    a_vec = np.dot(np.linalg.inv(m_mat), b_vec)
    return np.insert(
        a_vec, 0,
        np.sum((delta_obs - delta_sys_linear(a_vec, delta_sys))**2)
        / len(delta_obs))

def algebraic_parameter_covariance(delta_sys, delta_obs):
    """
    Get the parameter covariance matrix from the algebraic minimization

    :param delta_sys: The standardized systematics. The 0th axis should
        correspond to individual maps, and the 1st axis to individual pixels.
        The systematics should only include good pixels
    :type delta_sys: 2D array-like of ``float`` (``Nmaps``, ``Npix``)
    :param delta_obs: The observed density fluctuations for each pixel, only
        including good pixels
    :type delta_obs: array-like of ``float`` (``Npix``,)
    :return: The parameter covariance matrix for the likelihood, with shape
        (``Nmaps+1``,``Nmaps+1``)
    :rtype: 2D :class:`numpy.ndarray` of ``float``
    """
    sigmag2 = fit_algebraically(delta_sys, delta_obs)[0]
    cov_mat = np.zeros((len(delta_sys)+1, len(delta_sys)+1))
    cov_mat[0,0] = 2 * sigmag2**2 / len(delta_obs)
    cov_mat[1:,1:] = sigmag2 * np.linalg.inv(m_matrix(delta_sys))
    return cov_mat

@py_timer.detailed_time
def fit_systematics(delta_sys, delta_obs, nsteps, nwalkers=None,
                    chain_file=None, **kwargs):
    """
    This function performs the fit with no pixel covariances

    It first fits analytically, and then uses :mod:`emcee` to perform an MCMC
    from the analytic maximum likelihood parameters obtained in the first fit.
    The sampler is then returned for future use.

    The minimum number of walkers recommended by :mod:`emcee` (2 times the
    number of parameters) is used unless supplied by the ``nwalkers`` argument.

    By providing the optional parameter ``chain_file``, the MCMC results can
    also be stored in a file to be used later. This is helpful in case of errors
    when running or if extending the chains. In order to read the chain and
    remove the burn-in properly, the file type should support metadata
    information. Furthermore, if ``chain_file`` already exists and the number
    of walkers is consistent, the chain is read in and the sampler continues
    from the last location of the walkers in the file.

    Keyword arguments are passed to the :meth:`astropy.table.Table.write`
    method, if ``chain_file`` is not ``None``, and the
    :meth:`astropy.table.Table.read` method if ``chain_file`` already exists.

    :param delta_sys: The standardized systematics. The 0th axis should
        correspond to individual maps, and the 1st axis to individual pixels.
    :type delta_sys: 2D array-like of ``float`` (``Nmaps``, ``Npix``)
    :param delta_obs: The observed density fluctuations for each pixel
    :type delta_obs: array-like of ``float`` (``Npix``,)
    :param nsteps: The number of steps to take with the MCMC. More steps can be
        taken afterwards with the returned sampler as needed. Note that all
        steps are saved in the output file (if ``chain_file`` is not ``None``),
        without a burn-in phase being removed. If ``chain_file`` is not ``None``
        and already exists, and has the same numer of parameters and walkers as
        the current call, this will take ``nsteps`` steps starting from the
        last position saved
    :type nsteps: ``int``
    :param chain_file: If given, this is a file in which to save the MCMC chain
        (including the log-probability at each step). If it already exists and
        the number of parameters and number of walkers match the current call,
        the sampler will continue running from the end of the stored chain.
        Otherwise, a new fit will be run, and it will attempt to overwrite the
        already existing file. Default ``None``
    :type chain_file: ``str``, :class:`os.PathLike`, or ``NoneType``, optional
    :return: The MCMC sampler (for later use)
    :rtype: :class:`emcee.EnsembleSampler`
    :raises ValueError: If the length of the 1st axis of ``delta_sys`` does not
        match the length of ``delta_obs``
    """
    if not all([len(delta_sysi) == len(delta_obs) for delta_sysi in delta_sys]):
        raise ValueError("Shape mismatch for delta_obs and delta_sys")
    nparams = len(delta_sys) + 1
    nsteps = int(nsteps)
    if nsteps < 1:
        raise ValueError("Invalid nsteps: {}. What were you thinking?".format(
            nsteps))
    try:
        nwalkers = int(nwalkers)
    except ValueError:
        warnings.warn("Ignoring invalid nwalkers and using default")
        nwalkers = 2 * nparams
    except TypeError:
        nwalkers = 2 * nparams
    finally:
        if nwalkers < 2 * nparams:
            warnings.warn("Choosing to live dangerously because nwalkers is"
                          " too small")
            danger = True
        else:
            danger = False
    if chain_file is not None:
        chain_file = pathlib.Path(chain_file).expanduser().resolve()
        chain_file.parent.mkdir(parents=True, exist_ok=True)
        if chain_file.exists():
            copy_of_kwargs = kwargs.copy()
            existing_chain, existing_lnprob = read_chain(
                chain_file, **copy_of_kwargs)
            if (existing_chain.shape[0] != nwalkers or
                existing_chain.shape[2] != nparams):
                if not kwargs.get("overwrite", True):
                    raise ValueError(
                        "Chain file already exists, does not match current set"
                        " up, and is not going to be overwritten")
                existing_chain = None
                existing_lnprob = None
        else:
            existing_chain = None
            existing_lnprob = None
    sampler = emcee.EnsembleSampler(
        nwalkers, nparams, partial(
            lnlike, delta_sys=delta_sys, delta_obs=delta_obs),
        live_dangerously=danger)
    if existing_chain is None:
        init_guess = fit_algebraically(delta_sys, delta_obs)
        ball_width = np.sqrt(
            np.diag(algebraic_parameter_covariance(delta_sys, delta_obs)))
        init_step = np.array(
            [init_guess + ball_width * np.random.randn(nparams) for _ in
             range(nwalkers)])
    else:
        sampler._chain = existing_chain
        sampler._lnprob = existing_lnprob
        init_step = existing_chain[:,-1,:]
    sampler.run_mcmc(init_step, nsteps)
    if chain_file is not None:
        write_chain(
            sampler, chain_file, ["ln_sigmag_2"] + [
                "a_{}".format(i) for i in range(len(delta_sys))], **kwargs)
    return sampler

def fit_corr_func_power_law(theta, wtheta, theta_max=None):
    """
    Fit a power law to the correlation function at angles below ``theta_max``

    :param theta: The angular separations at which the correlation function is
        calculated
    :type theta: 1D array-like ``float`` (``Nbins``,)
    :param wtheta: The calculated correlation function to be fit
    :type wtheta: 1D array-like ``float`` (``Nbins``,)
    :param theta_max: The maximum angluar separation to be fit. If ``None``
        (default), the fit is performed for all angular separations in ``theta``
    :type theta_max: ``float`` or ``NoneType``, optional
    :return: The results from :fun:`scipy.optimize.minimize` for a power law fit
        to ``theta`` and ``wtheta``
    :rtype: :class:`scipy.optimize.OptimizeResult`
    """
    x = np.ravel(theta)
    x_order = np.argsort(x)
    y = np.ravel(wtheta)[x_order]
    x = x[x_order]
    if theta_max is not None:
        theta_filt = (x <= theta_max)
        y = y[theta_filt]
        x = x[theta_filt]
        del theta_filt
    nonzero_x = ~np.isclose(x, 0.0)
    logsafe_xy = np.logical_and(nonzero_x, y > 0.0)
    init_guess_p = -np.median(np.diff(np.log(y[logsafe_xy]))
                              / np.diff(np.log(x[logsafe_xy])))
    x_p = np.median(x[nonzero_x])
    arg_median_x = arg_median(x[nonzero_x])
    init_guess_a = y[nonzero_x][arg_median_x].mean()
    init_guess_b = y[~nonzero_x].mean()
    nll = lambda params, x_data=(x / x_p), y_data=y: np.sum(
        (y_data - const_plus_power_law(x_data, *params))**2)
    print("Run and return minimizer", flush=True)
    res = minimize(nll, [init_guess_a, init_guess_p, init_guess_b])
    # Do a transformation on the slope so that we don't have to use x / x_p
    # when calling
    res.x[0] *= x_p**res.x[1]
    return res

def corr_func_power_law_with_fit(
        theta_eval, theta_grid, wtheta_grid, theta_max):
    """
    Perform the power law fit to the correlation function, and then evaluate
    the modified power law at ``theta_eval``

    :param theta_eval: The angular separations at which to evaluate
    :type theta_eval: ``float`` or array-like ``float``
    :param theta_grid: The angular separations at which the correlation
        function is already calculated
    :type theta_grid: 1D array-like ``float`` (``Nbins``,)
    :param wtheta_grid: The pre-calculated correlation function to be fit
    :type wtheta_grid: 1D array-like ``float`` (``Nbins``,)
    :param theta_max: The maximum angular separation. For angular separations
        larger than this value, the power law is not fit and the function
        evaluates to zero rather than using the power law fit
    :type theta_max: ``float``
    :return: The modified power law function evaluated at ``theta_eval`` with
        the internally fitted parameters
    :rtype: ``float`` or :class:`numpy.ndarray` ``float`` with same shape as
        ``theta_eval``
    """
    res = fit_corr_func_power_law(theta_grid, wtheta_grid, theta_max)
    print("Fit finished, return correlation function evaluation", flush=True)
    return corr_func_power_law(theta_eval, res.x, theta_max)

@py_timer.detailed_time
def fit_constant_covariance(delta_sys, delta_obs, c_inv, nsteps, nwalkers=None,
                            nthreads=None, chain_file=None, **kwargs):
    """
    This function performs the fit for the Gaussian case with a constant
    non-diagonal covariance matrix

    It first fits with :func:`scipy.optimize.minimize`, and then uses
    :mod:`emcee` to perform an MCMC from the maximum likelihood parameters
    obtained in the first fit. The sampler is then returned for future use.

    The minimum number of walkers recommended by :mod:`emcee` (2 times the
    number of parameters) is used unless supplied by the ``nwalkers`` argument.

    By providing the optional parameter ``chain_file``, the MCMC results can
    also be stored in a file to be used later. This is helpful in case of errors
    when running or if extending the chains. In order to read the chain and
    remove the burn-in properly, the file type should support metadata
    information.

    Keyword arguments are passed to the :meth:`astropy.table.Table.write`
    method, if ``chain_file`` is not ``None``.

    :param delta_sys: The standardized systematics. The 0th axis should
        correspond to individual maps, and the 1st axis to individual pixels.
    :type delta_sys: 2D array-like of ``float`` (``Nmaps``, ``Npix``)
    :param delta_obs: The observed density fluctuations for each pixel
    :type delta_obs: array-like of ``float`` (``Npix``,)
    :param c_inv: The inverse of the pixel covariance matrix
    :type c_inv: 2D array-like of ``float`` (``Npix``, ``Npix``)
    :param nsteps: The number of steps to take with the MCMC. More steps can be
        taken afterwards with the returned sampler as needed. Note that all
        steps are saved in the output file (if ``chain_file`` is not ``None``),
        without a burn-in phase being removed. If ``chain_file`` is not ``None``
        and already exists, and has the same numer of parameters and walkers as
        the current call, this will take ``nsteps`` steps starting from the
        last position saved
    :type nsteps: ``int``
    :param nwalkers: The number of walkers to use. If ``None`` (default), this
        is set to 2 times the number of parameters
    :type nwalkers: ``int`` or ``NoneType``, optional
    :param nthreads: If given, this tells how many threads should be used to run
        MCMC in parallel. If ``None`` (default), the MCMC will not be run in
        parallel
    :type nthreads: ``int`` or ``NoneType``, optional
    :param chain_file: If given, this is a file in which to save the MCMC chain
        (including the log-probability at each step). If it already exists and
        the number of parameters and number of walkers match the current call,
        the sampler will continue running from the end of the stored chain.
        Otherwise, a new fit will be run, and it will attempt to overwrite the
        already existing file. Default ``None``
    :type chain_file: ``str``, :class:`os.PathLike`, or ``NoneType``, optional
    :return: The MCMC sampler (for later use)
    :rtype: :class:`emcee.EnsembleSampler`
    :raises ValueError: If the length of the 1st axis of ``delta_sys`` does not
        match the length of ``delta_obs``
    """
    if not all([len(delta_sysi) == len(delta_obs) for delta_sysi in delta_sys]):
        raise ValueError("Shape mismatch for delta_obs and delta_sys")
    if np.shape(c_inv) != (len(delta_obs), len(delta_obs)):
        raise ValueError("Shape mismatch for c_inv and delta_obs")
    nparams = len(delta_sys)
    nsteps = int(nsteps)
    if nsteps < 1:
        raise ValueError("Invalid nsteps: {}. What were you thinking?".format(
            nsteps))
    try:
        nwalkers = int(nwalkers)
    except ValueError:
        warnings.warn("Ignoring invalid nwalkers and using default")
        nwalkers = 2 * nparams
    except TypeError:
        nwalkers = 2 * nparams
    finally:
        if nwalkers < 2 * nparams:
            warnings.warn("Choosing to live dangerously because nwalkers is"
                          " too small")
            danger = True
        else:
            danger = False
    if chain_file is not None and pathlib.Path(chain_file).exists():
        chain, lnprob = read_chain(chain_file)
        init_step = chain[:,-1]
    else:
        chain = None
        lnprob = None
        analytic_params = fit_algebraically(delta_sys, delta_obs)[1:]
        analytic_cov = algebraic_parameter_covariance(
            delta_sys, delta_obs)[1:,1:]
        init_step = np.random.multivariate_normal(
            analytic_params, analytic_cov, nwalkers)
    partial_func = GaussMapClass()
    partial_func.set_data(delta_sys, delta_obs, c_inv)
    if nthreads is not None:
        with emcee.interruptible_pool.InterruptiblePool(nthreads) as p:
            sampler = emcee.EnsembleSampler(
                nwalkers, nparams, partial_func.lnprob, live_dangerously=danger,
                pool=p)
            sampler.run_mcmc(init_step, nsteps)
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers, nparams, partial_func.lnprob, live_dangerously=danger)
        sampler.run_mcmc(init_step, nsteps)
    if chain is not None:
        chain = np.concatenate([chain, sampler.chain], axis=1)
        lnprob = np.concatenate([lnprob, sampler.lnprobability], axis=1)
    else:
        chain = sampler.chain
        lnprob = sampler.lnprobability
    if chain_file is not None:
        write_chain(
            chain, lnprob, chain_file, [
                "a_{}".format(i) for i in range(len(delta_sys))], **kwargs)
    return sampler

def write_chain(chain, lnprob, chain_file, param_names, **kwargs):
    """
    A convenience function for writing a chain to a file that I can then
    easily read in later, using :func:`~.read_chain`.

    The file type should be supported by :meth:`astropy.table.Table.write`, and
    should support metadata.

    Keyword arguments are passed to :meth:`astropy.table.Table.write`

    :param chain: The (not flattened) MCMC chain to save
    :type chain: 3D array-like ``float``
    :param lnprob: The (not flattened) log-likelihood values for the chain
    :type lnprob: 2D array-like ``float``
    :param chain_file: The file path at which to save the chain
    :type chain_file: ``str`` or :class:`os.PathLike`
    :param param_names: The names of the parameters, for column names
    :type param_names: array-like ``str``
    """
    fpath = pathlib.Path(chain_file).expanduser().resolve()
    fpath.parent.mkdir(parents=True, exist_ok=True)
    chain_table = Table(chain.reshape((-1, chain.shape[-1])), names=param_names)
    chain_table["ln_prob"] = lnprob.flatten()
    chain_table.meta["nwalkers"] = chain.shape[0]
    chain_table.meta["nsteps"] = chain.shape[1]
    chain_table.meta["ndim"] = chain.shape[2]
    chain_table.write(fpath.as_posix(), **kwargs)

def read_chain(chain_file, nburnin=0, flat=False, **kwargs):
    """
    A convenience function for reading and reshaping a chain saved in
    :func:`~.fit_systematics`, optionally cutting out a burn-in phase and/or
    returning a flattened chain.

    Additional keyword arguments (``kwargs``) are passed to the
    :meth:`astropy.table.Table.read` method.

    :param chain_file: The file from which to read the chain. This is assumed to
        have been saved by :func:`~.fit_systematics`, and have metadata at least
        for the number of walkers
    :type chain_file: ``str`` or :class:`os.PathLike`
    :param nburnin: Optionally remove this many steps from each walker as a
        burn-in phase before returning the chain. Default 0
    :type nburnin: ``int``, optional
    :param flat: If ``True``, return the chain flattened to
        (``nwalkers*(nsteps-nburnin)``,``nparams``) rather than as a 3D array of
        shape (``nwalkers``,``nsteps-nburnin``,``nparams``). The log-probability
        is also flattened to (``nwalkers*(nsteps-nburnin)``,) rather than
        (``nwalkers``,``nsteps-nburnin``). Default ``False``
    :type flat: ``bool``, optional
    :return: A tuple of :class:`numpy.ndarray`'s. The first element of the tuple
        is the chain read from the file, and the second is the log-probability.
        The shapes of each are determined by the ``flat`` parameter
    :rtype: ``tuple`` of :class:`numpy.ndarray` of ``float``
    :raises IOError: If there is no metadata or the metadata is missing
        information about the number of walkers
    """
    fname = pathlib.Path(chain_file).resolve()
    input_chain = Table.read(fname, **kwargs)
    if "NWALKERS" not in input_chain.meta:
        raise IOError("Cannot obtain shape information from file")
    nwalkers = input_chain.meta["NWALKERS"]
    try:
        nparams = input_chain.meta["NPARAMS"]
    except KeyError:
        nparams = len(input_chain.colnames) - 1
    try:
        nsteps = input_chain.meta["NSTEPS"]
    except KeyError:
        nsteps = len(input_chain) // nwalkers
    lnprob_colname = ("ln_prob" if "ln_prob" in input_chain.colnames else
                      "ln_like")
    lnprob = np.array(input_chain[lnprob_colname]).reshape(
        (nwalkers, nsteps))[:,nburnin:]
    input_chain.remove_column(lnprob_colname)
    chain = np.array(input_chain)
    chain = chain.view(chain.dtype[0]).reshape(
        (nwalkers, nsteps, nparams))[:,nburnin:,:]
    if flat:
        lnprob = lnprob.flatten()
        chain = chain.reshape((-1, nparams))
    return (chain, lnprob)

#----------------------------------- Binning ----------------------------------#
def get_single_edges_and_weights(bins, fgood, x_alpha, min_count=10):
    """
    This function gets the bin edges, centers, and weights for binning in a
    single systematic.

    The ``min_count`` parameter sets the minimum number of pixels that must be
    in a bin for the bin to be included in plots. This is used to mask bins with
    few pixels and therefore large errors

    :param bins: Either the number of bins or the bin edges for the systematic
    :type bins: ``int`` or array-like of ``float``
    :param fgood: The pixel coverage fraction, used for weighting the pixels in
        the various means. Note that no masking is done on this, so it should be
        done by the user if needed
    :type fgood: array-like of ``float``
    :param x_alpha: The systematic being used for binning. Note that no masking
        is done on this, so it should be done by the user if needed
    :type x_alpha: array-like of ``float``
    :param min_count: The minimum number of pixels in a bin, below which the bin
        is masked out. Default 10
    :type min_count: ``int``
    :return: The bin edges, bin centers, sum of ``fgood`` in bins, and sum of
        ``fgood**2`` in bins. The edges are a regular array, but the centers and
        sums are masked arrays with bins not containing more than ``min_count``
        pixels masked out
    :rtype: ``tuple``(:class:`numpy.ndarray`, :class:`numpy.ma.MaskedArray`,
        :class:`numpy.ma.MaskedArray`, :class:`numpy.ma.MaskedArray`)
    """
    counts, edges = binned_statistic(x_alpha, x_alpha, "count", bins=bins)[:-1]
    centers = np.ma.array(
        0.5 * (edges[:-1] + edges[1:]), mask=(counts <= min_count))
    fsum = np.ma.array(
        binned_statistic(x_alpha, fgood, "sum", bins=edges)[0],
        mask=centers.mask)
    f2sum = np.ma.array(
        binned_statistic(x_alpha, fgood**2, "sum", bins=edges)[0],
        mask=centers.mask)
    return (edges, centers, fsum, f2sum)

def get_list_edges_and_weights(bins, fgood, all_x, min_count=10):
    """
    This function gets the bin edges, centers, and weights for binning in
    several systematics maps.

    The ``min_count`` parameter sets the minimum number of pixels that must be
    in a bin for the bin to be included in plots. This is used to mask bins with
    few pixels and therefore large errors

    :param bins: Either the number of bins or the bin edges for each systematic.
        If a single value is given, it is assumed to be the number of bins for
        all systematics. If a 1D array-like is given, it is assumed to be the
        number of bins for each systematic rather than edges to be used for all
        systematics
    :type bins: ``int``, 1D array-like of ``int``, or 2D array-like of ``float``
    :param fgood: The pixel coverage fraction, used for weighting the pixels in
        the various means. Note that no masking is done on this, so it should be
        done by the user if needed
    :type fgood: array-like of ``float``
    :param all_x: The various systematics being used for binning, with the 0th
        axis corresponding to different maps. Note that no masking is done on
        this, so it should be done by the user if needed
    :type all_x: 2D array-like of ``float``
    :param min_count: The minimum number of pixels in a bin, below which the bin
        is masked out. Default 10
    :type min_count: ``int``
    :return: The bin edges, bin centers, sum of ``fgood`` in bins, and sum of
        ``fgood**2`` in bins. The edges are a list of regular arrays, but the
        centers and sums are lists of masked arrays with bins not containing
        more than ``min_count`` pixels masked out
    :rtype: ``tuple``(``list`` of :class:`numpy.ndarray`, ``list`` of
        :class:`numpy.ma.MaskedArray`, ``list`` of
        :class:`numpy.ma.MaskedArray`, ``list`` of
        :class:`numpy.ma.MaskedArray`)
    """
    if np.ndim(bins) == 0:
        edges_and_counts = [
            binned_statistic(
                x_alpha, x_alpha, "count", bins=np.asarray(bins).item())[:-1]
            for x_alpha in all_x]
    else:
        edges_and_counts = [
            binned_statistic(
                x_alpha, x_alpha, "count", bins=bins_alpha)[:-1] for x_alpha,
            bins_alpha in zip(all_x, bins)]
    edges = np.asarray(edges_and_counts)[:,1].tolist()
    centers = [
        np.ma.array(
            0.5 * (edges_alpha[:-1] + edges_alpha[1:]),
            mask=(counts_alpha <= min_count)) for (counts_alpha, edges_alpha)
        in edges_and_counts]
    fsums = [
        np.ma.array(
            binned_statistic(x_alpha, fgood, "sum", bins=edges_alpha)[0],
            mask=(counts_alpha <= min_count)) for x_alpha,
        (counts_alpha, edges_alpha) in zip(all_x, edges_and_counts)]
    f2sums = [
        np.ma.array(
            binned_statistic(x_alpha, fgood**2, "sum", bins=edges_alpha)[0],
            mask=(counts_alpha <= min_count)) for x_alpha,
        (counts_alpha, edges_alpha) in zip(all_x, edges_and_counts)]
    return (edges, centers, fsums, f2sums)

def single_binned_means_errs(edges, delta, bin_x, fgood, fsum, f2sum,
                             verbose=True):
    """
    Get the means and errors on the means of ``delta`` in bins of ``bin_x``

    Use verbose to print the weighted average of the bins

    :param edges: The bin edges for the given map
    :type edges: 1D array-like of ``float`` (``Nbins``,)
    :param delta: The density fluctuations to be binned
    :type delta: 1D :class:`numpy.ndarray` of ``float`` (``Npix``,)
    :param bin_x: The systematic map over which to bin
    :type bin_x: 1D array-like of ``float`` (``Npix``,)
    :param fgood: The fractional pixel coverage for weighting the pixels
    :type fgood: 1D :class:`numpy.ndarray` of ``float`` (``Npix``,)
    :param fsum: The sum of ``fgood`` in the bins, possibly with some bins
        masked out
    :type fsum: 1D :class:`numpy.ma.MaskedArray` or :class:`numpy.ndarray` of
        ``float`` (``Nbins``,)
    :param f2sum: The sum of ``fgood**2`` in the bins, possibly with some bins
        masked out
    :type f2sum: 1D :class:`numpy.ma.MaskedArray` or :class:`numpy.ndarray` of
        ``float`` (``Nbins``,)
    :param verbose: If ``True``, print the weighted average of the binned
        ``delta``'s. Default ``True``
    :type verbose: ``bool``, optional
    :return: A tuple containing first the mean in bins and then the error on the
        mean
    :rtype: ``tuple``(:class:`numpy.ma.MaskedArray`,
        :class:`numpy.ma.MaskedArray`)
    """
    bin_mean = (
        np.ma.array(
            binned_statistic(bin_x, fgood * delta, "sum", bins=edges)[0],
            mask=np.ma.getmaskarray(fsum)) / fsum)
    bin_std = (np.ma.array(
        binned_statistic(bin_x, fgood * delta**2, "sum", bins=edges)[0],
        mask=np.ma.getmaskarray(fsum)) / fsum)
    bin_std = (f2sum * (bin_std - bin_mean**2)) / (fsum**2 - f2sum)
    if verbose:
        print(np.ma.average(bin_mean, weights=(1. / bin_std)))
    bin_std = np.ma.sqrt(bin_std)
    return (bin_mean, bin_std)

def binned_means_and_errs(edges, delta, all_x, fgood, fsums, f2sums,
                          verbose=True):
    """
    Get the means and errors on the means of ``delta`` in bins over each map in
    ``all_x``

    Use verbose to print the weighted average of the bins

    :param edges: The bin edges for the each map as elements of a list
    :type edges: ``list`` of array-like of ``float`` (``Nmaps``,``Nbins``)
    :param delta: The density fluctuations to be binned
    :type delta: 1D :class:`numpy.ndarray` of ``float`` (``Npix``,)
    :param all_x: The systematic maps over which to bin
    :type all_x: 2D array-like of ``float`` (``Nmaps``,``Npix``)
    :param fgood: The fractional pixel coverage for weighting the pixels
    :type fgood: 1D :class:`numpy.ndarray` of ``float`` (``Npix``,)
    :param fsums: A list of the sum of ``fgood`` in the bins, possibly with some
        bins masked out, for each systematic
    :type fsums: ``list`` of :class:`numpy.ma.MaskedArray` or
        :class:`numpy.ndarray` of ``float`` (``Nmaps``,``Nbins``)
    :param f2sums: A list of the sum of ``fgood**2`` in the bins, possibly with
        some bins masked out, for each systematic
    :type f2sums: ``list`` of :class:`numpy.ma.MaskedArray` or
        :class:`numpy.ndarray` of ``float`` (``Nmaps``,``Nbins``)
    :param verbose: If ``True``, print the weighted average of the binned
        ``delta``'s. Default ``True``
    :type verbose: ``bool``, optional
    :return: A tuple containing first the list of means in bins and then the
        list of errors on the means
    :rtype: ``tuple``(``list`` of :class:`numpy.ma.MaskedArray`, ``list`` of
        :class:`numpy.ma.MaskedArray`)
    """
    means_and_errs = [
        single_binned_means_errs(
            edges_alpha, delta, x_alpha, fgood, fsum_alpha, f2sum_alpha,
            verbose) for edges_alpha, x_alpha, fsum_alpha, f2sum_alpha in
        zip(edges, all_x, fsums, f2sums)]
    bin_means = np.asarray(means_and_errs)[:,0].tolist()
    bin_errs = np.asarray(means_and_errs)[:,1].tolist()
    return (bin_means, bin_errs)

def single_binned_mc_sim(edges, all_x, fgood, fsum, bin_x, chain, nreal=None):
    """
    Do an MC simulation of the parameters in ``chain``, and bin the results in
    the systematic ``bin_x``

    :param edges: The bin edges for the systematic
    :type edges: 1D array-like of ``float`` (``Nbins``,)
    :param all_x: The systematics to use for calculating the linear systematic
        fluctuations, which need not include ``bin_x``
    :type all_x: 2D array-like of ``float`` (``Nmaps``,``Npix``)
    :param fgood: The fractional pixel coverage for weighting the pixels
    :type fgood: 1D :class:`numpy.ndarray` of ``float`` (``Npix``,)
    :param fsum: The sum of ``fgood`` in each bin, possibly with some bins
        masked out
    :type fsum: :class:`numpy.ma.MaskedArray` or :class:`numpy.ndarray` of
        ``float`` (``Nbins``,)
    :param bin_x: The systematic values by which to bin, which need not be part
        of ``all_x``
    :type bin_x: 1D array-like of ``float`` (``Npix``,)
    :param chain: The parameter realizations from which to draw
    :type chain: 2D array-like of ``float`` (``Nsteps``,``Nmaps``)
    :param nreal: If not ``None``, the number of realizations to draw from
        ``chain``. Otherwise, use all ``Nsteps`` realizations. If this is
        greater than ``Nsteps``, it will be set to ``Nsteps``. Default ``None``
    :type nreal: ``int`` or ``NoneType``, optional
    :return: A tuple containing the mean and error in the bins over realizations
    :rtype: ``tuple``(:class:`numpy.ma.MaskedArray`,
        :class:`numpy.ma.MaskedArray`)
    """
    try:
        nreal = int(nreal)
    except TypeError:
        nreal = len(chain)
    finally:
        nreal = min(nreal, len(chain))
    if nreal == len(chain):
        chain_ = np.copy(chain)
    else:
        chain_ = np.asarray(chain)[np.random.choice(
            len(chain), nreal, replace=False)]
    delta_real = np.array(
        [delta_sys_linear(params, all_x) for params in chain_])
    bin_mean = (
        np.sum(
            [binned_statistic(bin_x, delta_i * fgood, "sum", bins=edges)[0] for
             delta_i in delta_real], axis=0) / (fsum * nreal))
    bin_std = (
        np.sum(
            [binned_statistic(bin_x, delta_i**2 * fgood, "sum", bins=edges)[0]
             for delta_i in delta_real], axis=0) / (fsum * nreal))
    bin_std = np.ma.sqrt(bin_std - bin_mean**2)
    return (bin_mean, bin_std)

def binned_mc_sim(edges, all_x, fgood, fsums, bins_x, chain, nreal=None):
    """
    Do an MC simulation of the parameters in ``chain``, and bin the results in
    the systematics ``bins_x``. The systematics by which to bin need not be the
    same systematics used to calculate the fluctuations.

    :param edges: A list of the bin edges for each systematic
    :type edges: ``list`` of array-like of ``float`` (``Nmap_bins``,``Nbins``)
    :param all_x: The systematics to use for calculating the linear systematic
        fluctuations, which need not include ``bins_x``
    :type all_x: 2D array-like of ``float`` (``Nmaps``,``Npix``)
    :param fgood: The fractional pixel coverage for weighting the pixels
    :type fgood: 1D :class:`numpy.ndarray` of ``float`` (``Npix``,)
    :param fsum: A list of the sum of ``fgood`` in each bin, possibly with some
        bins masked out, for each binned systematic
    :type fsum: ``list`` of :class:`numpy.ma.MaskedArray` or
        :class:`numpy.ndarray` of ``float`` (``Nmap_bins``,``Nbins``)
    :param bins_x: The systematic values by which to bin, which need not be part
        of ``all_x``
    :type bins_x: ``list`` of array-like of ``float`` (``Nmap_bins``,``Npix``)
    :param chain: The parameter realizations from which to draw
    :type chain: 2D array-like of ``float`` (``Nsteps``,``Nmaps``)
    :param nreal: If not ``None``, the number of realizations to draw from
        ``chain``. Otherwise, use all ``Nsteps`` realizations. If this is
        greater than ``Nsteps``, it will be set to ``Nsteps``. Default ``None``
    :type nreal: ``int`` or ``NoneType``, optional
    :return: A tuple containing the lists of mean and error in the bins of each
        systematic over realizations
    :rtype: ``tuple``(``list`` of :class:`numpy.ma.MaskedArray`, ``list`` of
        :class:`numpy.ma.MaskedArray`)
    """
    try:
        nreal = int(nreal)
    except TypeError:
        nreal = len(chain)
    finally:
        nreal = min(nreal, len(chain))
    if nreal == len(chain):
        chain_ = np.copy(chain)
    else:
        chain_ = np.asarray(chain)[np.random.choice(
            len(chain), nreal, replace=False)]
    delta_real = np.array(
        [delta_sys_linear(params, all_x) for params in chain_])
    map_bin_means = [
        np.sum(
            [binned_statistic(
                x_alpha, delta_i * fgood, "sum", bins=edges_alpha)[0] for
             delta_i in delta_real], axis=0) / (fsum_alpha * nreal) for x_alpha,
        edges_alpha, fsum_alpha in zip(bins_x, edges, fsums)]
    map_bin_errs = [
        np.ma.sqrt(
            (np.sum(
                [binned_statistic(
                    x_alpha, delta_i**2 * fgood, "sum", bins=edges_alpha)[0]
                 for delta_i in delta_real], axis=0) / (fsum_alpha * nreal))
            - mean_alpha**2) for x_alpha, edges_alpha, fsum_alpha, mean_alpha in
        zip(bins_x, edges, fsums, map_bin_means)]
    return (map_bin_means, map_bin_errs)

#--------------------------------- Eigenbasis ---------------------------------#
def calculate_rotation_matrix(x, bad_val=None, mask=None):
    """
    Get the rotation matrix to rotate x into its eigenbasis.

    Note that the rotation matrix should be applied later as :math:`R^T \cdot x`

    :param x: The unrotated values
    :type x: 2D array-like of ``float``
    :param bad_val: An optional value to exclude from ``x``, if it isn't already
        masked. Default ``None``
    :type bad_val: ``float`` or ``NoneType``, optional
    :param mask: If given, this should select observations from axis 1 to use in
        calculating the eigenbasis. Default ``None``
    :type mask: ``NoneType`` or array-like of ``bool`` or ``int``, optional
    :return: The rotation matrix
    :rtype: 2D :class:`numpy.ndarray` of ``float``
    """
    x_rot = np.asanyarray(x)
    if mask is not None:
        x_rot = x_rot[:,mask]
    if bad_val is not None:
        x_rot = x_rot[:,~np.isclose(x_rot[0], bad_val)]
    return np.linalg.eig(np.cov(x_rot))[1]

def rotate_with_matrix(x, rot_mat, bad_val=None, mask=None):
    """
    Rotate x with the given rotation matrix

    :param x: The unrotated values
    :type x: 2D array-like of ``float``
    :param rot_mat: The rotation matrix to be applied
    :type rot_mat: 2D array-like of ``float``
    :param bad_val: An optional value to exclude from ``x``, if it isn't already
        masked. Default ``None``
    :type bad_val: ``float`` or ``NoneType``, optional
    :param mask: If given, this should select observations to be kept from axis
        1. Default ``None``
    :type mask: ``NoneType`` or array-like of ``bool`` or ``int``, optional
    :return: The rotated values
    :rtype: 2D :class:`numpy.ndarray` of ``float``
    """
    x_rot = np.asanyarray(x)
    if mask is not None:
        x_rot = x_rot[:,mask]
    if bad_val is not None:
        x_rot = x_rot[:,~np.isclose(x_rot[0], bad_val)]
    return np.dot(np.transpose(rot_mat), x_rot)

def rotate_to_eigenbasis(x, bad_val=None, mask=None):
    """
    Rotate x into the systematic map eigenbasis, while preserving masked pixels

    :param x: The unrotated systematics
    :type x: 2D array-like of ``float``
    :param bad_val: The value for masked pixels, if not already masked. Default
        ``None``
    :type bad_val: ``float`` or ``NoneType``, optional
    :param mask: If given, this should select pixels to use in calculating the
        eigenbasis. Default ``None``
    :type mask: ``NoneType`` or array-like of ``bool`` or ``int``, optional
    :return: The rotated systematics
    :rtype: 2D :class:`numpy.ndarray` of ``float``
    """
    rotation_matrix = calculate_rotation_matrix(x, bad_val, mask)
    return rotate_with_matrix(x, rotation_matrix, bad_val, mask)

#---------------------------- Visualization helpers ---------------------------#
def plot_chains(chain, lnprob, nburnin, parameters, save_dir=None, show=False,
                is_eigenbasis=True, mock_num=None):
    """
    Convenience function for making plots of walkers and chains

    :param chain: The (3D) chain of walker steps
    :type chain: :class:`numpy.ndarray` of ``float``
        (``Nwalkers``,``Nsteps``,``Nparams``)
    :param lnprob: The (2D) chain of log-probabilities at each walker step
    :type lnprob: :class:`numpy.ndarray` of ``float`` (``Nwalkers``,``Nsteps``)
    :param nburnin: The length to cut from the chain for the burn-in. Must be
        less than ``Nsteps``. Could also be ``None``, to simply display the
        walker plots and exit (regardless of values of ``save_dir`` and
        ``show``)
    :type nburnin: ``int`` or ``NoneType``
    :param parameters: The names of the parameters in the plots
    :type parameters: ``list`` of ``str`` (``Nparams``,)
    :param save_dir: Optional location in which to save the resulting plots. If
        ``None`` (default), plots are not saved
    :type save_dir: ``str``, :class:`os.PathLike`, or ``NoneType``, optional
    :param show: If ``True``, show the generated plots. Default ``False``
    :type show: ``bool``, optional
    :param is_eigenbasis: This parameter is used only for plot names. If
        ``True`` (default), the plot names reflect that the parameters are for
        the eigen-maps.
    :type is_eigenbasis: ``bool``, optional
    :param mock_num: If given, this will be used to differentiate the plots for
        different mocks, where this is the number of the mock. If ``None``
        (default), assumes this is not a mock and that the file name without
        numbering will not clash
    :type mock_num: ``int`` or ``NoneType``, optional
    :return: ``None`` (no return)
    :rtype: ``NoneType``
    """
    from my_chain_plotter import plot_walkers_no_read as pw
    parameters_ = list(parameters)
    ncols = int(np.ceil((len(parameters_) + 1) / 5))
    if save_dir is not None:
        # Temporarily set fig3_file to a pathlib version of save_dir
        cfig_file = pathlib.Path(save_dir)
        cfig_file.mkdir(parents=True, exist_ok=True)
        eig_plt_specifier = "_eigenbasis" if is_eigenbasis else ""
        mock_num_specifier = ("_mock{}".format(mock_num) if mock_num is not None
                              else "")
        wfig_pre_file = cfig_file.joinpath(
            f"walkers_delta_lin_full{eig_plt_specifier}{mock_num_specifier}"
            f"_nsteps{chain.shape[1]}_nburnin{nburnin}_pre-burn.png")
        wfig_post_file = cfig_file.joinpath(
            f"walkers_delta_lin_full{eig_plt_specifier}{mock_num_specifier}"
            f"_nsteps{chain.shape[1]}_nburnin{nburnin}_post-burn.png")
        cfig_file = cfig_file.joinpath(
            f"corner_delta_lin_full{eig_plt_specifier}{mock_num_specifier}"
            f"_nsteps{chain.shape[1]}_nburnin{nburnin}_all-params.png")
    if nburnin is None:
        nburnin = -1
    nburnin = int(nburnin)
    if nburnin >= chain.shape[1]:
        raise ValueError(
            "Invalid nburnin {} for chain with {} steps".format(
                nburnin, chain.shape[1]))
    wfig_pre = pw(
        chain, parameters_, ncols, nburnin, post_burn=False, lnprob=lnprob, 
        filename=wfig_pre_file)
    if nburnin < 0:
        plt.show()
        plt.close("all")
        return
    wfig_post = pw(
        chain, parameters_, ncols, nburnin, post_burn=True, lnprob=lnprob, 
        filename=wfig_post_file)
    flatchain = chain[:,nburnin:,:].reshape((-1,len(parameters_)))
    flatlnprob = lnprob[:,nburnin:].flatten()
    c = ChainConsumer();
    c.add_chain(flatchain, parameters=parameters_, posterior=flatlnprob);
    c.configure(label_font_size=18, tick_font_size=16);
    cfig = c.plotter.plot(figsize="PAGE");
    if save_dir is not None:
        cfig.savefig(cfig_file)
    if show:
        plt.show()
    plt.close("all")
    return

def plot_importance(tau, tau_err, is_eigenbasis=True, ylim=None, save_dir=None, 
                    show=False):
    """
    Make a plot of the median absolute contribution of each systematic map 
    (`tau`) to the total systematics fluctuation
    
    :param tau: The median of the absolute value of the coefficient times the 
        map values for each systematic, which is the importance of the map. 
        There should be one of these for each map, so it should have 18 
        elements
    :type tau: 1D array-like ``float``
    :param tau_err: The error on the map importance `tau`
    :type tau_err: 1D array-like ``float``
    :param is_eigenbasis: This parameter is used for plot naming and for 
        setting the x-axis labels. If ``True`` (default), the plot name 
        reflects that the parameters are for the eigen-maps, and the x-axis
        is labeled by the eigen-number of each map. Otherwise, the x-axis is
        labeled with the full map name
    :type is_eigenbasis: ``bool``, optional
    :param ylim: If given, this is used to fix the limits on the y-axis, which
        is helpful if plotting for more than one redshift bin when all of them
        should have the same limits. If ``None`` (default), the limits are set
        by matplotlib using its default method
    :type ylim: 1D array-like ``float`` (2,) or ``NoneType``, optional
    :param save_dir: Optional location in which to save the resulting plot. If
        ``None`` (default), plot is not saved
    :type save_dir: ``str``, :class:`os.PathLike`, or ``NoneType``, optional
    :param show: If ``True``, show the generated plot. Default ``False``
    :type show: ``bool``, optional
    """
    fig = plt.figure()
    if is_eigenbasis:
        x_vals = np.array([fr"${xi}$" for xi in range(len(tau))])
        plt.xlabel(r"Eigen-map")
    else:
        x_vals = np.array(all_delta_systematics_plot_names)
        plt.xlabel(r"Map")
    plt.ylabel(r"$\tau$")
    tau_order = np.argsort(tau)[::-1]
    tau_ = np.asanyarray(tau)[tau_order]
    tau_err_ = np.asanyarray(tau_err)[tau_order]
    x_vals = x_vals[tau_order]
    plt.errorbar(x_vals, tau_, yerr=tau_err_, fmt="o")
    if save_dir is not None:
        fig_fname = pathlib.Path(save_dir).joinpath(
            f"map_importance{'_eigenbasis' if is_eigenbasis else ''}.png")
        fig_fname.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(fig_fname)
    if show:
        plt.show()
    plt.close("all")
    return
        

def plot_binned_points(centers, delta, delta_err, color="C0", marker="o",
                       x_offset=0, ax=None, xlim=None, ylim=None, label=None,
                       make_legend=False):
    """
    A helper function for plotting the binned points with errorbars, possibly
    with a slight offset to avoid overlapping points

    Note that ``xlim`` and ``ylim`` are ignored if ``ax`` is not ``None``

    :param centers: The center of the bin, as the x-coordinate
    :type centers: array-like ``float``
    :param delta: The average fluctuation in the bin, as the y-coordinate
    :type delta: array-like ``float``
    :param delta_err: The error on the average fluctuation, as the error in the
        y-coordinate
    :type delta_err: array-like ``float``
    :param color: An optional string specifying what color to make the points.
        This can be any string that :mod:`matplotlib` can identify as a color.
        Default 'C0'
    :type color: ``str``, optional
    :param marker: An optional parameter specifying the marker type to use for
        the points. This should be a valid :mod:`matplotlib` marker specifier.
        Default 'o'
    :type marker: ``str`` or ``int``, optional
    :param x_offset: An amount by which to shift the points in the x-coordinate,
        to avoid overlapping points. Default 0.0
    :type x_offset: ``float``, optional
    :param ax: Specify the axes on which to plot the points. If ``None``
        (default), a new set of axes will be created.
    :type ax: :class:`matplotlib.axes.Axes` or ``NoneType``, optional
    :param xlim: Optionally supply the x-limits to use to make plots with the
        same axes. This parameter is ignored if ``ax`` is not ``None``. If
        ``None`` (default), the x-limits not manually changed
    :type xlim: array-like of ``float`` (2,) or ``NoneType``, optional
    :param ylim: Optionally supply the y-limits to use to make plots with the
        same axes. This parameter is ignored if ``ax`` is not ``None``. If
        ``None`` (default), the y-limits not manually changed
    :type ylim: array-like of ``float`` (2,) or ``NoneType``, optional
    :param label: The label to use for the points in the legend. If ``None``
        (default), '_nolegend_' will be used resulting in the points being
        excluded from the legend if created
    :type label: object or ``NoneType``, optional
    :param make_legend: If ``True``, create a legend for the axes before
        returning. Default ``False``
    :type make_legend: ``bool``, optional
    :return: The axes are returned for adding to the plot with later commands
    :rtype: :class:`matplotlib.axes.Axes`
    """
    if label is None:
        label = "_nolegend_"
    if ax is None:
        ax = plt.axes()
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
    ax.errorbar(
        centers+x_offset, delta, yerr=delta_err, fmt=marker, c=color,
        label=label)
    if make_legend:
        ax.legend()
    return ax

def plot_binned_line(centers, delta, delta_err, color="C0", line="-", alpha=1,
                     x_offset=0, ax=None, xlim=None, ylim=None, label=None,
                     make_legend=False):
    """
    A helper function for plotting the binned line with error regions, possibly
    with a slight offset to match offset points

    Note that ``xlim`` and ``ylim`` are ignored if ``ax`` is not ``None``

    :param centers: The center of the bin, as the x-coordinate
    :type centers: array-like ``float``
    :param delta: The average fluctuation in the bin, as the y-coordinate of the
        line
    :type delta: array-like ``float``
    :param delta_err: The error on the average fluctuation, as the error in the
        y-coordinate for setting the shaded region boundaries
    :type delta_err: array-like ``float``
    :param color: An optional string specifying what color to make the points.
        This can be any string that :mod:`matplotlib` can identify as a color.
        Default 'C0'
    :type color: ``str``, optional
    :param line: An optional parameter specifying the line type to use for
        the points. This should be a valid :mod:`matplotlib` line specifier.
        Default '-'
    :type line: ``str``, optional
    :param alpha: Set the transparency of the fill. Default 1.0 (not
        transparent)
    :type alpha: ``float``, optional
    :param x_offset: An amount by which to shift the points in the x-coordinate,
        to avoid overlapping points. Default 0.0
    :type x_offset: ``float``, optional
    :param ax: Specify the axes on which to plot the points. If ``None``
        (default), a new set of axes will be created.
    :type ax: :class:`matplotlib.axes.Axes` or ``NoneType``, optional
    :param xlim: Optionally supply the x-limits to use to make plots with the
        same axes. This parameter is ignored if ``ax`` is not ``None``. If
        ``None`` (default), the x-limits not manually changed
    :type xlim: array-like of ``float`` (2,) or ``NoneType``, optional
    :param ylim: Optionally supply the y-limits to use to make plots with the
        same axes. This parameter is ignored if ``ax`` is not ``None``. If
        ``None`` (default), the y-limits not manually changed
    :type ylim: array-like of ``float`` (2,) or ``NoneType``, optional
    :param label: The label to use for the line with fill in the legend. If
        ``None`` (default), '_nolegend_' will be used resulting in the line
        being excluded from the legend if created. Note that the legend will
        always be created if ``label`` is not ``None``
    :type label: object or ``NoneType``, optional
    :param make_legend: If ``True``, create a legend for the axes before
        returning. If ``label`` is not ``None``, this is always changed to
        ``True``. Default ``False``
    :type make_legend: ``bool``, optional
    :return: The axes are returned for adding to the plot with later commands
    :rtype: :class:`matplotlib.axes.Axes`
    """
    if label is None:
        label = "_nolegend_"
    else:
        make_legend = True
    if ax is None:
        ax = plt.axes()
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
    fill = ax.fill_between(
        centers + x_offset, delta - delta_err, delta + delta_err, color=color,
        alpha=alpha)
    line, = ax.plot(centers + x_offset, delta, ls=line, c=color)
    if make_legend:
        h, l = ax.get_legend_handles_labels()
        h.append((fill, line))
        l.append(label)
        ax.legend(h, l)
    return ax

def plot_binned_line_and_points(centers, delta_points, delta_err_points,
                                delta_line, delta_err_line, color="C0",
                                marker="o", line="-", alpha=1, x_offset=0,
                                ax=None, xlim=None, ylim=None,
                                label_points=None, label_line=None,
                                make_legend=False):
    """
    A convenience function for plotting binned points and matching line with the
    same color and offset

    Note that ``xlim`` and ``ylim`` are ignored if ``ax`` is not ``None``

    :param centers: The center of the bin, as the x-coordinate
    :type centers: array-like ``float``
    :param delta_points: The average fluctuation in the bin for the points, as
        the y-coordinate
    :type delta_points: array-like ``float``
    :param delta_err_points: The error on the average fluctuation of the points,
        as the error in the y-coordinate
    :type delta_err_points: array-like ``float``
    :param delta_line: The average fluctuation in the bin, as the y-coordinate
        of the line
    :type delta_line: array-like ``float``
    :param delta_err_line: The error on the average fluctuation, as the error in
        the y-coordinate for setting the shaded region boundaries
    :type delta_err_line: array-like ``float``
    :param color: An optional string specifying what color to make the points.
        This can be any string that :mod:`matplotlib` can identify as a color.
        Default 'C0'
    :type color: ``str``, optional
    :param marker: An optional parameter specifying the marker type to use for
        the points. This should be a valid :mod:`matplotlib` marker specifier.
        Default 'o'
    :type marker: ``str`` or ``int``, optional
    :param line: An optional parameter specifying the line type to use for
        the points. This should be a valid :mod:`matplotlib` line specifier.
        Default '-'
    :type line: ``str``, optional
    :param alpha: Set the transparency of the fill for the shaded region.
        Default 1.0 (not transparent)
    :type alpha: ``float``, optional
    :param x_offset: An amount by which to shift the points in the x-coordinate,
        to avoid overlapping points. Default 0.0
    :type x_offset: ``float``, optional
    :param ax: Specify the axes on which to plot the points. If ``None``
        (default), a new set of axes will be created.
    :type ax: :class:`matplotlib.axes.Axes` or ``NoneType``, optional
    :param xlim: Optionally supply the x-limits to use to make plots with the
        same axes. This parameter is ignored if ``ax`` is not ``None``. If
        ``None`` (default), the x-limits not manually changed
    :type xlim: array-like of ``float`` (2,) or ``NoneType``, optional
    :param ylim: Optionally supply the y-limits to use to make plots with the
        same axes. This parameter is ignored if ``ax`` is not ``None``. If
        ``None`` (default), the y-limits not manually changed
    :type ylim: array-like of ``float`` (2,) or ``NoneType``, optional
    :param label_points: The label to use for the points in the legend. If
        ``None`` (default), '_nolegend_' will be used resulting in the points
        being excluded from the legend if created
    :type label_points: object or ``NoneType``, optional
    :param label_line: The label to use for the line with fill in the legend. If
        ``None`` (default), '_nolegend_' will be used resulting in the line
        being excluded from the legend if created. Note that the legend will
        always be created if ``label`` is not ``None``
    :type label_line: object or ``NoneType``, optional
    :param make_legend: If ``True``, create a legend for the axes before
        returning. If ``label_line`` is not ``None``, this is always changed to
        ``True``. Default ``False``
    :type make_legend: ``bool``, optional
    :return: The axes are returned for adding to the plot with later commands
    :rtype: :class:`matplotlib.axes.Axes`
    """
    if ax is None:
        ax = plt.axes()
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
    ax = plot_binned_points(
        centers, delta_points, delta_err_points, color, marker, x_offset, ax,
        label=label_points)
    ax = plot_binned_line(
        centers, delta_line, delta_err_line, color, line, alpha, x_offset, ax,
        label=label_line, make_legend=make_legend)
    return ax

def plot_correlation_functions(
        corr_params, zbin, des_spec, wtheta_uw, plot_path, wtheta_analytic=None,
        wtheta_average=None, wtheta_error_correction=None,
        analytic_plot_path=None, show_fig=True):
    """
    Plot the correlation functions and the differences from the DES Y1
    correlation function for a given redshift bin. This also handles plot
    formatting to make things look pretty

    :param corr_params: A dictionary containing parameters including the plot
        details such as which angular scales are included, what the limits
        should be on the y-axis, where the text indicating the redshift bin
        should be placed on the y-axis, and what to label the x-axis. Any
        parameters not pertaining to the plots are ignored
    :type corr_params: ``dict``
    :param zbin: The current redshift bin being plotted. This is used for
        labeling the plot as well as finding the correct output path(s)
    :type zbin: ``int``
    :param des_spec: The spectrum object containing the correlation function as
        measured in DES Y1.
    :type des_spec: :class:`SpectrumMeasurement`
    :param wtheta_uw: The unweighted correlation function
    :type wtheta_uw: array-like of ``float``
    :param plot_path: The location at which to save the final plot
    :type plot_path: ``str`` or :class:`os.PathLike`
    :param wtheta_analytic: If given, this is the correlation function measured
        using the analytic fit parameters to calculate weights. If ``None``
        (default), the analytic weight correlation function is not included
    :type wtheta_analytic: ``NoneType`` or array-like of ``float``, optional
    :param wtheta_average: If given, this is the correlation function measured
        using the average of the MCMC chains. It will be plotted with corrected
        errorbars if ``wtheta_error_correction`` is also given. If ``None``
        (default), the average weight correlation function is not included
    :type wtheta_average: ``NoneType`` or array-like of ``float``, optional
    :param wtheta_error_correction: If given, this is the correction to the DES
        Y1 errors. Note that it should be the square root of the diagonal of the
        correction to the covariance matrix, not a covariance matrix itself.
        This is ignored if ``wtheta_average`` is ``None``. If this is ``None``
        (default), the DES Y1 errors are used without correction
    :type wtheta_error_correction: ``NoneType`` or array-like of ``float``,
        optional
    :param analytic_plot_path: Where to save the plot before adding the average
        weight correlation function, if any. Ignored if ``wtheta_analytic`` is
        ``None``. If this is ``None`` (default), only the final plot is saved
    :type analytic_plot_path: ``NoneType``, ``str``, or :class:`os.PathLike`,
        optional
    :param show_fig: If ``True`` (default), show the completed plot. Otherwise,
        just save and exit
    :type show_fig: ``bool``, optional
    :return: ``None`` (no return)
    :rtype: ``NoneType``
    """
    theta, wtheta_des = des_spec.get_pair(zbin, zbin)
    wtheta_des_err = des_spec.get_error(zbin, zbin)
    fig, [tax, bax] = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"hspace": 0, "height_ratios": [3, 1]})
    bax.set_xlabel(corr_params["xlabel"])
    tax.set_ylabel(r"$\theta\, w\!\left(\theta\right)$")
    bax.set_ylabel(r"$\theta\, \Delta w\!\left(\theta\right)$")
    tax.set_ylim(corr_params["ylim"][zbin])
    tax.set_yticks(corr_params["yticks"][zbin])
    tax.set_yticklabels(corr_params["yticklabels"][zbin])
    tax.tick_params(axis="y", which="both", direction="inout")
    bax.tick_params(axis="y", which="both", direction="inout")
    tax.tick_params(
        axis="x", which="both", direction="inout", bottom=False, top=True,
        labelbottom=False)
    bax.tick_params(
        axis="x", which="both", direction="inout", bottom=True, top=True)
    tax.text(
        85, corr_params["text_yloc"][zbin], r"${z_bin}, {z_bin}$".format(
            z_bin=zbin))
    tax.axhline(0, c="k", ls="-", alpha=0.2)
    tax.axvspan(
        0, corr_params["invalid_region"][zbin], facecolor="k", alpha=0.1)
    bax.axvspan(
        0, corr_params["invalid_region"][zbin], facecolor="k", alpha=0.1)
    bax.set_xscale("log")
    tax.plot(theta, theta * wtheta_uw, "k--", alpha=0.3, label=r"Unweighted")
    bax.plot(theta, theta * (wtheta_uw - wtheta_des), "k--", alpha=0.3)
    tax.plot(theta, theta * wtheta_des, "C0-", label=r"DES Y1")
    bax.axhline(0, c="C0", ls="-")
    gs = tax.get_gridspec()
    gs.tight_layout(fig)
    gs.update(hspace=0)
    if wtheta_analytic is not None:
        tax.errorbar(
            theta, theta * wtheta_analytic,
            yerr=(theta * des_spec.get_error(zbin, zbin)), fmt="C1d",
            label=r"Analytic weights")
        bax.errorbar(
            theta, theta * (wtheta_analytic - wtheta_des),
            yerr=(theta * des_spec.get_error(zbin, zbin)), fmt="C1d")
        if analytic_plot_path is not None:
            tax.legend()
            plt.savefig(analytic_plot_path, bbox_inches="tight")
        tepsilon = transforms.offset_copy(
            tax.transData, fig, (4 / 72), 0, units="inches")
        bepsilon = transforms.offset_copy(
            bax.transData, fig, (4 / 72), 0, units="inches")
    else:
        tepsilon = transforms.offset_copy(
            tax.transData, fig, 0, 0, units="inches")
        bepsilon = transforms.offset_copy(
            bax.transData, fig, 0, 0, units="inches")
    if wtheta_average is not None:
        if wtheta_error_correction is not None:
            wtheta_error = np.sqrt(
                des_spec.get_error(zbin, zbin)**2 + wtheta_error_correction)
        else:
            wtheta_error = des_spec.get_error(zbin, zbin)
        tax.errorbar(
            theta, theta * wtheta_average, yerr=(theta * wtheta_error),
            fmt="C2o", label=r"Average weights", transform=tepsilon)
        bax.errorbar(
            theta, theta * (wtheta_average - wtheta_des),
            yerr=(theta * wtheta_error), fmt="C2o", transform=bepsilon)
    tax.legend()
    plt.savefig(plot_path, bbox_inches="tight")
    if show_fig:
        plt.show()
    plt.close()
    return

#--------------------------------- Statistics ---------------------------------#
def calc_median_absolute_deviation(x, mask=None, scale=1.4826):
    """
    A helper function for calculating the median absolute deviation of a masked
    sample

    :param x: The value to calculate the MAD for
    :type x: array-like
    :param mask: An optional mask to apply before calculating, where ``True``
        elements are masked out. Default ``None``
    :type mask: array-like of ``bool`` or ``NoneType``, optional
    :param scale: A scaling factor applied to the median absolute deviation. 
        The default (1.4826) is consistent with the standard deviation when `x` 
        is normally distributed
    :type scale: float, optional
    :return: The MAD for only non-masked values
    :rtype: ``float``
    """
    x_ma = np.ma.array(x, mask=mask, copy=True)
    return mad(x_ma.compressed(), scale=scale)

def arg_median(arr):
    """
    Find the index corresponding to the median of the array. If the array has an
    even number of elements, this gets the indices of the elements on either 
    side of the median. The return is always an array, so to get the actual 
    median, you can either select the item from the returned index array or take
    the mean of the element(s) indexed by the result
    
    :param arr: The array for which to get the median index/indices
    :type arr: array-like
    :return: The index array that gives the median element of `arr` if it has an 
        odd number of elements, or which gives the elements of `arr` on either
        side of the median if it has an even number of elements
    :rtype: :class:`numpy.ndarray` of ``int``
    """
    mid_point = len(arr) // 2
    if len(arr) % 2 == 1:
        return np.argsort(arr)[[mid_point]]
    return np.argsort(arr)[mid_point - 1:mid_point + 1]

#----------------------------- Catalog utilities ------------------------------#
def add_weights(cat, weights):
    cat.w = weights
    cat.nontrivial_w = True
    use = cat.w != 0
    cat.nobj = np.sum(use)
    cat.sumw = np.sum(cat.w)
    if cat.sumw == 0.:
        raise ValueError("Catalog has invalid sumw == 0")
    if cat.g1 is not None:
        cat.varg = np.sum(cat.w[use]**2 * (cat.g1[use]**2 + cat.g2[use]**2))
        cat.varg /= 2. * cat.sumw
    else:
        cat.varg = 0.
    if cat.k is not None:
        cat.vark = np.sum(cat.w[use]**2 * cat.k[use]**2)
        cat.vark /= cat.sumw
    else:
        cat.vark = 0.
    cat.clear_cache()
    return cat

#------------------------------------ Other -----------------------------------#
def create_delta_lin_mask(fracgood_good_mask, delta_lin_max, *delta_lin_maps):
    """
    Determine the mask for excluding large values of systematics correction.

    Values of ``True`` in the mask are pixels that are in the survey footprint
    and do not have large systematics corrections.

    Arguments after the first two are all maps of the systematics correction,
    which are considered jointly: a pixel that has a large correction in any map
    is masked out, while only pixels that have a correction less than the
    maximum in all maps are kept. Any number of these maps can be given, and
    each should be one-dimensional with length equal to either the total number
    of pixels (i.e. the same length as ``fracgood_good_mask``) or the number of
    pixels in the footprint (i.e. ``numpy.count_nonzero(fracgood_good_mask)``).
    However, at least one such map must be given or an error is raised.

    :param fracgood_good_mask: The survey footprint mask, with ``True``
        indicating that a pixel is in the footprint
    :type fracgood_good_mask: 1D array-like ``bool`` (``Npix``,)
    :param delta_lin_max: The maximum systematics correction value to keep
    :type delta_lin_max: ``float``
    :arg delta_lin_maps: One or more arrays containing maps of the systematics
        correction values per pixel. These can either be for all pixels, with
        shape (``Npix``,), or only for pixels within the footprint, with shape
        (``numpy.count_nonzero(fracgood_good_mask)``,)
    :type delta_lin_maps: 1D array-like(s) ``float``
    :return: The combined mask including the survey footprint and the cut on
        systematics corrections in all maps. ``True`` refers to pixels that are
        in the combined footprint, while ``False`` means the pixel is either
        outside the original footprint or was cut because of the systematics
        correction value in one or more maps
    :rtype: 1D :class:`numpy.ndarray` ``bool`` (``Npix``,)
    :raises ValueError: If nothing is passed for ``delta_lin_maps`` or if any of
        the ``delta_lin_maps`` has a length not equal to ``Npix`` or
        ``numpy.count_nonzero(fracgood_good_mask)``
    """
    if len(delta_lin_maps) == 0:
        raise ValueError("At least one map must be passed to delta_lin_maps")
    for i, mapi in enumerate(delta_lin_maps):
        if (len(mapi) != len(fracgood_good_mask) and
            len(mapi) != np.count_nonzero(fracgood_good_mask)):
            raise ValueError(
                ("Invalid size {} for map {}: all maps must have size of either"
                 " {} or {}").format(len(mapi), i, len(fracgood_good_mask),
                                     np.count_nonzero(fracgood_good_mask)))
    full_mask = np.copy(fracgood_good_mask)
    masked_maps = [np.asanyarray(mapi)[fracgood_good_mask]
                   if len(mapi) == len(fracgood_good_mask) else
                   np.asanyarray(mapi) for mapi in delta_lin_maps]
    full_mask[fracgood_good_mask] = np.prod(
        [mapi < delta_lin_max for mapi in masked_maps], axis=0, dtype=bool)
    return full_mask

def map_importance(chain, sys_maps):
    """
    Calculate the map importance and the error on the importance for each map
    given the values in the MCMC chain
    
    :param chain: An MCMC chain of the parameter values, which should have any
        burn-in phase already removed and should be 2-dimensional, with the 1st
        axis having length equal to the number of systematics maps in 
        `sys_maps`
    :type chain: 2D array-like of ``float`` (Nsteps, Nmaps)
    :param sys_maps: The systematics maps, with the 0th axis being indexed by
        map number. This should already be masked
    :type sys_maps: 2D array-like of ``float`` (Nmaps, Npix)
    :return: The importance value, which is the median absolute value of the
        contribution to the total systematics fluctuation, for each map and the
        error on the importance value for each map
    :rtype: ``tuple`` of 2 1D :class:`numpy.ndarray` of ``float``
    """
    abs_mean_params = np.abs(chain.mean(axis=0))
    err_mean_params = chain.std(axis=0, ddof=1) / np.sqrt(chain.shape[0])
    median_abs_sys = np.median(np.atleast_2d(np.abs(sys_maps)), axis=1)
    return abs_mean_params * median_abs_sys, err_mean_params * median_abs_sys

def read_standard_sys_maps(sys_path, nside=None, nside_shift=None,
                           shift_fname=None, weight_fname=None, 
                           rot_mat_fname=None, fgood=None, minfrac=0.001):
    """
    A convenience function for reading the pre-standardized, pre-masked, and 
    possibly pre-rotated systematics maps

    :param sys_path: The path at which the pre-standardized systematics should
        be stored. Try to read the systematics from this path, or create it
        if it doesn't exist
    :type sys_path: ``str`` or :class:`os.PathLike`
    :param nside: The resolution of the maps to read. This is only needed if
        ``sys_path`` does not exist. Default ``None``
    :type nside: ``int`` or ``NoneType``, optional
    :param nside_shift: The resolution of the maps from which to get the shfit
        and scale parameters, and possibly the roation matrix. This is only
        needed if ``sys_path`` does not exist. Default ``None``
    :type nside_shift: ``int`` or ``NoneType``, optional
    :param shift_fname: The file name template for the shift parameters, which
        should at most need to be filled with the map resolution of the shift
        parameters. This is only needed if ``sys_path`` does not exist. Default
        ``None``
    :type shift_fname: ``str`` or ``NoneType``, optional
    :param weight_fname: The file name template for the weight parameters, which
        should at most need to be filled with the map resolution of the weight
        parameters. This is only needed if ``sys_path`` does not exist. Default
        ``None``
    :type weight_fname: ``str`` or ``NoneType``, optional
    :param rot_mat_fname: The file name template for the rotation matrix, which
        should at most need to be filled with the map resolution of the rotation
        matrix. This is only needed if ``sys_path`` does not exist and the
        systematics maps should be rotated. Default ``None``
    :type rot_mat_fname: ``str`` or ``NoneType``, optional
    :param fgood: The pixel coverage map, which is needed for reading and 
        masking the systematics maps if the standardized maps are not already 
        saved. Not used if ``sys_path`` exists. Should be at resolution 
        Nside=4096. Default ``None``
    :type fgood: :class:`lsssys.Mask` or ``NoneType``, optional
    :param minfrac: The minimum pixel coverage fraction to allow. Unused if 
        ``sys_path`` exists. Default 0.001
    :type minfrac: ``float``, optional
    :return: The shifted, re-weighted, and possibly rotated high-resolution
        systematics. The 0th axis is indexed by maps, and the 1st axis by pixels
        in the footprint
    :rtype: 2D :class:`numpy.ndarray` ``float``
    :raises ValueError: If ``sys_path`` does not exist and any of ``nside``,
        ``nside_shift``, ``good_mask``, ``shift_fname``, or ``weight_fname`` is
        ``None``, or if the length of ``good_mask`` doesn't match the number of
        pixels indicated by ``nside``
    :raises IOError: If the files obtained from ``shift_fname``,
        ``weight_fname``, or ``rot_mat_fname`` do not exist in the same
        directory as ``sys_path``
    """
    try:
        return np.load(sys_path, allow_pickle=True)
    except IOError:
        spath = pathlib.Path(sys_path).expanduser().resolve()
        sdir = spath.parent
        if nside is None:
            raise ValueError("Need nside when systematics file does not exist")
        if nside_shift is None:
            raise ValueError(
                "Need nside_shift when systematics file does not exist")
        if shift_fname is None:
            raise ValueError(
                "Need shift_fname when systematics file does not exist")
        if weight_fname is None:
            raise ValueError(
                "Need weight_fname when systematics file does not exist")
        if fgood is None:
            raise ValueError("Need fgood when systematics file does not exist")
        if nside != 4096:
            fgood_final = fgood.degrade(nside, minfrac, False)
        else:
            fgood_final = lsssys.Mask(None, empty=True)
            fgood_final.nside = fgood.nside
            fgood_final.maskpix = fgood.maskpix
            fgood_final.fracpix = fgood.fracpix
            fgood_final.mask = fgood.mask
            fgood_final.fracdet = fgood.fracdet
            try:
                fgood_final.zmax = fgood.zmax
            except AttributeError:
                pass
        try:
            shifts = np.load(
                sdir.joinpath(shift_fname.format(nside_shift)),
                allow_pickle=True)
            weights = np.load(
                sdir.joinpath(weight_fname.format(nside_shift)),
                allow_pickle=True)
        except (FileNotFoundError, IOError):
            shifts = np.empty(len(all_systematics))
            weights = np.empty_like(shifts)
            for i, syst in enumerate(all_systematics):
                smap = read_sysmap(syst, nside_shift, fgood.fracdet, minfrac)
                shifts[i] = smap.weightedmean()
                weights[i] = calc_median_absolute_deviation(
                    smap.data, smap.mask)
                del smap
            if sdir.joinpath(shift_fname.format(nside_shift)).suffix == ".pkl":
                sdir.joinpath(shift_fname.format(nside_shift)).write_bytes(
                    shifts.dumps())
            else:
                np.save(sdir.joinpath(shift_fname.format(nside_shift)), shifts)
            if sdir.joinpath(weight_fname.format(nside_shift)).suffix == ".pkl":
                sdir.joinpath(weight_fname.format(nside_shift)).write_bytes(
                    weights.dumps())
            else:
                np.save(
                    sdir.joinpath(weight_fname.format(nside_shift)), weights)
        if rot_mat_fname is not None:
            try:
                rot_mat = np.load(
                    sdir.joinpath(rot_mat_fname.format(nside_shift)), 
                allow_pickle=True)
            except (FileNotFoundError, IOError):
                npix_shift = fgood.degrade(
                    nside_shift, minfrac, False).maskpix.size
                x_shift = np.empty((len(all_systematics), npix_shift))
                for i, (syst, m, s) in enumerate(
                        zip(all_systematics, shifts, weights)):
                    smap = read_sysmap(
                        syst, nside_shift, fgood.fracdet, minfrac)
                    x_shift[i] = (smap.data[~smap.mask] - m) / s
                    del smap
                rot_mat = calculate_rotation_matrix(x_shift)
                del x_shift
                sdir.joinpath(rot_mat_fname.format(nside_shift)).write_bytes(
                    rot_mat.dumps())
        else:
            rot_mat = np.diag(np.ones(len(all_systematics)))
        x = np.array([
            (read_sysmap(
                syst, nside, fgood.fracdet, minfrac).data[
                    fgood_final.maskpix] - m) / s 
            for syst, m, s in zip(all_systematics, shifts, weights)])
        x = rotate_with_matrix(x, rot_mat)
        spath.write_bytes(pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL))
        return x

#----------------------------- Single bin routines ----------------------------#
def do_single_analytic_fit(store_path, delta_sys, delta_obs):
    try:
        return np.load(store_path, allow_pickle=True)
    except IOError:
        analytic_params = fit_algebraically(delta_sys, delta_obs)
        if pathlib.Path(store_path).suffix == ".pkl":
            pathlib.Path(store_path).write_bytes(analytic_params.dumps())
        else:
            np.save(store_path, analytic_params)
        return analytic_params

def cf_to_precision_matrix(delta_true, good_mask, theta_edges, pix_seps,
                           theta_max, num_threads, cov_path):
    with update_env(OMP_NUM_THREADS=str(num_threads)):
        cf = healcorr.compute_corr(
            delta_true, good_mask, theta_edges,
            len(delta_true[0]) == np.count_nonzero(good_mask), False)
    upper_triangle_seps = pix_seps[np.triu_indices_from(pix_seps)]
    theta_means = binned_statistic(
        upper_triangle_seps[upper_triangle_seps <= theta_edges[-1]],
        upper_triangle_seps[upper_triangle_seps <= theta_edges[-1]],
        "mean", theta_edges)[0]
    del upper_triangle_seps
    print("Fit correlation function", flush=True)
    cov_mat = corr_func_power_law_with_fit(pix_seps, theta_means, cf, theta_max)
    print("Write covariance matrix", flush=True)
    if pathlib.Path(cov_path).suffix == ".pkl":
        pathlib.Path(cov_path).write_bytes(cov_mat.dumps())
    else:
        np.save(cov_path, cov_mat)
    del cf, theta_means
    print("Invert covariance matrix", flush=True)
    inv_cov_mat = np.linalg.inv(cov_mat)
    return inv_cov_mat

def get_single_inv_covariance(cov_path, analytic_path, delta_sys, delta_obs,
                              good_mask, theta_edges, pix_seps, theta_max,
                              num_threads):
    params = do_single_analytic_fit(analytic_path, delta_sys, delta_obs)
    delta_true = np.atleast_2d(delta_obs - delta_sys_linear(params, delta_sys))
    del params
    return cf_to_precision_matrix(
        delta_true, good_mask, theta_edges, pix_seps, theta_max, num_threads,
        cov_path)

def run_const_cov_fit_once(zbin, fracgood, delta_sys, delta, cov_path,
                           analytic_path, chain_path, pix_seps, theta_edges,
                           theta_max, nsteps, mock_num=None, num_threads=None):
    print("Get inverse covariance matrix", flush=True)
    try:
        cov_mat = np.load(cov_path, allow_pickle=True)
    except IOError:
        inv_cov_mat = get_single_inv_covariance(
            cov_path, analytic_path, delta_sys, delta, ~fracgood.mask,
            theta_edges, pix_seps, theta_max, num_threads)
        print("Received inverse covariance matrix", flush=True)
    else:
        print("Invert covariance matrix from file", flush=True)
        inv_cov_mat = py_timer.detailed_time(np.linalg.inv)(cov_mat)
        del cov_mat
        print("Finished inverting covariance matrix from file", flush=True)
    print("Running fit for redshift bin {}{}".format(
        zbin, ", mock {}".format(mock_num) if mock_num is not None else ""))
    sampler = fit_constant_covariance(
        delta_sys, delta, inv_cov_mat, nsteps, chain_file=chain_path,
        overwrite=True, nthreads=params["common"]["max_nprocesses"])
    chain = sampler.chain.copy()
    lnprob = sampler.lnprobability.copy()
    return chain, lnprob

def run_const_cov_fit(zbin, params):
    nside = params["common"]["nside_fit"][zbin]
    fgood4096 = lsssys.Mask(
        params["common"]["cat_root_path"].joinpath(
            params["common"]["fracgood_fname"].format(4096)).as_posix())
    try:
        fgood = lsssys.Mask(
            params["common"]["cat_root_path"].joinpath(
                params["common"]["fracgood_fname"].format(nside)).as_posix(), 
            nside=nside)
    except (FileNotFoundError, IOError):
        fgood = fgood4096.degrade(nside, 0.0, False)
        fgood.save(
            params["common"]["cat_root_path"].joinpath(
                params["common"]["fracgood_fname"].format(nside)).as_posix(), 
            colnames=["HPIX", "FRACGOOD"])
    fgood.cutmask(minfrac=0.8)
    x = read_standard_sys_maps(
        params["common"]["sys_root_path"].joinpath(
            params["common"]["sys_fname"].format(fit=nside, nside=nside)), 
        nside, nside, params["common"]["sys_shift_fname"], 
        params["common"]["sys_weight_fname"], params["common"]["rot_mat_fname"], 
        fgood4096, 0.8)
    del fgood4096
    plot_dir = params["common"]["plot_root_path"] / "zbin{}".format(zbin)
    plot_dir.mkdir(parents=True, exist_ok=True)
    resol = hp.nside2resol(fgood.nside)
    power_of_ten = 10.**(int(np.log10(resol)) - 1)
    delta_theta = int(resol / power_of_ten) * power_of_ten
    theta_edges = (np.arange(
        0, int(1.5 * params["fit"]["cov_mat_theta_max"] / delta_theta))
                   - 0.5) * delta_theta
    del resol, power_of_ten, delta_theta
    map_idxs = fgood.maskpix
    map_angles = np.array(hp.pix2ang(fgood.nside, map_idxs))
    ang_seps = np.array(
        [hp.rotator.angdist(anglei, map_angles) for anglei in map_angles.T])
    del map_idxs, map_angles
    if not params["common"]["is_mock"]:
        # Names for mean and maximum likelihood parameter output files
        mean_fname = f"mean_parameters_nside{fgood.nside}.pkl"
        ml_fname = f"max_like_parameters_nside{fgood.nside}.pkl"
        cat_path = params["common"]["dcat_path"]
        cat = lsssys.Redmagic(params["common"]["dcat_path"].as_posix())
        # Generate the galaxy count map
        dmap = lsssys.cat2galmap(
            *cat.eqinbin(
                *params["common"]["zedges"][zbin]), 
            fgood, minfrac=0.8, rescale=True)
        # Get the overdensity on good pixels from the galaxy count map
        delta = (dmap.data[fgood.maskpix] / dmap.weightedmean()) - 1.
        # We no longer need dmap
        del dmap
        analytic_path = plot_dir.joinpath(
            params["fit"]["analytic_output_fname"].format(
                nside=params["common"]["nside_fit"][zbin]))
        cov_path = plot_dir / params["fit"]["const_cov_fname"].format(
            nside=fgood.nside)
        chain_path = params["common"]["chain_root_path"].joinpath(
            params["common"]["chain_fname"].format(zbin=zbin, nside=nside))
        if params["fit"]["force_do_fits"] or not chain_path.exists():
            chain, lnprob = run_const_cov_fit_once(
                zbin, fgood, x, delta, cov_path, analytic_path, chain_path,
                ang_seps, theta_edges, params["fit"]["cov_mat_theta_max"],
                params["fit"]["nsteps"], num_threads=params["common"].get(
                    "max_nthreads", None))
        else:
            chain, lnprob = read_chain(chain_path)
        flat_chain = chain[:, params["common"]["nburnin"]:].reshape(
            (-1, chain.shape[-1]))
        flat_lnprob = lnprob[:, params["common"]["nburnin"]:].flatten()
        plot_dir.joinpath(mean_fname).write_bytes(
            flat_chain.mean(axis=0).dumps())
        plot_dir.joinpath(ml_fname).write_bytes(
            flat_chain[flat_lnprob.argmax()].dumps())
        tau, tau_err = map_importance(flat_chain, x)
        np.save(
            plot_dir.joinpath(
                f"map_importance_{params['common']['fit_type']}_fit{nside}_"
                f"unsorted_v{params['common']['chain_version']}.npy"), tau)
        np.save(
            plot_dir.joinpath(
                f"map_importance_err_{params['common']['fit_type']}_fit{nside}"
                f"_unsorted_v{params['common']['chain_version']}.npy"), tau_err)
        np.save(
            plot_dir.joinpath(
                f"map_importance_order_{params['common']['fit_type']}_"
                f"fit{nside}_v{params['common']['chain_version']}.npy"), 
            np.argsort(tau)[::-1])
        plot_importance(
            tau, tau_err, params["common"]["rotated"], save_dir=plot_dir, 
            show=params["fit"]["show_chain_plots"])
        del flat_chain, flat_lnprob, tau, tau_err
        if params["fit"]["plot_chains"]:
            plot_chains(
                chain, lnprob, params["common"]["nburnin"],
                params["fit"]["param_names"], plot_dir,
                params["fit"]["show_chain_plots"], params["common"]["rotated"])
    else:
        for mock_num in params["common"]["mock_nums"]:
            # Names for mean and maximum likelihood parameter output files
            mean_fname = (f"mean_parameters_mock{mock_num}_"
                          f"nside{fgood.nside}.pkl")
            ml_fname = (f"max_like_parameters_mock{mock_num}_"
                        f"nside{fgood.nside}.pkl")
            # The mocks can't be read by lsssys, so just read the fits file 
            # directly
            mock_path = params["common"]["mock_root_path"].joinpath(
                params["common"]["mcat_zbin_fname"].format(
                    num=mock_num, zlim=params["common"]["zedges"][zbin]))
            cat = Table.read(mock_path)
            # Generate the galaxy count map
            dmap = lsssys.cat2galmap(
                cat["RA"], cat["DEC"], fgood, minfrac=0.8, rescale=True)
            # Get the overdensity on good pixels from the galaxy count map
            delta = (dmap.data[fgood.maskpix] / dmap.weightedmean()) - 1.
            # We no longer need dmap
            del dmap
            analytic_path = plot_dir.joinpath(
                params["fit"]["analytic_output_fname"].format(
                    num=mock_num, nside=params["common"]["nside_fit"][zbin]))
            cov_path = plot_dir / params["fit"]["const_cov_fname"].format(
                num=mock_num, nside=fgood.nside)
            chain_path = params["common"]["chain_root_path"].joinpath(
                params["common"]["chain_fname"].format(zbin=zbin, num=mock_num,
                                                       nside=nside))
            if params["fit"]["force_do_fits"] or not chain_path.exists():
                chain, lnprob = run_const_cov_fit_once(
                    zbin, fgood, x, delta, cov_path, analytic_path,
                    chain_path, ang_seps, theta_edges,
                    params["fit"]["cov_mat_theta_max"], params["fit"]["nsteps"],
                    mock_num, params["common"].get("max_nthreads", None))
            else:
                chain, lnprob = read_chain(chain_path)
            flat_chain = chain[:, params["common"]["nburnin"]:].reshape(
                (-1, chain.shape[-1]))
            flat_lnprob = lnprob[:, params["common"]["nburnin"]:].flatten()
            plot_dir.joinpath(mean_fname).write_bytes(
                flat_chain.mean(axis=0).dumps())
            plot_dir.joinpath(ml_fname).write_bytes(
                flat_chain[flat_lnprob.argmax()].dumps())
            del flat_chain, flat_lnprob
            if params["fit"]["plot_chains"]:
                plot_chains(
                    chain, lnprob, params["common"]["nburnin"],
                    params["fit"]["param_names"], plot_dir,
                    params["fit"]["show_chain_plots"],
                    params["common"]["rotated"], mock_num)
    return

def get_wtheta_unweighted(save_path, nn_config, dcat_file, rcat_file, 
                          cat_config, dcat_zedges, rcat_zedges, 
                          rr_file=None, nthreads=0, **kwargs):
    if pathlib.Path(save_path).exists():
        return Table.read(save_path, **kwargs)["xi"].data
    else:
        tqdm.tqdm.write("Calculating unweighted correlation function")
        dd = treecorr.NNCorrelation(nn_config, num_threads=nthreads)
        dr = dd.copy()
        rr = dd.copy()
        if dcat_zedges is not None:
            data_cat = lsssys.Redmagic(pathlib.Path(dcat_file).as_posix())
            data_cat.cutz(*dcat_zedges)
            dcat = treecorr.Catalog(
                ra=data_cat.ra, dec=data_cat.dec, **cat_config)
            del data_cat
        else:
            dcat = treecorr.Catalog(
                pathlib.Path(dcat_file).as_posix(), cat_config, 
                file_type="FITS")
        if rcat_zedges is not None:
            rand_cat = lsssys.Randoms(pathlib.Path(rcat_file).as_posix())
            rand_cat.cutz(*rcat_zedges)
            rcat = treecorr.Catalog(
                ra=rand_cat.ra, dec=rand_cat.dec, **cat_config)
            del rand_cat
        else:
            rcat = treecorr.Catalog(
                pathlib.Path(rcat_file).as_posix(), cat_config, 
                file_type="FITS")
        if rr_file is not None and pathlib.Path(rr_file).is_file():
            rr.read(pathlib.Path(rr_file).as_posix())
        else:
            rr.process(rcat)
            if rr_file is not None:
                rr.write(pathlib.Path(rr_file).as_posix())
        dd.process(dcat)
        dr.process(dcat, rcat)
        dd.write(pathlib.Path(save_path).as_posix(), rr, dr)
        return dd.calculateXi(rr, dr)[0]

def get_wtheta_with_weight(delta_sys, params, dcat, rcat, rr, nn_config,
                           nthreads=0, fname=None, force=False):
    if fname is not None and fname.exists() and not force:
        return Table.read(fname)["xi"]
    dcat = add_weights(dcat, linear_weights(params, delta_sys))
    dd = treecorr.NNCorrelation(nn_config, num_threads=nthreads)
    dr = treecorr.NNCorrelation(nn_config, num_threads=nthreads)
    dd.process(dcat)
    dr.process(dcat, rcat)
    if fname is not None:
        dd.write(pathlib.Path(fname).as_posix(), rr, dr)
    return dd.calculateXi(rr, dr)[0]

def run_treecorr_once(zbin, params, delta_sys, rcat, fgood, weights_mask, rr,
                      nthreads=0, mock_num=None):
    plot_root_path = params["common"]["plot_root_path"]
    chain_root_path = params["common"]["chain_root_path"]
    if mock_num is not None:
        data_cat = Table.read(
            params["common"]["mock_root_path"].joinpath(
                params["common"]["mcat_zbin_fname"].format(
                    num=mock_num, zlim=params["common"]["zedges"][zbin])))
        dcat_hpix = hp.ang2pix(
            fgood.nside, data_cat["RA"], data_cat["DEC"], lonlat=True)
        data_cat = data_cat[weights_mask[dcat_hpix]]
        dra = data_cat["RA"].copy()
        ddec = data_cat["DEC"].copy()
        dcat_hpix = dcat_hpix[weights_mask[dcat_hpix]]
        del data_cat
    else:
        data_cat = lsssys.Redmagic(params["common"]["dcat_path"].as_posix())
        data_cat.cutz(*params["common"]["zedges"][zbin])
        dcat_hpix = hp.ang2pix(
            fgood.nside, data_cat.ra, data_cat.dec, lonlat=True)
        data_cat.cut(weights_mask[dcat_hpix])
        dra = data_cat.ra.copy()
        ddec = data_cat.dec.copy()
        dcat_hpix = dcat_hpix[weights_mask[dcat_hpix]]
        del data_cat
    idxs = np.full(fgood.mask.size, -1)
    assert (~fgood.mask).sum() == delta_sys.shape[1]
    assert fgood.maskpix.size == delta_sys.shape[1]
    idxs[~fgood.mask] = np.arange((~fgood.mask).sum())
    assert idxs.max() < delta_sys.shape[1]
    dcat_idxs = idxs[dcat_hpix].copy()
    assert dcat_idxs.max() < delta_sys.shape[1]
    dcat = treecorr.Catalog(
        ra=dra.copy(), dec=ddec.copy(), ra_units="deg", dec_units="deg")
    del dra, ddec, dcat_hpix, idxs
    x = delta_sys[:,dcat_idxs]
    if params["corr"]["treecorr"]["with_analytic"]:
        analytic_params_path = params["common"]["plot_root_path"].joinpath(
            "zbin{}".format(zbin),
            params["fit"]["analytic_output_fname"].format(
                num=mock_num, nside=params["common"]["nside_fit"][zbin]))
        analytic_wtheta_path = params["common"]["plot_root_path"].joinpath(
            "zbin{}".format(zbin),
            params["corr"]["analytic_output_fname"].format(
                num=mock_num, nside=params["common"]["nside_fit"][zbin]))
        wtheta_analytic = get_wtheta_with_weight(
            x, np.load(analytic_params_path, allow_pickle=True), dcat, rcat, rr,
            params["corr"]["treecorr"]["nn_config"], nthreads,
            analytic_wtheta_path, params["corr"]["force_do_corr"])
        del analytic_params_path, analytic_wtheta_path
    else:
        wtheta_analytic = None
    if params["corr"]["treecorr"]["with_mean"]:
        chain_fname = chain_root_path.joinpath(
            params["common"]["chain_fname"].format(
                zbin=zbin, num=mock_num,
                nside=params["common"]["nside_fit"][zbin]))
        mean_wtheta_path = plot_root_path.joinpath(
            "zbin{}".format(zbin),
            params["corr"]["treecorr"]["mean_output_fname"].format(
                num=mock_num, nside=params["common"]["nside_fit"][zbin]))
        chain = read_chain(chain_fname, params["common"]["nburnin"], True)[0]
        wtheta_mean = get_wtheta_with_weight(
            x, chain.mean(axis=0), dcat, rcat, rr,
            params["corr"]["treecorr"]["nn_config"], nthreads,
            mean_wtheta_path, params["corr"]["force_do_corr"])
        del chain_fname, mean_wtheta_path
        if params["corr"]["treecorr"]["with_samples"]:
            wtheta_real_path = plot_root_path.joinpath(
                "zbin{}".format(zbin),
                "wtheta_real{}_fit{}_nside{}_nsteps{}_v{}.npy".format(
                    "_mock{}".format(mock_num) if mock_num is not None else "",
                    params["common"]["nside_fit"][zbin],
                    params["corr"]["nside"], params["corr"]["nreal"], 
                    params["common"]["chain_version"]))
            wtheta_cov_path = plot_root_path.joinpath(
                "zbin{}".format(zbin),
                "wtheta_stat_cov{}_fit{}_nside{}_nsteps{}_v{}.pkl".format(
                    "_mock{}".format(mock_num) if mock_num is not None else "",
                    params["common"]["nside_fit"][zbin],
                    params["corr"]["nside"], params["corr"]["nreal"], 
                    params["common"]["chain_version"]))
            if params["corr"]["force_do_corr"] or not wtheta_real_path.exists():
                chain_steps = chain[np.random.choice(
                    len(chain), params["corr"]["nreal"], replace=False)]
                wtheta_real = np.zeros((len(chain_steps), rr.nbins))
                for i, this_step in enumerate(tqdm.tqdm(
                        chain_steps, desc="zbin{}".format(zbin))):
                    wtheta_real[i] = get_wtheta_with_weight(
                        x, this_step, dcat, rcat, rr,
                        params["corr"]["treecorr"]["nn_config"], nthreads)
                # print("", flush=True)
                np.save(wtheta_real_path, wtheta_real)
                del chain_steps
            else:
                wtheta_real = np.load(wtheta_real_path)
            del wtheta_real_path
            if params["corr"]["force_do_corr"] or not wtheta_cov_path.exists():
                wtheta_cov = np.cov(wtheta_real, rowvar=False, bias=True)
                wtheta_cov_path.write_bytes(wtheta_cov.dumps())
            else:
                wtheta_cov = np.load(wtheta_cov_path, allow_pickle=True)
            del wtheta_cov_path, wtheta_real
            wtheta_err = np.sqrt(np.diag(wtheta_cov))
        else:
            wtheta_cov = None
            wtheta_err = None
        del chain
    else:
        wtheta_mean = None
        wtheta_err = None
        wtheta_cov = None
    return wtheta_analytic, wtheta_mean, wtheta_cov, wtheta_err

def run_treecorr(zbin, params, des_spec, weights_mask, fgood):
    nthreads = params["common"].get("max_nthreads", 0)
    rr = treecorr.NNCorrelation(
        params["corr"]["treecorr"]["nn_config"], num_threads=nthreads)
    rand_zedges = params["common"]["zedges"][
        params["corr"]["treecorr"]["rr_zbin"]]
    rand_cat = lsssys.Randoms(params["common"]["rcat_path"].as_posix())
    if not np.allclose(rand_cat.z, rand_cat.z[0]):
        rand_cat.cutz(*rand_zedges)
    else:
        rand_zedges = None
    rcat_hpix = hp.ang2pix(
        fgood.nside, rand_cat.ra, rand_cat.dec, lonlat=True)
    rand_cat.mask_array(weights_mask[rcat_hpix])
    del rcat_hpix
    rcat = treecorr.Catalog(
        ra=rand_cat.ra, dec=rand_cat.dec, ra_units="deg", dec_units="deg")
    try:
        if params["corr"]["force_do_corr"]:
            raise IOError("Must not read RR from file")
        rr.read(params["corr"]["treecorr"]["rr_paircounts_file"].as_posix())
    except (OSError, IOError):
        print("Running RR paircounts", flush=True)
        rr.process(rcat)
        rr.write(params["corr"]["treecorr"]["rr_paircounts_file"].as_posix())
    params["corr"]["treecorr"]["nn_config"]["nthreads"] = nthreads
    x = read_standard_sys_maps(
        params["common"]["sys_root_path"].joinpath(
            params["common"]["sys_fname"].format(
                fit=params["common"]["nside_fit"][zbin], nside=fgood.nside)))
    if not params["common"]["is_mock"]:
        wtheta_uw = get_wtheta_unweighted(
            params["common"]["plot_root_path"].joinpath(
                "zbin{}".format(zbin),
                params["corr"]["uw_fname"]),
            params["corr"]["treecorr"]["nn_config"],
            params["common"]["dcat_path"],
            params["common"]["rcat_path"],
            params["corr"]["treecorr"]["cat_config"],
            params["common"]["zedges"][zbin], rand_zedges, 
            params["corr"]["treecorr"]["rr_uw_paircounts_file"], nthreads)
        w_ana, w_mean, w_cov, w_err = run_treecorr_once(
            zbin, params, x, rcat, fgood, weights_mask, rr, nthreads)
        if params["corr"]["make_plots"]:
            plot_correlation_functions(
                params["corr"], zbin, des_spec, wtheta_uw,
                params["common"]["plot_root_path"].joinpath(
                    "zbin{}".format(zbin),
                    params["corr"]["treecorr"]["plot_fname"].format(
                        nside=params["common"]["nside_fit"][zbin])), w_ana, 
                w_mean, w_err, params["common"]["plot_root_path"].joinpath(
                    "zbin{}".format(zbin),
                    params["corr"]["treecorr"]["analytic_plot_fname"].format(
                        nside=params["common"]["nside_fit"][zbin])),
                params["corr"]["show_corr_plot"])
    else:
        wtheta_uw = np.empty((
            len(params["common"]["mock_nums"]),
            params["corr"]["treecorr"]["nn_config"]["nbins"]))
        w_mean = np.empty_like(wtheta_uw)
        w_cov = np.empty_like(wtheta_uw)
        for i, mocki in enumerate(tqdm.tqdm(
                params["common"]["mock_nums"], desc="Mock", 
                dynamic_ncols=True)):
            wtheta_uw[i] = get_wtheta_unweighted(
                params["common"]["plot_root_path"].joinpath(
                    "zbin{}".format(zbin),
                    params["corr"]["uw_fname"].format(num=mocki)),
                params["corr"]["treecorr"]["nn_config"],
                params["common"]["mock_root_path"].joinpath(
                    params["common"]["mcat_zbin_fname"].format(
                        zlim=params["common"]["zedges"][zbin], 
                        num=mocki)),
                params["common"]["rcat_path"],
                params["corr"]["treecorr"]["cat_config"],
                None, rand_zedges, 
                params["corr"]["treecorr"]["rr_uw_paircounts_file"], nthreads)
            w_ana, w_mean[i], w_cov[i], w_err = run_treecorr_once(
                zbin, params, x, rcat, fgood, weights_mask, rr, nthreads, mocki)
            if params["corr"]["make_plots"]:
                plot_correlation_functions(
                    params["corr"], zbin, des_spec, wtheta_uw[i],
                    params["common"]["plot_root_path"].joinpath(
                        "zbin{}".format(zbin),
                        params["corr"]["treecorr"]["plot_fname"].format(
                            num=mocki, 
                            nside=params["common"]["nside_fit"][zbin])),
                    w_ana, w_mean[i], w_err,
                    params["common"]["plot_root_path"].joinpath(
                        "zbin{}".format(zbin),
                        params["corr"]["treecorr"][
                            "analytic_plot_fname"].format(
                                num=mocki, 
                                nside=params["common"]["nside_fit"][zbin])),
                    params["corr"]["show_corr_plot"])
    return w_mean, w_cov

#----------------------------------- loops ------------------------------------#
def run_fits(params):
    params["common"]["chain_root_path"].mkdir(parents=True, exist_ok=True)
    params["common"]["plot_root_path"].mkdir(parents=True, exist_ok=True)
    if params["common"]["rotated"]:
        params["fit"]["param_names"] = [
            fr"$a^\prime_{{{i}}}$" for i in range(len(all_systematics))]
    else:
        params["fit"]["param_names"] = [
            fr"$a_{{{i}}}$" for i in range(len(all_systematics))]
    for zbin in params["common"]["zbins"]:
        if params["common"]["fit_type"] == "diag":
            run_fit(zbin, params)
        elif params["common"]["fit_type"] == "const_cov":
            run_const_cov_fit(zbin, params)
        else:
            raise ValueError(
                "Unrecognized fit_type: {}".format(
                    params["common"]["fit_type"]))

def run_corr(params):
    nside = params["corr"]["nside"]
    fgood4096 = lsssys.Mask(
        params["common"]["cat_root_path"].joinpath(
            params["common"]["fracgood_fname"].format(4096)).as_posix())
    try:
        fgood = lsssys.Mask(
            params["common"]["cat_root_path"].joinpath(
                params["common"]["fracgood_fname"].format(nside)).as_posix())
    except (FileNotFoundError, IOError):
        fgood = fgood4096.degrade(nside, 0.8, False)
        fgood.save(
            params["common"]["cat_root_path"].joinpath(
                params["common"]["fracgood_fname"].format(nside)).as_posix())
    for key in all_systematics:
        read_sysmap(key, nside, fgood4096.fracdet)
    if not params["corr"]["wmask_path"].exists():
        if "sys_fname" not in params["common"]:
            params["common"]["sys_fname"] = (
                "standard_systematics_{}_fit{{fit}}_nside{{nside}}.pkl".format(
                    "eigenbasis" if params["common"]["rotated"] else 
                    "unrotated"))
        sys_params = [np.load(
            params["common"]["plot_top_path"].joinpath(
                f"zbin{zbin}", 
                f"mean_parameters_nside{params['common']['nside_fit'][zbin]}"
                ".pkl"), 
            allow_pickle=True) for zbin in np.sort(
                list(params["common"]["zedges"].keys()))]
        sys_vals = [read_standard_sys_maps(
            params["common"]["sys_root_path"].joinpath(
                params["common"]["sys_fname"].format(
                    fit=params["common"]["nside_fit"][zbin], 
                    nside=fgood.nside)), 
            fgood.nside, params["common"]["nside_fit"][zbin], 
            params["common"]["sys_shift_fname"], 
            params["common"]["sys_weight_fname"], 
            params["common"]["rot_mat_fname"] if params["common"]["rotated"] 
            else None, fgood4096, 0.8) for zbin in np.sort(
                list(params["common"]["zedges"].keys()))]
        delta_elin_maps = [
            delta_sys_linear(p, v) for p, v in zip(sys_params, sys_vals)]
        del sys_params, sys_vals
        wmask = create_delta_lin_mask(
            ~fgood.mask, params["corr"]["delta_elin_max"], *delta_elin_maps)
        np.save(params["corr"]["wmask_path"], wmask)
    else:
        wmask = np.load(params["corr"]["wmask_path"], allow_pickle=True)
    del fgood4096
    try:
        des_dvec = twopoint.TwoPointFile.from_fits(params["corr"]["des_data"])
        des_wtheta_spec = des_dvec.get_spectrum("wtheta")
    except FileNotFoundError:
        des_wtheta_spec = None
    for zbin in params["common"]["zbins"]:
        wtheta_tc, cov_tc = run_treecorr(
            zbin, params, des_wtheta_spec, wmask, fgood)

#------------------------------------ Main ------------------------------------#
def main(params):
    """
    Main script function, which calls the other functions. Edit the function
    calls here to include or exclude various portions of the pipeline. The input
    parameters should be a dictionary-like object
    """
    if params["common"]["do_fits"]:
        run_fits(params)
    if params["common"]["do_corr"]:
        run_corr(params)

#-------------------------------- Script runner -------------------------------#
if __name__ == "__main__":
    # plt.style.use(["paper", "seaborn-talk", "colorblind"])
    # plt.rcParams["figure.figsize"] = (10.0, 4.96)
    des_year = 3
    if des_year == 1:
        zedges = np.around(np.arange(15, 91, 15) / 100, 2)
    elif des_year == 3:
        zedges = np.around([0.15, 0.35, 0.5, 0.65, 0.8, 0.9], 2)
    else:
        raise ValueError(f"Unknown des_year: {des_year}")
    theta_edges = np.logspace(np.log(2.5), np.log(250.0), num=21, base=np.e)
    contamination = [0, len(all_systematics)]
    params = dict()
    params["common"] = {
        "rotated": True,
        "zbins": [1, 2, 3, 4, 5],
        "zedges": dict((i, zedges[i-1:i+1]) for i in range(1, zedges.size)),
        "is_mock": True,
        "mock_nums": np.arange(100),
        "des_root_path": pathlib.Path(
            "/", "spiff", "wagoner47", "des", f"y{des_year}"),
        "fracgood_fname": ("y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_"
                           "maglim_v2.2_mask_nside{}.fits.gz"),
        "fit_type": "const_cov",  # "diag", "const_cov"
        "chain_version": 6,
        "dcat_zcol": "ZREDMAGIC",
        "rcat_zcol": "Z",
        "mcat_zbin_fname": "cat_mock_{num}_z{zlim[0]}-{zlim[1]}.fits",
        "nside_fit": dict((i+1, ni) for i, ni in enumerate(
            [128, 128, 128, 128, 128])),
        "do_fits": True,
        "do_corr": True,
        "rot_mat_fname": "rotation_matrix_nside{}.pkl",
        "sys_weight_fname": "systematics_weights_nside{}.npy",
        "sys_shift_fname": "systematics_means_nside{}.npy",
        "max_nthreads": 16,
        "max_nprocesses": 8
    }
    params["common"]["mock_run"] = (
        f"full_mocks_v{params['common']['chain_version']}")
    params["common"]["mock_top_path"] = pathlib.Path(
    	"/", "spiff", "wagoner47", "mock_runs", f"y{des_year}", 
    	params["common"]["mock_run"], "lognormal_mock_output", "catalogs")
    params["common"]["sys_root_path"] = (
        params["common"]["des_root_path"].joinpath("systematics"))
    params["common"]["cat_root_path"] = (
        params["common"]["des_root_path"].joinpath("redmagic"))
    results_top_path = pathlib.Path(
        "/", "spiff", "wagoner47", f"finalized_systematics_results_y{des_year}", 
        f"v{params['common']['chain_version']}_results")
    params["common"]["chain_top_path"] = results_top_path.joinpath(
        f"finalized_systematics_chains_y{des_year}")
    params["common"]["plot_top_path"] = results_top_path.joinpath(
        f"finalized_systematics_plots_y{des_year}")
    params["common"]["dcat_path"] = params["common"]["cat_root_path"].joinpath(
        "y3_gold_2.2.1_wide_sofcol_run_redmapper_v0.5.1_combined_hd3_hl2_sample"
        ".fits.gz")
    params["common"]["rcat_path"] = params["common"]["cat_root_path"].joinpath(
        "y3_gold_2.2.1_wide_sofcol_run_redmapper_v0.5.1_combined_hd3_hl2_sample"
        "_randoms_zbin3_n60.fits.gz")
    params["common"]["sys_fname"] = (
        "standard_systematics_{}_fit{{fit}}_nside{{nside}}.pkl".format(
            "eigenbasis" if params["common"]["rotated"] else "unrotated"))
    params["fit"] = {
        "force_do_fits": False,
        "plot_chains": True,
        "show_chain_plots": True and (
            mpl.get_backend() in mpl.rcsetup.interactive_bk),
        "cov_mat_theta_max": 0.1
    }
    if params["common"]["is_mock"]:
        params["common"]["nburnin"] = 300
        params["fit"]["nsteps"] = 700
    else:
        params["common"]["nburnin"] = 0
        params["fit"]["nsteps"] = 1000
    params["fit"]["nsteps"] += params["common"]["nburnin"]
    params["common"]["chain_fname"] = (
        "zbin{{zbin}}{}_linear_{}{}_nside{{nside}}_nsteps{}_v{}.fits").format(
            "_mock{num}" if params["common"]["is_mock"] else "",
            "eigenbasis" if params["common"]["rotated"] else "unrotated",
            "" if params["common"]["fit_type"] == "diag" else "_{}".format(
                params["common"]["fit_type"]), params["fit"]["nsteps"],
            params["common"]["chain_version"])
    params["fit"]["analytic_output_fname"] = (
        "analytic_parameters{}_nside{{nside}}.pkl").format(
            "_mock{num}" if params["common"]["is_mock"] else "")
    params["fit"]["analytic_covariance_fname"] = (
        "analytic_parameter_covariance_matrix{}_nside{{nside}}.pkl").format(
            "_mock{num}" if params["common"]["is_mock"] else "")
    params["fit"]["const_cov_fname"] = (
        "delta_t_cov_mat_analytic{}_nside{{nside}}_v{}.pkl").format(
            "_mock{num}" if params["common"]["is_mock"] else "", 
        params["common"]["chain_version"])
    params["corr"] = {
        "force_do_corr": False,
        "delta_elin_max": 0.2,
        "nside": 4096,
        "make_plots": False,
        "xlabel": r"$\theta \left[\mathrm{arcmin}\right]$",
        "invalid_region": dict((i+1, xi) for i, xi in enumerate(
            [43, 27, 20, 16, 14])),
        "ylim": dict((i+1, ylimi) for i, ylimi in enumerate(
            [[0, 2.5], [-0.1, 1.7], [-0.1, 1.5], [-0.5, 1.5], [-0.5, 1.5]])),
        "yticks": dict((i+1, yticki) for i, yticki in enumerate(
            [[0, 0.5, 1, 1.5, 2, 2.5], [0, 0.5, 1, 1.5], [0, 0.5, 1, 1.5],
             [-0.5, 0, 0.5, 1, 1.5], [-0.5, 0, 0.5, 1, 1.5]])),
        "text_yloc": dict((i+1, yloci) for i, yloci in enumerate(
            [0.25, 0.25, 1.2, 1, 1])),
        "show_corr_plot": True and (
            mpl.get_backend() in mpl.rcsetup.interactive_bk),
        "des_data": params["common"]["des_root_path"].joinpath(
            "2pt_NG_mcal_1110.fits"),
        "nreal": 250,
        "uw_fname": "wtheta_unweighted{}.fits".format(
            "_mock{num}" if params["common"]["is_mock"] else "")
    }
    params["corr"]["yticklabels"] = dict(
        (key, [r"${}$".format(y) if y >= 0 else r"" for y in val]) for key,
        val in params["corr"]["yticks"].items())
    params["corr"]["wmask_path"] = params["common"]["cat_root_path"].joinpath(
        "systematics_weights_mask_max{}_nside{}_v{}.npy".format(
            params["corr"]["delta_elin_max"], params["corr"]["nside"],
            params["common"]["chain_version"]))
    params["corr"]["analytic_output_fname"] = (
        "wtheta_analytic{}_fit{{nside}}_nside{}_v{}.fits").format(
            "_mock{num}" if params["common"]["is_mock"] else "",
            params["corr"]["nside"], params["common"]["chain_version"])
    params["corr"]["treecorr"] = {
        "nn_config": {
            "min_sep": theta_edges[0],
            "max_sep": theta_edges[-1],
            "nbins": theta_edges.size - 1,
            "bin_slop": 0,
            "sep_units": "arcmin",
            "metric": "Arc"
            },
        "cat_config": {
            "ra_col": "RA",
            "dec_col": "DEC",
            "ra_units": "deg",
            "dec_units": "deg"
            },
        "with_analytic": True,
        "with_mean": True,
        "with_samples": True and not params["common"]["is_mock"],
        "corrected_errors": False,
        "rr_zbin": 3,
        "rr_paircounts_file": params["common"]["plot_top_path"].joinpath(
            "zbin3", "rr_paircounts_nside{}_v{}.fits".format(
                params["corr"]["nside"], params["common"]["chain_version"])),
        "rr_uw_paircounts_file": params["common"]["plot_top_path"].joinpath(
            "zbin3", "rr_paircounts_unmasked_finalized.fits"),
        "mean_output_fname": (
            "wtheta_mean{}{}_fit{{nside}}_nside{}_v{}.fits").format(
                "" if params["common"]["fit_type"] == "diag" else "_{}".format(
                    params["common"]["fit_type"]),
                "_mock{num}" if params["common"]["is_mock"] else "",
                params["corr"]["nside"], params["common"]["chain_version"]),
        "analytic_plot_fname": (
            "wtheta_analytic{}_fit{{nside}}_nside{}_v{}.png").format(
                "_mock{num}" if params["common"]["is_mock"] else "",
                params["corr"]["nside"], params["common"]["chain_version"]),
        "plot_fname": (
            "wtheta_analytic_mcmc{}_errs{}_fit{{nside}}_nside{}_nsteps{}_v{}."
            "png").format(
                "" if params["common"]["fit_type"] == "diag" else "_{}".format(
                    params["common"]["fit_type"]), "_mock{num}" if
                params["common"]["is_mock"] else "", params["corr"]["nside"],
                params["corr"]["nreal"], params["common"]["chain_version"])
    }
    if params["common"]["is_mock"]:
        for n_cont in np.ravel(np.atleast_1d(contamination)):
            params["common"]["mock_root_path"] = (
                params["common"]["mock_top_path"].joinpath(
                    f"n_contaminate_{n_cont}"))
            params["common"]["chain_root_path"] = (
                params["common"]["chain_top_path"].joinpath(
                    params["common"]["mock_run"], f"contamination_{n_cont}"))
            params["common"]["plot_root_path"] = (
                params["common"]["plot_top_path"].joinpath(
                    params["common"]["mock_run"], f"contamination_{n_cont}"))
            main(params)
    else:
        params["common"]["plot_root_path"] = params["common"]["plot_top_path"]
        params["common"]["chain_root_path"] = params["common"]["chain_top_path"]
        main(params)
