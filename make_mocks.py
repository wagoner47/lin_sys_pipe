import sys
import re
import numpy as np
import healpy as hp
import pathlib
from astropy.table import Table
import argparse
import healpix_util as hu
import copy
from tqdm.autonotebook import tqdm, trange
lsssys_path = "/home/wagoner47/lss_sys"
if lsssys_path not in sys.path:
    sys.path.insert(0, lsssys_path)
import lsssys

def next_power_of_2(n):
    return 2**int(np.ceil(np.log2(n)))

def calculate_number_density(catalog, mask, z_bin_edges=None):
    """
    Calculate the number density of galaxies on the sky, possibly in redshift
    bins.

    :param catalog: The catalog of galaxies for which to get the density
    :type catalog: :class:`lsssys.Catalog`
    :param mask: The pixel coverage mask object for calculating the total area
    :type mask: :class:`lsssys.HealMask` or :class:`lsssys.Mask`
    :param z_bin_edges: If given, specifies the redshift bin edges in which to
        get the number density. If 1D with length 2, only a single redshift bin
        is used, and the output will be a scalar. If 1D with length 3 or more,
        ``Nbins = len(z_bin_edges) - 1`` redshift bins are assumed with the
        upper edge of one bin corresponding to the lower edge of the next, and
        the output will be 1D with length ``Nbins``. If 2D, the shape must be
        (``Nbins``, 2), where ``Nbins`` is the number of redshift bins. In this
        case, the output will also be 1D with length ``Nbins``. If ``None``
        (default), all galaxies are assumed to be in a single redshift bin,
        so the output will again be scalar
    :type z_bin_edges: ``NoneType`` or (``Nbins+1``,) or (``Nbins``, 2)
        array-like of ``float``, optional
    :return: The number of galaxies in the redshift bin(s) and the number
        density per square arcminute of galaxies in the redshift bin(s)
    :rtype: 2-``tuple`` of ``float`` or (``Nbins``,) :class:`numpy.ndarray` of
        ``float``
    :rtype: ``float`` or (``Nbins``,) :class:`numpy.ndarray` of ``float``
    :raises ValueError: If ``z_bin_edges`` is 1D with length less than 2 or
        if it is 2D and the length of the first axis is not 2 or if it is not
        1D or 2D
    """
    mask_area = mask.area(units="arcmin")
    if z_bin_edges is None:
        return len(catalog.ra) / mask_area
    elif np.ndim(z_bin_edges) == 1:
        if len(z_bin_edges) < 2:
            raise ValueError("z_bin_edges must have length at least 2 if 1D")
        elif len(z_bin_edges) == 2:
            num_in_bin = catalog.eqinbin(
                np.min(z_bin_edges), np.max(z_bin_edges))[0].size
            return num_in_bin, num_in_bin / mask_area
        else:
            sorted_z = np.sort(z_bin_edges)
            num_in_bin = np.array(
                [catalog.eqinbin(this_z, next_z)[0].size for this_z, next_z in
                 zip(sorted_z[:-1], sorted_z[1:])])
            return num_in_bin, num_in_bin / mask_area
    elif np.ndim(z_bin_edges) == 2:
        if np.shape(z_bin_edges)[1] != 2:
            raise ValueError(
                "z_bin_edges must have length 2 along last axis if 2D")
        sorted_z = np.sort(z_bin_edges)
        num_in_bin = np.array(
            [catalog.eqinbin(this_bin[0], this_bin[1])[0].size for this_bin in
             sorted_z])
        return num_in_bin, num_in_bin / mask_area
    else:
        raise ValueError(
            "Invalid dimensions {} for z_bin_edges".format(
                np.ndim(z_bin_edges)))

def initialize_theory(theory_dir, lmax=3500, zbins=None, k0=None):
    """
    Initialize (and possibly log-normalize) a :class:`lsssys.Theory` object

    :param theory_dir: The top level directory containing the CosmoSIS theory
        output. Must contain a directory 'galaxy_cl' with files 'ell.txt' and
        'bin_{i}_{j}.txt' where 'i' and 'j' are redshift bin indices
    :type theory_dir: ``str`` or :class:`os.PathLike`
    :param zbins: The redshift bin(s) for which to create mocks. Assumes there
        are ``Nbins`` total redshift bins, and the indexing starts from 1
        (not 0!). If given, this must have length of at least 1 and no larger
        than ``Nbins``. If ``None`` (default), makes a mock for all redshift
        bins
    :type zbins: ``NoneType`` or 1D array-like of ``int``, optional
    :param k0: Parameter for the skewness of the lognormal field, which can be
        different for each redshift bin, in which case it should have length
        equal to ``zbins``. If ``None`` (default), creates a Gaussian density
        field rather than a lognormal one
    :type k0: ``NoneType`` or ``int`` or 1D array-like of ``int``, optional
    :return: The theory object created and possibly log-normalized
    :rtype: :class:`lsssys.Theory`
    :raises IOError: If no 'galaxy_cl' directory exists under ``theory_dir``
    :raises ValueError: If ``zbins`` is not ``None`` and ``k0`` is not ``None``
        and ``len(k0) != 1`` and ``len(k0) != len(zbins)``
    """
    cldir = pathlib.Path(theory_dir).expanduser().resolve().joinpath(
        "galaxy_cl")
    if not cldir.exists():
        raise IOError("No galaxy_cl directory in given theory_dir")
    if zbins is None:
        zbins = "all"
    theory = lsssys.Theory(cldir.as_posix() + "/", lmax=lmax, pos_binlist=zbins)
    if k0 is not None:
        if hasattr(k0, "__len__"):
            if zbins != "all" and len(k0) > 1 and len(k0) != len(zbins):
                raise ValueError("Mismatch between zbins and k0")
            elif len(k0) == 1:
                theory.lognormalise(k0=k0[0])
            else:
                theory.lognormalise(k0=k0)
        else:
            theory.lognormalise(k0=k0)
    return theory

def random_points_on_sphere(num_to_generate, ra_range=None, dec_range=None):
    """
    Generate random points on a sphere, with the option to specify the range
    of coordinates to allow. Uses sphere point picking.

    :param num_to_generate: The number of random points to generate
    :type num_to_generate: ``int``
    :param ra_range: Optional parameter to limit the allowed range in right
        ascension (in degrees). The total range should not be more than 360. If
        ``None`` (default), uses the range ``[0.0, 360.0]``
    :type ra_range: (2,) array-like of ``float`` or ``NoneType``, optional
    :param dec_range: Optional parameter to limit the allowed range in
        declination (in degrees). The total range should not be more than 180.
        If ``None`` (default), uses the range ``[-90.0, 90.0]``
    :type dec_range: (2,) array-like of ``float`` or ``NoneType``, optional
    :return: A tuple of arrays of size ``num_to_generate``, with the first
        specifying the right ascension of the points and the second the
        declination
    :rtype: ``tuple`` 2 of (``num_to_generate``,) :class:`numpy.ndarray` of
        ``float``
    :raises ValueError: If the difference in ``ra_range`` is more than 360 or
        the difference in ``dec_range`` is more than 180
    """
    if ra_range is not None:
        delta_ra = ra_range[1] - ra_range[0]
        if np.abs(delta_ra) > 360.0:
            raise ValueError("ra_range is too wide")
        min_ra = ra_range[0]
    else:
        delta_ra = 360.0
        min_ra = 0.0
    if dec_range is not None:
        delta_x = (np.sin(np.deg2rad(dec_range[0]))
                   - np.sin(np.deg2rad(dec_range[1])))
        if np.abs(delta_x) > 2:
            raise ValueError("dec_range is too wide")
        min_x = -np.sin(np.deg2rad(dec_range[0]))
    else:
        delta_x = -2
        min_x = 1
    ra = delta_ra * np.random.rand(num_to_generate) + min_ra
    dec = -np.rad2deg(
        np.arcsin(delta_x * np.random.rand(num_to_generate) + min_x))
    return ra, dec

def generate_randoms(mask, max_ndraw, ra_min=0, ra_max=360, dec_min=-90,
                     dec_max=90, factor=2):
    """
    Generate random points within the footprint

    This can be used to generate points distributed uniformly over the entire
    footprint for generating a CDF from which to draw mock galaxy positions. It
    does not generate the CDF, it merely draws positions from a uniform density
    field with the mask applied. Unlike :func:`~.generate_random_points`, this
    function generates the points itself rather than calling :mod:`healpix_util`

    You should make sure that ``factor`` is large enough that the number of
    points generated here, ``int(max_ndraw * factor)``, is enough to sample the
    CDF well when ``max_ndraw`` is the maximum final number of mock galaxies to
    be drawn in any redshift bin. It needs to be larger than 1, but there is no
    guarantee that the default (2) is enough.

    :param mask: The pixel coverage mask
    :type mask: :class:`lsssys.Mask` or :class:`lsssys.HealMask`
    :param max_ndraw: The maximum number of mock galaxies that will be drawn
        in any redshift bin. This way, the set of points generated here can be
        used to generate the CDF for all of the redshift bins
    :type max_ndraw: ``int``
    :param ra_min: The minimum right ascension (in degrees) in which the data
        will be, to prevent drawing points far from the region of interest.
        Default 0.0
    :type ra_min: ``float``, optional
    :param ra_max: The maximum right ascension (in degrees) in which the data
        will be, to prevent drawing points far from the region of interest.
        Default 360.0
    :type ra_max: ``float``, optional
    :param dec_min: The minimum declination (in degrees) in which the data
        will be, to prevent drawing points far from the region of interest.
        Default -90.0
    :type dec_min: ``float``, optional
    :param dec_max: The maximum declination (in degrees) in which the data
        will be, to prevent drawing points far from the region of interest.
        Default 90.0
    :type dec_max: ``float``, optional
    :param factor: The multiplicative factor on ``max_ndraw`` that sets how many
        points should be generated here. This should probably be at least 2, but
        that is not checked. Default 2
    :type factor: ``int`` or ``float``, optional
    :return: An array of right ascension and an array of declination that can be
        used with weights to draw mock galaxy positions from the non-uniform
        density field with or without systematics
    :rtype: 2-``tuple`` of (``int(factor * max_ndraw)``,) :class:`numpy.ndarray`
        of ``float``
    """
    ra = np.zeros(int(factor * max_ndraw))
    dec = np.zeros_like(ra)
    nleft = ra.size
    nside = mask.nside
    if 0 < ra_max < ra_min:
        shift_ra = True
        ra_range = [ra_min - 360, ra_max]
    else:
        shift_ra = False
        ra_range = [ra_min, ra_max]
    while nleft > 0:
        # print("Number of points still needed:", nleft, flush=True)
        # print("Sphere point pick", flush=True)
        new_ra, new_dec = random_points_on_sphere(
            10 * ra.size, ra_range, [dec_min, dec_max])
        if shift_ra:
            new_ra[new_ra < 0] += 360.0
        # print("Figure out which points are on good pixels", flush=True)
        keep = np.where(
            ~mask.mask[hp.ang2pix(nside, new_ra, new_dec, lonlat=True)])[0]
        if keep.size > 0:
            # print(
            #     "Number of points on good pixels:", keep.size, flush=True)
            i_start = ra.size - nleft
            nkeep = min(nleft, keep.size)
            i_stop = i_start + nkeep
            if nkeep < keep.size:
                keep = np.random.choice(keep, size=nkeep, replace=False)
            # print("Add new points", flush=True)
            ra[i_start:i_stop] = new_ra[keep]
            dec[i_start:i_stop] = new_dec[keep]
            # print("Adjust nleft", flush=True)
            nleft -= nkeep
    return ra, dec

def generate_random_points(mask, max_ndraw, ra_min=0, ra_max=360, dec_min=-90,
                           dec_max=90, factor=2):
    """
    Generate random points within the footprint

    This can be used to generate points distributed uniformly over the entire
    footprint for generating a CDF from which to draw mock galaxy positions. It
    does not generate the CDF, it merely draws positions from a uniform density
    field with the mask applied.

    You should make sure that ``factor`` is large enough that the number of
    points generated here, ``int(max_ndraw * factor)``, is enough to sample the
    CDF well when ``max_ndraw`` is the maximum final number of mock galaxies to
    be drawn in any redshift bin. It needs to be larger than 1, but there is no
    guarantee that the default (2) is enough.

    :param mask: The pixel coverage mask
    :type mask: :class:`lsssys.Mask` or :class:`lsssys.HealMask`
    :param max_ndraw: The maximum number of mock galaxies that will be drawn
        in any redshift bin. This way, the set of points generated here can be
        used to generate the CDF for all of the redshift bins
    :type max_ndraw: ``int``
    :param ra_min: The minimum right ascension (in degrees) in which the data
        will be, to prevent drawing points far from the region of interest.
        Default 0.0
    :type ra_min: ``float``, optional
    :param ra_max: The maximum right ascension (in degrees) in which the data
        will be, to prevent drawing points far from the region of interest.
        Default 360.0
    :type ra_max: ``float``, optional
    :param dec_min: The minimum declination (in degrees) in which the data
        will be, to prevent drawing points far from the region of interest.
        Default -90.0
    :type dec_min: ``float``, optional
    :param dec_max: The maximum declination (in degrees) in which the data
        will be, to prevent drawing points far from the region of interest.
        Default 90.0
    :type dec_max: ``float``, optional
    :param factor: The multiplicative factor on ``max_ndraw`` that sets how many
        points should be generated here. This should probably be at least 2, but
        that is not checked. Default 2
    :type factor: ``int`` or ``float``, optional
    :return: An array of right ascension and an array of declination that can be
        used with weights to draw mock galaxy positions from the non-uniform
        density field with or without systematics
    :rtype: 2-``tuple`` of (``int(factor * max_ndraw)``,) :class:`numpy.ndarray`
        of ``float``
    """
    dmap = hu.DensityMap("RING", mask.fracdet)
    ra, dec = dmap.genrand(
        10 * int(factor * max_ndraw), ra_range=[ra_min, ra_max],
        dec_range=[dec_min, dec_max])
    pix = dmap.hpix.eq2pix(ra, dec)
    ra = ra[~mask.mask[pix]]
    dec = dec[~mask.mask[pix]]
    while ra.size < int(factor * max_ndraw):
        new_ra, new_dec = dmap.genrand(
            10 * int(factor * max_ndraw), ra_range=[ra_min, ra_max],
            dec_range=[dec_min, dec_max])
        new_pix = dmap.hpix.eq2pix(new_ra, new_dec)
        ra = np.append(ra, new_ra[~mask.mask[new_pix]])
        dec = np.append(dec, new_dec[~mask.mask[new_pix]])
    return ra[:int(factor * max_ndraw)], dec[:int(factor * max_ndraw)]

def contaminate_bin(delta, sys, coeffs):
    """
    Add systematics contamination for a single redshift bin

    :param delta: The overdensity field which we are contaminating
    :type delta: :class:`lsssys.Map`
    :param sys: The systematics maps
    :type sys: array-like of ``float`` (``Nmaps``, ``Npix``)
    :param coeffs: The systematics coefficients
    :type coeffs: array-like of ``float`` (``Nmaps``,)
    :return: The contaminated overdensity field
    :rtype: :class:`lsssys.Map`
    """
    sys_arr = np.atleast_2d(sys)
    coeffs_arr = np.atleast_1d(coeffs)
    assert sys_arr.shape[0] == coeffs_arr.size, ("Shape mismatch between maps"
                                                 " and coefficients")
    if sys_arr.shape[1] == delta.data.size:
        delta.data[~delta.mask] += np.sum(
            coeffs_arr[:,None] * sys_arr[~delta.mask], axis=0)
    elif sys_arr.shape[1] == delta.data[~delta.mask].size:
        delta.data[~delta.mask] += np.sum(coeffs_arr[:,None] * sys_arr, axis=0)
    else:
        raise ValueError("Shape mismatch between systematics and delta")
    return delta

def contaminate_mock(mock, n, zbins, fit_resols, coeff_dir, sys_dir, 
                     chain_version, order_dir=None, coeff_fname=None, 
                     order_fname=None, sys_fname=None):
    """
    Contaminate each bin in the mock. Redshift bins and resolutions are needed
    for accessing the correct systematics maps, coefficients, and orders

    :param mock: The mock to contaminate
    :type mock: :class:`lsssys.Mock`
    :param n: The number of maps with which to contaminate
    :type n: ``int``
    :param zbins: The redshift bin numbers for each bin in ``mock``, in order
    :type zbins: array-like ``int`` (``Nbins``,)
    :param fit_resols: The fitting resolution for each bin in ``mock``, in order
    :type fit_resols: array-like ``int`` (``Nbins``,)
    :param coeff_dir: Where the systematics coefficient files can be found. The
        file for each redshift bin should be located in subdirectories under
        this location called 'zbin#'
    :type coeff_dir: ``str`` or :class:`os.PathLike`
    :param sys_dir: Where the systematics map files can be found. The
        various systematics map files should be here and each should contain
        a 2D array where the 0th axis is indexed by map number
    :type sys_dir: ``str`` or :class:`os.PathLike`
    :param chain_version: The run version of the data chains that are used as 
        the "truth" for the contamination
    :type chain_version: ``int``
    :param order_dir: Where the map order files can be found, with files for
        each redshift bin contained within subdirectories 'zbin#'. If ``None``
        (default), this is assumed to be the same as ``coeff_dir``
    :type order_dir: ``str`` or :class:`os.PathLike` or ``NoneType``, optional
    :param coeff_fname: The file name template for coefficient files. If
        ``None`` (default), this is taken to be
        'mean_parameters_nside{res}.pkl' where res is the fit resolution
    :type coeff_fname: ``str`` or ``NoneType``, optional
    :param order_fname: The file name template for map order files. If
        ``None`` (default), this is taken to be
        'map_importance_order_const_cov_fit{{res}}_v{v}.npy' where 
        res is the fit resolution and v is the version of the data fit specified 
        by `chain_version`
    :type order_fname: ``str`` or ``NoneType``, optional
    :param sys_fname: The file name template for systematics map files. If
        ``None`` (default), this is taken to be
        'standard_systematics_eigenbasis_fit{{res}}_nside{mock.nside}.pkl'
        where res is the fit resolution and mock is the input mock
    :type sys_fname: ``str`` or ``NoneType``, optional
    :return: The contaminated mock
    :rtype: :class:`lsssys.Mock`
    """
    cdir = pathlib.Path(coeff_dir).resolve()
    sdir = pathlib.Path(sys_dir).resolve()
    if order_dir is None:
        odir = cdir
    else:
        odir = pathlib.Path(order_dir).resolve()
    if coeff_fname is None:
        coeff_fname = "mean_parameters_nside{res}.pkl"
    if order_fname is None:
        order_fname = (f"map_importance_order_const_cov_fit{{res}}_"
                       f"v{chain_version}.npy")
    if sys_fname is None:
        sys_fname = ("standard_systematics_eigenbasis_fit{{res}}_"
                     "nside{mock.nside}.pkl").format(mock=mock)
    cont_mock = copy.deepcopy(mock)
    for i, (zbin, nside) in enumerate(zip(zbins, fit_resols)):
        maps = np.load(
            odir / f"zbin{zbin}" / order_fname.format(res=nside),
            allow_pickle=True)[:n]
        cont_mock.delta[i] = contaminate_bin(
            cont_mock.delta[i],
            np.load(
                sdir / sys_fname.format(res=nside), allow_pickle=True)[maps],
            np.load(
                cdir / f"zbin{zbin}" / coeff_fname.format(res=nside),
                allow_pickle=True)[maps])
    return cont_mock

def gen_catalog_bin(ngal):
    """
    Generate a random catalog for a single bin using healpy routines
    
    :param ngal: The number of galaxies per pixel
    :type ngal: :class:`lsssys.Map`
    :return: The right ascension and declination of the generated catalog
    :rtype: ``tuple`` of 2 :class:`numpy.ndarray` of ``float``
    """
    pix = np.where(ngal.data > 0.)[0]
    n_gal = ngal.data[pix].astype(int)
    highres_nside = ngal.nside * next_power_of_2(2 * n_gal.max())
    pix_nest = hp.ring2nest(ngal.nside, pix)
    corners = np.array([c.T for c in hp.boundaries(
        ngal.nside, pix_nest, nest=True)])
    high_res_pix = np.concatenate([np.random.choice(
        hp.query_polygon(
            highres_nside, corn, nest=True), n) for corn, n in zip(
        corners, n_gal)])
    hpix_highres = hu.HealPix("nest", highres_nside)
    return hpix_highres.pix2eq(high_res_pix)
        
def gen_mock_catalogs(mock, cat_out_dir, mock_num, zedges, force):
    """
    Generate and save mock galaxy catalogs in each redshift bin, using the
    galaxy count maps

    This draws points using a higher resolution pixel map and the pixel 
    querying functionality of :mod:`healpy`.

    :param mock: The mock map object containing the density maps for the mocks
        in the redshift bin(s)
    :type mock: :class:`lsssys.Mock`
    :param cat_out_dir: The directory in which to store the mock catalogs that
        are generated
    :type cat_out_dir: ``str`` or :class:`os.PathLike`
    :param mock_num: The number of the current mock, for file naming
    :type mock_num: ``int``
    :param zedges: The edges of the redshift bin(s), for file naming
    :type zedges: (``Nbins+1``,) or (``Nbins``, 2) array-like of ``float``
    :param force: If ``True``, force catalog creation and writing even if the 
        catalogs already exist. Otherwise, don't overwrite any catalog files 
        that already exist, or skip generation completely if all of them exist
    :type force: ``bool``
    """
    save_dir = pathlib.Path(cat_out_dir).expanduser().resolve()
    if len(mock.ngal) == 1:
        if np.ndim(zedges) > 1:
            if np.shape(zedges)[0] > 1:
                raise ValueError("Too many redshift bin edges for mock")
            fout_list = [save_dir.joinpath(
                "cat_mock_{num}_z{zlim[0]}-{zlim[1]}.fits".format(
                    num=mock_num, zlim=zedges[0]))]
        elif np.ndim(zedges) == 0 or np.ndim(zedges) > 2:
            raise ValueError(
                "Invalid dimensions for zedges: {}".format(np.ndim(zedges)))
        else:
            if len(zedges) > 2:
                raise ValueError("Too many redshift bin edges for mock")
            fout_list = [save_dir.joinpath(
                "cat_mock_{num}_z{zlim[0]}-{zlim[1]}.fits".format(
                    num=mock_num, zlim=zedges))]
    else:
        if ((np.ndim(zedges) == 1 and len(zedges) != len(mock.ngal) + 1)
            or (np.ndim(zedges) == 2 and np.shape(zedges)[0]
                != len(mock.ngal))):
            raise ValueError(
                "Mismatch between number of redshift bin edges and number of"
                " mocks")
        elif np.ndim(zedges) == 0 or np.ndim(zedges) > 2:
            raise ValueError(
                "Invalid dimensions for zedges: {}".format(np.ndim(zedges)))
        else:
            if np.ndim(zedges) == 1:
                fout_list = [
                    save_dir.joinpath(
                        "cat_mock_{num}_z{zmin}-{zmax}.fits".format(
                            num=mock_num, zmin=zl, zmax=zu)) for zl, zu in zip(
                        zedges[:-1], zedges[1:])]
            else:
                fout_list = [
                    save_dir.joinpath(
                        "cat_mock_{num}_z{zlim[0]}-{zlim[1]}.fits".format(
                            num=mock_num, zlim=zbin)) for zbin in zedges]
    if force or not all([fouti.exists() for fouti in fout_list]):
        for ngali, fouti in zip(mock.ngal, fout_list):
            ra, dec = gen_catalog_bin(ngali)
            cat = Table([ra, dec], names=["RA", "DEC"])
            cat.write(fouti.as_posix(), overwrite=force)
        
def _load_deltas(mock, mock_num, delta_dir, n_bins):
    mock.delta = np.array([lsssys.Map() for _ in range(n_bins)])
    for i in range(n_bins):
        mock.delta[i].load(
            delta_dir.joinpath(
                f"mock_{mock_num}_bin{i}.fits").as_posix(), mock.nside)
        mock.delta[i].mask = mock.mask
        mock.delta[i].fracdet = mock.fracdet
    return mock

def _load_ngals(mock, mock_num, ngals_dir):
    temp_mock = lsssys.Mock(None, mock.nside, True)
    temp_mock.load(ngals_dir.joinpath(f"mock_{mock_num}.fits").as_posix())
    mock.ngal = copy.deepcopy(temp_mock.ngal)
    del temp_mock
    for i in range(len(mock.ngal)):
        mock.ngal[i].mask = mock.mask
        mock.ngal[i].fracdet = mock.fracdet
    return mock

def main(cat_path, mask_path, theory_dir, nside, z_bin_edges, out_dir, 
         chain_version=None, data_result_dir=None, nside_fit=None, zbins=None, 
         k0=None, nmocks=1, min_mock=0, lmax=3500, n_contam=0, 
         gen_catalogs=False, test=False, force_truth=None, force=None):
    """
    Create mock density field(s) and/or catalog(s)

    Note that the theory directory should be the upper-most directory for
    CosmoSIS output, and should contain a directory 'galaxy_cl'

    :param cat_path: The path to the catalog file
    :type cat_path: ``str`` or :class:`os.PathLike`
    :param mask_path: The path to the mask file
    :type mask_path: ``str`` or :class:`os.PathLike`
    :param theory_dir: The top level directory containing the CosmoSIS theory
        output. Must contain a directory 'galaxy_cl' with files 'ell.txt' and
        'bin_{i}_{j}.txt' where 'i' and 'j' are redshift bin indices
    :type theory_dir: ``str`` or :class:`os.PathLike`
    :param nside: The resolution at which to create the mock density field(s)
    :type nside: ``int``
    :param z_bin_edges: The edges of the redshift bins for all redshift bins.
        Only the edges for the selected bins will be used, but all must be given
    :type z_bin_edges: (``Nbins+1``,) array-like of ``float``
    :param out_dir: The parent directory in which to save the mock density
        map(s) and catalog(s). Make sure write permissions are available. The
        actual results will be stored under here in a directory
        'gaussian_mock_output' if ``k0`` is ``None`` or 'lognormal_mock_output'
        otherwise: counts maps will be under that in 'ngal_maps' and
        catalogs in 'catalogs'
    :type out_dir: ``str`` or :class:`os.PathLike`
    :param chain_version: The run version of the data chains that are used as 
        the "truth" for the contamination. Ignored if no contamination is added, 
        otherwise it **must** be specified. Default ``None``
    :type chain_version: ``int`` or ``NoneType``, optional
    :param data_result_dir: The directory within which the results from the data 
        systematics fits can be found. This is needed to ensure that the proper 
        coefficients are used to contaminate the mocks. The mean fit parameters 
        as well as the map importance order for each redshift bin should be 
        stored in subdirectories 'zbin#' under this directory, where '#' is 
        replaced by the bin numbers included from `zbins`. The names of these 
        files are assumed to be 'mean_parameters_nside{res}.pkl' and 
        'map_importance_order_const_cov_fit{res}_v{v}.npy', where 'res' is the 
        fitting resolution for the redshift bin and 'v' is the version of the 
        data fit specified by `chain_version`. Ignored if no contamination is 
        added, otherwise it **must** be specified. Default ``None``
    :type data_result_dir: ``str`` or :class:`os.PathLike` or ``NoneType``, 
        optional
    :param nside_fit: The resolutions used when fitting for the redshift bins.
        Ignored if no contamination is being applied, otherwise it **must** be 
        specified. Default ``None``
    :type nside_fit: ``int`` or (``Nbins``,) array-like of ``int`` or 
        ``NoneType``, optional
    :param zbins: The redshift bin(s) for which to create mocks. Assumes there
        are ``Nbins`` total redshift bins, and the indexing starts from 1
        (not 0!). If given, this must have length of at least 1 and no larger
        than ``Nbins``. If ``None`` (default), makes a mock for all redshift
        bins
    :type zbins: ``NoneType`` or 1D array-like of ``int``, optional
    :param k0: Parameter for the skewness of the lognormal field, which can be
        different for each redshift bin, in which case it should have length
        equal to ``zbins``. If ``None`` (default), creates a Gaussian density
        field rather than a lognormal one
    :type k0: ``NoneType`` or ``int`` or 1D array-like of ``int``, optional
    :param nmocks: The number of mocks to create. Default 1
    :type nmocks: ``int``, optional
    :param min_mock: The number of the first mock, used to edit the random seed
        and file names in the case where some mocks have already been created.
        If some mocks have already been created and this is not used, an error
        will be raised when trying to write the catalog. Default 0
    :type min_mock: ``int``, optional
    :param lmax: The maximum scale beyond which to truncate C(l). Default 3500
    :type lmax: ``float``, optional
    :param n_contam: The number of systematics maps by which to contaminate.
        Use an array-like for multiple levels of contamination. Default 0.
    :type n_contam: ``int`` or array-like of ``int``, optional
    :param gen_catalogs: If ``True``, also generate mock galaxy catalog(s) from
        the density field(s). Default ``False``
    :type gen_catalogs: ``bool``, optional
    :param test: If ``True``, set a random seed for reproducibility. Default
        ``False``
    :type test: ``bool``, optional
    :param force_truth: If ``True`` and true mock overdensity fields with the 
        specified numbers already exist, create new ones and overwrite the 
        files. This should be done with caution if adding contamination levels 
        to existing mocks, as it will mean the true overdensity field for 
        existing catalogs is overwritten. But random seed setting when `test` is 
        ``True`` will be off when adding contamination levels unless everything 
        that is already stored is redone. If ``None`` (default), this is set to 
        ``False`` if `test` is ``False`` or ``True`` if `test` is ``True``
    :type force_truth: ``bool`` or ``NoneType``, optional
    :param force: If ``True`` and files exist for mocks with the specified 
        numbers (other than the true overdensities, see `force_truth`), they 
        will be overwritten with the newly generated mocks rather than being 
        skipped. If ``None`` (default), set to the same as `force_truth`
    :type force: ``bool`` or ``NoneType``, optional
    """
    mask = lsssys.Mask(
        pathlib.Path(mask_path).expanduser().resolve().as_posix(), ZMAXcol=None)
    cat = lsssys.Redmagic(
        pathlib.Path(cat_path).expanduser().resolve().as_posix())
    sorted_z = np.sort(z_bin_edges)
    if zbins is not None:
        if (len(zbins) >= len(sorted_z) or np.min(zbins) < 0 or
              np.max(zbins) > len(sorted_z)):
            raise ValueError("Invalid size or value(s) for zbins")
        sorted_zbins = np.sort(zbins)
        zedges = np.dstack(
            (sorted_z[sorted_zbins - 1], sorted_z[sorted_zbins])).squeeze()
        number_density = calculate_number_density(cat, mask, zedges)[1]
    else:
        ngal_tot, number_density = calculate_number_density(cat, mask, sorted_z)
        sorted_zbins = np.arange(len(z_bin_edges) - 1) + 1
        zedges = np.dstack((sorted_z[:-1], sorted_z[1:])).squeeze()
    ngal_mean = np.atleast_1d(number_density) * hp.nside2pixarea(
        nside, degrees=True) * 60.**2
    output_dir = pathlib.Path(out_dir).expanduser().resolve()
    del cat
    if k0 is not None:
        if hasattr(k0, "__len__"):
            if len(k0) > 1:
                if zbins is not None and len(k0) != len(zbins):
                    raise ValueError("Mismatch between zbins and k0")
                elif zbins is None and len(k0) != (len(z_bin_edges) - 1):
                    raise ValueError("Mismatch between z_bin_edges and k0")
            else:
                k0 = k0[0]
        output_dir = output_dir.joinpath("lognormal_mock_output")
        ngal_fill = None
    else:
        output_dir = output_dir.joinpath("gaussian_mock_output")
        ngal_fill = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir.joinpath("mean_ngals.npy"), ngal_mean)
    theory = initialize_theory(theory_dir, lmax, zbins, k0)
    delta_output_dir = output_dir.joinpath("delta_maps")
    delta_output_dir.mkdir(exist_ok=True)
    ngals_output_dir = output_dir.joinpath("ngal_maps")
    ngals_output_dir.mkdir(exist_ok=True)
    for n in np.unique(np.append(np.atleast_1d(n_contam), 0)):
        delta_output_dir.joinpath(f"n_contaminate_{n}").mkdir(exist_ok=True)
        ngals_output_dir.joinpath(f"n_contaminate_{n}").mkdir(exist_ok=True)
    if gen_catalogs:
        noisy_ngals_output_dir = output_dir.joinpath("poisson_ngal_maps")
        noisy_ngals_output_dir.mkdir(exist_ok=True)
        cats_output_dir = output_dir.joinpath("catalogs")
        cats_output_dir.mkdir(exist_ok=True)
        for n in np.atleast_1d(n_contam):
            noisy_ngals_output_dir.joinpath(
                f"n_contaminate_{n}").mkdir(exist_ok=True)
            cats_output_dir.joinpath(f"n_contaminate_{n}").mkdir(exist_ok=True)
    if hasattr(n_contam, "__len__") or n_contam != 0:
        sys_dir = pathlib.Path("/spiff/wagoner47/des/y3/systematics")
        if chain_version is None:
            raise ValueError(
                "Must specify chain_version if contamination is required")
        if data_result_dir is None:
            raise ValueError(
                "Must specify data_result_dir if contamination is required")
        coeff_dir = pathlib.Path(data_result_dir)
        if nside_fit is None:
            raise ValueError(
                "Must specify nside_fit if contamination is required")
        if not hasattr(nside_fit, "__len__"):
            nside_fit = np.full(sorted_zbins.size, nside_fit)
        elif len(nside_fit) != sorted_zbins.size:
            if len(nside_fit) == 1:
                nside_fit = np.full(sorted_zbins.size, nside_fit[0])
            else:
                raise ValueError(
                    "Different number of specified nside_fit and redshift bins")
    mock = lsssys.Mock(theory, nside)
    mock.mask = mask.mask
    if force_truth is None:
        force_truth = test
    if force is None:
        force = force_truth
    if test:
        rand_states = np.arange(min_mock, nmocks + min_mock)
    else:
        rand_states = [None] * nmocks
    for i, r_state in enumerate(tqdm(
            rand_states, desc="Mock", dynamic_ncols=True), min_mock):
        try:
            assert not force_truth
            mock = _load_deltas(
                mock, i, delta_output_dir / "n_contaminate_0", 
                sorted_zbins.size)
        except (AssertionError, IOError):
            mock.gen_maps(rs=np.random.RandomState(r_state))
            if k0 is not None:
                mock.lognormalise(
                    k0, [
                        deltai.data[~mask.mask].std() for deltai in mock.delta])
            for z, deltai in enumerate(mock.delta):
                deltai.save(
                    delta_output_dir.joinpath(
                        "n_contaminate_0", f"mock_{i}_bin{z}.fits").as_posix(),
                    clobber=True)
        try:
            assert not force
            mock = _load_ngals(mock, i, ngals_output_dir / "n_contaminate_0")
        except (AssertionError, IOError):
            mock.gen_ngal(ngal_mean, ngal_fill)
            for z in range(len(mock.ngal)):
                mock.ngal[z].data[~mask.mask] *= mask.fracdet[~mask.mask]
            mock.save(
                ngals_output_dir.joinpath(
                    "n_contaminate_0", f"mock_{i}.fits").as_posix(),
                clobber=True)
        if gen_catalogs:
            for n in np.sort(np.atleast_1d(n_contam)):
                if n == 0:
                    mock = _load_deltas(
                        mock, i, delta_output_dir / "n_contaminate_0", 
                        sorted_zbins.size)
                    mock = _load_ngals(
                        mock, i, ngals_output_dir / "n_contaminate_0")
                else:
                    try:
                        assert not force
                        mock = _load_deltas(
                            mock, i, delta_output_dir / f"n_contaminate_{n}", 
                            sorted_zbins.size)
                    except (AssertionError, IOError):
                        mock = _load_deltas(
                            mock, i, delta_output_dir / "n_contaminate_0", 
                            sorted_zbins.size)
                        mock = contaminate_mock(
                            mock, n, sorted_zbins, nside_fit, coeff_dir, 
                            sys_dir, chain_version)
                        for z, deltai in enumerate(mock.delta):
                            deltai.save(
                                delta_output_dir.joinpath(
                                    f"n_contaminate_{n}",
                                    f"mock_{i}_bin{z}.fits").as_posix(),
                                clobber=True)
                    try:
                        assert not force
                        mock = _load_ngals(
                            mock, i, ngals_output_dir / f"n_contaminate_{n}")
                    except (AssertionError, IOError):
                        mock.gen_ngal(ngal_mean, ngal_fill)
                        for z in range(len(mock.ngal)):
                            mock.ngal[z].data[~mask.mask] *= mask.fracdet[
                                ~mask.mask]
                        mock.save(
                            ngals_output_dir.joinpath(
                                f"n_contaminate_{n}",
                                f"mock_{i}.fits").as_posix(),
                            clobber=True)
                try:
                    assert not force
                    mock = _load_ngals(
                        mock, i, noisy_ngals_output_dir / f"n_contaminate_{n}")
                except (AssertionError, IOError):
                    mock.poisson_sample()
                    mock.save(
                        noisy_ngals_output_dir.joinpath(
                            f"n_contaminate_{n}", f"mock_{i}.fits").as_posix(),
                        clobber=True)
                ngal_tot = np.array([
                    ng.data[~mask.mask].astype(int).sum() for ng in mock.ngal])
                gen_mock_catalogs(
                    mock, cats_output_dir / f"n_contaminate_{n}", i, zedges, 
                    force)
        
class FlagAction(argparse.Action):
    """
    GNU style --foo/--no-foo flag action for argparse
    (via http://bugs.python.org/issue8538 and
    https://stackoverflow.com/a/26618391/1256452).

    This provides a GNU style flag action for argparse.  Use
    as, e.g., parser.add_argument('--foo', action=FlagAction).
    The destination will default to 'foo' and the default value
    if neither --foo or --no-foo are specified will be None
    (so that you can tell if one or the other was given).
    """
    def __init__(self, option_strings, dest, default=None,
                 required=False, help=None, metavar=None,
                 positive_prefixes=['--'], negative_prefixes=['--no-']):
        self.positive_strings = set()
        # self.negative_strings = set()
        # Order of strings is important: the first one is the only
        # one that will be shown in the short usage message!  (This
        # is an annoying little flaw.)
        strings = []
        for string in option_strings:
            assert re.match(r'--[a-z]+', string, re.IGNORECASE)
            suffix = string[2:]
            for positive_prefix in positive_prefixes:
                s = positive_prefix + suffix
                self.positive_strings.add(s)
                strings.append(s)
            for negative_prefix in negative_prefixes:
                s = negative_prefix + suffix
                # self.negative_strings.add(s)
                strings.append(s)
        super(FlagAction, self).__init__(option_strings=strings, dest=dest,
                                         nargs=0, default=default,
                                         required=required, help=help,
                                         metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.positive_strings:
            setattr(namespace, self.dest, True)
        else:
            setattr(namespace, self.dest, False)

def path_dir(arg):
    """
    Converts the argument to a :class:`pathlib.Path` and checks that it is a
    directory, then returns the path object
    """
    path = pathlib.Path(arg).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError("Path must be a directory")
    return path

def existing_dir(arg):
    """
    Converts the argument to a :class:`pathlib.Path` and checks that it is a
    directory that already exists, then returns the path object
    """
    path = path_dir(arg)
    if not path.exists():
        raise argparse.ArgumentTypeError("Directory path must already exist")
    return path

def existing_file(arg):
    """
    Converts the argument to a :class:`pathlib.Path` and checks that it is a
    file that exists, then returns the path object
    """
    path = pathlib.Path(arg).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise argparse.ArgumentTypeError("Must be a file that exists")
    return path

def valid_nside(arg):
    """
    Check that the argument can be a valid Nside parameter, and return it as
    an int
    """
    try:
        val = int(arg)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("Value must be an integer")
    if not hp.isnsideok(val):
        raise argparse.ArgumentTypeError("Value must be a valid Nside")
    return val

def required_length(min_size=None, max_size=None):
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if min_size is not None and len(values) < min_size:
                raise argparse.ArgumentTypeError(
                    ("argument '{obj.dest}' must have at least {nmin}"
                     " elements").format(obj=self, nmin=min_size))
            if max_size is not None and len(values) > max_size:
                raise argparse.ArgumentTypeError(
                    ("argument '{obj.dest}' can have at most {nmax}"
                     " elements").format(obj=self, nmax=max_size))
            setattr(args, self.dest, values)
    return RequiredLength

def int_bigger_than(arg, min_val):
    """
    Converts the argument to an integer and checks that it is bigger than the
    given minimum, then returns the integer
    """
    try:
        val = int(arg)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("Value must be an integer")
    if val <= min_val:
        raise argparse.ArgumentTypeError(
            "Value must be bigger than {}".format(min_val))
    return val

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Generate mock density maps and catalogs", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "cat_path", type=existing_file, help="Path to catalog file")
    parser.add_argument(
        "mask_path", type=existing_file, help="Path to mask file")
    parser.add_argument(
        "theory_dir", type=existing_dir,
        help="Parent directory of CosmoSIS theory output")
    parser.add_argument(
        "nside", type=valid_nside,
        help="The resolution at which to make the mock")
    parser.add_argument(
        "out_dir", type=path_dir,
        help="The directory under which to store the output")
    parser.add_argument(
        "--zedges", nargs="+", action=required_length(min_size=2),
        type=float, required=True, help="The edges of the redshift bins")
    parser.add_argument(
        "--zbins", nargs="*", type=int, default=None,
        help="The redshift bins for which to make mocks")
    parser.add_argument(
        "--k0", nargs="*", type=float, default=None,
        help=("The skew parameter for making a lognormal field. Use None for a"
              " Gaussian field"))
    parser.add_argument(
        "-n", "--nmocks", type=lambda a: int_bigger_than(a, 0), default=1,
        help="The number of mocks to generate per redshift bin")
    parser.add_argument(
        "--nmin", type=lambda a: int_bigger_than(a, -1), default=0,
        help=("The minimum mock number, for naming the files when some mocks"
              " have already been generated"))
    parser.add_argument(
        "--lmax", type=float, default=3500,
        help="The maximum scale beyond which C(l) will be truncated")
    parser.add_argument(
        "--n_contaminate", type=int, default=0, nargs="*",
        help=("The number of maps by which to contaminate. Use multiple"
              " arguments to do multiple contamination levels."))
    parser.add_argument(
        "-v", "--chain_version", type=int, default=None, 
        help=("The version of the data fit to use as the truth for "
              "contamination, only needed if contamination is being done"))
    parser.add_argument(
        "--result_dir", type=path_dir, default=None, 
        help=("The location of the data fit results to use as the true "
              "coefficients, only needed if contamination is being done"))
    parser.add_argument(
        "--resol", type=int, nargs="*", default=None,
        help=("The fitting resolution, needed for contamination"))
    parser.add_argument(
        "-c", "--generate_catalogs", action="store_true",
        help=("Flag to specify that mock galaxy catalogs should also be"
              " generated"))
    parser.add_argument(
        "-t", "--test", action="store_true",
        help=("Flag specifying that the mocks should be generated with a"
              " predictable random seed (namely, the seed is the number of the"
              " mock) for reproducibility while testing"))
    parser.add_argument(
        "--force-truth", action=FlagAction, 
        help=("Flag to specify whether the true overdensity field should be "
              "overwritten if it already exists for some or all of the mocks. "
              "This is True if '--force-truth' is given, False if "
              "'--no-force-truth' is given, or None if neither is given. If "
              "None, forcing the true fields to be generated is done if the "
              "test flag is True"))
    parser.add_argument(
        "--force", action=FlagAction, 
        help=("Flag to specify whether the other mock outputs should be "
              "overwritten if they already exist for some or all of the mocks. "
              "This is True if '--force' is given, False if '--no-force' is "
              "given, or None if neither is given. If None, forcing is done if "
              "and only if it is being done for the true overdensity fields"))
    args = parser.parse_args()
    main(args.cat_path, args.mask_path, args.theory_dir, args.nside,
         args.zedges, args.out_dir, args.chain_version, args.result_dir, 
         args.resol, args.zbins, args.k0, args.nmocks, args.nmin, args.lmax, 
         args.n_contaminate, args.generate_catalogs, args.test, 
         args.force_truth, args.force)
