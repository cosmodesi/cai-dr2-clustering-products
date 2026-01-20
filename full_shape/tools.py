import os
import logging
from pathlib import Path
import warnings
from collections.abc import Callable
import itertools
import functools

import numpy as np

from mpi4py import MPI
from mockfactory import Catalog, sky_to_cartesian, setup_logging


logger = logging.getLogger('tools')


desi_dir = Path('/dvs_ro/cfs/cdirs/desi/')


def default_mpicomm(func: Callable):
    """Wrapper to provide a default MPI communicator."""
    @functools.wraps(func)
    def wrapper(*args, mpicomm=None, **kwargs):
        if mpicomm is None:
            from mpi4py import MPI
            mpicomm = MPI.COMM_WORLD
        return func(*args, mpicomm=mpicomm, **kwargs)

    return wrapper


def load_footprint():
    #global footprint
    from regressis import footprint
    footprint = footprint.DR9Footprint(256, mask_lmc=False, clear_south=True, mask_around_des=False, cut_desi=False)
    return footprint


def join_tracers(tracers):
    """Given list/tuple of input tracers, return joined string with 'x' separator."""
    if not isinstance(tracers, str):
        return 'x'.join(tracers)
    return tracers


def get_simple_tracer(tracer):
    """Given import tracer, return simple tracer name; e.g. 'ELG_LOPnotqso' would result in 'ELG'."""
    if 'BGS' in tracer:
        return 'BGS'
    elif 'LRG+ELG' in tracer:
        return 'LRG+ELG'
    elif 'LRG' in tracer:
        return 'LRG'
    elif 'ELG' in tracer:
        return 'ELG'
    elif 'QSO' in tracer:
        return 'QSO'
    else:
        raise NotImplementedError(f'tracer {tracer} is unknown')


def select_region(ra, dec, region=None):
    """
    Return mask of corresponding R.A./Dec. region.

    Parameters
    ----------
    ra : array_like
        R.A. coordinates in degrees.
    dec : array_like
        Dec. coordinates in degrees.
    region : str, optional
        Region to select. Options are:
        - None, 'ALL', 'GCcomb': all-sky
        - 'NGC': North Galactic Cap
        - 'SGC': South Galactic Cap
        - 'N': Northern region (Dec > 32.375 and in NGC)
        - 'S': Southern region (Dec > -25 and in SGC)
        - 'SNGC': Southern part of NGC
        - 'SSGC': Southern part of SGC
        - 'NGCnoN': NGC excluding Northern region
        - 'DES': DES footprint
        - 'SnoDES': Southern region excluding DES footprint
        - 'SSGCnoDES': Southern part of SGC excluding DES footprint
        - 'SGCnoDES': SGC excluding DES footprint

    Returns
    -------
    mask : array_like
        Boolean mask array indicating the selected region.
    """
    import healpy as hp
    # print('select', region)
    if region in [None, 'ALL', 'GCcomb']:
        return np.ones_like(ra, dtype='?')
    mask_ngc = (ra > 100 - dec)
    mask_ngc &= (ra < 280 + dec)
    mask_n = mask_ngc & (dec > 32.375)
    mask_s = (~mask_n) & (dec > -25.)
    if region == 'NGC':
        return mask_ngc
    if region == 'SGC':
        return ~mask_ngc
    if region == 'N':
        return mask_n
    if region == 'S':
        return mask_s
    if region == 'SNGC':
        return mask_ngc & mask_s
    if region == 'SSGC':
        return (~mask_ngc) & mask_s
    if region == 'NGCnoN':
        return mask_ngc & (~mask_n)
    # if region == 'GCcomb_noNorth':
    #     return ~mask_n
    north, south, des = load_footprint().get_imaging_surveys()
    mask_des = des[hp.ang2pix(256, ra, dec, nest=True, lonlat=True)]
    if region == 'DES':
        return mask_des
    if region == 'SnoDES':
        return mask_s & (~mask_des)
    if region == 'SSGCnoDES':
        return (~mask_ngc) & mask_s & (~mask_des)
    if region == 'SGCnoDES':
        return (~mask_ngc) & (~mask_des)
    # if region == 'GCcomb_noDES':
    #     return ~mask_des
    raise ValueError('unknown region {}'.format(region))


def _make_tuple(item, n=None):
    if not isinstance(item, (list, tuple)):
        item = (item,)
    item = tuple(item)
    if n is not None:
        item = item + (item[-1],) * (n - len(item))
    return item


def compute_fiducial_png_weights(ell, catalog, tracer='LRG', p=1.):
    """Return total optimal weights for local PNG analysis."""

    def bias(z, tracer='QSO'):
        """Bias model for the different DESI tracer (measured from DR2 data (loa/v2))."""
        params = {'BGS_BRIGHT-21.35': (0.60646037, 0.52389492),
                  'LRG': (0.23553567, 1.3458994),
                  'ELG_LOPnotqso': (0.15066781, 0.59463735),
                  'ELGnotqso': (0.15487521, 0.59464828),
                  'ELG': (0.15487521, 0.59464828),
                  'QSO': (0.25207547, 0.71020952)}
        params.update({f'{key}_zcmb': value for key, value in params.items()})

        if tracer in params:
            alpha, beta = params[tracer]
        else:
            raise ValueError(f'Bias for {tracer} is not ready!')
        return alpha * (1 + z)**2 + beta

    from cosmoprimo.fiducial import DESI
    from cosmoprimo.utils import Interpolator1D

    cosmo = DESI()
    zmax, nz = 100., 512
    zgrid = 1. / jnp.geomspace(1. / (1. + zmax), 1., nz)[::-1] - 1.
    growth_factor = Interpolator1D(zgrid, cosmo.growth_factor(zgrid), k=3)
    growth_rate = Interpolator1D(zgrid, cosmo.growth_rate(zgrid), k=3)

    tracers = _make_tuple(tracer, n=2)
    catalogs = _make_tuple(catalog, n=2)
    ps = _make_tuple(p, n=2)

    def _get_weights(catalogs, tracers, ps):
        wtilde = bias(catalogs[0]['Z'], tracer=tracers[0]) - ps[0]
        w0 = growth_factor(catalogs[0]['Z']) * (bias(catalogs[1]['Z'], tracer=tracers[1]) + growth_rate(catalogs[1]['Z']) / 3)
        w2 = 2 / 3 * growth_factor(catalogs[1]['Z']) * growth_rate(catalogs[1]['Z'])
        yield catalogs[0]['INDWEIGHT'] * wtilde, catalogs[1]['INDWEIGHT'] * {0: w0, 2: w2}[ell]

    yield _get_weights(catalogs, tracers, ps)
    if tracers[1] != tracers[0]:
        yield _get_weights(catalogs[::-1], tracers[::-1], ps[::-1])


def propose_fiducial(kind, tracer, zrange=None, analysis='full_shape'):
    """
    Propose fiducial measurement parameters for given tracer and statistic kind.

    Parameters
    ----------
    kind : str
        Statistic kind. Options are 'zranges', 'nran', 'particle2_correlation', 'mesh2_spectrum', 'mesh3_spectrum', 'recon'.
    tracer : str
        Tracer name. Options are 'BGS', 'LRG', 'ELG', 'LRG+ELG', 'QSO'.
    zrange : tuple, optional
        Redshift range. If provided, it will override the default zrange.

    Returns
    -------
    params : dict
        Dictionary of proposed fiducial parameters for the specified statistic kind and tracer.
    """
    from jaxpower import get_mesh_attrs
    base = {'catalog': {'weight': 'default_FKP'}, 'particle2_correlation': {}, 'mesh2_spectrum': {}, 'mesh3_spectrum': {}}
    propose_fiducial = {
        'BGS': {'zranges': [(0.1, 0.4)], 'nran': 3, 'recon': {'bias': 1.5, 'smoothing_radius': 15., 'zrange': (0.1, 0.4)}},
        'LRG+ELG': {'zranges': [(0.8, 1.1)], 'nran': 13, 'recon': {'bias': 1.6, 'smoothing_radius': 15.}, 'zrange': (0.8, 1.1)},
        'LRG': {'zranges': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 'nran': 10, 'recon': {'bias': 2.0, 'smoothing_radius': 15., 'zrange': (0.4, 1.1)}},
        'ELG': {'zranges': [(0.8, 1.1), (1.1, 1.6)], 'nran': 15, 'recon': {'bias': 1.2, 'smoothing_radius': 15., 'zrange': (0.8, 1.6)}},
        'QSO': {'zranges': [(0.8, 2.1)], 'nran': 4, 'recon': {'bias': 2.1, 'smoothing_radius': 30., 'zrange': (0.8, 2.1)}}
    }
    tracers = _make_tuple(tracer)
    tracer = join_tracers(tracers)
    tracer = get_simple_tracer(tracer)
    propose_fiducial = base | propose_fiducial[tracer]
    if 'png' in analysis:
        propose_meshsizes = {'BGS': 700, 'LRG': 700, 'ELG': 700, 'LRG+ELG': 700, 'QSO': 700}
        propose_cellsize = 20.
    else:
        propose_meshsizes = {'BGS': 750, 'LRG': 750, 'ELG': 960, 'LRG+ELG': 750, 'QSO': 1152}
        propose_cellsize = 7.5
    for stat in ['mesh2_spectrum', 'mesh3_spectrum']:
        propose_fiducial[stat]['mattrs'] = {'meshsize': propose_meshsizes[tracer], 'cellsize': propose_cellsize}
    if 'png' in analysis:
        propose_fiducial['mesh2_spectrum'].update(ells=(0, 2), optimal_weights=functools.partial(compute_fiducial_png_weights, tracer=tracers))
    else:
        propose_fiducial['mesh2_spectrum'].update(ells=(0, 2, 4))
        propose_fiducial['mesh3_spectrum'].update(ells=[(0, 0, 0), (2, 0, 2)], basis='sugiyama-diagonal')
    for stat in ['recon']:
        recon_cellsize = propose_fiducial[stat]['smoothing_radius'] / 3.
        primes, divisors = (2, 3, 5), (2,)
        propose_fiducial[stat]['mattrs'] = {'boxpad': 1.2, 'cellsize': recon_cellsize, 'primes': primes, 'divisors': divisors}
    for name in list(propose_fiducial):
        propose_fiducial[f'recon_{name}'] = propose_fiducial[name]  # same for post-recon measurements
    return propose_fiducial[kind]


def get_catalog_fn(version=None, cat_dir=None, kind='data', tracer='LRG',
                   region='NGC', weight='default_FKP', nran=10, imock=0, ext='h5', **kwargs):
    """
    Return catalog filename(s) for given parameters.

    Parameters
    ----------
    version : str
        Catalog version. Options are 'data-dr1-v1.5', 'data-dr2-v2', 'holi-v1-complete', 'holi-v1-altmtl'.
    cat_dir : str, Path, optional
        Directory containing the catalogs. If None, pre-registered paths will be used based on version.
    kind : str
        Catalog kind. Options are 'data', 'randoms', 'full_data', 'full_randoms'.
    tracer : str
        Tracer name. Options are 'BGS', 'LRG', 'ELG', 'LRG+ELG', 'QSO'.
    region : str
        Region name. Options are 'NGC', 'SGC', 'N', 'S', 'ALL', 'NGCnoN', 'SGCnoDES'.
    weight : str
        Weight type. Options are 'default_FKP', 'defaut_FKP_bitwise', etc.
    nran : int
        Number of random catalogs.
    imock : int
        Mock index (for mock catalogs). Default is 0.
    ext : str
        File extension. Default is 'h5'.

    Returns
    -------
    fn : str, Path, list
        Catalog filename(s).
        Multiple filenames are returned as a list when region is 'ALL' or when kind is 'randoms' or 'full_randoms'.
    """
    if region in ['N', 'NGC', 'NGCnoN']: region = 'NGC'
    elif region in ['SGC', 'SGCnoDES']: region = 'SGC'
    elif 'full' not in kind:
        if region in ['S', 'ALL']: regions = ['NGC', 'SGC']
        else: raise NotImplementedError(f'{region} is unknown')
        return [get_catalog_fn(version=version, cat_dir=cat_dir, kind=kind, tracer=tracer,
                               region=region, weight=weight, nran=nran, imock=imock, ext=ext, **kwargs) for region in regions]

    if cat_dir is None:  # pre-registered paths
        if version == 'data-dr1-v1.5':
            cat_dir = desi_dir / f'survey/catalogs/Y1/LSS/iron/LSScats'
            if 'bitwise' in weight:
                cat_dir = cat_dir / 'v1.5pip'
            else:
                cat_dir = cat_dir / 'v1.5'
            ext = 'fits'
        elif version == 'data-dr2-v2':
            cat_dir = desi_dir / f'survey/catalogs/DA2/LSS/loa-v1/LSScats/v2'
            if 'bitwise' in weight:
                data_dir = cat_dir / 'PIP'
            else:
                data_dir = cat_dir / 'nonKP'
            if kind == 'data':
                return data_dir / f'{tracer}_{region}_clustering.dat.fits'
            if kind == 'randoms':
                return [data_dir / f'{tracer}_{region}_{iran:d}_clustering.ran.fits' for iran in range(nran)]
            if kind == 'full_data':
                return cat_dir / f'{tracer}_full_HPmapcut.dat.fits'
            if kind == 'full_randoms':
                return [cat_dir / f'{tracer}_{iran:d}_full_HPmapcut.ran.fits' for iran in range(nran)]
        elif version == 'holi-v1-complete':
            cat_dir = desi_dir / f'mocks/cai/LSS/DA2/mocks/holi_v1/altmtl{imock:d}/loa-v1/mock{imock:d}/LSScats'
            if kind == 'data':
                return cat_dir / f'{tracer}_complete_{region}_clustering.dat.h5'
            if kind == 'randoms':
                return [cat_dir / f'{tracer}_complete_{region}_{iran:d}_clustering.ran.h5' for iran in range(nran)]
        elif version == 'holi-v1-altmtl':
            cat_dir = desi_dir / f'mocks/cai/LSS/DA2/mocks/holi_v1/altmtl{imock:d}/loa-v1/mock{imock:d}/LSScats'
            ext = 'fits' if 'full' in kind else 'h5'
    cat_dir = Path(cat_dir)
    if kind == 'data':
        return cat_dir / f'{tracer}_{region}_clustering.dat.{ext}'
    if kind == 'randoms':
        return [cat_dir / f'{tracer}_{region}_{iran:d}_clustering.ran.{ext}' for iran in range(nran)]
    if kind == 'full_data':
        return cat_dir / f'{tracer}_full_HPmapcut.dat.{ext}'
    if kind == 'full_randoms':
        return [cat_dir / f'{tracer}_{iran:d}_full_HPmapcut.ran.{ext}' for iran in range(nran)]


def get_measurement_fn(meas_dir=Path(os.getenv('SCRATCH')) / 'measurements', version=None, kind='mesh2_spectrum', recon=None,
                       tracer='LRG', region='NGC', zrange=None, auw=None, cut=None, weight='default_FKP', imock=None, extra='', ext='h5', **kwargs):
    """
    Return measurement filename for given parameters.

    Parameters
    ----------
    meas_dir : str, Path
        Directory containing the measurements.
    version : str, optional
        Measurement version.
    kind : str
        Measurement kind. Options are 'particle2_correlation', 'mesh2_spectrum', 'mesh3_spectrum', etc.
    recon : str, optional
        Reconstruction type, tyipcally 'recsym' or 'reciso'.
    tracer : str
        Tracer name.
    region : str
        Region name.
    zrange : tuple, optional
        Redshift range.
    auw : bool, optional
        Whether to include angular upweighting.
    cut : bool, optional
        Whether to include theta cut.
    weight : str
        Weight type. Options are 'default_FKP', 'defaut_FKP_bitwise', etc.
    imock : int, str, optional
        Mock index (for mock catalogs). If '*', return all existing mock filenames.
    extra : str, optional
        Extra string to append to filename.
    ext : str
        File extension. Default is 'h5'.

    Returns
    -------
    fn : str, Path, list
        Measurement filename(s).
        Multiple filenames are returned as a list when imock is '*'.
    """
    if imock == '*':
        fns = [get_measurement_fn(meas_dir=meas_dir, kind=kind, version=version, recon=recon, tracer=tracer, region=region, zrange=zrange, auw=auw, cut=cut, weight=weight, imock=imock, ext=ext, **kwargs) for imock in range(1000)]
        return [fn for fn in fns if os.path.exists(fn)]
    if cut: cut = '_thetacut'
    else: cut = ''
    if auw: auw = '_auw'
    else: auw = ''
    meas_dir = Path(meas_dir)
    if version is not None:
        meas_dir = meas_dir / version
    if recon:
        meas_dir = meas_dir / recon
    if imock is None:
        imock = ''
    else:
        imock = f'_{imock:d}'
    if extra:
        extra = f'_{extra}'
    tracer = join_tracers(tracer)
    if zrange is not None:
        zrange = f'_z{zrange[0]:.1f}-{zrange[1]:.1f}'
    else:
        zrange = ''
    corr_type = 'smu'
    battrs = kwargs.get('battrs', None)
    if battrs is not None: corr_type = ''.join(list(battrs))
    kind = {'mesh2_spectrum': 'mesh2_spectrum_poles',
            'mesh3_spectrum': 'mesh3_spectrum_poles',
            'particle2_correlation': f'particle2_correlation_{corr_type}'}.get(kind, kind)
    basename = f'{kind}_{tracer}{zrange}_{region}_{weight}{auw}{cut}{extra}{imock}.{ext}'
    return meas_dir / basename


def checks_if_exists_and_readable(get_fn, test_if_readable=True, **kwargs):
    """
    Return lists of existing, missing and not readable files for all combinations of input kwargs.
    Input :func:`get_fn` must provide the filename associated to the input kwargs.
    """
    def is_unreadable(fn):
        fn = str(fn)
        if any(fn.endswith(ext) for ext in ['hdf', 'h4', 'hdf4', 'he2', 'h5', 'hdf5', 'he5', 'h5py']):
            try:
                import hdf5plugin
                import h5py
                with h5py.File(fn, 'r', locking=False):
                    pass
                return False
            except Exception as exc:
                return exc
        elif fn.endswith('fits') or fn.endswith('fit') or fn.endswith('fts'):
            try:
                with fitsio.FITS(fn) as file:
                    for i, hdu in enumerate(file): pass
                return False
            except Exception as exc:
                return exc
        else:
            warnings.warn(f'cannot check readability of file {fn} with unknown extension')
            return False

    names, values = zip(*kwargs.items())
    exists, missing, unreadable = [([], {name: [] for name in names}) for i in range(3)]
    unreadable = unreadable + ([],)  # add exceptions

    def _append(toret, fn, kwargs, exc=None):
        toret[0].append(fn)
        for name in names: toret[1][name].append(kwargs[name])
        if exc is not None: toret[2].append(exc)

    for values in itertools.product(*values):
        fn_kwargs = dict(zip(names, values))
        fn = get_fn(**fn_kwargs)
        if os.path.exists(fn):
            _append(exists, fn, fn_kwargs)
            if test_if_readable:
                exc = is_unreadable(fn)
                if exc:
                    _append(unreadable, fn, fn_kwargs, exc=exc)
        else:
            _append(missing, fn, fn_kwargs)
    return exists, missing, unreadable


# Create a lookup table for set bits per byte
_popcount_lookuptable = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)


def popcount(*arrays):
    """
    Return number of 1 bits in each value of input array.
    Inspired from https://github.com/numpy/numpy/issues/16325.
    """
    # if not np.issubdtype(array.dtype, np.unsignedinteger):
    #     raise ValueError('input array must be an unsigned int dtype')
    toret = _popcount_lookuptable[arrays[0].view((np.uint8, (arrays[0].dtype.itemsize,)))].sum(axis=-1)
    for array in arrays[1:]: toret += popcount(array)
    return toret


def _format_bitweights(bitweights):
    if bitweights is None:
        return []
    if isinstance(bitweights, (tuple, list)):
        return list(bitweights)
    if bitweights.ndim == 2:
        return list(bitweights.T)
    return [bitweights]


@default_mpicomm
def _read_catalog(fn, mpicomm=None):
    """Wrapper around :meth:`Catalog.read` to read catalog(s)."""
    one_fn = fn[0] if isinstance(fn, (tuple, list)) else fn
    kw = {}
    if str(one_fn).endswith('.h5'): kw['group'] = 'LSS'
    catalog = Catalog.read(fn, mpicomm=mpicomm, **kw)
    if str(one_fn).endswith('.fits'): catalog.get(catalog.columns())  # Faster to read all columns at once
    return catalog


def _compute_missing_power(ntile, bitweights, loc_assigned, method='missing_power'):
    """
    Compute "missing power weights", called "NTMP" in Davide's paper below.

    Reference
    ---------
    https://arxiv.org/pdf/2411.12025v2, Section 5.2.2


    Parameters
    ----------
    ntile : array_like
        NTILE values.
    bitweights : array_like, list
        Bitweights array or list of bitweights arrays.
    loc_assigned : array_like
        LOCATION_ASSIGNED boolean array.
    method : str
        Method to compute missing power. Options are 'missing_power' or 'zero_prob'.

    Returns
    -------
    toret : array_like
        Missing power weights per NTILE.
    """
    bitweights = _format_bitweights(bitweights)
    # Input: list of bitweights
    nbits = 8 * sum(weight.dtype.itemsize for weight in bitweights)
    recurr = popcount(*bitweights)
    wiip = (nbits + 1) / (recurr + 1)
    zero_prob = (recurr == 0) & (~loc_assigned)

    #print(np.sum(zerop_msk))
    sum_ntile = np.bincount(ntile)
    sum_zero_prob = np.bincount(ntile, weights=zero_prob)
    sum_loc_assigned = np.bincount(ntile, weights=loc_assigned)
    sum_wiip = np.bincount(ntile, weights=loc_assigned * wiip)
    mask_zero_ntile = sum_ntile == 0
    frac_zero_prob = np.divide(sum_zero_prob, sum_ntile, out=np.ones_like(sum_wiip), where=~mask_zero_ntile)
    frac_missing_power = np.divide(sum_ntile - sum_wiip, sum_ntile, out=np.ones_like(sum_wiip), where=~mask_zero_ntile)
    if method == 'missing_power':
        toret = 1 - frac_missing_power
    elif method == 'zero_prob':
        toret = 1 - frac_zero_prob
    else:
        raise NotImplementedError(f'unknown method {method}')
    return toret


def _compute_binned_weight(ntile, weight):
    """Compute weights per ntile."""
    sum_ntile = np.bincount(ntile)
    sum_weight = np.bincount(ntile, weights=weight)
    mask_zero_ntile = sum_ntile == 0
    return np.divide(sum_weight, sum_ntile, out=np.ones_like(sum_weight), where=~mask_zero_ntile)


def get_binned_weight(catalog, binned_weight):
    """Get values of binned weights."""
    toret = 1.
    for column, weight in binned_weight.items():
        toret *= weight[catalog[column]]  # e.g. completeness[ntile]
    return toret


def get_positions_from_rdz(catalog):
    """Return Cartesian positions from R.A., Dec., and redshift."""
    from cosmoprimo.fiducial import TabulatedDESI, DESI
    fiducial = TabulatedDESI()  # faster than DESI/class (which takes ~30 s for 10 random catalogs)
    dist = fiducial.comoving_radial_distance(catalog['Z'])
    catalog['POSITION'] = sky_to_cartesian(dist, catalog['RA'], catalog['DEC'], dtype=dist.dtype)
    return catalog


def expand_randoms(randoms, parent_randoms, data, from_randoms=('RA', 'DEC'), from_data=('Z',)):
    """
    Expand randoms by adding columns from parent randoms and data catalogs via TARGETID matches.

    Parameters
    ----------
    randoms : Catalog
        Catalog of randoms, containing at least the columns ['TARGETID', 'TARGETID_DATA', 'WEIGHT', 'NX'].
    parent_randoms : Catalog
        Path to randoms that are a superset of 'randoms' to expand and contain at least ['TARGETID', 'RA', 'DEC'].
    data : Catalog
        Data catalogs to take redshift information from (concatenation of NGC and SGC).
    from_randoms : list, tuple
        List of the column names to add to ``randoms`` from the parent random catalog via TARGETID match.
    from_data : list, tuple
        List of the column names to add to ``randoms`` from the data catalog via TARGETID_DATA to TARGETID match.

    Returns
    -------
    randoms : Catalog
        Expanded randoms catalog.
    """
    _, randoms_index, parent_index = np.intersect1d(randoms['TARGETID'], parent_randoms['TARGETID'], return_indices=True)
    randoms = randoms[randoms_index]
    for column in from_randoms:
        if column != 'TARGETID':
            randoms[column] = parent_randoms[column]

    if isinstance(data, (list, tuple)):  # NGC + SGC
        data = Catalog.concatenate([dd[list(from_data) + ['TARGETID']] for dd in data])
    else:
        data = data[list(from_data) + ['TARGETID']]
    data['TARGETID_DATA'] = data.pop('TARGETID')

    if data['TARGETID_DATA'].max() < int(1e9):  # faster method
        lookup = np.arange(1 + data['TARGETID_DATA'].max())
        lookup[data['TARGETID_DATA']] = np.arange(len(data))
        index = lookup[data['TARGETID_DATA']]
    else:
        sorted_index = np.argsort(data['TARGETID_DATA'])
        index_in_sorted = np.searchsorted(data['TARGETID_DATA'], randoms['TARGETID_DATA'], sorter=sorted_index)
        index = sorted_index[index_in_sorted]
    for column in data:
        if column != 'TARGETID':
            randoms[column] = data[column][index]

    return randoms


@default_mpicomm
def read_clustering_catalog(kind=None, concatenate=True, get_catalog_fn=get_catalog_fn, get_positions_from_rdz=get_positions_from_rdz,
                            expand=None, binned_weight=None, mpicomm=None, **kwargs):
    """
    Read clustering catalog (data or randoms) with given parameters.

    Parameters
    ----------
    kind : str
        Catalog kind. Options are 'data' or 'randoms'.
    concatenate : bool
        Whether to concatenate catalogs from different regions or multiple randoms.
    get_catalog_fn : callable
        Function to get catalog filenames.
    get_positions_from_rdz : callable
        Function to compute Cartesian positions from R.A., Dec., and redshift.
    expand : callable, dict, optional
        If callable, function to expand randoms catalog.
        If dict, parameters to expand randoms catalog via :func:`expand_randoms`.
        In this case, modified in-place to add the parent randoms catalog with key its file name.
    binned_weight : dict, optional
        Binned weights to apply. Keys are column names, values are weight arrays.
    mpicomm : MPI.Comm, optional
        MPI communicator.
    kwargs : dict
        Additional keyword arguments to pass to :func:`get_catalog_fn`.

    Returns
    -------
    catalog : Catalog, list
        Catalog object or list of Catalog objects (if ``concatenate`` is False).
        Contains 'RA', 'DEC', 'Z', 'NX', 'TARGETID', 'POSITION', 'INDWEIGHT' (individual weight), 'BITWEIGHT' columns.
    """
    assert kind in ['data', 'randoms'], 'provide kind'

    zrange, region, weight_type = (kwargs.get(key) for key in ['zrange', 'region', 'weight'])
    fns = get_catalog_fn(kind=kind, **kwargs)
    if not isinstance(fns, (tuple, list)): fns = [fns]
    exists = {os.path.exists(fn): fn for fn in fns}
    if not all(exists):
        raise IOError(f'Catalogs {[fn for ex, fn in exists.items() if not ex]} do not exist!')

    if kind == 'randoms' and isinstance(expand, dict):
        from_data = expand.get('from_data', ['Z', 'WEIGHT_FKP', 'FRAC_TLOBS_TILES', 'NTILE'])
        from_randoms = expand.get('from_randoms', ['RA', 'DEC'])
        parent_randoms_fn = expand['parent_randoms_fn']
        if parent_randoms_fn not in expand:
            expand[parent_randoms_fn] = _read_catalog(parent_randoms_fn, mpicomm=MPI.COMM_SELF)
        parent_randoms = expand[parent_randoms_fn]
        data_fn = expand.get('data_fn', None)
        if data_fn is None:
            data_fn = [get_catalog_fn(kind='data', **(kwargs | dict(region=region))) for region in ['NGC', 'SGC']]
        data = _read_catalog(data_fn, mpicomm=MPI.COMM_SELF)
        expand = functools.partial(expand_randoms, parent_randoms=parent_randoms, data=data, from_randoms=from_randoms, from_data=from_data)

    catalogs = [None] * len(fns)
    for ifn, fn in enumerate(fns):
        irank = ifn % mpicomm.size
        catalogs[ifn] = (irank, None)
        if mpicomm.rank == irank:  # Faster to read catalogs from one rank
            catalog = _read_catalog(fn, mpicomm=MPI.COMM_SELF)
            if expand is not None:
                catalog = expand(catalog, *expand)
            columns = ['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_COMP', 'WEIGHT_FKP', 'BITWEIGHTS', 'FRAC_TLOBS_TILES', 'NTILE', 'NX', 'TARGETID']
            columns = [column for column in columns if column in catalog.columns()]
            catalog = catalog[columns]
            if zrange is not None:
                mask = (catalog['Z'] >= zrange[0]) & (catalog['Z'] < zrange[1])
                catalog = catalog[mask]
            if 'bitwise' in weight_type:
                mask = (catalog['FRAC_TLOBS_TILES'] != 0)
                catalog = catalog[mask]
            if region is not None:
                mask = select_region(catalog['RA'], catalog['DEC'], region)
                catalog = catalog[mask]
            catalogs[ifn] = (irank, catalog)

    rdzw = []
    for irank, catalog in catalogs:
        if mpicomm.size > 1:
            catalog = Catalog.scatter(catalog, mpicomm=mpicomm, mpiroot=irank)
        individual_weight = catalog['WEIGHT']
        bitwise_weights = None
        if 'bitwise' in weight_type:
            if kind == 'data':
                individual_weight = catalog['WEIGHT'] / catalog['WEIGHT_COMP']
                bitwise_weights = catalog['BITWEIGHTS']
            elif kind == 'randoms':
                individual_weight = catalog['WEIGHT'] * get_binned_weight(catalog, binned_weight['missing_power'])
        if 'FKP' in weight_type.upper():
            individual_weight *= catalog['WEIGHT_FKP']
        if 'comp' in weight_type:
            individual_weight *= get_binned_weight(catalog, binned_weight['completeness'])
        catalog = catalog[['RA', 'DEC', 'Z', 'NX', 'TARGETID']]
        catalog['INDWEIGHT'] = individual_weight
        for column in catalog:
            if not np.issubdtype(catalog[column].dtype, np.integer):
                catalog[column] = catalog[column].astype('f8')
        if bitwise_weights is not None:
            catalog['BITWEIGHT'] = bitwise_weights
        catalog = get_positions_from_rdz(catalog)
        rdzw.append(catalog)
    if concatenate:
        if len(rdzw) > 1: return rdzw[0]
        return Catalog.concatenate(rdzw)
    else:
        return rdzw


@default_mpicomm
def read_full_catalog(kind, wntile=None, concatenate=True,
                     get_catalog_fn=get_catalog_fn, mpicomm=None, **kwargs):
    """
    Read full data or randoms catalog with given parameters.

    Parameters
    ----------
    kind : str
        Catalog kind. Options are 'parent_data', 'fibered_data', 'parent_randoms', 'fibered_randoms'.
    wntile : Path, str, default=None
        Filename of precomputed wntile weights. If None, compute from data clustering catalog (using :func:`get_catalog_fn`).
    concatenate : bool
        Whether to concatenate catalogs from different regions or multiple randoms.
    get_catalog_fn : callable
        Function to get catalog filenames.
    mpicomm : MPI.Comm, optional
        MPI communicator.
    kwargs : dict
        Additional keyword arguments to pass to :func:`get_catalog_fn`.

    Returns
    -------
    catalog : Catalog, list
        Catalog object or list of Catalog objects (if ``concatenate`` is False).
        Contains 'RA', 'DEC', 'TARGETID', 'INDWEIGHT' (individual weight), 'BITWEIGHT' columns.
    """
    assert kind in ['parent_data', 'fibered_data', 'parent_randoms', 'fibered_randoms'], 'provide kind'
    region, weight_type = (kwargs.get(key) for key in ['region', 'weight'])
    fns = get_catalog_fn(kind='full_data' if 'data' in kind else 'full_randoms', **kwargs)
    if not isinstance(fns, (tuple, list)): fns = [fns]

    exists = {os.path.exists(fn): fn for fn in fns}
    if not all(exists):
        raise IOError('Catalogs {[fn for ex, fn in exists.items() if not ex]} do not exist!')

    def get_wntile(wntile):
        if wntile is None:
            clustering_data_fn = get_catalog_fn(kind='data', **kwargs)
        else:
            clustering_data_fn = wntile
        return compute_wntile(clustering_data_fn, mpicomm=mpicomm)

    catalogs = [None] * len(fns)
    for ifn, fn in enumerate(fns):
        irank = ifn % mpicomm.size
        catalogs[ifn] = (irank, None)
        if mpicomm.rank == irank:  # Faster to read catalogs from one rank
            catalog = Catalog.read(fn, mpicomm=MPI.COMM_SELF)
            catalog.get(catalog.columns())  # Faster to read all columns at once
            columns = ['RA', 'DEC', 'LOCATION_ASSIGNED', 'BITWEIGHTS', 'NTILE', 'WEIGHT_NTILE', 'FRACZ_TILELOCID', 'FRAC_TLOBS_TILES']
            columns = [column for column in columns if column in catalog.columns()]
            catalog = catalog[columns]
            catalog.attrs['missing_power'] = {column: _compute_missing_power(catalog[column], catalog['BITWEIGHTS'], catalog['LOCATION_ASSIGNED']) for column in ['NTILE']}
            catalog.attrs['completeness'] = {column: _compute_binned_weight(catalog[column], catalog['FRACZ_TILELOCID'] * catalog['FRAC_TLOBS_TILES']) for column in ['NTILE']}
            if 'fibered' in kind:
                mask = catalog['LOCATION_ASSIGNED']
                catalog = catalog[mask]
            if region is not None:
                mask = select_region(catalog['RA'], catalog['DEC'], region)
                catalog = catalog[mask]
            catalogs[ifn] = (irank, catalog)

    rdw = []
    for irank, catalog in catalogs:
        if mpicomm.size > 1:
            catalog = Catalog.scatter(catalog, mpicomm=mpicomm, mpiroot=irank)
        if 'WEIGHT_NTILE' in catalog:
            individual_weight = catalog['WEIGHT_NTILE']
        else:
            individual_weight = apply_wntile(catalog['NTILE'], get_wntile(wntile))
        bitwise_weights = None
        if 'fibered' in kind and 'data' in kind:
            if 'bitwise' in weight_type:
                individual_weight /= get_binned_weight(catalog, catalog.attrs['missing_power'])
                bitwise_weights = catalog['BITWEIGHTS']
            else:
                individual_weight /= (catalog['FRACZ_TILELOCID'] * catalog['FRAC_TLOBS_TILES'])
                bitwise_weights = None
        catalog = catalog[['RA', 'DEC']]
        catalog['INDWEIGHT'] = individual_weight
        for column in catalog:
            catalog[column] = catalog[column].astype('f8')
        if bitwise_weights is not None: catalog['BITWEIGHT'] = bitwise_weights
        rdw.append(catalog)
    if concatenate:
        if len(rdw) > 1: return rdw[0]
        return Catalog.concatenate(rdw)
    else:
        return rdw


def write_summary_statistics(fn, stats):
    """Write summary statistics to file from process 0 only."""
    import jax
    if jax.process_index() == 0:
        logger.info(f'Writing to {fn}')
        stats.write(fn)


def possible_combine_regions(regions):
    """Return potential combinations of regions."""
    regions = sorted(regions)
    region_combs = {'GCcomb': ['NGC', 'SGC'],
                    'NS': ['N', 'S'],
                    'GCcomb_noN': ['NGCnoN', 'SGC'],
                    'GCcomb_noDES': ['NGC', 'SGCnoDES']}
    combs = {}
    for _region_comb, _regions in region_combs.items():
        if all(region in _regions for region in regions):
            combs[_region_comb] = _regions
    return combs


def compute_fkp_effective_redshift(*fkps, cellsize=10., order=2, split=None, func_of_z=lambda x: x,
                                   resampler='cic'):
    """
    Return effective redshift given input :class:`FKPField` of :class:`ParticleField` fields.

    Parameters
    ----------
    cellsize : float, default=10.
        Cellsize to use for mesh assignment in Mpc/h.
    order : int, default=2
        Weight redshift by the mesh density to that power (2 for 2PCF, 3 for 3PCF).
    split : int, tuple, optional
        Random seed to split particles to obtain ``order`` samples.
    func_of_z : callable, default=lambda x: x
        Optionally, function of redshift. E.g. growth rate.
    resampler : str, default='cic'
        Resampler to use for mesh assignment.

    Returns
    -------
    zeff : float
        Effective redshift.
    """
    # FIXME
    from jax import numpy as jnp
    from cosmoprimo.fiducial import TabulatedDESI, DESI
    from cosmoprimo.utils import Interpolator1D
    from jaxpower import split_particles, FKPField
    from jaxpower.mesh import _iter_meshes

    fiducial = TabulatedDESI()
    zmax, nz = 100., 512
    zgrid = 1. / np.geomspace(1. / (1. + zmax), 1., nz)[::-1] - 1.
    rgrid = fiducial.comoving_radial_distance(zgrid)
    d2z = Interpolator1D(jnp.array(rgrid), jnp.array(func_of_z(zgrid)), k=1)  #FIXME k = 1, otherwise memory error

    fkps_none =  list(fkps) + [None] * (order - len(fkps))

    def get_randoms(fkp):
        return fkp.randoms if isinstance(fkp, FKPField) else fkp

    randoms = [get_randoms(fkp) for fkp in fkps_none]

    def compute_fkp_normalization_z(*particles, cellsize=cellsize, split=None):
        if split is not None:
            particles = split_particles(*particles, seed=split)
        reduce = 1
        for mesh in _iter_meshes(*particles, resampler=resampler, cellsize=cellsize, compensate=False, interlacing=0):
            reduce *= mesh
        reduce /= reduce.sum()
        distance = jnp.sqrt(sum(xx**2 for xx in mesh.attrs.xcoords(kind='position', sparse=True)))
        reduce *= d2z(distance)
        return reduce.sum()

    return compute_fkp_normalization_z(*randoms, cellsize=cellsize, split=split)