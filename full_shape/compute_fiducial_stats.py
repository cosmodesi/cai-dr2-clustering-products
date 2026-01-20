import os
import logging
import functools
from pathlib import Path

import numpy as np
import lsstypes as types

import tools
from tools import Catalog, setup_logging
from correlation2_tools import compute_angular_upweights, compute_particle2_correlation
from spectrum2_tools import prepare_jaxpower_particles, compute_mesh2_spectrum
from spectrum3_tools import compute_mesh3_spectrum
from recon_tools import compute_reconstruction


logger = logging.getLogger('summary-statistics')


def _merge_options(options1, options2):
    """Merge two options dictionaries. Nested dictionaries are updated."""
    options = {key: dict(value) for key, value in options1.items()}
    for key, value in options2.items():
        if key not in options: options[key] = {}
        options[key].update(value)
    return options


def fill_fiducial_options(kwargs, analysis='full_shape'):
    """Fill missing options with fiducial values."""
    options = {key: dict(value) for key, value in kwargs.items()}
    mattrs = options.pop('mattrs', {})
    tracer = options['catalog']['tracer']
    tracers = tuple(tracer) if not isinstance(tracer, str) else (tracer,)
    options['catalog']['tracer'] = tracers
    fiducial_options = tools.propose_fiducial('catalog', tracer=tools.join_tracers(tracers), analysis=analysis)
    options['catalog'] = fiducial_options | options['catalog']
    if options['catalog'].get('nran', None) is None:
        options['catalog']['nran'] = tools.propose_fiducial('nran', tracer=tools.join_tracers(tracers), analysis=analysis)
    recon_options = options.pop('recon', {})
    # recon for each tracer
    options['recon'] = {}
    for tracer in tracers:
        fiducial_options = tools.propose_fiducial('recon', tracer=tracer, analysis=analysis)
        options['recon'][tracer] = fiducial_options | recon_options.get(tracer, recon_options)
        if mattrs: options['recon'][tracer]['mattrs'] = mattrs
        options['recon'][tracer]['nran'] = options['recon'][tracer].get('nran', options['catalog']['nran'])
        assert options['recon'][tracer]['nran'] >= options['catalog']['nran'], 'must use more randoms for reconstruction than clustering measurements'
    for recon in ['', 'recon_']:
        for stat in ['particle2_correlation', 'mesh2_spectrum', 'mesh3_spectrum']:
            stat = f'{recon}{stat}'
            fiducial_options = tools.propose_fiducial(stat, tracer=tools.join_tracers(tracers), analysis=analysis)
            options[stat] = fiducial_options | options.get(stat, {})
            if 'mesh' in stat:
                if mattrs: options[stat]['mattrs'] = mattrs
    return options


def _expand_cut_auw_options(stat, options):
    """Expand options for cut and auw spectra, returning a dictionary of options with keys 'raw', 'cut', 'auw' as needed."""
    if 'spectrum' in stat:
        keys = ['cut', 'auw']
        kw = dict(options)
        for key in keys: kw.pop(key, None)
        args = {'raw': kw}
        for key in keys:
            kw = dict(options)
            if not kw.get(key, False):
                continue
            else:
                for name in keys:
                    if name != key: kw.pop(name, None)  # keep only if spectrum is with cut (resp. auw)
                args[key] = kw
    else:
        args = {'stat': options}
    return args


def clone_catalog(catalog, **kwargs):
    """Clone a catalog and update columns with given values."""
    catalog = catalog.copy()
    for column, value in kwargs.items():
        catalog[column] = value
    return catalog


def apply_fiducial_selection_weight(catalog, stat):
    """
    Apply fiducial selection weight to the catalog based on the statistic being computed.
    For the bispectrum (mesh3), the individual weights are scaled by NX^(-1/3) to make the
    effective redshift comparable to that of the power spectrum (mesh2).
    """
    if 'mesh3' in stat:
        catalog = clone_catalog(catalog, INDWEIGHT=catalog['INDWEIGHT'] * catalog['NX']**(-1. / 3.))
    return catalog


def compute_fiducial_stats_from_options(stats, cache=None,
                                        get_measurement_fn=tools.get_measurement_fn,
                                        get_catalog_fn=None,
                                        read_clustering_catalog=tools.read_clustering_catalog,
                                        read_full_catalog=tools.read_full_catalog,
                                        **kwargs):
    """
    Compute summary statistics based on the provided options.

    Parameters
    ----------
    stats : str or list of str
        Summary statistics to compute.
    cache : dict, optional
        Cache to store intermediate results (binning class and parent/reference random catalog).
        See :func:`spectrum2_tools.compute_mesh2_spectrum`, :func:`spectrum3_tools.compute_mesh3_spectrum`,
        and func:`tools.read_clustering_catalog` for details.
    get_measurement_fn : callable, optional
        Function to get the filename for storing the measurement.
    get_catalog_fn : callable, optional
        Function to get the filename for reading the catalog.
        If provided, it is given to ``read_clustering_catalog`` and ``read_full_catalog``.
    read_clustering_catalog : callable, optional
        Function to read the clustering catalog.
    read_full_catalog : callable, optional
        Function to read the full catalog.
    **kwargs : dict
        Options for catalog, reconstruction, and summary statistics.
    """
    if isinstance(stats, str):
        stats = [stats]

    cache = cache or {}
    kwargs = fill_fiducial_options(kwargs)
    catalog_options = kwargs['catalog']
    tracers = catalog_options['tracer']
    zranges = catalog_options['zrange']
    if np.ndim(zranges[0]) == 0:
        zranges = [zranges]
    zrange_max = (min(zrange[0] for zrange in zranges), max(zrange[1] for zrange in zranges))
    if get_catalog_fn is not None:
        read_clustering_catalog = functools.partial(read_clustering_catalog, get_catalog_fn=get_catalog_fn)
        read_full_catalog = functools.partial(read_full_catalog, get_catalog_fn=get_catalog_fn)

    with_recon = any('recon' in stat for stat in stats)

    data, randoms = {}, {}
    for tracer in tracers:
        _catalog_options = dict(catalog_options)
        _catalog_options['zrange'] = zrange_max
        _catalog_options['tracer'] = tracer
        if any(name in catalog_options.get('weight', '') for name in ['bitwise', 'compntile']):
            # sets NTILE-MISSING-POWER (missing_power) and per-tile completeness (completeness)
            _catalog_options['binned_weight'] = read_full_catalog(kind='parent_data', **_catalog_options).attrs

        data[tracer] = read_clustering_catalog(kind='data', **_catalog_options, concatenate=True)
        if with_recon:
            recon_options = kwargs['recon'][tracer]
            # pop as we don't need it anymore
            _catalog_options |= {key: recon_options.pop(key) for key in list(recon_options) if key in ['nran', 'zrange']}
        randoms[tracer] = read_clustering_catalog(kind='randoms', **_catalog_options, cache=cache, concatenate=False)

    from jaxpower import create_sharding_mesh
    with create_sharding_mesh() as sharding_mesh:
        if with_recon:
            data_rec, randoms_rec = {}, {}
            for tracer in tracers:
                recon_options = kwargs['recon'][tracer]
                # local sizes to select positions
                data[tracer]['POSITION_REC'], randoms_rec_positions = compute_reconstruction(lambda: (data[tracer], Catalog.concatenate(randoms[tracer])), **recon_options)
                start = 0
                for random in randoms[tracer]:
                    size = len(random['POSITION'])
                    random['POSITION_REC'] = randoms_rec_positions[start:start + size]
                    start += sizeget_catalog_fn
                randoms[tracer] = randoms[tracer][:catalog_options['nran']]  # keep only relevant random files

        def get_sliced(catalog, sizes):
            sliced = []
            start = 0
            for size in sizes:
                sliced.append(catalog[slice(start, start + size)])
                start += size
            return sliced

        # Compute angular upweights
        if any(kwargs[stat].get('auw', False) for stat in stats):

            def get_data(tracer):
                _catalog_options = catalog_options | dict(tracer=tracer, zrange=None)
                fibered = read_full_catalog(kind='fibered_data', **_catalog_options)
                full = read_full_catalog(kind='parent_data', **_catalog_options)
                return (fibered, full)

            auw = compute_angular_upweights(*[functools.partial(get_data, tracer) for tracer in tracers])
            _catalog_options = catalog_options | dict(zrange=None)
            fn = get_measurement_fn(kind='particle2_angular_upweights', **_catalog_options)
            tools.write_summary_statistics(fn, auw)
            for key, kw in kwargs.items():
                if kw.get('auw', False): kw['auw'] = auw  # update with angular upweights

        for zrange in zranges:

            def get_zcatalog(catalog):
                mask = (catalog['Z'] >= zrange[0]) & (catalog['Z'] < zrange[1])
                return catalog[mask]

            zdata, zrandoms = {}, {}
            for tracer in tracers:
                zdata[tracer] = get_zcatalog(data[tracer])
                zrandoms[tracer] = [get_zcatalog(random) for random in randoms[tracer]]
            zcatalog_options = catalog_options | dict(zrange=zrange)

            def get_catalog_recon(catalog):
                return clone_catalog(catalog, POSITION=catalog['POSITION_REC'])

            for recon in ['', 'recon_']:
                stat = f'{recon}particle2_correlation'
                if stat in stats:
                    correlation_options = kwargs[stat]

                    def get_data(tracer):
                        if recon:
                            return (get_catalog_recon(zdata[tracer]),
                                    zrandoms[tracer],
                                    [get_catalog_recon(zrandom) for zrandom in zrandoms[tracer]])
                        return (zdata[tracer], zrandoms[tracer])

                    correlation = compute_particle2_correlation(*[functools.partial(get_data, tracer) for tracer in tracers], **correlation_options)
                    fn = get_measurement_fn(kind=stat, **zcatalog_options, **correlation_options)
                    tools.write_summary_statistics(fn, correlation)

                # Prepare jax-power particles: spatial sharding
                funcs = {f'{recon}mesh2_spectrum': compute_mesh2_spectrum, f'{recon}mesh3_spectrum': compute_mesh3_spectrum}

                for stat, func in funcs.items():
                    if stat in stats:

                        def get_data(tracer):
                            czrandoms = Catalog.concatenate(zrandoms[tracer])
                            if recon:
                                toret = (get_catalog_recon(zdata[tracer]), czrandoms,
                                         get_catalog_recon(czrandoms))
                            else:
                                toret = (zdata[tracer], czrandoms)
                            return tuple(apply_fiducial_selection_weight(catalog, stat=stat) for catalog in toret)

                        spectrum_options = kwargs[stat]
                        spectrum = func(*[functools.partial(get_data, tracer) for tracer in tracers], cache=cache, **spectrum_options)
                        if not isinstance(spectrum, dict): spectrum = {'raw': spectrum}
                        for key, kw in _expand_cut_auw_options(stat, spectrum_options).items():
                            fn = get_measurement_fn(kind=stat, **zcatalog_options, **kw)
                            tools.write_summary_statistics(fn, spectrum[key])


def combine_fiducial_stats_from_options(stats, region_comb, regions, get_measurement_fn=tools.get_measurement_fn, **kwargs):
    """
    Combine summary statistics from multiple regions based on the provided options.

    Parameters
    ----------
    stats : str or list of str
        Summary statistics to combine.
    region_comb : str
        Combined region name, e.g. 'GCcomb'.
    regions : list of str
        Regions to combine, e.g. ['NGC', 'SGC'].
    get_measurement_fn : callable, optional
        Function to get the filename for storing the measurement.
    **kwargs : dict
        Options for catalog and summary statistics. For example:
            catalog = dict(version='holi-v1-altmtl', tracer='LRG', zrange=[(0.4, 0.6), (0.8, 1.1)], imock=451)
            mesh2_spectrum = dict(cut=True, auw=True, ells=(0, 2, 4), mattrs=dict(boxsize=7000., cellsize=8.))  # all arguments for compute_mesh2_spectrum
            mesh3_spectrum = dict(basis='sugiyama-diagonal', ells=[(0, 0, 0)], mattrs=dict(boxsize=7000., cellsize=10.))  # all arguments for compute_mesh3_spectrum
    """
    if isinstance(stats, str):
        stats = [stats]

    kwargs = fill_fiducial_options(kwargs)
    catalog_options = kwargs['catalog']
    zranges = catalog_options['zrange']
    if np.ndim(zranges[0]) == 0:
        zranges = [zranges]
    for zrange in zranges:
        for stat in stats:
            for kw in _expand_cut_auw_options(stat, kwargs[stat]).values():
                fns = [get_measurement_fn(kind=stat, **(catalog_options | dict(region=region, zrange=zrange)), **kw) for region in regions]
                fn_comb = get_measurement_fn(kind=stat, **(catalog_options | dict(region=region_comb, zrange=zrange)), **kw)
                exists = {os.path.exists(fn): fn for fn in fns}
                if all(exists):
                    combined = types.sum([types.read(fn) for fn in fns])
                    tools.write_summary_statistics(fn_comb, combined)
                else:
                    logger.debug(f'Skipping {fn_comb} as {[fn for ex, fn in exists.items() if not ex]} do not exist')


def main(**kwargs):
    r"""
    This is an example main, which can be run from command line to compute fiducial statistics.
    Let's try to keep it simple; write your own if you need anything fancier.
    Or just use :func:`compute_fiducial_stats_from_options` directly; see example in `job_scripts/desipipe_holi_mocks.py`.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats', help='what do you want to compute?', type=str, nargs='*', choices=['mesh2_spectrum', 'mesh3_spectrum', 'recon_particle2_correlation'], default=['mesh2_spectrum'])
    parser.add_argument('--version', help='catalog version; e.g. holi-v1-altmtl', type=str, default=None)
    parser.add_argument('--cat_dir', help='where to find catalogs', type=str, default=None)
    parser.add_argument('--tracer', help='tracer(s) to be selected - e.g. LRGxELG for cross-correlation', type=str, default='LRG')
    parser.add_argument('--zrange', help='redshift bins; 0.4 0.6 0.8 1.1 to run (0.4, 0.6), (0.8, 1.1)', nargs='*', type=float, default=None)
    parser.add_argument('--imock', help='mock number', type=int, nargs='*', default=[None])
    parser.add_argument('--region', help='regions', type=str, nargs='*', choices=['N', 'S', 'NGC', 'SGC', 'NGCnoN', 'SGCnoDES'], default=['NGC', 'SGC'])
    parser.add_argument('--analysis', help='type of analysis', type=str, choices=['full_shape', 'png_local'], default='full_shape')
    parser.add_argument('--weight',  help='type of weights to use for tracer; "default" just uses WEIGHT column', type=str, default='default_FKP')
    parser.add_argument('--thetacut',  help='Apply theta-cut', action='store_true', default=None)
    parser.add_argument('--auw',  help='Apply angular upweighting', action='store_true', default=None)
    parser.add_argument('--boxsize',  help='box size', type=float, default=None)
    parser.add_argument('--cellsize', help='cell size', type=float, default=None)
    parser.add_argument('--nran', help='number of random files to combine together (1-18 available)', type=int, default=None)
    parser.add_argument('--expand_randoms', help='expand catalog of randoms; provide version of parent randoms (must be registered in get_catalog_fn)', type=str, default=None)
    parser.add_argument('--meas_dir',  help='base directory for measurements, default is SCRATCH', type=str, default=Path(os.getenv('SCRATCH')) / 'measurements')
    parser.add_argument('--meas_extra',  help='extra string to include in measurement filename', type=str, default='')
    parser.add_argument('--combine', help='combine measurements in two regions', action='store_true')

    args = parser.parse_args()
    if args.stats:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
        import jax
        jax.distributed.initialize()

    setup_logging()
    tracer = args.tracer
    if args.zrange is None:
        zranges = tools.propose_fiducial('zranges', tracer=tools.join_tracers(tracer), analysis=args.analysis)
    else:
        assert len(args.zrange) % 2 == 0
        zranges = list(zip(args.zrange[::2], args.zrange[1::2]))
    mattrs = {key: value for key, value in dict(boxsize=args.boxsize, cellsize=args.cellsize).items() if value is not None}
    options = {'mattrs': mattrs}
    for stat in ['mesh2_spectrum', 'particle2_correlation']:
        options.setdefault(stat, {})
        options[stat].update(cut=args.thetacut, auw=args.auw)
    options = _merge_options(options, kwargs)
    get_catalog_fn = tools.get_catalog_fn
    get_measurement_fn = functools.partial(tools.get_measurement_fn, meas_dir=args.meas_dir, extra=args.meas_extra)
    cache = {}
    for imock in args.imock:
        catalog_options = dict(version=args.version, cat_dir=args.cat_dir, tracer=args.tracer, zrange=zranges,
                               weight=args.weight, nran=args.nran, imock=imock)
        options_imock = fill_fiducial_options(dict(catalog=catalog_options) | options, analysis=args.analysis)
        for region in args.region:
            _options_imock = dict(options_imock)
            _options_imock['catalog'] = _options_imock['catalog'] | dict(region=region)
            if args.expand_randoms:
                _options_imock['expand_randoms'] = {'parent_randoms_fn': get_catalog_fn(kind='randoms', version=args.expand_randoms, cat_dir=args.cat_dir, tracer=args.tracer, region=region)}
            compute_fiducial_stats_from_options(args.stats, get_catalog_fn=get_catalog_fn, get_measurement_fn=get_measurement_fn, cache=cache, **_options_imock)
            jax.experimental.multihost_utils.sync_global_devices('measurements')
        if args.combine and jax.process_index() == 0:
            for region_comb, regions in tools.possible_combine_regions(args.region).items():
                combine_fiducial_stats_from_options(args.stats, region_comb, regions, get_measurement_fn=get_measurement_fn, **_options_imock)
    if args.stats:
        jax.distributed.shutdown()


if __name__ == '__main__':

    main()
