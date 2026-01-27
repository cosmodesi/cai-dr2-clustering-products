import os
import logging
import functools
from pathlib import Path

import numpy as np
import jax
import lsstypes as types

from . import tools
from .tools import fill_fiducial_options, _merge_options, Catalog, setup_logging
from .correlation2_tools import compute_angular_upweights, compute_particle2_correlation
from .spectrum2_tools import compute_mesh2_spectrum, compute_window_mesh2_spectrum
from .spectrum3_tools import compute_mesh3_spectrum, compute_window_mesh3_spectrum
from .recon_tools import compute_reconstruction


logger = logging.getLogger('summary-statistics')


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


def _make_list_zrange(zranges):
    if np.ndim(zranges[0]) == 0:
        zranges = [zranges]
    return list(zranges)


def compute_stats_from_options(stats, analysis='full_shape', cache=None,
                                get_stats_fn=tools.get_stats_fn,
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
    analysis : str, optional
        Type of analysis, 'full_shape' or 'png_local', to set fiducial options.
    cache : dict, optional
        Cache to store intermediate results (binning class and parent/reference random catalog).
        See :func:`spectrum2_tools.compute_mesh2_spectrum`, :func:`spectrum3_tools.compute_mesh3_spectrum`,
        and func:`tools.read_clustering_catalog` for details.
    get_stats_fn : callable, optional
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
    kwargs = fill_fiducial_options(kwargs, analysis=analysis)
    catalog_options = kwargs['catalog']
    tracers = list(catalog_options.keys())

    zranges = {tracer: _make_list_zrange(catalog_options[tracer]['zrange']) for tracer in tracers}

    if get_catalog_fn is not None:
        read_clustering_catalog = functools.partial(read_clustering_catalog, get_catalog_fn=get_catalog_fn)
        read_full_catalog = functools.partial(read_full_catalog, get_catalog_fn=get_catalog_fn)

    with_recon = any('recon' in stat for stat in stats)

    data, randoms = {}, {}
    for tracer in tracers:
        _catalog_options = dict(catalog_options[tracer])
        _catalog_options['zrange'] = (min(zrange[0] for zrange in zranges[tracer]), max(zrange[1] for zrange in zranges[tracer]))
        if any(name in catalog_options.get('weight', '') for name in ['bitwise', 'compntile']):
            # sets NTILE-MISSING-POWER (missing_power) and per-tile completeness (completeness)
            _catalog_options['binned_weight'] = read_full_catalog(kind='parent_data', **_catalog_options, attrs_only=True)

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
                    start += size
                randoms[tracer] = randoms[tracer][:catalog_options['nran']]  # keep only relevant random files

        # Compute angular upweights
        if any(kwargs[stat].get('auw', False) for stat in stats):

            def get_data(tracer):
                _catalog_options = catalog_options[tracer] | dict(zrange=None)
                fibered = read_full_catalog(kind='fibered_data', **_catalog_options)
                full = read_full_catalog(kind='parent_data', **_catalog_options)
                return (fibered, full)

            auw = compute_angular_upweights(*[functools.partial(get_data, tracer) for tracer in tracers])
            fn_catalog_options = {tracer: catalog_options[tracer] | dict(zrange=None) for tracer in tracers}
            fn = get_stats_fn(kind='particle2_angular_upweights', catalog=fn_catalog_options)
            tools.write_stats(fn, auw)
            for key, kw in kwargs.items():
                if kw.get('auw', False): kw['auw'] = auw  # update with angular upweights

        for zvals in zip(*(zranges[tracer] for tracer in tracers)):
            zrange = dict(zip(tracers, zvals))

            def get_zcatalog(catalog, zrange):
                mask = (catalog['Z'] >= zrange[0]) & (catalog['Z'] < zrange[1])
                return catalog[mask]

            zdata, zrandoms = {}, {}
            for tracer in tracers:
                zdata[tracer] = get_zcatalog(data[tracer], zrange[tracer])
                zrandoms[tracer] = [get_zcatalog(random, zrange[tracer]) for random in randoms[tracer]]
            fn_catalog_options = {tracer: catalog_options[tracer] | dict(zrange=zrange[tracer]) for tracer in tracers}

            def get_catalog_recon(catalog):
                return catalog.clone(POSITION=catalog['POSITION_REC'])

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
                    fn = get_stats_fn(kind=stat, catalog=fn_catalog_options, **correlation_options)
                    tools.write_stats(fn, correlation)

                funcs = {f'{recon}mesh2_spectrum': compute_mesh2_spectrum, f'{recon}mesh3_spectrum': compute_mesh3_spectrum}

                for stat, func in funcs.items():
                    if stat in stats:
                        spectrum_options = dict(kwargs[stat])
                        selection_weights = spectrum_options.pop('selection_weights', None)

                        def get_data(tracer):
                            czrandoms = Catalog.concatenate(zrandoms[tracer])
                            if recon:
                                toret = (get_catalog_recon(zdata[tracer]), czrandoms,
                                         get_catalog_recon(czrandoms))
                            else:
                                toret = (zdata[tracer], czrandoms)
                            if selection_weights:
                                return tuple(selection_weights[tracer](catalog) for catalog in toret)
                            return toret

                        spectrum = func(*[functools.partial(get_data, tracer) for tracer in tracers], cache=cache, **spectrum_options)
                        if not isinstance(spectrum, dict): spectrum = {'raw': spectrum}
                        for key, kw in _expand_cut_auw_options(stat, spectrum_options).items():
                            fn = get_stats_fn(kind=stat, catalog=fn_catalog_options, **kw)
                            tools.write_stats(fn, spectrum[key])

                jax.experimental.multihost_utils.sync_global_devices('spectrum')  # such that spectrum ready for window
                funcs = {'window_mesh2_spectrum': compute_window_mesh2_spectrum, 'window_mesh3_spectrum': compute_window_mesh3_spectrum}

                for stat, func in funcs.items():
                    if stat in stats:
                        window_options = dict(kwargs[stat])
                        selection_weights = window_options.pop('selection_weights', None)

                        def get_data(tracer):
                            czrandoms = Catalog.concatenate(zrandoms[tracer])
                            toret = (zdata[tracer], czrandoms)
                            if selection_weights:
                                return tuple(selection_weights[tracer](catalog) for catalog in toret)
                            return toret

                        spectrum_fn = window_options.pop('spectrum', None)
                        if spectrum_fn is None:
                            spectrum_stat = stat.replace('window_', '')
                            spectrum_fn = get_stats_fn(kind=spectrum_stat, catalog=fn_catalog_options, **(kwargs[spectrum_stat] | dict(auw=False, cut=False)))
                        spectrum = types.read(spectrum_fn)

                        def get_extra(ibatch, nbatch):
                            return f'batch-{ibatch:d}-{nbatch:d}'

                        ibatch = window_options.get('ibatch', None)
                        extra = get_extra(*ibatch) if ibatch is not None else None

                        nbatch = window_options.get('computed_batches', False)
                        if nbatch:
                            fns = [get_stats_fn(kind=key, catalog=fn_catalog_options, **(window_options | dict(auw=False, cut=False, extra=get_extra(ibatch, nbatch)))) for ibatch in range(nbatch)]
                            window_options['computed_branches'] = [types.read(fn) for fn in fns]

                        window = func(*[functools.partial(get_data, tracer) for tracer in tracers], spectrum=spectrum, **window_options)
                        for key, kw in _expand_cut_auw_options(stat, window_options).items():
                            fn = get_stats_fn(kind=stat, catalog=fn_catalog_options, **kw)
                            if key in window:
                                tools.write_stats(fn, window[key])
                        for key in window:
                            if 'correlation' in key:  # window functions
                                fn = get_stats_fn(kind=key, catalog=fn_catalog_options, **(window_options | dict(auw=False, cut=False, extra=extra)))
                                tools.write_stats(fn, window[key])


def list_stats(stats, get_stats_fn=tools.get_stats_fn, **kwargs):
    """
    List measurements produced by :func:`compute_stats_from_options`.

    Parameters
    ----------
    stats : str or list of str
        Summary statistics to list.
    get_stats_fn : callable, optional
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

    tracers = list(catalog_options.keys())
    zranges = {tracer: _make_list_zrange(catalog_options[tracer]['zrange']) for tracer in tracers}

    toret = {stat: [] for stat in stats}
    for zvals in zip(*(zranges[tracer] for tracer in tracers)):
        zrange = dict(zip(tracers, zvals))
        _catalog_options = {tracer: catalog_options[tracer] | dict(zrange=zrange[tracer]) for tracer in tracers}
        for stat in stats:
            for kw in _expand_cut_auw_options(stat, kwargs[stat]).values():
                kw = dict(catalog=_catalog_options, **kw)
                fn = get_stats_fn(kind=stat, **kw)
                toret[stat].append((fn, kw))
    return toret


def combine_stats_from_options(stats, region_comb, regions, get_stats_fn=tools.get_stats_fn, **kwargs):
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
    get_stats_fn : callable, optional
        Function to get the filename for storing the measurement.
    **kwargs : dict
        Options for catalog and summary statistics. For example:
            catalog = dict(version='holi-v1-altmtl', tracer='LRG', zrange=[(0.4, 0.6), (0.8, 1.1)], imock=451)
            mesh2_spectrum = dict(cut=True, auw=True, ells=(0, 2, 4), mattrs=dict(boxsize=7000., cellsize=8.))  # all arguments for compute_mesh2_spectrum
            mesh3_spectrum = dict(basis='sugiyama-diagonal', ells=[(0, 0, 0)], mattrs=dict(boxsize=7000., cellsize=10.))  # all arguments for compute_mesh3_spectrum
    """
    options = fill_fiducial_options(kwargs)
    regions = list(regions)
    all_fns = {}
    for region in regions + [region_comb]:
        kwargs = dict(options)
        kwargs['catalog'] = {tracer: options['catalog'][tracer] | dict(region=region) for tracer in options['catalog']}
        all_fns[region] = list_stats(stats, get_stats_fn=get_stats_fn, **kwargs)

    stats = next(iter(all_fns.values())).keys()
    for stat in stats:
        for ifn, (fn_comb, _) in enumerate(all_fns[region_comb][stat]):
            fns = [all_fns[region][stat][ifn][0] for region in regions]  # [1] is kwargs
            exists = {os.path.exists(fn): fn for fn in fns}
            if all(exists):
                combined = tools.combine_stats([types.read(fn) for fn in fns])
                tools.write_stats(fn_comb, combined)
            else:
                logger.debug(f'Skipping {fn_comb} as {[fn for ex, fn in exists.items() if not ex]} do not exist')


def main(**kwargs):
    r"""
    This is an example main, which can be run from command line to compute fiducial statistics.
    Let's try to keep it simple; write your own if you need anything fancier.
    Or just use :func:`compute_stats_from_options` directly; see example in `job_scripts/desipipe_holi_mocks.py`.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats', help='what do you want to compute?', type=str, nargs='*', choices=['mesh2_spectrum', 'mesh3_spectrum', 'recon_particle2_correlation', 'window_mesh2_spectrum', 'window_mesh3_spectrum'], default=['mesh2_spectrum'])
    parser.add_argument('--version', help='catalog version; e.g. holi-v1-altmtl', type=str, default=None)
    parser.add_argument('--cat_dir', help='where to find catalogs', type=str, default=None)
    parser.add_argument('--tracer', help='tracer(s) to be selected - e.g. LRG ELG for cross-correlation', nargs='*', type=str, default='LRG')
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
    parser.add_argument('--expand_randoms', help='expand catalog of randoms; provide version of parent randoms (must be registered in get_catalog_fn)', type=str, choices=['data-dr2-v2'], default=None)
    parser.add_argument('--stats_dir',  help='base directory for measurements, default is SCRATCH', type=str, default=Path(os.getenv('SCRATCH')) / 'measurements')
    parser.add_argument('--stats_extra',  help='extra string to include in measurement filename', type=str, default='')
    parser.add_argument('--combine', help='combine measurements in two regions', action='store_true')

    args = parser.parse_args()
    if args.stats:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
        import jax
        jax.distributed.initialize()

    setup_logging()
    if args.zrange is None:
        zranges = tools.propose_fiducial('zranges', tracer=tools.join_tracers(args.tracer), analysis=args.analysis)
    else:
        assert len(args.zrange) % 2 == 0
        zranges = list(zip(args.zrange[::2], args.zrange[1::2]))
    mattrs = {key: value for key, value in dict(boxsize=args.boxsize, cellsize=args.cellsize).items() if value is not None}
    options = {'mattrs': mattrs}
    for stat in ['mesh2_spectrum', 'particle2_correlation']:
        options.setdefault(stat, {})
        options[stat].update(cut=args.thetacut, auw=args.auw)
    get_catalog_fn = tools.get_catalog_fn
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=args.stats_dir, extra=args.stats_extra)
    cache = {}
    for imock in args.imock:
        catalog_options = dict(version=args.version, cat_dir=args.cat_dir, tracer=args.tracer, zrange=zranges,
                               weight=args.weight, nran=args.nran, imock=imock)
        options_imock = _merge_options(fill_fiducial_options(dict(catalog=catalog_options) | options, analysis=args.analysis), kwargs)

        for region in args.region:
            _options_imock = dict(options_imock)
            for tracer in _options_imock['catalog']:
                _options_imock['catalog'][tracer] = _options_imock['catalog'][tracer] | dict(region=region)
                if args.expand_randoms:
                    _options_imock['catalog'][tracer]['expand'] = {'parent_randoms_fn': get_catalog_fn(kind='parent_randoms', version=args.expand_randoms, tracer=tracer, region=region, nran=max(value['nran'] for value in _options_imock['recon'].values()))}
            compute_stats_from_options(args.stats, get_catalog_fn=get_catalog_fn, get_stats_fn=get_stats_fn, cache=cache, **_options_imock)
            jax.experimental.multihost_utils.sync_global_devices('measurements')
        if args.combine and jax.process_index() == 0:
            for region_comb, regions in tools.possible_combine_regions(args.region).items():
                combine_stats_from_options(args.stats, region_comb, regions, get_stats_fn=get_stats_fn, **_options_imock)
    if args.stats:
        jax.distributed.shutdown()


if __name__ == '__main__':

    main()
