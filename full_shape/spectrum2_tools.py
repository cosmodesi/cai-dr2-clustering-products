import logging

import numpy as np
import jax
import lsstypes as types

from tools import default_mpicomm, _format_bitweights


logger = logging.getLogger('spectrum2')


@default_mpicomm
def prepare_jaxpower_particles(*get_data_randoms, mattrs=None, add_data=tuple(), add_randoms=tuple(), **kwargs):
    """
    Prepare :class:`jaxpower.ParticleField` objects from data and randoms catalogs.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return tuples of (data, randoms, [shifted]) catalogs.
        Each catalog must contain 'POSITION' and 'INDWEIGHT', and optionally 'BITWEIGHT' for bitwise weights and 'TARGETID'
        for randoms IDs to allow process-invariant random split in bispectrum normalization.
    mattrs : dict, optional
        Mesh attributes to define the :class:`ParticleField` objects. If None, default attributes are used.
    kwargs : dict, optional
        Additional keyword arguments to pass to :class:`ParticleField`.

    Returns
    -------
    all_particles : list of tuples
        List of tuples (data, randoms, shifted) ParticleField objects for each input catalog.
    """
    from jaxpower.mesh import get_mesh_attrs, ParticleField, make_array_from_process_local_data
    backend = 'mpi'
    mpicomm = kwargs['mpicomm']

    all_data, all_randoms, all_shifted = [], [], []
    for _get_data_randoms in get_data_randoms:
        # data, randoms (optionally shifted) are tuples (positions, weights)
        data, randoms, *shifted = _get_data_randoms()
        all_data.append(data)
        all_randoms.append(randoms)
        if shifted:
            all_shifted.append(shifted[0])

    if all_shifted:
        assert len(all_shifted) == len(all_data), 'Give as many shifted randoms as data/randoms'

    # Define the mesh attributes; pass in positions only
    mattrs = get_mesh_attrs(*[data['POSITION'] for data in all_data + all_shifted + all_randoms], check=True, **(mattrs or {}))
    if jax.process_index() == 0:
        logger.info(f'Using mesh {mattrs}.')

    def collective_arange(local_size):
        sizes = mpicomm.allgather(local_size)
        return sum(sizes[:mpicomm.rank]) + np.arange(local_size)

    all_particles = []
    for i, (data, randoms) in enumerate(zip(all_data, all_randoms)):
        _add_data, _add_randoms = {}, {}
        indweights, bitweights = data['INDWEIGHT'], None
        if 'BITWEIGHT' in data and 'BITWEIGHT' in add_data:
            bitweights = _format_bitweights(data['BITWEIGHT'])
            from cucount.jax import BitwiseWeight
            iip = BitwiseWeight(weights=bitweights, p_correction_nbits=False)(bitweights)
            _add_data['BITWEIGHT'] = [indweights] + bitweights  # add individual weight (photometric, spectro systematics) without PIP
            indweights = indweights * iip  # multiply by IIP to correct fiber assignment at large scales
        data = ParticleField(data['POSITION'], indweights, attrs=mattrs, exchange=True, backend=backend, **kwargs)
        #ids = collective_arange(len(randoms['POSITION']))
        if 'TARGETID' in randoms and 'IDS' in add_randoms:
            _add_randoms['IDS'] = randoms['TARGETID']
        randoms = ParticleField(randoms['POSITION'], randoms['INDWEIGHT'], attrs=mattrs, exchange=True, backend=backend, **kwargs)
        if backend == 'jax':  # first convert to JAX Array
            sharding_mesh = mattrs.sharding_mesh
            for key, value in _add_data.items():
                _add_data[key] = make_array_from_process_local_data(value, pad=0, sharding_mesh=sharding_mesh)
            for key, value in _add_randoms.items():
                _add_randoms[key] = make_array_from_process_local_data(value, pad=0, sharding_mesh=sharding_mesh)
        for key, value in _add_data.items():
            if isinstance(value, list): value = [data.exchange_direct(value, pad=0) for value in value]
            else: value = data.exchange_direct(value, pad=0)
            data.__dict__[key] = value
        for key, value in _add_randoms.items():
            if isinstance(value, list): value = [randoms.exchange_direct(value, pad=0) for value in value]
            else: value = randoms.exchange_direct(value, pad=0)
            randoms.__dict__[key] = value
        if all_shifted:
            shifted = all_shifted[i]
            shifted = ParticleField(shifted['POSITION'], shifted['INDWEIGHT'], attrs=mattrs, exchange=True, backend=backend, **kwargs)
        else:
            shifted = None
        all_particles.append((data, randoms, shifted))
    if jax.process_index() == 0:
        logger.info(f'All particles on the device')

    return all_particles


def _get_jaxpower_attrs(*particles):
    """Return summary attributes from :class:`jaxpower.ParticleField` objects: total weight and size."""
    mattrs = particles[0][0].attrs
    # Creating FKP fields
    attrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    for i, (data, randoms, shifted) in enumerate(particles):
        attrs[f'size_data{i:d}'], attrs[f'wsum_data{i:d}'] = data.size, data.sum()
        attrs[f'size_randoms{i:d}'], attrs[f'wsum_randoms{i:d}'] = randoms.size, randoms.sum()
        if shifted is not None:
            attrs[f'size_shifted{i:d}'], attrs[f'wsum_shifted{i:d}'] = shifted.size, shifted.sum()
    return attrs


def compute_mesh2_spectrum(*get_data_randoms, mattrs=None, cut=None, auw=None,
                           ells=(0, 2, 4), edges=None, los='firstpoint', optimal_weights=None,
                           cache=None):
    r"""
    Compute the 2-point spectrum multipoles using mesh-based FKP fields with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return tuples of (data, randoms, [shifted]) catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects. If None, default attributes are used.
        See :func:`prepare_jaxpower_particles` for details.
    cut : bool, optional
        If True, apply a theta-cut of (0, 0.05) in degrees.
    auw : ObservableTree, optional
        Angular upweights to apply. If ``None``, no angular upweights are applied.
    ells : list of int, optional
        List of multipole moments to compute. Default is (0, 2, 4).
    edges : dict, optional
        Edges for the binning; array or dictionary with keys 'start' (minimum :math:`k`), 'stop' (maximum :math:`k`), 'step' (:math:`\Delta k`).
        If ``None``, default step of :math:`0.001 h/\mathrm{Mpc}` is used.
        See :class:`jaxpower.BinMesh2SpectrumPoles` for details.
    los : {'local', 'firstpoint', 'x', 'y', 'z', array-like}, optional
        Line-of-sight definition. 'local' uses local LOS, 'firstpoint' uses the position of the first point in the pair,
        'x', 'y', 'z' use fixed axes, or provide a 3-vector.
    optimal_weights : callable or None, optional
        Function taking (ell, catalog) as input and returning total weights to apply to data and randoms.
        It can have an optional attribute 'columns' that specifies which additional columns are needed to compute the optimal weights.
        As a default, ``optimal_weights.columns = ['Z']`` to indicate that redshift information is needed.
        A dictionary ``catalog`` of columns is provided, containing 'INDWEIGHT' and the requested columns.
        If ``None``, no optimal weights are applied.
    cache : dict, optional
        Cache to store binning class (can be reused if ``meshsize`` and ``boxsize`` are the same).
        If ``None``, a new cache is created.

    Returns
    -------
    spectrum : Mesh2SpectrumPoles or dict of Mesh2SpectrumPoles
        The computed 2-point spectrum multipoles. If `cut` or `auw` are provided, returns a dict with keys 'raw', 'cut', and/or 'auw'.
    """

    from jaxpower import (FKPField, compute_fkp2_normalization, compute_fkp2_shotnoise, BinMesh2SpectrumPoles, compute_mesh2_spectrum,
                          BinParticle2SpectrumPoles, BinParticle2CorrelationPoles, compute_particle2, compute_particle2_shotnoise)

    columns_optimal_weights = []
    if optimal_weights is not None:
        columns_optimal_weights += getattr(optimal_weights, 'columns', ['Z'])   # to compute optimal weights, e.g. for fnl
    all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_data=['BITWEIGHT'] + columns_optimal_weights, add_randoms=columns_optimal_weights)

    if cache is None: cache = {}
    if edges is None: edges = {'step': 0.001}

    def _compute_spectrum(all_particles, ells, ifields=None):
        # Compute power spectrum for input given multipoles
        attrs = _get_jaxpower_attrs(*all_particles)
        mattrs = all_particles[0][0].attrs

        # Define the binner
        key = 'bin_mesh2_spectrum_{}'.format('_'.join(map(str, ells)))
        bin = cache.get(key, None)
        if bin is None or not np.all(bin.mattrs.meshsize == mattrs.meshsize) or not np.allclose(bin.mattrs.boxsize, mattrs.boxsize):
            bin = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)
        cache.setdefault(key, bin)

        # Computing normalization
        all_fkp = [FKPField(data, randoms) for (data, randoms, _) in all_particles]
        norm = compute_fkp2_normalization(*all_fkp, bin=bin, cellsize=10)

        # Computing shot noise
        all_fkp = [FKPField(data, shifted if shifted is not None else randoms) for (data, randoms, shifted) in all_particles]
        del all_particles
        num_shotnoise = compute_fkp2_shotnoise(*all_fkp, bin=bin, fields=ifields)

        jax.block_until_ready((norm, num_shotnoise))
        if jax.process_index() == 0:
            logger.info('Normalization and shotnoise computation finished')

        results = {}
        # First compute the theta-cut pairs
        if cut is not None:
            sattrs = {'theta': (0., 0.05)}
            #pbin = BinParticle2SpectrumPoles(mattrs, edges=bin.edges, xavg=bin.xavg, sattrs=sattrs, ells=ells)
            pbin = BinParticle2CorrelationPoles(mattrs, edges={'step': 0.1}, sattrs=sattrs, ells=ells)
            from jaxpower.particle2 import convert_particles
            all_particles = [convert_particles(fkp.particles) for fkp in all_fkp]
            close = compute_particle2(*all_particles, bin=pbin, los=los)
            close = close.clone(num_shotnoise=compute_particle2_shotnoise(*all_particles, bin=pbin), norm=norm)
            close = close.to_spectrum(bin.xavg)
            results['cut'] = -close.value()

        # Then compute the AUW-weighted pairs
        with_bitweights = 'BITWEIGHT' in all_fkp[0].data.__dict__
        if auw is not None or with_bitweights:
            from cucount.jax import WeightAttrs
            from jaxpower.particle2 import convert_particles
            sattrs = {'theta': (0., 0.1)}
            bitwise = angular = None
            if with_bitweights:
                # Order of weights matters
                # fkp.data.__dict__['BITWEIGHT'] includes IIP in the first position
                all_data = [convert_particles(fkp.data, weights=list(fkp.data.__dict__['BITWEIGHT']) + [fkp.data.weights], exchange_weights=False) for fkp in all_fkp]
                bitwise = dict(weights=all_data[0].get('bitwise_weight'))  # sets nrealizations, etc.: fine to use the first
                if jax.process_index() == 0:
                    logger.info(f'Applying PIP weights {bitwise}.')
            else:
                all_data = [convert_particles(fkp.data, weights=[fkp.data.weights] * 2, exchange_weights=False, index_value=dict(individual_weight=1, negative_weight=1)) for fkp in all_fkp]
            if auw is not None:
                angular = dict(sep=auw.get('DD').coords('theta'), weight=auw.get('DD').value())
                if jax.process_index() == 0:
                    logger.info(f'Applying AUW {angular}.')
            wattrs = WeightAttrs(bitwise=bitwise, angular=angular)
            pbin = BinParticle2SpectrumPoles(mattrs, edges=bin.edges, xavg=bin.xavg, sattrs=sattrs, wattrs=wattrs, ells=ells)
            DD = compute_particle2(*all_data, bin=pbin, los=los)
            DD = DD.clone(num_shotnoise=compute_particle2_shotnoise(*all_data, bin=pbin), norm=norm)
            results['auw'] = DD.value()

        jax.block_until_ready(results)
        if jax.process_index() == 0:
            logger.info(f'Particle-based calculation finished')

        kw = dict(resampler='tsc', interlacing=3, compensate=True)
        # out='real' to save memory
        meshes = [fkp.paint(**kw, out='real') for fkp in all_fkp]
        del all_fkp

        # JIT the mesh-based spectrum computation; helps with memory footprint
        jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'])
        #jitted_compute_mesh2_spectrum = compute_mesh2_spectrum
        spectrum = jitted_compute_mesh2_spectrum(*meshes, bin=bin, los=los)
        spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
        spectrum = spectrum.map(lambda pole: pole.clone(attrs=attrs))
        jax.block_until_ready(spectrum)
        if jax.process_index() == 0:
            logger.info('Mesh-based computation finished')

        # Add theta-cut and AUW contributes
        for name, value in results.items():
            results[name] = spectrum.clone(value=spectrum.value() + value)
        results['raw'] = spectrum

        return results

    if optimal_weights is None:
        results = _compute_spectrum(*all_particles, ells=ells)
    else:
        results = {}
        for ell in ells:
            logger.info(f'Applying optimal weights for ell {ell}')

            ifields = tuple(range(len(all_particles)))
            ifields = ifields + (ifields[-1],) * (2 - len(ifields))
            all_particles = all_particles + (all_particles[-1],) * (2 - len(all_particles))

            def _get_optimal_weights(all_data):
                if all_data is None: # shifted, yield None
                    while True:
                        yield None
                for all_weights in optimal_weights(ell, [{'INDWEIGHT': data.weights} | {column: data.__dict__[column] for column in columns_optimal_weights} for data in all_data]):
                    yield tuple(data.clone(weights=weights) for data, weights in zip(all_data, all_weights))

            result_ell = {}
            for all_data, all_randoms, all_shifted in zip(*[*_get_optimal_weights([particles[i] for particles in all_particles]) for i in range(3)]):
                # all_data, all_randoms, all_shifted are tuples of ParticleField with optimal weights applied
                _all_particles = list(zip(all_data, all_randoms, all_shifted))
                _result = _compute_spectrum(_all_particles, ells=ell, ifields=ifields)
                for key in result:  # raw, cut, auw
                    result_ell.setdefault(key, [])
                    result_ell[key].append(_result[key])
            for key, value in result_ell.items():  # sum 1<->2
                result.setdefault(key, [])
                result[key].append(types.sum(value))
        for key in results:  # join multipoles
            results[key] = types.join(results[key])

    if len(results) == 1:
        return next(iter(results.values()))
    return results



def compute_window_mesh2_spectrum(*get_randoms, spectrum, kind='smooth', optimal_weights=None):
    # TODO FIXME
    from jax import numpy as jnp
    from jaxpower import (ParticleField, compute_mesh2_spectrum_window, BinMesh2SpectrumPoles, BinMesh2CorrelationPoles, compute_mesh2_correlation, compute_fkp2_shotnoise, compute_smooth2_spectrum_window, MeshAttrs, get_smooth2_window_bin_attrs, interpolate_window_function, compute_mesh2_spectrum, split_particles, read)
    mattrs = MeshAttrs(**{name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    los = spectrum.attrs['los']
    pole = next(iter(spectrum))
    ells, norm, edges = spectrum.ells, pole.values('norm')[0], pole.edges('k')
    bin = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)
    step = bin.edges[-1, 1] - bin.edges[-1, 0]
    edgesin = np.arange(0., 1.2 * bin.edges.max(), step)
    edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])
    ellsin = [0, 2, 4]

    prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_data=tuple(), add_randoms=tuple(), **kwargs)
    zeff = compute_fkp_effective_redshift(randoms, order=2)
    randoms = spectrum.attrs['wsum_data1'] / randoms.sum() * randoms

    kind = 'smooth'
    #kind = 'infinite'
    kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)

    if kind == 'smooth':
        correlations = []
        kw = get_smooth2_window_bin_attrs(ells, ellsin)
        compute_mesh2_correlation = jax.jit(compute_mesh2_correlation, static_argnames=['los'], donate_argnums=[0, 1])
        # Window computed in configuration space, summing Bessel over the Fourier-space mesh
        coords = jnp.logspace(-3, 5, 4 * 1024)
        for scale in [1, 4]:
            mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize) #, meshsize=800)
            meshes = []
            for _ in split_particles(randoms.clone(attrs=mattrs2, exchange=True, backend='jax'), None, seed=42):
                alpha = spectrum.attrs['wsum_data1'] / _.sum()
                meshes.append(alpha * _.paint(**kw_paint, out='real'))
            sbin = BinMesh2CorrelationPoles(mattrs2, edges=np.arange(0., mattrs2.boxsize.min() / 2., mattrs2.cellsize.min()), **kw, basis='bessel') #, kcut=(0., mattrs2.knyq.min()))
            #num_shotnoise = compute_fkp2_shotnoise(randoms, bin=sbin)
            correlation = compute_mesh2_correlation(*meshes, bin=sbin, los=los).clone(norm=[norm] * len(sbin.ells)) #, num_shotnoise=num_shotnoise)
            del meshes
            correlation_fn = output_fn.replace('window_mesh2_spectrum', f'window_correlation{scale:d}_bessel_mesh2_spectrum')
            if jax.process_index() == 0:
                logger.info(f'Writing to {correlation_fn}')
                correlation.write(correlation_fn)
            correlation = interpolate_window_function(correlation, coords=coords, order=3)
            correlations.append(correlation)
        limits = [0, 0.4 * mattrs.boxsize.min(), 2. * mattrs.boxsize.max()]
        weights = [jnp.maximum((coords >= limits[i]) & (coords < limits[i + 1]), 1e-10) for i in range(len(limits) - 1)]
        correlation = correlations[0].sum(correlations, weights=weights)
        flags = ('fftlog',)
        if output_fn is not None and jax.process_index() == 0:
            correlation_fn = output_fn.replace('window_mesh2_spectrum', 'window_correlation_bessel_mesh2_spectrum')
            logger.info(f'Writing to {correlation_fn}')
            correlation.write(correlation_fn)
        window = compute_smooth2_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=flags)
    else:
        kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
        meshes = []
        for _ in split_particles(randoms, None, seed=42):
            alpha = spectrum.attrs['wsum_data1'] / _.sum()
            meshes.append(alpha * _.paint(**kw_paint, out='real'))
        window = compute_mesh2_spectrum_window(*meshes, edgesin=edgesin, ellsin=ellsin, los=los, bin=bin, pbar=True, flags=('infinite',), norm=norm)
        #window = compute_mesh2_spectrum_window(mesh, edgesin=edgesin[:3], ellsin=ellsin, los=los, bin=bin, pbar=True, flags=[], norm=norm)
    window = window.clone(observable=window.observable.map(lambda pole: pole.clone(norm=norm * np.ones_like(pole.values('norm')))))
    for pole in window.theory: pole._meta['z'] = zeff
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        window.write(output_fn)
    return window