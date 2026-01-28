import logging

import numpy as np
import jax
from jax import numpy as jnp
import lsstypes as types

from .tools import compute_fkp_effective_redshift
from .spectrum2_tools import prepare_jaxpower_particles, _get_jaxpower_attrs


logger = logging.getLogger('spectrum3')


def compute_mesh3_spectrum(*get_data_randoms, mattrs=None,
                            basis='sugiyama-diagonal', ells=[(0, 0, 0), (2, 0, 2)], edges=None, los='local',
                            buffer_size=0, cache=None):
    r"""
    Compute the 3-point spectrum multipoles using mesh-based FKP fields with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return tuples of (data, randoms, [shifted]) catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects. If None, default attributes are used.
        See :func:`prepare_jaxpower_particles` for details.
    basis : str, optional
        Basis for the 3-point spectrum computation. Default is 'sugiyama-diagonal'.
    ells : list of tuples, optional
        List of multipole moments to compute. Default is [(0, 0, 0), (2, 0, 2)] (for the sugiyama basis).
    edges : dict, optional
        Edges for the binning; array or dictionary with keys 'start' (minimum :math:`k`), 'stop' (maximum :math:`k`), 'step' (:math:`\Delta k`).
        If ``None``, default step of :math:`0.005 h/\mathrm{Mpc}` is used for the sugiyama basis, :math:`0.01 h/\mathrm{Mpc}` for the scoccimarro basis.
        See :class:`jaxpower.BinMesh3SpectrumPoles` for details.
    los : {'local', 'x', 'y', 'z', array-like}, optional
        Line-of-sight definition. 'local' uses local LOS, 'x', 'y', 'z' use fixed axes, or provide a 3-vector.
    buffer_size : int, optional
        Buffer size when binning; if the binning is multidimensional, increase for faster computation at the cost of memory.
    cache : dict, optional
        Cache to store binning class (can be reused if ``meshsize`` and ``boxsize`` are the same).
        If ``None``, a new cache is created.

    Returns
    -------
    spectrum : Mesh3SpectrumPoles
        The computed 3-point spectrum multipoles.
    """
    from jaxpower import (FKPField, compute_fkp3_normalization, compute_fkp3_shotnoise, BinMesh3SpectrumPoles, compute_mesh3_spectrum)

    all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'])
    attrs = _get_jaxpower_attrs(*all_particles)
    attrs.update(los=los)
    mattrs = all_particles[0][0].attrs
    # Define the binner
    if cache is None: cache = {}
    bin = cache.get(f'bin_mesh3_spectrum_{basis}', None)
    if edges is None: edges = {'step': 0.02 if 'scoccimarro' in basis else 0.005}
    if bin is None or not np.all(bin.mattrs.meshsize == mattrs.meshsize) or not np.allclose(bin.mattrs.boxsize, mattrs.boxsize):
        bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis=basis, ells=ells, buffer_size=buffer_size)
    cache.setdefault(f'bin_mesh3_spectrum_{basis}', bin)

    # Computing normalization
    all_fkp = [FKPField(data, randoms) for (data, randoms, _) in all_particles]
    norm = compute_fkp3_normalization(*all_fkp, bin=bin, split=[(42, fkp.randoms.__dict__['IDS']) for fkp in all_fkp],  # index for process invariance
                                      cellsize=10)

    # Computing shot noise
    all_fkp = [FKPField(data, shifted if shifted is not None else randoms) for (data, randoms, shifted) in all_particles]
    del all_particles
    kw = dict(resampler='tsc', interlacing=3, compensate=True)
    num_shotnoise = compute_fkp3_shotnoise(*all_fkp, los=los, bin=bin, **kw)

    jax.block_until_ready((norm, num_shotnoise))
    if jax.process_index() == 0:
        logger.info('Normalization and shotnoise computation finished')

    # out='real' to save memory
    meshes = [fkp.paint(**kw, out='real') for fkp in all_fkp]
    del all_fkp

    jitted_compute_mesh3_spectrum = jax.jit(compute_mesh3_spectrum, static_argnames=['los'])

    # out='real' to save memory
    spectrum = jitted_compute_mesh3_spectrum(*meshes, los=los, bin=bin)
    spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
    spectrum = spectrum.map(lambda pole: pole.clone(attrs=attrs))
    spectrum = spectrum.clone(attrs=attrs)

    jax.block_until_ready(spectrum)
    if jax.process_index() == 0:
        logger.info('Mesh-based computation finished')

    return spectrum



def compute_window_mesh3_spectrum(*get_data_randoms, spectrum, ibatch: tuple=None, computed_batches: list=None, buffer_size=10):
    r"""
    Compute the 3-point spectrum window with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return tuples of (data, randoms) catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    spectrum : Mesh3SpectrumPoles
        Measured 3-point spectrum multipoles.
    ibatch : tuple, optional
        To split the window function multipoles to compute in batches, provide (0, nbatches) for the first batch,
        (1, nbatches) for the second, etc; up to (nbatches - 1, nbatches).
        ``None`` to compute the final window matrix.
    computed_batches : list, optional
        The window function multipoles that have been computed thus far.

    Returns
    -------
    spectrum : WindowMatrix or dict of WindowMatrix
        The computed 3-point spectrum window.
    """
    # FIXME: data is not used, could be dropped, add auw
    from jaxpower import (BinMesh3SpectrumPoles, BinMesh3CorrelationPoles, compute_mesh3_correlation,
                           compute_smooth3_spectrum_window, get_smooth3_window_bin_attrs, interpolate_window_function, split_particles)
    ells = spectrum.ells
    mattrs = {name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    los = spectrum.attrs['los']
    kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)

    all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'])
    all_randoms = [particles[1] for particles in all_particles]
    del all_particles
    mattrs = all_randoms[0].attrs

    pole = next(iter(spectrum))
    ells, edges, basis = spectrum.ells, pole.edges('k'), pole.basis
    norm = jnp.concatenate([spectrum.get(ell).values('norm') for ell in spectrum.ells])
    k, index = np.unique(pole.coords('k', center='mid_if_edges')[..., 0], return_index=True)
    edges = edges[index, 0]
    edges = np.insert(edges[:, 1], 0, edges[0, 0])
    bin = BinMesh3SpectrumPoles(mattrs, edges=edges, ells=ells, basis=basis, mask_edges='')   # mask_edges useless if cellsize is large enough
    stop = bin.edges1d[0].max()
    step = np.diff(bin.edges1d[0], axis=-1).min()
    edgesin = np.arange(0., 1.5 * stop, step / 2.)
    edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])

    fields = list(range(len(all_randoms)))
    fields += [fields[-1]] * (3 - len(all_randoms))
    seed = [(42, randoms.__dict__['IDS']) for randoms in all_randoms]
    zeff = compute_fkp_effective_redshift(*all_randoms, order=3, split=seed)

    correlations = []
    kw, ellsin = get_smooth3_window_bin_attrs(ells, ellsin=2, fields=fields, return_ellsin=True)
    jitted_compute_mesh3_correlation = jax.jit(compute_mesh3_correlation, static_argnames=['los'], donate_argnums=[0])

    coords = jnp.logspace(-3, 5, 1024)
    scales = [1, 4]
    b, c = mattrs.boxsize.min(), mattrs.cellsize.min()
    edges = [np.concatenate([np.arange(11) * c, np.arange(11 * c, 0.3 * b, 4 * c)]),
            np.concatenate([np.arange(11) * scales[1] * c, np.arange(11 * scales[1] * c, 2 * b, 4 * scales[1] * c)])]

    ells = kw['ells']
    if ibatch is not None:
        start, stop = ibatch[0] * len(ells) // ibatch[1], (ibatch[0] + 1) * len(ells) // ibatch[1]
        kw['ells'] = ells[start:stop]
    if ells and not bool(computed_batches):
        # multigrid calculation
        for scale, edges in zip(scales, edges):
            if jax.process_index() == 0:
                logger.info(f'Processing scale x{scale:.0f}')
            mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize)
            kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
            sbin = BinMesh3CorrelationPoles(mattrs2, edges=edges, **kw, buffer_size=buffer_size)  # kcut=(0., mattrs2.knyq.min()))
            meshes = []
            for iran, randoms in enumerate(split_particles(all_randoms + [None] * (3 - len(all_randoms)),
                                                           seed=seed, fields=fields)):
                    randoms = randoms.exchange()
                    alpha = pole.attrs[f'wsum_data{min(iran, len(all_randoms) - 1):d}'] / randoms.weights.sum()
                    meshes.append(alpha * randoms.paint(**kw_paint, out='real'))
            correlation = jitted_compute_mesh3_correlation(meshes, bin=sbin, los=los).clone(norm=[np.mean(norm)] * len(sbin.ells))
            jax.block_until_ready(correlation)
            correlation = interpolate_window_function(correlation.unravel(), coords=coords, order=3)
            correlations.append(correlation)

        coords = list(next(iter(correlations[0])).coords().values())
        limit = 0.25 * mattrs.boxsize.min()
        mask = (coords[0] < limit)[:, None] * (coords[1] < limit)[None, :]
        weights = [jnp.maximum(mask, 1e-6), jnp.maximum(~mask, 1e-6)]
        correlation = correlations[0].sum(correlations, weights=weights)

    if computed_batches:
        correlation = types.join(computed_batches)
        correlation = types.join([correlation.get(ells=[ell]) for ell in ells])  # reorder

    jax.block_until_ready(correlation)
    if jax.process_index() == 0:
        logger.info('Window functions computed.')

    results = {}
    results['window_mesh3_correlation_raw'] = correlation
    if ibatch is None:
        if jax.process_index() == 0:
            logger.info('Building window matrix.')
        window = compute_smooth3_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=('fftlog',), batch_size=4)
        observable = window.observable.map(lambda pole, label: pole.clone(norm=spectrum.get(**label).values('norm'), attrs=pole.attrs | dict(zeff=zeff)), input_label=True)
        window = window.clone(observable=observable, value=window.value() / (norm[..., None] / np.mean(norm)))  # just in case norm is k-dependent
        results['raw'] = window
    return results


def compute_box_mesh3_spectrum(get_data, get_shifted=None, basis='scoccimarro', ells=[0, 2], los='z', mask_edges=None, cache=None, **attrs):
    """
    Compute the 3-point spectrum multipoles for a cubic box using :mod:`jaxpower`.

    Parameters
    ----------
    get_data : callable
        Function that returns a tuple of (positions, weights) for the data catalog.
    get_shifted : callable, optional
        Function that returns a tuple of (positions, weights) for shifted randoms.
        If None, no shifted randoms are used.
    basis : str, optional
        Basis for the 3-point spectrum computation. Default is 'scoccimarro'.
        Can also be 'sugiyama-diagonal' or other supported bases.
    ells : list, optional
        Multipole moments to compute. Default is [0, 2].
        For scoccimarro basis, use integer ells. For sugiyama basis, use tuples like [(0, 0, 0), (2, 0, 2)].
    los : {'x', 'y', 'z'}, optional
        Line-of-sight direction. Default is 'z'.
    mask_edges : str or None, optional
        Edge masking specification. Default is None.
    cache : dict, optional
        Cache to store binning class (can be reused if ``meshsize`` and ``boxsize`` are the same).
        If ``None``, a new cache is created.
    **attrs : dict
        Mesh attributes (boxsize, cellsize, etc.) to pass to :func:`jaxpower.get_mesh_attrs`.

    Returns
    -------
    spectrum : Mesh3SpectrumPoles
        The computed 3-point spectrum multipoles.
    """
    import jax
    from jaxpower import (ParticleField, FKPField, compute_box3_normalization, compute_fkp3_shotnoise, BinMesh3SpectrumPoles, get_mesh_attrs, compute_mesh3_spectrum)
    mattrs = get_mesh_attrs(boxcenter=0., **attrs)
    data = ParticleField(*get_data(), attrs=mattrs, exchange=True, backend='jax')
    edges = {'step': 0.01 if 'scoccimarro' in basis else 0.005}
    if cache is None: cache = {}
    bin = cache.get(f'bin_mesh3_spectrum_{basis}', None)
    if bin is None: bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis=basis, ells=ells, buffer_size=16, mask_edges=mask_edges)
    cache.setdefault(f'bin_mesh3_spectrum_{basis}', bin)
    norm = compute_box3_normalization(data, bin=bin)
    if get_shifted is not None:
        data = FKPField(data, ParticleField(*get_shifted(), attrs=mattrs, exchange=True, backend='jax'))
    kw = dict(resampler='tsc', interlacing=3, compensate=True)
    num_shotnoise = compute_fkp3_shotnoise(data, los=los, bin=bin, **kw)
    jax.block_until_ready((norm, num_shotnoise))
    if jax.process_index() == 0:
        logger.info('Normalization and shotnoise computation finished')
    mesh = data.paint(**kw, out='real')
    mesh = mesh - mesh.mean()
    spectrum = compute_mesh3_spectrum(mesh, los=los, bin=bin)
    spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
    jax.block_until_ready(spectrum)
    if jax.process_index() == 0:
        logger.info('Mesh-based computation finished')
    return spectrum


def compute_box_mesh3_cross_spectrum(get_data, get_data2, basis='scoccimarro', ells=[0, 2], los='z', mask_edges=None, cache=None, **attrs):
    """
    Compute the 3-point cross-spectrum multipoles between two fields in a cubic box using :mod:`jaxpower`.

    Parameters
    ----------
    get_data : callable
        Function that returns a tuple of (positions, weights) for the first data catalog.
    get_data2 : callable
        Function that returns a tuple of (positions, weights) for the second data catalog.
    basis : str, optional
        Basis for the 3-point spectrum computation. Default is 'scoccimarro'.
        Can also be 'sugiyama-diagonal' or other supported bases.
    ells : list, optional
        Multipole moments to compute. Default is [0, 2].
        For scoccimarro basis, use integer ells. For sugiyama basis, use tuples like [(0, 0, 0), (2, 0, 2)].
    los : {'x', 'y', 'z'}, optional
        Line-of-sight direction. Default is 'z'.
    mask_edges : str or None, optional
        Edge masking specification. Default is None.
    cache : dict, optional
        Cache to store binning class (can be reused if ``meshsize`` and ``boxsize`` are the same).
        If ``None``, a new cache is created.
    **attrs : dict
        Mesh attributes (boxsize, cellsize, etc.) to pass to :func:`jaxpower.get_mesh_attrs`.

    Returns
    -------
    spectrum : Mesh3SpectrumPoles
        The computed 3-point cross-spectrum multipoles.
    """
    import jax
    from jaxpower import (ParticleField, compute_box3_normalization, BinMesh3SpectrumPoles, get_mesh_attrs, compute_mesh3_spectrum)
    mattrs = get_mesh_attrs(boxcenter=0., **attrs)
    data = ParticleField(*get_data(), attrs=mattrs, exchange=True, backend='jax')
    data2 = ParticleField(*get_data2(), attrs=mattrs, exchange=True, backend='jax')
    edges = {'step': 0.01 if 'scoccimarro' in basis else 0.005}
    if cache is None: cache = {}
    bin = cache.get(f'bin_mesh3_spectrum_{basis}', None)
    if bin is None: bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis=basis, ells=ells, buffer_size=16, mask_edges=mask_edges)
    cache.setdefault(f'bin_mesh3_spectrum_{basis}', bin)
    norm = compute_box3_normalization(data, data2, bin=bin)
    jax.block_until_ready(norm)
    if jax.process_index() == 0:
        logger.info('Normalization and shotnoise computation finished')
    kw = dict(resampler='tsc', interlacing=3, compensate=True)
    mesh = data.paint(**kw, out='real')
    mesh = mesh - mesh.mean()
    del data
    mesh2 = data2.paint(**kw, out='real')
    mesh2 = mesh2 - mesh2.mean()
    del data2
    spectrum = compute_mesh3_spectrum(mesh, mesh2, los=los, bin=bin)
    spectrum = spectrum.clone(norm=norm)
    jax.block_until_ready(spectrum)
    if jax.process_index() == 0:
        logger.info('Mesh-based computation finished')
    return spectrum
