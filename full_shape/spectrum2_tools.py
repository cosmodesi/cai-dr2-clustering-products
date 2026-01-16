import logging

import numpy as np
import jax

from tools import default_mpicomm, _format_bitweights


logger = logging.getLogger('spectrum2')


@default_mpicomm
def prepare_jaxpower_particles(*get_data_randoms, mattrs=None, **kwargs):
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
        indweights, bitweights = data['INDWEIGHT'], None
        if 'BITWEIGHT' in data:
            bitweights = _format_bitweights(data['BITWEIGHT'])
            from cucount.jax import BitwiseWeight
            iip = BitwiseWeight(weights=bitweights, p_correction_nbits=False)(bitweights)
            bitweights = [indweights] + bitweights  # add individual weight (photometric, spectro systematics) without PIP
            indweights = indweights * iip  # multiply by IIP to correct fiber assignment at large scales
        data = ParticleField(data['POSITION'], indweights, attrs=mattrs, exchange=True, backend=backend, **kwargs)
        #ids = collective_arange(len(randoms['POSITION']))
        ids = randoms['TARGETID']
        randoms = ParticleField(randoms['POSITION'], randoms['INDWEIGHT'], attrs=mattrs, exchange=True, backend=backend, **kwargs)
        if backend == 'jax':  # first convert to JAX Array
            sharding_mesh = mattrs.sharding_mesh
            ids = make_array_from_process_local_data(ids, pad=-1, sharding_mesh=sharding_mesh)
            if bitweights is not None:
                bitweights = [make_array_from_process_local_data(bitweight, pad=0, sharding_mesh=sharding_mesh) for bitweight in bitweights]
        randoms.__dict__['ids'] = randoms.exchange_direct(ids, pad=-1)  # index in the input catalog; for random split in bispectrum normalization
        if bitweights is not None:
            data.__dict__['BITWEIGHT'] = [data.exchange_direct(bitweight, pad=0) for bitweight in bitweights]
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
                           ells=(0, 2, 4), edges=None, los='firstpoint', cache=None):

    from jaxpower import (FKPField, compute_fkp2_normalization, compute_fkp2_shotnoise, BinMesh2SpectrumPoles, compute_mesh2_spectrum,
                          BinParticle2SpectrumPoles, BinParticle2CorrelationPoles, compute_particle2, compute_particle2_shotnoise)

    all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs)
    attrs = _get_jaxpower_attrs(*all_particles)
    mattrs = all_particles[0][0].attrs
    # Define the binner
    if cache is None: cache = {}
    bin = cache.get('bin_mesh2_spectrum', None)
    if edges is None: edges = {'step': 0.001}
    if bin is None: bin = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)
    cache.setdefault('bin_mesh2_spectrum', bin)

    # Computing normalization
    all_fkp = [FKPField(data, randoms) for (data, randoms, _) in all_particles]
    norm = compute_fkp2_normalization(*all_fkp, bin=bin, cellsize=10)

    # Computing shot noise
    all_fkp = [FKPField(data, shifted if shifted is not None else randoms) for (data, randoms, shifted) in all_particles]
    del all_particles
    num_shotnoise = compute_fkp2_shotnoise(*all_fkp, bin=bin)

    jax.block_until_ready((norm, num_shotnoise))
    if jax.process_index() == 0:
        logger.info('Normalization and shotnoise computation finished')

    results = {}
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

    with_bitweights = 'BITWEIGHT' in all_fkp[0].data.__dict__
    if auw is not None or with_bitweights:
        from cucount.jax import WeightAttrs
        from jaxpower.particle2 import convert_particles
        sattrs = {'theta': (0., 0.1)}
        bitwise = None
        if with_bitweights:
            # Order of weights matters
            # fkp.data.__dict__['BITWEIGHT'] includes IIP in the first position 
            all_data = [convert_particles(fkp.data, weights=list(fkp.data.__dict__['BITWEIGHT']) + [fkp.data.weights], exchange_weights=False) for fkp in all_fkp]
            bitwise = dict(weights=all_data[0].get('bitwise_weight'))  # sets nrealizations, etc.: fine to use the first
            if jax.process_index() == 0:
                logger.info(f'Applying PIP weights {bitwise}.')
        else:
            all_data = [convert_particles(fkp.data, weights=[fkp.data.weights] * 2, exchange_weights=False, index_value=dict(individual_weight=1, negative_weight=1)) for fkp in all_fkp]
            if jax.process_index() == 0:
                logger.info(f'Applying AUW weights.')
        wattrs = WeightAttrs(bitwise=bitwise,
                             angular=dict(sep=auw.get('DD').coords('theta'), weight=auw.get('DD').value()) if auw is not None else None)
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

    jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'])
    #jitted_compute_mesh2_spectrum = compute_mesh2_spectrum
    spectrum = jitted_compute_mesh2_spectrum(*meshes, bin=bin, los=los)
    spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise, attrs=attrs)
    jax.block_until_ready(spectrum)
    if jax.process_index() == 0:
        logger.info('Mesh-based computation finished')

    for name, value in results.items():
        results[name] = spectrum.clone(value=spectrum.value() + value)
    results['raw'] = spectrum

    if len(results) == 1:
        return next(iter(results.values()))
    return results