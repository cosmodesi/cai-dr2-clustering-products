import logging

import jax

from tools import compute_fkp_effective_redshift
from spectrum2_tools import prepare_jaxpower_particles


logger = logging.getLogger('reconstruction')


def compute_reconstruction(get_data_randoms, mattrs=None, mode='recsym', bias=2.0, smoothing_radius=15.):
    from jaxpower import FKPField
    from jaxrecon.zeldovich import IterativeFFTReconstruction, estimate_particle_delta

    particles = prepare_jaxpower_particles(get_data_randoms, mattrs=mattrs, return_inverse=True)[0]

    # Define FKP field = data - randoms
    fkp = FKPField(*particles)
    delta = estimate_particle_delta(fkp, smoothing_radius=smoothing_radius)
    # Line-of-sight "los" can be local (None, default) or an axis, 'x', 'y', 'z', or a 3-vector
    # In case of IterativeFFTParticleReconstruction, and multi-GPU computation, provide the size of halo regions in cell units. E.g., maximum displacement is ~ 40 Mpc/h => 4 * chosen cell size => provide halo_add=2
    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    growth_rate = compute_fkp_effective_redshift(fkp.data, order=1, cellsize=None, func_of_z=cosmo.growth_rate)
    recon = jax.jit(IterativeFFTReconstruction, static_argnames=['los', 'halo_add', 'niterations'], donate_argnums=[0])(delta, growth_rate=growth_rate, bias=bias, los=None, halo_add=0)
    data_positions_rec = recon.read_shifted_positions(fkp.data.positions)
    assert mode in ['recsym', 'reciso']
    # RecSym = remove large scale RSD from randoms
    kwargs = {}
    if mode == 'recsym': kwargs['field'] = 'disp'
    randoms_positions_rec = recon.read_shifted_positions(fkp.randoms.positions, **kwargs)
    if jax.process_index() == 0:
        logger.info('Reconstruction finished.')

    data_positions_rec = fkp.data.exchange_inverse(data_positions_rec)
    randoms_positions_rec = fkp.randoms.exchange_inverse(randoms_positions_rec)
    if jax.process_index() == 0:
        logger.info('Exchange finished.')

    return data_positions_rec, randoms_positions_rec