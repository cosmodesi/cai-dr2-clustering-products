import logging

import numpy as np
import jax

from tools import _format_bitweights
import lsstypes as types


logger = logging.getLogger('correlation2')


def compute_angular_upweights(*get_data):
    from cucount.jax import Particles, BinAttrs, WeightAttrs, count2, setup_logging
    from lsstypes import ObservableLeaf, ObservableTree

    all_fibered_data, all_parent_data = [], []

    def get_rdw(catalog):
        positions = (catalog['RA'], catalog['DEC'])
        weights = [catalog['INDWEIGHT']] + _format_bitweights(catalog['BITWEIGHT'] if 'BITWEIGHT' in catalog else None)
        return positions, weights

    for _get_data in get_data:
        fibered_data, parent_data = _get_data()
        fibered_data = Particles(*get_rdw(fibered_data), positions_type='rd', exchange=True)
        parent_data = Particles(*get_rdw(parent_data), positions_type='rd', exchange=True)
        all_fibered_data.append(fibered_data)
        all_parent_data.append(parent_data)

    theta = 10**np.arange(-5, -1 + 0.1, 0.1)  # TODO: update
    battrs = BinAttrs(theta=theta)
    bitwise = None
    if all_fibered_data[0].get('bitwise_weight'):
        bitwise = dict(weights=all_fibered_data[0].get('bitwise_weight'))
        if jax.process_index() == 0:
            logger.info(f'Applying PIP weights {bitwise}.')
    wattrs = WeightAttrs(bitwise=bitwise)

    def get_counts(*particles):
        #setup_logging('error')
        autocorr = len(particles) == 1
        weight = count2(*(particles * 2 if autocorr else particles), battrs=battrs, wattrs=wattrs)['weight']
        if autocorr:
            norm = wattrs(particles[0]).sum()**2 - wattrs(*(particles * 2)).sum()
        else:
            norm = wattrs(particles[0]).sum() * wattrs(particles[1]).sum()
        # No need to remove auto-pairs, as edges[0] > 0
        return weight / norm
        #return Count2(counts=weight, norm=norm, theta=battrs.coords('theta'), theta_edges=battrs.edges('theta'), coords=['theta'])

    DDfibered = get_counts(*all_fibered_data)
    wattrs = WeightAttrs()
    DDparent = get_counts(*all_parent_data)

    kw = dict(theta=battrs.coords('theta'), theta_edges=battrs.edges('theta'), coords=['theta'])
    auw = {}
    auw['DD'] = ObservableLeaf(value=np.where(DDfibered == 0., 1., DDparent / DDfibered), **kw)
    #auw['DR'] = ObservableLeaf(value=np.where(DRfibered == 0., 1., DRparent / DRfibered), **kw)
    auw = ObservableTree(list(auw.values()), pairs=list(auw.keys()))
    return auw


def compute_particle2_correlation(*get_data_randoms, auw=None, cut=None, battrs=None):
    from cucount.jax import Particles, BinAttrs, WeightAttrs, SelectionAttrs, MeshAttrs, count2, setup_logging
    from lsstypes import Count2, Count2Correlation

    all_data, all_randoms, all_shifted = [], [], []

    def get_pw(catalog):
        positions = catalog['POSITION']
        weights = [catalog['INDWEIGHT']] + _format_bitweights(catalog['BITWEIGHT'] if 'BITWEIGHT' in catalog else None)
        return positions, weights

    def get_all_particles(catalog):
        if not isinstance(catalog, (tuple, list)):
            catalog = [catalog]  # list of randoms
        return [Particles(*get_pw(catalog), exchange=True) for catalog in catalog]

    for _get_data_randoms in get_data_randoms:
        # data, randoms (optionally shifted) are tuples (positions, weights)
        data, randoms, *shifted = _get_data_randoms()
        data = Particles(*get_pw(data), exchange=True)
        randoms = get_all_particles(randoms)
        if shifted:
            shifted = get_all_particles(shifted[0])
        else:
            shifted = [None] * len(randoms)
        all_data.append(data)
        all_randoms.append(randoms)
        all_shifted.append(shifted)
    if jax.process_index() == 0:
        logger.info(f'All particles on the GPU.')

    if battrs is None:
        battrs = dict(s=np.linspace(0., 180., 181), mu=(np.linspace(-1., 1., 201), 'midpoint'))

    battrs = BinAttrs(**battrs)
    sattrs = None
    if cut is not None:
        sattrs = SelectionAttrs(theta=(0., 0.05))
        if jax.process_index() == 0:
            logger.info(f'Applying theta-cut {sattrs}.')
    bitwise = angular = None
    if data.get('bitwise_weight'):
        bitwise = dict(weights=data.get('bitwise_weight'))
        if jax.process_index() == 0:
            logger.info(f'Applying PIP weights {bitwise}.')
    if auw is not None:
        angular = dict(sep=auw.get('DD').coords('theta'), weight=auw.get('DD').value())
        if jax.process_index() == 0:
            logger.info(f'Applying AUW {angular}.')
    wattrs = WeightAttrs(bitwise=bitwise, angular=angular)
    mattrs = None  # automatic setting for mesh

    # Helper to convert to lsstypes Count2
    def to_lsstypes(battrs: BinAttrs, counts: np.ndarray, norm: np.ndarray, attrs: dict) -> Count2:
        coords = battrs.coords()
        edges = battrs.edges()
        edges = {f'{k}_edges': v for k, v in edges.items()}
        return Count2(counts=counts, norm=norm * np.ones_like(counts), **coords, **edges, coords=list(coords), attrs=attrs)

    # Hepler to get counts as Count2
    def get_counts(*particles: Particles, wattrs: WeightAttrs=None) -> Count2:
        if wattrs is None: wattrs = WeightAttrs()
        autocorr = len(particles) == 1
        counts = count2(*(particles * 2 if autocorr else particles), battrs=battrs, wattrs=wattrs, mattrs=mattrs, sattrs=sattrs)['weight']
        attrs = {'size1': particles[0].size, 'wsum1': wattrs(particles[0]).sum()}
        if autocorr:
            auto_sum = wattrs(*(particles * 2)).sum()
            norm = wattrs(particles[0]).sum()**2 - auto_sum
            # Correct auto-pairs
            zero_index = tuple(np.flatnonzero((0 >= edges[:, 0]) & (0 < edges[:, 1])) for edges in battrs.edges().values())
            counts = counts.at[zero_index].add(-auto_sum)
        else:
            norm = wattrs(particles[0]).sum() * wattrs(particles[1]).sum()
            attrs.update({'size2': particles[1].size, 'wsum2': wattrs(particles[1]).sum()})
        return to_lsstypes(battrs, counts, norm, attrs=attrs)

    DD = get_counts(*all_data, wattrs=wattrs)
    data = data.clone(weights=wattrs(data))  # clone data, with IIP weights (in case we provided bitwise weights)

    DS, SD, SS, RR = [], [], [], []
    iran = 0
    for all_randoms, all_shifted in zip(zip(*all_randoms, strict=True), zip(*all_shifted, strict=True), strict=True):
        if jax.process_index() == 0:
            logger.info(f'Processing random {iran:d}.')
        iran += 1
        RR.append(get_counts(*all_randoms))
        if all(shifted is not None for shifted in all_shifted):
            SS.append(get_counts(*all_shifted))
        else:
            all_shifted = all_randoms
            SS.append(RR[-1])
        DS.append(get_counts(all_data[0], all_shifted[-1]))
        SD.append(get_counts(all_shifted[0], all_data[-1]))

    DS, SD, SS, RR = (types.sum(XX) for XX in [DS, SD, SS, RR])
    correlation = Count2Correlation(estimator='landyszalay', DD=DD, DS=DS, SD=SD, SS=SS, RR=RR)

    return correlation