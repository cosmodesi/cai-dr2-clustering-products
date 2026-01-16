"""
salloc -N 1 -C "gpu&hbm80g" -t 02:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
srun -n 4 python check_fiducial_setup.py
"""
import os
import functools
from pathlib import Path

import jax
import numpy as np

import tools
from tools import setup_logging
from compute_fiducial_stats import compute_fiducial_stats_from_options


def propose_nran_boxsize_from_catalogs():
    from jaxpower import get_mesh_attrs
    zranges = [('BGS_BRIGHT-21.35', (0.1, 0.4)), ('LRG', (0.4, 1.1)), ('ELG_LOPnotqso', (0.8, 1.6)), ('QSO', (0.8, 2.1))][::-1]
    for tracer, zrange in zranges:
        for region in ['NGC', 'SGC'][:1]:
            catalog_options = dict(version='data-dr2-v2', tracer=tracer, zrange=zrange, region=region)
            data_fn = tools.get_catalog_fn(kind='data', **catalog_options)
            data = tools.read_clustering_catalog(data_fn, kind='data', **catalog_options)
            _nran = 10
            randoms_fn = tools.get_catalog_fn(kind='randoms', nran=_nran, **catalog_options)
            randoms = tools.read_clustering_catalog(*randoms_fn, kind='randoms', **catalog_options, concatenate=True)
            alpha = len(data) / (len(randoms) / _nran)
            nran = 50. * alpha
            mattrs = get_mesh_attrs(data['POSITION'], randoms['POSITION'], boxpad=1.2, cellsize=7.5, primes=(2, 3, 5), divisors=(2,))
            print(f'For {tracer} in {zrange[0]:.1f}-{zrange[1]:.1f}, {region}')
            print(f'nran = {nran:.1f}')
            print(f'mattrs = {mattrs}')


def check_boxsize_spectrum(stats=['mesh2_spectrum']):
    meas_dir = Path(Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks')
    boxsizes = {'LRG': [5000., 6000., 7000., 8000., 9000., 10000.],
                'ELG_LOP': [6000., 7000., 8000., 9000., 10000.],
                'QSO': [7000., 8000., 9000., 10000.]}
    for tracer in boxsizes:
        zranges = tools.propose_fiducial('zranges', tracer)
        for region in ['NGC', 'SGC']:
            for boxsize in boxsizes[tracer]:
                catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zranges, region=region, imock=451)
                cellsize = 7.8
                #mattrs = dict(boxsize=boxsize, cellsize=cellsize)
                mattrs = dict(boxsize=boxsize, cellsize=cellsize)
                extra = f'boxsize{boxsize:.0f}_cellsize{cellsize:.1f}'
                compute_fiducial_stats_from_options(stats, catalog=catalog_options, mattrs=mattrs,
                                                    get_measurement_fn=functools.partial(tools.get_measurement_fn, meas_dir=meas_dir, extra=extra), mesh2_spectrum={'cut': True, 'auw': True})


def check_nran_spectrum(stats=['mesh2_spectrum']):
    meas_dir = Path(Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks')
    nrans = {'LRG': [8, 9, 11, 18],
             'ELG_LOP': [11, 13, 16, 18],
             'QSO': [8, 9, 11, 18]}
    for tracer in nrans:
        zranges = tools.propose_fiducial('zranges', tracer)
        for region in ['NGC', 'SGC']:
            for nran in nrans[tracer]:
                catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zranges, region=region, nran=nran, imock=451)
                extra = f'nran{nran:d}'
                compute_fiducial_stats_from_options(stats, catalog=catalog_options,
                                                    get_measurement_fn=functools.partial(tools.get_measurement_fn, meas_dir=meas_dir, extra=extra), mesh2_spectrum={'cut': True, 'auw': True})


if __name__ == '__main__':

    todo = ['propose']
    setup_logging()
    if 'propose' in todo:
        propose_nran_boxsize_from_catalogs()
    if 'check' in todo:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
        jax.distributed.initialize()
        #check_boxsize_spectrum(stats=['mesh3_spectrum'])
        check_nran_spectrum(stats=['mesh2_spectrum', 'mesh3_spectrum'])
        jax.distributed.shutdown()