"""
salloc -N 1 -C "gpu&hbm80g" -t 02:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
srun -n 4 python test.py
"""
import os
import sys
import functools
from pathlib import Path

import jax
import numpy as np
import lsstypes as types

sys.path.insert(0, '../')
import tools
from tools import setup_logging
from compute_fiducial_stats import compute_fiducial_stats_from_options


def test_bitwise(stats=['mesh2_spectrum']):
    meas_dir = Path(Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks')
    for tracer in ['LRG']:
        zranges = tools.propose_fiducial('zranges', tracer)
        for region in ['NGC', 'SGC']:
            #catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zranges, region=region, imock=451)
            catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight_type='default_FKP_bitwise', nran=1)
            compute_fiducial_stats_from_options(stats, catalog=catalog_options, get_measurement_fn=functools.partial(tools.get_measurement_fn, meas_dir=meas_dir), mesh2_spectrum={'cut': True, 'auw': True}, particle2_correlation={'auw': True})


def test_norm():
    meas_dir = Path(Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks')
    stat = 'mesh3_spectrum'
    for tracer in ['BGS_BRIGHT-21.5']:
        zrange = tools.propose_fiducial('zranges', tracer)[0]
        for region in ['NGC']:
            catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zrange, region=region, weight_type='default_FKP', nran=2)
            compute_fiducial_stats_from_options(stat, catalog=catalog_options, get_measurement_fn=functools.partial(tools.get_measurement_fn, meas_dir=meas_dir))
            fn = tools.get_measurement_fn(kind=stat, meas_dir=meas_dir, **catalog_options)
            if jax.process_index() == 0:
                spectrum = types.read(fn)
                print(spectrum.get((0, 0, 0)).values('norm').mean())
                assert np.allclose(spectrum.get((0, 0, 0)).values('norm').mean(), 1.28543918)


if __name__ == '__main__':

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    jax.distributed.initialize()
    setup_logging()
    #test_bitwise(stats=['mesh2_spectrum'])
    test_norm()
    jax.distributed.shutdown()