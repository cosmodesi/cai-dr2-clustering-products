

import os
import sys
import glob
import itertools
import argparse
import logging
import numpy as np
from mpi4py import MPI
mpicomm = MPI.COMM_WORLD
rank = mpicomm.Get_rank()

# from mockfactory import Catalog
from desilike.samplers.emcee import EmceeSampler
from desilike import setup_logging
setup_logging()  # for logging messages

sys.path.append('/')
from fitting_tools import load_bins, get_observable_likelihood
logger = logging.getLogger('fit_blinded_data') 

RESULT_DIR = f'/global/cfs/projectdirs/desi/mocks/cai/mock-challenge-cutsky-dr2/blinded_data/dr2-v2/data_splits/results'

########################################################################################################################################################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--tracers', nargs='+', type=str, default=['LRG'], choices=['BGS','LRG','ELG','QSO'], help='Tracers')
    # parser.add_argument('--zrange', nargs='+', type=str, default=None, help='Redshift bins')
    parser.add_argument('--indx', type=int, default=6, help='index for redshift bin')    
    parser.add_argument('--version', type=str,  default='dr2-v2', choices=['test', 'dr2-v2'], help='Catalog versions to use.')
    parser.add_argument('--regions', nargs='+', type=str, default=['GCcomb'], help='Sky regions to include.')
    parser.add_argument('--weight_types', nargs='+', type=str, default=['default_fkp'],
                        choices=['default', 'default_fkp', 'default_thetacut', 'default_auw', 'bitwise', 'bitwise_auw'], help='Weighting schemes to use.')
    parser.add_argument("--theory", type=str, default="vel", help="Galaxy clustering theory", choices=['folps', 'vel', 'folpsrc', 'tns'])
    parser.add_argument("--cosmology", type=str, default="LCDM", help="Cosmology to fit", choices=['LCDM', 'nuCDM', 'nsCDM', 'w0waCDM'])
    parser.add_argument("--approaches", nargs='+', type=str, default=["FM"], help="Fitting approach", choices=['SF','FM','BAO'])
    parser.add_argument("--option", type=str, default='', help="option", choices=['','_wq_prior'])
    args = parser.parse_args()
    use_emulator = True
    setup_logging()
    if rank == 0:
        logger.info(f"Received arguments: {args}")
    version = args.version
    option = args.option
    default_bins = [
    ('BGS', (0.1, 0.4)),
    ('LRG', (0.4, 0.6)),
    ('LRG', (0.6, 0.8)),
    ('LRG', (0.8, 1.1)),
    ('ELG_LOPnotqso', (0.8, 1.1)),
    ('ELG_LOPnotqso', (1.1, 1.6)),
    ('QSO', (0.8, 2.1))
    ]
    tracers_bins = [default_bins[args.indx]]
    for (tracer, zrange), region, weight_type, approach in itertools.product(tracers_bins, args.regions, args.weight_types, args.approaches):
        if option == '_wq_prior' and approach != 'SF':
            ValueError(f"{option} option only set for ShapeFit (SF)")
        if 'BGS' in tracer: tracer = 'BGS_BRIGHT-21.35'
        if 'ELG' in tracer: tracer = 'ELG_LOPnotqso'
        if weight_type.endswith('_thetacut'): cut = '_thetacut'
        else: cut = ''
        if weight_type.endswith('_auw'): auw = '_auw'
        else: auw = ''
        task = f'{approach}fit_{args.cosmology}_{args.theory}'
        data_args = {'tracer':tracer, 'zrange':zrange, 'region':region, 'weight_type':weight_type, 'version':version}
        chain_fn    = RESULT_DIR+f'/full-shape/mcmc/chain_{task}_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{region}_{weight_type}{auw}{cut}{option}.npy'
        if approach in ['FM', 'SF']:
            corr_type = 'pk'
            bins_type = 'y3_blinding'
            (kmin, kmax, kbin, lenk) = load_bins(corr_type, bins_type)
            if use_emulator:
                emulator_fn = RESULT_DIR+f'/emulator/emulator_{task}_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{region}_{weight_type}{auw}{cut}_k{kmin}-{kmax}{option}.npy'
                if not os.path.exists(emulator_fn):
                    from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine
                    fit_args = {"corr_type":corr_type, "bins_type":bins_type, "option":option}
                    (likelihood, _, theory) = get_observable_likelihood(task, data_args, fit_args)
                    emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(order=3, method='finite')) # Taylor expansion, up to a given order
                    emulator.set_samples() # evaluate the theory derivatives (with jax auto-differentiation if possible, else finite differentiation)
                    emulator.fit()
                    emulator.save(emulator_fn)
                fit_args = {"corr_type":corr_type, "bins_type":bins_type, "emulator_fn": emulator_fn, "option":option}
            else: 
                fit_args = {"corr_type":corr_type, "bins_type":bins_type, "option":option}
            nwalkers= 64; interations = 30001 # save every 300 iterations
        mpicomm.barrier()
        if not os.path.exists(chain_fn):
            logger.info(f"Sampling with data {data_args}")
            logger.info(f"Fitting arguments {fit_args}")
            (likelihood, _, _) = get_observable_likelihood(task, data_args, fit_args)
            # MCMC sampling
            sampler = EmceeSampler(likelihood, seed=42, nwalkers=nwalkers, save_fn = chain_fn)
            sampler.run(check={'max_eigen_gr': 0.03, 'min_ess': 300}, max_iterations = interations) # save every 300 iterations
            logger.info(f'Save to {chain_fn}')
