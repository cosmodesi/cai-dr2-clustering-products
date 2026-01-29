import os
import sys
import glob
import logging
import numpy as np
from pathlib import Path
import lsstypes as types

from cosmoprimo.fiducial import DESI, AbacusSummit
from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate, DampedBAOWigglesTracerCorrelationFunctionMultipoles
from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate, StandardPowerSpectrumTemplate
from desilike.theories.galaxy_clustering import FOLPSTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerPowerSpectrumMultipoles, TNSTracerPowerSpectrumMultipoles, REPTVelocileptorsTracerPowerSpectrumMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, TracerCorrelationFunctionMultipolesObservable, BAOCompressionObservable
from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine
from desilike.likelihoods import ObservablesGaussianLikelihood

logger = logging.getLogger('fitting_tools') 

def load_bins(corr_type, bins_type = 'test'):
    if corr_type == 'xi':
        if bins_type in ['test']:
            rmin, rmax, rbin, lenr = 20, 200, 4, 45
        elif bins_type in ['y3_bao', 'y3_sys']:
            rmin, rmax, rbin, lenr = 60, 150, 4, 23
        else:
            raise ValueError(f"Unknown bins_type '{bins_type}' for correlation type 'xi'.")
        return (rmin, rmax, rbin, lenr)
    elif corr_type == 'pk':
        if bins_type in ['y3_bao', 'test']:
            kmin, kmax, kbin, lenk = 0.02, 0.3, 0.005, 56
        elif bins_type in ['y3_fs', 'y3_sys', 'y3_blinding']: 
            kmin, kmax, kbin, lenk = 0.02, 0.2, 0.005, 36
        elif bins_type in ['test_covbox']:
            kmin, kmax, kbin, lenk = 0.03, 0.2, 0.005, 34     
        else:
            raise ValueError(f"Unknown bins_type '{bins_type}' for correlation type 'pk'.")
        return (kmin, kmax, kbin, lenk)
    elif corr_type == 'bk':
        if bins_type in ['test']:
            kmin, kmax, kbin, lenk = 0, 0.2, 0.01, 20 #Sigiyama space
        else:
            raise ValueError(f"Unknown bins_type '{bins_type}' for correlation type 'bk'.")
        return (kmin, kmax, kbin, lenk)
    else:
        raise ValueError(f"Invalid corr_type '{corr_type}'. Expected one of ['xi', 'pk', 'mpslog', 'wp', 'bk'].")

def get_measurement_fn(kind='mesh2_spectrum_poles', version='dr2-v2', recon=None, tracer='LRG', region='NGC', zrange=(0.8, 1.1), cut=None, auw=None, nran = 18, weight_type='default', **kwargs):
    # base_dir = Path(f'/global/cfs/projectdirs/desi/mocks/cai/mock-challenge-cutsky-dr2/')
    # base_dir = base_dir / (f'blinded_{recon}' if recon else 'blinded')
    base_dir = Path(f'/global/cfs/projectdirs/desi/mocks/cai/mock-challenge-cutsky-dr2/blinded_data/{version}/data_splits')
    if cut: cut = '_thetacut'
    else: cut = ''
    if auw: auw = '_auw'
    else: auw = ''
    return str(base_dir / f'{kind}_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{region}_{weight_type}{auw}{cut}_nran{nran}.h5')

def get_effective_redshift(args):
    window = types.read(get_measurement_fn(**args, kind='window_mesh2_spectrum_poles'))
    mono = window.theory.get(ells=0)
    zeff = getattr(mono, "z", mono._meta.get("z", None))
    if zeff is None:
        raise AttributeError(f"No z_eff found in window function")
    return zeff

def load_blinded_data_pip(args, ells = (0,2), bins_type = 'y3_blinding', blinding=True, dir = None):
    if dir == None:
        DIR = '/pscratch/sd/s/shengyu/Y3/blinded/test'
    """
    Dispatch to the blinded data files, window function and covariance based on `data args`.
    Returns
    -------
    data, wmatrix, covariance
    """
    (kmin, kmax, kbin, lenk) = load_bins('pk', bins_type)
    prefix='blinded_' if blinding == True else ''
    data = types.read(get_measurement_fn(**args, kind=f'{prefix}mesh2_spectrum_poles'))
    window = types.read(get_measurement_fn(**args, kind=f'window_mesh2_spectrum_poles'))
    covariance = types.read(get_measurement_fn(**args, kind=f'covariance_mesh2_spectrum_poles'))
    ells = list(ells)
    sl = slice(0, None, 5)  # rebin to dk = 0.005 h/Mpc
    oklim = (kmin, kmax)  # fitted k-range, no need to go to higher k
    mean = data.select(k=sl).select(k=oklim).get(ells)
    wmatrix = window.at.observable.match(mean).at.theory.match(data.select(k=(0., 1.1 * oklim[1])).get(ells))
    covariance = covariance.at.observable.match(mean)
    return mean, wmatrix, covariance

def get_template(task, z_eff = 1.0, ells = (0,2), cosmo=DESI(), option=None, **data_args):
    '''
    task: {FM/SF}_{fit_cosmology}_{fiducial_cosmology}_{theory}
    return: initialized template
    '''
    # for param in ['h', 'omega_cdm', 'omega_b', 'logA', 'n_s', 'm_ncdm', 'Omega_m']:
    # print(f'\'{param}\': {cosmo[param]},')
    if 'FM' in task:
        template = DirectPowerSpectrumTemplate(z=z_eff, fiducial=cosmo)
        template.init.params['h'].update(prior={'dist': 'uniform', 'limits': [0.2, 1.0]})
        template.init.params['omega_cdm'].update(delta=0.01)
        template.init.params['logA'].update(delta=0.07, prior={'dist': 'uniform', 'limits': [1.61, 3.91]})
        # template.init.params['omega_b'].update(prior={'dist': 'norm', 'loc': 0.02218, 'scale': (3.025e-7)**0.5})
        template.init.params['omega_b'].update(prior={'dist': 'norm', 'loc': 0.02237, 'scale': 0.00055})
        template.init.params['n_s'].update(fixed=False, prior={'dist': 'norm', 'loc': 0.9649, 'scale': 0.042}, delta=0.01)
        template.init.params['sigma8'] = {'derived': True, 'latex': r'\sigma_8'}
        # template.init.params['Omega_m'] = {'derived': True, 'latex': r'\Omega_m'}
        # template.init.params['rs_drag'] = {'derived': True, 'latex': r'r_s'}
        if 'nuCDM' in task:
            # template.init.params['m_ncdm'].update(fixed=False)
            template.init.params['m_ncdm'].update(fixed=False, prior=dict(dist='uniform', limits=[0.0, 4.0]))
        if 'wCDM' in task:
            template.init.params['w_fld'].update(fixed=False)
        if 'w0waCDM' in task:
            template.init.params['w0_fld'].update(fixed=False)
            template.init.params['wa_fld'].update(fixed=False)
    elif 'SF' in task:
        template = ShapeFitPowerSpectrumTemplate(z=z_eff, fiducial=cosmo)
        template.init.update(apmode='qisoqap')
        template.init.params['qiso'].update(delta=0.02, prior={'limits': [0.8, 1.2]})
        template.init.params['qap'].update(delta=0.02, prior={'limits': [0.8, 1.2]})
        if option == '_wq_prior':
            logger.info('Set the scaling parameter to wide priors')
            template.init.params['qiso'].update(delta=0.02, prior={'limits': [0.2, 1.8]})
            template.init.params['qap'].update(delta=0.02, prior={'limits': [0.2, 1.8]})
        template.init.params['df'].update(delta=0.05)
        template.init.params['dm'].update(prior={'limits': [-0.8, 0.8]})
    elif 'BAO' in task:
        apmode = 'qisoqap'
        if 2 not in ells: apmode = 'qiso'
        template = BAOPowerSpectrumTemplate(z=z_eff, cosmo=cosmo, apmode=apmode)
    else:
        raise ValueError(f"Unknown template type (BAO, SF, FM): {task}")
    return template

def get_theory(task, template=None, ells = (0,2), emulator_fn=None, smoothing_radius=15):
    if 'FM' in task or 'SF' in task:
        theory_keys = ['folpsrc', 'folps', 'vel', 'tns'] 
        theory_map = {
            'vel': REPTVelocileptorsTracerPowerSpectrumMultipoles,
            'folps': FOLPSTracerPowerSpectrumMultipoles,
            'tns': TNSTracerPowerSpectrumMultipoles,
        }
        matched_key = next((key for key in theory_keys if key in task), None)
        if matched_key is None:
            raise ValueError(f"Unsupported theory type in task: {task}")
        theory = theory_map[matched_key]
        if emulator_fn != None:
            emulator = EmulatedCalculator.load(emulator_fn)
            return theory(pt=emulator)
        else:
            if template is None:
                template = DirectPowerSpectrumTemplate(fiducial = DESI())
            return theory(template=template)
    elif 'BAO' in task:
        # theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(template=template, ells=ells, broadband='pcs2')
        theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(template=template, ells=ells, smoothing_radius = 15, 
                                                                    broadband='pcs2', mode = 'recsym',)
        return theory

def get_observable_likelihood(task, data_args, fit_args):
    """
    Defines the modeling task and relevant configuration for fitting.
    Task format:
        {FM|SF}_{fit_cosmology}_{fiducial_cosmology}_{theory}
        where:
            - FM: Full Modeling
            - SF: ShapeFit
            - BAO: BAO fitting
    Parameters:
        data_args (dict): Contains the following keys:
            - 'tracer', 'zrange', 'region', 'weight_type', 'version'

        fit_args (dict): Contains the following keys:
            - 'corr_type', 'bins_type', 
            - 'emulator_fn': Path to the emulator function (if applicable)

    Returns:
        tuple: (likelihood, observable, theory)
    """
    (tracer, region) = (data_args[key] for key in ["tracer", "region"])
    (corr_type, bins_type) = (fit_args[key] for key in ["corr_type", "bins_type"])
    z_eff = fit_args.get('z_eff', get_effective_redshift(data_args))
    grid_cosmo = fit_args.get('grid_cosmo', '000')
    # seed = data_args.get("seed", None)
    # rsf = data_args.get("rsf", 1)
    if grid_cosmo == '000' or grid_cosmo == 'DESI':
        cosmo = DESI()
    elif grid_cosmo == '002':
        cosmo = AbacusSummit(2)
    if 'BAO' in task:
        if corr_type != 'xi':
            raise ValueError(f"BAO task currently requires corr_type='xi', but got '{corr_type}'")
        option = fit_args.get("option", 'No')
        (rmin, rmax, rbin, lenr) = load_bins(corr_type, bins_type)
        slim={0: (rmin, rmax, rbin), 2: (rmin, rmax, rbin)}
        ells = (0,2); apmode = 'qisoqap'
        broadband = None
        if 'recon' in task:
            data_args["recon"] = True
        if '1d' in option: ells = (0,)
        if tracer == 'QSO':
            smoothing_radius = 30
        else:
            smoothing_radius = 15
        ddata, wmatrix, covariance = load_blinded_data_pip(data_args, ells=ells)
        template = get_template(task, z_eff=z_eff, ells=ells, cosmo=cosmo)
        theory = get_theory(task, template=template, ells=ells, smoothing_radius=smoothing_radius)
        if broadband == 'fixed':
            for param in theory.init.params.select(basename=['al*_*', 'bl*_*']):
                param.update(fixed=True)
        if 2 not in ells:
            slim={0: (rmin, rmax, rbin)}
            for param in theory.init.params.select(basename='*l2_*'):
                param.update(fixed=True)
            for param in theory.init.params.select(basename='qap'):
                param.update(fixed=True)
        # observable = TracerCorrelationFunctionMultipolesObservable(data = data_fns, covariance = cov, 
        #                                                         slim=slim, theory=theory)
        # likelihood = ObservablesGaussianLikelihood(observables=observable, theory=theory, scale_covariance = 1/rsf)
        # likelihood.all_params[f'sigmaper'].update(fixed = False, prior=dict(dist='norm', loc=3.0, scale=1.0))
        # likelihood.all_params[f'sigmapar'].update(fixed = False, prior=dict(dist='norm', loc=6.0, scale=1.0))
        # likelihood.all_params[f'sigmas'].update(fixed = False, prior=dict(dist='norm', loc=2.0, scale=2.0))
    elif 'FM' in task or 'SF' in task:
        if corr_type != 'pk':
            raise ValueError(f"{task} task requires corr_type='pk', but got '{corr_type}'")
        option = fit_args.get("option", 'no')
        (kmin, kmax, kbin, lenk) = load_bins(corr_type, bins_type)
        klim = {ell*2: (kmin,kmax,kbin) for ell in range(2)}
        ells = (0,2)
        if '1d' in option:
            ells = (0)
            klim = {ell*2: (kmin,kmax,kbin) for ell in range(1)}
        data, wmatrix, covariance = load_blinded_data_pip(data_args, ells=ells)
        if 'emulator_fn' in fit_args:
            # print('[LOADING EMULATOR]:', fit_args['emulator_fn'])
            theory = get_theory(task, emulator_fn = fit_args['emulator_fn'])
        else:
            template = get_template(task, z_eff = z_eff, cosmo=cosmo, option=option)
            theory = get_theory(task, template=template, ells=ells)
        observable = TracerPowerSpectrumMultipolesObservable(data=data.value(concatenate=True), wmatrix=wmatrix.value(), ells=data.ells, 
                                                             k=[pole.coords('k') for pole in data], kin=wmatrix.theory.get(ells=0).coords('k'), ellsin=wmatrix.theory.ells, theory=theory)
        likelihood = ObservablesGaussianLikelihood(observables=observable, theory=theory, covariance=covariance.value())
    else:
        raise ValueError(f"Unknown fit type (BAO, SF, FM): {task}")
    likelihood()
    return (likelihood, observable, theory)

