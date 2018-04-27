import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from universe import *
# import seaborn as sns
from utils import *
from fisher import *
from IPython import embed
import copy
from scipy import linalg as la

import warnings
warnings.filterwarnings("ignore")

# Fiducial cosmology, params, 'n' stuff
FID_COSMO  = {'w':-1., 
			  'mnu':0.1, 
			  'ombh2':0.0222, 
			  'omch2':0.1197, 
			  'ns':0.9655, 
			  '100theta':1.04119, 
			  'tau':0.06, 
			  # 'As':2.1955e-9, 
			  'sigma8':0.82, 
			  'gamma0':0.55,
			  'nnu':3.046}
PARAMS_CMB = ['tau',
			  # 'mnu',
			  'ombh2',
			  'omch2',
			  'ns',
			  '100theta',
			  'w',
			  'sigma8']
			  # 'nnu']
STEPS      = {'w':0.15,
			  'mnu':0.01, 
			  'ombh2':0.0001, 
			  'omch2':0.001, 
			  'ns':0.005,  
			  '100theta':0.005,   
			  'tau':0.01, 
			  # 'As': 2.0e-11,
			  'sigma8':0.05,
			  'nnu':0.04}

exps = {'PlanckTT': {'obs':['TT'], 
					  'fid_surv': {'fsky':0.8, 
								   'lminT':2, 
								   'lmaxT':30, 
								   'DeltaT':[145., 149., 137., 65., 43., 66., 200.], 
								   'fwhm':[33., 23., 14., 10., 7., 5., 5.], 
								   'DeltaP':[1e9,1e9]},
	   				   'path_file_NLKK': '/Users/fbianchini/Research/CosMojo/cosmojo/data/nlkk_planck2015' 
	   				}
					 , 
		'PlanckTTTEEE': {'obs':['TT', 'TE', 'EE'], 
					  'fid_surv': {'fsky':0.2, 
								   'lminT':30, 
								   'lmaxT':2500, 
								   'lminP':30, 
								   'lmaxP':2500, 
								   'DeltaT':[145., 149., 137., 65., 43., 66., 200.], 
								   'fwhm':[33., 23., 14., 10., 7., 5., 5.], 
								   'DeltaP':[1e9, 1e9, 450., 103., 81., 134., 406.]},
				        'path_file_NLKK': '/Users/fbianchini/Research/CosMojo/cosmojo/data/nlkk_planck2015' 
					 },
		'SA': { 'obs': ['TT', 'TE', 'EE', 'KK'],
					'fid_surv': {'fsky':0.5, 
							   'lminT':30, 
							   'lminP':30, 
							   'lminK':30, 
							   'lmaxT':3000, 
							   'lmaxP':5000, 
							   'lmaxK':2500, 
							   'DeltaT':[14.4, 11.8, 40.3], 
							   'DeltaP':[14.4*np.sqrt(2), 11.8*np.sqrt(2), 40.3*np.sqrt(2)],  
							   'fwhm':[5.2, 3.5, 2.7]},
					'path_file_NLKK': '/Users/fbianchini/Softwares/lensingbiases/nlkk_cmb_SO_1muK_1.4fwhm_lmax3000T_lmax5000P_lmin30.dat'}
					 
	   }

fishy_SA = {}
fishy_Planck = {}

# embed()

# Precompute CMB derivatives
fishyCMB = FisherCMB(fid_cosmo=FID_COSMO.copy(),
					 obs=['TT','EE','TE', 'KK'], 
					 fid_surv={'fsky':0.6, 
							 'lminT':2, 
							 'lmaxT':5000, 
							 'lminP':2, 
							 'lmaxP':5000,
							 'lminK':2, 
							 'lmaxK':5000},
					 params=copy.copy(PARAMS_CMB), 
					 steps=STEPS.copy())

for tau_prior in [0.012, 0.004, 0.002]:
	print '****** SA ****'
	print '-> tau = %.4f' %tau_prior
	myexp = copy.copy(exps)
	fishy_SA[tau_prior] = {}
	fishy_SA[tau_prior] = MultiFisherCMB(fid_cosmo=FID_COSMO.copy(), 
									  params=copy.copy(PARAMS_CMB), 
									  exps=myexp, 
									  steps=STEPS.copy(), 
									  priors={'tau':tau_prior}, 
									  inv_routine=la.pinv2,
									  dcldp=copy.copy(fishyCMB._dcldp))
	fishy_SA[tau_prior].mat()

	for p in fishy_SA[tau_prior].params:
		print p, '%f' %fishy_SA[tau_prior].sigma_marg(p)
	print '*****************'

	# ~~~~~~~~~~~~~~~~~~~~~~
	print '      Planck  '
	print '-> tau = %.4f' %tau_prior
	myexp = copy.copy(exps)
	del myexp['SA']
	myexp['PlanckTTTEEE']['fid_surv']['fsky']=0.8
	fishy_Planck[tau_prior] = {}
	fishy_Planck[tau_prior] = MultiFisherCMB(fid_cosmo=FID_COSMO.copy(), 
									  params=copy.copy(PARAMS_CMB), 
									  exps=myexp, 
									  steps=STEPS.copy(), 
									  priors={'tau':tau_prior}, 
									  inv_routine=la.pinv2,
									  dcldp=copy.copy(fishyCMB._dcldp))
	fishy_Planck[tau_prior].mat()


embed()

