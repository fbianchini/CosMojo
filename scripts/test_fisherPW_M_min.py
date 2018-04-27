import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from cosmojo.universe import *
# import seaborn as sns
from cosmojo.utils import *
from cosmojo.fisher import *
from IPython import embed
import copy
import pickle as pk

import warnings
warnings.filterwarnings("ignore")

cov = pk.load(open('cov_stage4.pk','r'))
dvdp = pk.load(open('ders_stage4.pk','r'))

fishyCMB = FisherCMB(fid_cosmo={'mnu':0.06,
								'ombh2':0.0222, 
								'omch2':0.1197, 
								'ns':0.9655, 
								'H0':69., 
								'tau':0.06, 
								'As':2.1955e-9,  
								'w':-1.},\
					fid_surv={'fsky':0.6, 
							 'lminT':30, 
							 'lmaxT':2500, 
							 'lminP':30, 
							 'lmaxP':2500, 
							 'DeltaT':[145., 149., 137.,  65., 43.,  66., 200.],
							 'DeltaP':[1e9,  1e9,  450., 103., 81., 134., 406.], 
							 'fwhm':  [33.,  23.,   14.,  10.,  7.,   5.,   5.]},\
						steps={'w':0.15, 'ombh2':0.0001, 'omch2':0.001, 'ns':0.005, 'H0':0.1, 'tau':0.01, 'As': 2.0e-11},\
						params=['ombh2', 'omch2', 'ns', 'H0', 'As', 'w'],
						margin_params=['tau'],
						# priors={'tau':0.01},
						obs=['TT','TE','EE'])


fishyCMB.mat()


fishyPW = {}

M_mins = [1e13, 3e13, 6e13, 9e13, 2e14] 

for M_min in M_mins:  
	fishyPW[M_min] = FisherPairwise(fid_cosmo={'mnu':0.06,
									'ombh2':0.0222, 
									'omch2':0.1197, 
									'ns':0.9655, 
									'H0':69., 
									'As':2.1955e-9,
									'w':-1.,
									'gamma0':0.55},\
						 fid_surv={'fsky':1e4/42000., 
								   'M_min':M_min, 
								   'sigma_v':120,
								   'Nr':50, 
								   'rmin':30, 
								   'rmax':250, 
								   'zmin':0.1, 
								   'zmax':0.6, 
								   'Nz':5},\
						 steps={'w':0.15,'ombh2':0.0001, 'omch2':0.001, 'ns':0.005, 'H0':0.1, 'tau':0.01, 'As': 2.0e-11,'gamma0':0.01},\
						 params=['ombh2', 'omch2', 'ns', 'H0', 'As', 'w', 'gamma0'], \
                         # c,\
                         dvdp=dvdp,\
#                        priors={'H0':0.01},\
						 cmb_prior=fishyCMB) 

	fishyPW[M_min].mat()

embed()

