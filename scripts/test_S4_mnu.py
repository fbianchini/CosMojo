import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from cosmojo.universe import *
# import seaborn as sns
from cosmojo.utils import *
from cosmojo.fisher import *
from IPython import embed
import copy

import warnings
warnings.filterwarnings("ignore")


exps = {'PlanckTT': {'obs':['TT'], 
                      'fid_surv': {'fsky':0.8, 
                                   'lminT':2, 
                                   'lmaxT':30, 
                                   'DeltaT':[145., 149., 137., 65., 43., 66., 200.], 
                                   'fwhm':[33., 23., 14., 10., 7., 5., 5.], 
                                   'DeltaP':[1e9,1e9]}
                     }, 
        'PlanckTTTEEE': {'obs':['TT', 'TE', 'EE'], 
                      'fid_surv': {'fsky':0.2, 
                                   'lminT':30, 
                                   'lmaxT':2500, 
                                   'lminP':30, 
                                   'lmaxP':2500, 
                                   'DeltaT':[145., 149., 137., 65., 43., 66., 200.], 
                                   'fwhm':[33., 23., 14., 10., 7., 5., 5.], 
                                   'DeltaP':[1e9, 1e9, 450., 103., 81., 134., 406.]} 
                     },
        'S4': { 'fid_surv': {'fsky':0.4, 
                               'lminT':30, 
                               'lminP':30, 
                               'lminK':30, 
                               'lmaxT':3000, 
                               'lmaxP':5000, 
                               'lmaxK':2500, 
                               'DeltaT':1., 
                               'DeltaP':1.4,  
                               'fwhm':3.}
                     }
       }

fishy = {}

S4obs = {'TT':['TT'], 'EE':['EE'], 'EETE':['EE','TE'], 'TTTEEE':['TT','EE','TE'], 'TTTEEEKK':['TT','EE','TE','KK'] }

for key, val in S4obs.iteritems():
	print '~~~~~~~~~~~~~~~' + key + '~~~~~~~~~~~~~~~~~~~'
	myexp = copy.copy(exps)
	myexp['S4']['obs'] = val
	print myexp['S4']['obs']
	fishy[key] = MultiFisherCMB(fid_cosmo={'w':-1., 'mnu':0.06,'ombh2':0.0222, 'omch2':0.1197, 'ns':0.9655, 'H0':69., 'tau':0.06, 'As':2.1955e-9},\
	                        params=['mnu', 'ombh2', 'omch2', 'ns', 'H0', 'tau', 'As'],\
	                        exps=myexp,\
	                        steps={'mnu':0.01, 'ombh2':0.0001, 'omch2':0.001, 'ns':0.005, 'H0':0.1, 'tau':0.01, 'As': 2.0e-11},\
	                        priors={'tau':0.01}
	                       )
	fishy[key].mat()
	for p in fishy[key].params:
	    print p, '%f' %fishy[key].sigma_marg(p)
	print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

embed()

