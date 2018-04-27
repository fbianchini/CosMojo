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
from scipy import linalg as la

import warnings
warnings.filterwarnings("ignore")


# Fiducial cosmology, params, 'n' stuff
FID_COSMO  = {'w':-1., 'mnu':0.06, 'ombh2':0.0222, 'omch2':0.1197, 'ns':0.9655, '100theta':1.04119, 'tau':0.06, 'As':2.1955e-9, 'gamma0':0.55}
PARAMS_CMB = ['w',     'mnu',      'ombh2',        'omch2',        'ns',        '100theta',                     'As']
PARAMS_kSZ = ['w',     'mnu',      'ombh2',        'omch2',        'ns',        '100theta',         'gamma0',   'As']
STEPS      = {'w':0.15,'mnu':0.01, 'ombh2':0.0001, 'omch2':0.001,  'ns':0.005,  '100theta':0.005,   'tau':0.01, 'As': 2.0e-11, 'gamma0':0.01, 'b_tau':0.005}
PRIORS     = {'tau':0.01}

# Ze experiments
exps = {'PlanckTT': {'obs':['TT'], 
					  'fid_surv': {'fsky':0.8, 
								   'lminT':2, 
								   'lmaxT':30, 
								   'DeltaT':[145., 149., 137., 65., 43., 66., 200.], 
								   'fwhm':[33., 23., 14., 10., 7., 5., 5.], 
								   'DeltaP':[1e9,1e9]},
					'path_file_NLKK': None},
 
		'PlanckTTTEEE': {'obs':['TT', 'TE', 'EE'], 
					  'fid_surv': {'fsky':0.2, 
								   'lminT':30, 
								   'lmaxT':2500, 
								   'lminP':30, 
								   'lmaxP':2500, 
								   'DeltaT':[145., 149., 137., 65., 43., 66., 200.], 
								   'fwhm':[33., 23., 14., 10., 7., 5., 5.], 
								   'DeltaP':[1e9, 1e9, 450., 103., 81., 134., 406.]},
					'path_file_NLKK': None},

		'SPT-3G': { 'obs':['TT','EE','TE','KK'],
				    'fid_surv': {'fwhm':[1.8, 1.2, 1.], 
							     'lminT':30, 
							     'lminP':30, 
							     'lminK':30, 
							     'lmaxT':3000, 
							     'lmaxP':5000, 
							     'lmaxK':2500}, 
				    'path_file_NLKK': None}
		}

# Ze kSZ
kSZ_survey = {
			2500:{'fsky':2500./42000., 
				  'M_min':1e14, 
				  'M_max':3e14, 
				  'Nr':50, 
				  'rmin':50, 
				  'rmax':400, 
				  'zmin':0.2, 
				  'zmax':1., 
				  'Nz':8,
				  'fwhm_arcmin':1.2,
				  'noise_uK_arcmin':2.5,
				  'b_tau':None},
			1500:{'fsky':1500./42000., 
				  'M_min':1e14, 
				  'M_max':3e14, 
				  'Nr':50, 
				  'rmin':50, 
				  'rmax':400, 
				  'zmin':0.2, 
				  'zmax':1., 
				  'Nz':8,
				  'fwhm_arcmin':1.2,
				  'noise_uK_arcmin':2.5,
				  'b_tau':None},
			  }

fishy = {}
fishy_kSZ = {}
fishy_kSZ_fg = {}

areas = [1500, 2500]
noises = {1500: {90:2.8, 150:2.6, 220:6.6}, 
		  2500: {90:3.6, 150:3.3, 220:8.5}
		 }

# Precompute CMB derivatives
fishyCMB = FisherCMB(fid_cosmo=FID_COSMO.copy(), 
					 fid_surv={'fsky':0.6, 
							 'lminT':2, 
							 'lmaxT':5000, 
							 'lminP':2, 
							 'lmaxP':5000,
							 'lminK':2, 
							 'lmaxK':5000},
					 params=copy.copy(PARAMS_CMB), 
					 steps=STEPS.copy(), 
					 margin_params=['tau'])

# Precompute kSZ derivatives and covariances
fishyksz = {}
fishyksz[2500] = FisherPairwise(fid_cosmo=FID_COSMO.copy(), 
						  fid_surv=kSZ_survey[2500], 
						  steps=STEPS.copy(), 
						  params=copy.copy(PARAMS_kSZ),)
						  # margin_params=['b_tau'],
						  # priors={'b_tau':0.01}) 
fishyksz[1500] = FisherPairwise(fid_cosmo=FID_COSMO.copy(), 
						  fid_surv=kSZ_survey[1500], 
						  steps=STEPS.copy(), 
						  params=copy.copy(PARAMS_kSZ),)
						  # margin_params=['b_tau'],
						  # priors={'b_tau':0.01}) 



for area in areas:
	print "noise: ", noises[area], "area: ", area

	fishy[area] = {}
	fishy_kSZ[area] = {}
	fishy_kSZ_fg[area] = {}

	myexp = copy.copy(exps)
	myexp['SPT-3G']['fid_surv']['fsky'] = area/42000.
	myexp['SPT-3G']['fid_surv']['DeltaT'] = [noises[area][90], noises[area][150], noises[area][220]]
	myexp['SPT-3G']['fid_surv']['DeltaP'] = [noises[area][90]*np.sqrt(2.), noises[area][150]*np.sqrt(2.), noises[area][220]*np.sqrt(2.)]

	fishy[area] = MultiFisherCMB(fid_cosmo=FID_COSMO.copy(), 
								params=copy.copy(PARAMS_CMB), 
								exps=myexp, 
								steps=STEPS.copy(), 
								priors=PRIORS.copy(), 
								margin_params=['tau'], 
								inv_routine=la.pinv2,
								dcldp=copy.copy(fishyCMB._dcldp))
	fishy[area].mat()

	del myexp

	myksz = copy.copy(kSZ_survey)
	myksz['noise_uK_arcmin'] = noises[area][150]

	fishy_kSZ[area] = FisherPairwise(fid_cosmo=FID_COSMO.copy(), 
									fid_surv=myksz, 
									steps=STEPS.copy(), 
									params=copy.copy(PARAMS_kSZ), 
									cmb_prior=fishy[area],
									dvdp=copy.copy(fishyksz._dvdp),
									cs=fishyksz.cov_gauss_shot,
									cv=fishyksz.cov_cosmic)
									# margin_params=['b_tau'],
									# priors={'b_tau':0.1}) 
	fishy_kSZ[area].mat()

	del myksz

	myksz = copy.copy(kSZ_survey[area])
	myksz['noise_uK_arcmin'] = noises[area][150]
	myksz['fg'] = 'all'

	fishy_kSZ_fg[area] = FisherPairwise(fid_cosmo=FID_COSMO.copy(), 
										fid_surv=myksz, 
										steps=STEPS.copy(), 
										params=copy.copy(PARAMS_kSZ), 
										cmb_prior=fishy[area],
										dvdp=copy.copy(fishyksz[area]._dvdp),
										cs=fishyksz[area].cov_gauss_shot,
										cv=fishyksz[area].cov_cosmic)
										# margin_params=['b_tau'],
										# priors={'b_tau':0.1}) 
	fishy_kSZ_fg[area].mat()

	del myksz

	print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

embed()

# pk.dump(fishy, open('../fisherCMB_S0/fishCMB_S0.pkl','w'), protocol=2)

# # ~~~~~~~~~~~~~~~ mnu vs w ~~~~~~~~~~~~~~~~~~~~~~~~
# plt.figure(figsize=(6,4))

# plt.suptitle(r'$f_{\rm sky}=0.33 \quad M > 6\times 10^{13}M_{\odot} \quad 0.0 < z < 0.6$')#' + FG')# + $b_{\tau}$ weak prior')

# ax1 = plt.subplot(121)

# ax1.set_title(r'$\Delta_T = 3.0\mu$K-arcmin')

# # fishy_kSZ[3.0][3.0].plot('w','mnu', tag=r'$D=3$m', color='#555B6E', howmanysigma=[1])
# # fishy_kSZ[3.0][4.0].plot('w','mnu', tag=r'$D=4$m', color='#89B0AE', howmanysigma=[1])
# # fishy_kSZ[3.0][5.0].plot('w','mnu', tag=r'$D=5$m', color='#BEE3DB', howmanysigma=[1])
# fishy_kSZ[3.0][3.0].plot('w','mnu', tag=r'$D=3$m', color='#89CE94', howmanysigma=[1])
# fishy_kSZ[3.0][4.0].plot('w','mnu', tag=r'$D=4$m', color='#86A59C', howmanysigma=[1])
# fishy_kSZ[3.0][5.0].plot('w','mnu', tag=r'$D=5$m', color='#7D5BA6', howmanysigma=[1])
# fishy_kSZ[3.0][6.0].plot('w','mnu', tag=r'$D=6$m', color='#643173', howmanysigma=[1])
# fishy_kSZ[3.0][7.0].plot('w','mnu', tag=r'$D=7$m', color='#333333', howmanysigma=[1])

# ax1.set_ylim(0,0.19)
# ax1.set_xlim([-1.5,-0.5])
# ax1.legend(frameon=False)
# ax1.set_ylabel(r'$M_{\nu}$ [eV]', size=10)


# # plt.subplot(122)
# ax2 = plt.subplot(122, sharex=ax1)

# ax2.set_title(r'$D=7$ m')

# plt.plot(1e3,1e3,label=r'CMB', color='#8980F5')
# fishy_kSZ[5.0][7.0].fcmb.plot('w','mnu', tag=r'CMB', color='#8980F5', howmanysigma=[1])
# fishy_kSZ[5.0][7.0].plot('w','mnu', tag=r'$\Delta_T=5\mu$K-arcmin', color='#BEE3DB', howmanysigma=[1])
# fishy_kSZ[4.0][7.0].plot('w','mnu', tag=r'$\Delta_T=4\mu$K-arcmin', color='#89B0AE', howmanysigma=[1])
# fishy_kSZ[3.0][7.0].plot('w','mnu', tag=r'$\Delta_T=3\mu$K-arcmin', color='#555B6E', howmanysigma=[1])

# ax2.set_ylim(0,0.19)
# ax2.set_xlim([-1.5,-0.5])
# ax2.set_ylabel('')
# plt.setp(ax2.get_yticklabels(), visible=False)

# ax2.legend(frameon=False)

# plt.subplots_adjust(wspace=0, hspace=0, top=0.86)

# plt.savefig('/Users/fbianchini/Research/kSZ/w_mnu_SO_lowz.pdf', bboxes_inches='tight')
# plt.close()

# # ~~~~~~~~~~~~~~ gamma vs w ~~~~~~~~~~~~~~~~~~~~~~
# plt.figure(figsize=(6,4))

# plt.suptitle(r'$f_{\rm sky}=0.33 \quad M > 6\times 10^{13}M_{\odot} \quad 0.0 < z < 0.6$')#' + FG')# + $b_{\tau}$ weak prior')

# ax1 = plt.subplot(121)

# ax1.set_title(r'$\Delta_T = 3.0\mu$K-arcmin')

# # fishy_kSZ[3.0][3.0].plot('w','gamma0', tag=r'$D=3$m', color='#C5E6A6', howmanysigma=[1])
# # fishy_kSZ[3.0][4.0].plot('w','gamma0', tag=r'$D=4$m', color='#BDD2A6', howmanysigma=[1])
# # fishy_kSZ[3.0][5.0].plot('w','gamma0', tag=r'$D=5$m', color='#B9BEA5', howmanysigma=[1])
# # fishy_kSZ[3.0][6.0].plot('w','gamma0', tag=r'$D=6$m', color='#A7AAA4', howmanysigma=[1])
# # fishy_kSZ[3.0][7.0].plot('w','gamma0', tag=r'$D=7$m', color='#9899A6', howmanysigma=[1])

# fishy_kSZ[3.0][3.0].plot('w','gamma0', tag=r'$D=3$m', color='#89CE94', howmanysigma=[1])
# fishy_kSZ[3.0][4.0].plot('w','gamma0', tag=r'$D=4$m', color='#86A59C', howmanysigma=[1])
# fishy_kSZ[3.0][5.0].plot('w','gamma0', tag=r'$D=5$m', color='#7D5BA6', howmanysigma=[1])
# fishy_kSZ[3.0][6.0].plot('w','gamma0', tag=r'$D=6$m', color='#643173', howmanysigma=[1])
# fishy_kSZ[3.0][7.0].plot('w','gamma0', tag=r'$D=7$m', color='#333333', howmanysigma=[1])

# ax1.legend(frameon=False)
# ax1.set_ylim([0.45,0.65])
# ax1.set_xlim([-1.5,-0.5])

# # plt.subplot(122)
# ax2 = plt.subplot(122, sharex=ax1)

# ax2.set_title(r'$D=7$ m')

# fishy_kSZ[5.0][7.0].plot('w','gamma0', tag=r'$\Delta_T=5\mu$K-arcmin', color='#BEE3DB', howmanysigma=[1])
# fishy_kSZ[4.0][7.0].plot('w','gamma0', tag=r'$\Delta_T=4\mu$K-arcmin', color='#89B0AE', howmanysigma=[1])
# fishy_kSZ[3.0][7.0].plot('w','gamma0', tag=r'$\Delta_T=3\mu$K-arcmin', color='#555B6E', howmanysigma=[1])

# ax2.set_ylim([0.45,0.65])
# ax2.set_xlim([-1.5,-0.5])
# ax2.set_ylabel('')
# plt.setp(ax2.get_yticklabels(), visible=False)

# ax2.legend(frameon=False)

# plt.subplots_adjust(wspace=0, hspace=0, top=0.86)

# plt.savefig('/Users/fbianchini/Research/kSZ/w_gamma0_SO_lowz.pdf', bboxes_inches='tight')
# plt.close()



# # ~~~~~~~~~~~~~~ gamma vs mnu ~~~~~~~~~~~~~~~~~~~~~~
# plt.figure(figsize=(6,4))

# plt.suptitle(r'$f_{\rm sky}=0.33 \quad M > 6\times 10^{13}M_{\odot} \quad 0.0 < z < 0.6$')#' + FG')# + $b_{\tau}$ weak prior')

# ax1 = plt.subplot(121)

# ax1.set_title(r'$\Delta_T = 3.0\mu$K-arcmin')

# # fishy_kSZ[3.0][3.0].plot('mnu','gamma0', tag=r'$D=3$m', color='#555B6E', howmanysigma=[1])
# # fishy_kSZ[3.0][4.0].plot('mnu','gamma0', tag=r'$D=4$m', color='#89B0AE', howmanysigma=[1])
# # fishy_kSZ[3.0][5.0].plot('mnu','gamma0', tag=r'$D=5$m', color='#BEE3DB', howmanysigma=[1])

# fishy_kSZ[3.0][3.0].plot('mnu','gamma0', tag=r'$D=3$m', color='#89CE94', howmanysigma=[1])
# fishy_kSZ[3.0][4.0].plot('mnu','gamma0', tag=r'$D=4$m', color='#86A59C', howmanysigma=[1])
# fishy_kSZ[3.0][5.0].plot('mnu','gamma0', tag=r'$D=5$m', color='#7D5BA6', howmanysigma=[1])
# fishy_kSZ[3.0][6.0].plot('mnu','gamma0', tag=r'$D=6$m', color='#643173', howmanysigma=[1])
# fishy_kSZ[3.0][7.0].plot('mnu','gamma0', tag=r'$D=7$m', color='#333333', howmanysigma=[1])

# ax1.legend(frameon=False)

# ax1.set_xlim([0.,0.2])
# ax1.set_ylim([0.45,0.65])
# ax1.set_xlabel(r'$M_{\nu}$ [eV]', size=10)

# ax2 = plt.subplot(122, sharex=ax1)

# ax2.set_title(r'$D=7$ m')

# fishy_kSZ[5.0][5.0].plot('mnu','gamma0', tag=r'$\Delta_T=5\mu$K-arcmin', color='#BEE3DB', howmanysigma=[1])
# fishy_kSZ[4.0][5.0].plot('mnu','gamma0', tag=r'$\Delta_T=4\mu$K-arcmin', color='#89B0AE', howmanysigma=[1])
# fishy_kSZ[3.0][5.0].plot('mnu','gamma0', tag=r'$\Delta_T=3\mu$K-arcmin', color='#555B6E', howmanysigma=[1])

# ax2.set_ylabel('')
# plt.setp(ax2.get_yticklabels(), visible=False)

# ax2.set_xlim([0.,0.2])
# ax2.set_ylim([0.45,0.65])
# ax2.set_xlabel(r'$M_{\nu}$ [eV]', size=10)

# ax2.legend(frameon=False)

# plt.subplots_adjust(wspace=0, hspace=0, top=0.86)

# plt.savefig('/Users/fbianchini/Research/kSZ/gamma0_mnu_SO_lowz.pdf', bboxes_inches='tight')
# plt.close()

# # ~~~~~~~~~~~~~~~~~
# ws = np.zeros((len(noises), len(diameters)))

# for i, noise  in enumerate(noises):
# 	for j, diameter in enumerate(diameters):
# 		ws[i,j] = fishy_kSZ[noise][diameter].sigma_marg('w')

# heatmap = plt.pcolor(ws, cmap='Blues')
# for y in range(ws.shape[0]):
#     for x in range(ws.shape[1]):
#         plt.text(x + 0.5, y + 0.5, '%.2f' % ws[y, x],
#                  horizontalalignment='center',
#                  verticalalignment='center',
#                  )

# plt.colorbar(heatmap)
# plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5], diameters)
# plt.yticks([0.5, 1.5, 2.5], noises)
# plt.xlabel(r'Diameter [m]',size=13)
# plt.ylabel(r'$\Delta_T [\mu$K-arcmin]',size=13)
# plt.title(r'$\sigma(w) $- w/o FG')# + $b_{\tau}$ weak prior')

# plt.savefig('/Users/fbianchini/Research/kSZ/w_constraints_SO_lowz.pdf', bboxes_inches='tight')
# plt.close()

# # ~~~~~~~~~~~~~~~~~
# gammas = np.zeros((len(noises), len(diameters)))

# for i, noise  in enumerate(noises):
# 	for j, diameter in enumerate(diameters):
# 		gammas[i,j] = fishy_kSZ[noise][diameter].sigma_marg('gamma0')

# heatmap = plt.pcolor(gammas, cmap='Blues')
# for y in range(gammas.shape[0]):
#     for x in range(gammas.shape[1]):
#         plt.text(x + 0.5, y + 0.5, '%.3f' % gammas[y, x],
#                  horizontalalignment='center',
#                  verticalalignment='center',
#                  )

# plt.colorbar(heatmap)
# plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5], diameters)
# plt.yticks([0.5, 1.5, 2.5], noises)
# plt.xlabel(r'Diameter [m]',size=13)
# plt.ylabel(r'$\Delta_T [\mu$K-arcmin]',size=13)
# plt.title(r'$\sigma(\gamma_0) $- w/o FG')# + $b_{\tau}$ weak prior')

# plt.savefig('/Users/fbianchini/Research/kSZ/gamma_constraints_SO_lowz.pdf', bboxes_inches='tight')
# plt.close()

# # ~~~~~~~~~~~~~~~~~
# mnus = np.zeros((len(noises), len(diameters)))

# for i, noise  in enumerate(noises):
# 	for j, diameter in enumerate(diameters):
# 		mnus[i,j] = fishy_kSZ[noise][diameter].sigma_marg('mnu')*1000

# heatmap = plt.pcolor(mnus, cmap='Blues')
# for y in range(mnus.shape[0]):
#     for x in range(mnus.shape[1]):
#         plt.text(x + 0.5, y + 0.5, '%d' % mnus[y, x],
#                  horizontalalignment='center',
#                  verticalalignment='center',
#                  )

# plt.colorbar(heatmap)
# plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5], diameters)
# plt.yticks([0.5, 1.5, 2.5], noises)
# plt.xlabel(r'Diameter [m]',size=13)
# plt.ylabel(r'$\Delta_T [\mu$K-arcmin]',size=13)
# plt.title(r'$\sigma(\sum m_{\nu}) $ [meV]- w/o FG')# + $b_{\tau}$ weak prior')

# plt.savefig('/Users/fbianchini/Research/kSZ/mnu_constraints_SO_lowz.pdf', bboxes_inches='tight')
# plt.close()

# embed()

