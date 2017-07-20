import numpy as np
from scipy import linalg
from astropy import constants as const
from defaults import *
from utils import nl_cmb
from universe import Cosmo, GuessH0
from pairwise import BinPairwise
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

Labels = {'omch2'   : r'$\Omega_c h^2$',
		  'ombh2'   : r'$\Omega_b h^2$',
		  'ns'      : r'$n_s$',
		  'As'      : r'$10^9 A_s$',
		  'tau'     : r'$\tau$',
		  'H0'      : r'$h$',
		  'w'       : r'$w_0$',
		  'wa'      : r'$w_a$',
		  'mnu'     : r'$\sum m_{\nu}$',
		  'nnu'     : r'$N_{\rm eff}$',
		  'YHe'     : r'$Y_{\rm He}$',
		  '100theta': r'$100\theta_{\rm MC}$',
		  'gamma0'  : r'$\gamma_0$',
		  'gammaa'  : r'$\gamma_a$'}

class Fisher(object):
	def __init__(self, fid_cosmo=None, 
					   fid_surv=None, 
					   params=None,
					   margin_params=[], 
					   priors={}, 
					   steps={},
					   inv_routine=np.linalg.inv,
					   verbose=False):

		self.fid_cosmo = fid_cosmo.copy()
		self.fid_surv = fid_surv.copy()
		self.params = []
		self.priors = {}
		self.steps  = {}
		self.step = 0.003
		self.inv_routine = inv_routine
		self.verbose = verbose

		self.D = np.diag(np.ones(len(params)))

		self.margin_params = []
		for p in margin_params:
			if p in self.fid_cosmo.keys() or p in self.fid_surv.keys():
				self.margin_params.append(p)
			else:
				print("Warning, requested marginalisation parameter " + p + " is not included in the analysis")


		# Precomputed Fisher matrix
		self._fullMat = None
		self._fullInvMat = None
		self._mat = None
		self._invmat = None

		self._fullMat_scaled = None
		self._fullInvMat_scaled = None
		self._mat_scaled = None
		self._invmat_scaled = None

	def _computeObservables(self):
		pass

	def _computeCovariance(self):
		pass

	def _computeFullMatrix(self):
		pass

	def _computeDerivatives(self):
		pass

	def Fij(self, param_i, param_j, scaled=True):
		"""
		Returns the matrix element of the Fisher matrix for parameters
		param_i and param_j
		"""
		i = self.params.index(param_i)
		j = self.params.index(param_j)

		if scaled:
			return reduce(np.dot, [self.D, self.mat(scaled=True), self.D])[i, j]
		else:
			return self.mat()[i, j]

	def invFij(self, param_i, param_j, scaled=True):
		"""
		Returns the matrix element of the inverse Fisher matrix for
		parameters param_i and param_j
		"""
		i = self.params.index(param_i)
		j = self.params.index(param_j)

		if scaled:
			return reduce(np.dot, [self.D, self.invmat(scaled=True), self.D])[i, j]
		else:
			return self.invmat()[i, j]

	def sigma_fix(self, param, scaled=True):
		return 1.0 / np.sqrt(self.Fij(param, param, scaled=scaled))

	def sigma_marg(self, param, scaled=True):
		return np.sqrt(self.invFij(param, param, scaled=scaled))

	def _marginalise(self, params):
		""" 
		Marginalises the Fisher matrix over unwanted parameters.
		
		Parameters
		----------
		params: list
			List of parameters that should *not* be marginalised over.
		
		Returns
		-------
		(mat, invmat): ndarray
			Marginalised Fisher matrix and covariance
		"""
		# Builds covariance matrix
		marg_inv = np.zeros((len(params), len(params)))
		D_ = np.diag(np.ones(len(params)))#np.zeros((len(params), len(params)))
		for i in xrange(len(params)):
			indi = self.params.index(params[i])
			for j in xrange(len(params)):
				indj = self.params.index(params[j])
				marg_inv[i, j] = self.invmat(scaled=True)[indi, indj]
				D_[i, j] = self.D[indi, indj]

		marg_mat = reduce(np.dot, [D_, self.inv_routine(marg_inv), D_])
		marg_inv = reduce(np.dot, [D_, marg_inv, D_])

		return (marg_mat, marg_inv)

	@property
	def FoM_DETF(self):
		"""
			Computes the figure of merit from the Dark Energy Task Force
			Albrecht et al 2006
			FoM = 1/sqrt(det(F^-1_{w0,wa}))
		"""
		det = (self.invFij('w', 'w') * self.invFij('wa', 'wa') -
			   self.invFij('wa', 'w') * self.invFij('w', 'wa'))
		return 1.0 / np.sqrt(det)

	@property
	def FoM(self):
		"""
		Total figure of merit : ln (1/det(F^{-1}))
		"""
		return np.log(1.0 / abs(linalg.det(self.invmat)))

	# @property
	def invmat(self, scaled=False):
		"""
		Returns the inverse Fisher matrix (= covariance)
		"""
		if scaled:
			if self._invmat_scaled is None:
				self._invmat_scaled = self.inv_routine(self.mat(scaled=True))
			return self._invmat_scaled
		else:
			if self._invmat is None:
				self._invmat = self.inv_routine(self.mat)
			return self._invmat

	# @staticmethod
	def _rescaleFisherMatrix(self, F, simple=True, return_scaling_matrix=True):
		if simple:
			D = np.diag(1./np.sqrt(np.diag(F)))
			F_scaled = reduce(np.dot, [D, F, D]) # rescaled Fisher matrix
		else:
			# see https://www.cs.ubc.ca/~inutard/files/cs517-project.pdf
			T = np.tril(F)
			d = np.zeros(T.shape[0])
			for i in xrange(T.shape[0]):
			    # print np.sqrt(T[i,i]), np.dot(d[:i], T[i,:i])
			    d[i] = 1. / np.max( [np.sqrt(T[i,i]), np.dot(d[:i], T[i,:i])] )
			D = np.diag(d)
			F_scaled = reduce(np.dot, [D, F, D]) # rescaled Fisher matrix

		if return_scaling_matrix:
			return F_scaled, D
		else:
			return F_scaled

	# @property
	def mat(self, scaled=False, simple=True):
		"""
		Returns the Fisher matrix marginalised over nuisance parameters
		"""
		# If the Fisher matrix is not already computed, compute it
		if (self._mat is None) or (self._mat_scaled is None):
			self._fullMat = self._computeFullMatrix() # Full (w/ margin params) Fisher matrix
			self._fullMat_scaled, self.D_fullMat = self._rescaleFisherMatrix(self._fullMat, simple=simple, return_scaling_matrix=True) # rescaled Full (w/ margin params) Fisher matrix  
			self._fullInvMat_scaled = self.inv_routine(self._fullMat_scaled) # rescaled Full (w/ margin params) Covariance matrix 
			self._fullInvMat = reduce(np.dot, [self.D_fullMat, self._fullInvMat_scaled, self.D_fullMat]) # Full (w/ margin params) Covariance matrix 

			# Apply marginalisation over nuisance parameters 
			self._invmat = self._fullInvMat[0:len(self.params),0:len(self.params)]
			self._mat = self.inv_routine(self._invmat)
			self._mat_scaled, self.D = self._rescaleFisherMatrix(self._mat, simple=simple, return_scaling_matrix=True) # rescaled marginalizes Fisher matrix  
			self._invmat_scaled = self.inv_routine(self._mat_scaled)

		if scaled:
			return self._mat_scaled
		else:
			return self._mat

	def clearall(self):
		print("...!!! erasing Fisher matrix & covariance !!!...")
		self._fullMat = None
		self._fullInvMat = None
		self._mat = None
		self._invmat = None

		self._fullMat_scaled = None
		self._fullInvMat_scaled = None
		self._mat_scaled = None
		self._invmat_scaled = None
		print("...done...")

	def corner_plot(self, nstd=1, labels=None, **kwargs):
		""" 
		Makes a corner plot including all the parameters in the Fisher analysis
		"""

		if labels is None:
			labels = self.params

		for i in xrange(len(self.params)):
			for j in range(i):
				ax = plt.subplot(len(self.params)-1, len(self.params)-1 , (i - 1)*(len(self.params)-1) + (j+1))

				self.plot(self.params[j], self.params[i], nstd=nstd, ax=ax, labels=False, **kwargs)

				if i == len(self.params) - 1:
					# ax.set_xlabel(labels[j])
					ax.set_xlabel(Labels[self.params[j]])
				else:
					ax.set_xticklabels([])
				if j == 0:
					# ax.set_ylabel(labels[i])
					ax.set_ylabel(Labels[self.params[i]])
				else:
					ax.set_yticklabels([])

		plt.subplots_adjust(wspace=0)
		plt.subplots_adjust(hspace=0)

	def ellipse_pars(self, p1, p2, howmanysigma=1):
		params = [p1, p2]

		def eigsorted(cov):
			vals, vecs = linalg.eigh(cov)
			order = vals.argsort()[::-1]
			return vals[order], vecs[:, order]

		mat, COV = self._marginalise(params)
		# print COV

		# First find the fiducial value for the parameter in question
		fid_param = None
		pos = [0, 0]
		for p in params:
			if p in dir(self.fid_surv):
				fid_param = self.fid_surv[p]
			else:
				fid_param = self.fid_cosmo[p]
				if p == 'As':
					fid_param *= 1e9 
				if p == 'H0':
					fid_param /= 100. 

			pos[params.index(p)] = fid_param

		vals, vecs = eigsorted(COV)
		theta = np.arctan2(*vecs[:, 0][::-1])
		theta = np.degrees(theta)

		assert COV.shape == (2,2)

		confsigma_dic = {1:2.3, 2:6.17, 3: 11.8}

		sig_x2, sig_y2 = COV[0,0], COV[1,1]
		sig_xy = COV[0,1]

		t1 = (sig_x2 + sig_y2)/2.
		t2 = np.sqrt( (sig_x2 - sig_y2)**2. /4. + sig_xy**2. )
		a = np.sqrt( abs(t1 + t2) )
		b = np.sqrt( abs(t1 - t2) )

		t1 = 2 * sig_xy
		t2 = sig_x2 - sig_y2

		theta = np.degrees(np.arctan2(t1,t2) / 2.)
		alpha = np.sqrt(confsigma_dic[howmanysigma])

		return pos, a*alpha, b*alpha, theta

	def plot(self, p1, p2, nstd=1, ax=None, howmanysigma=[1,2], labels=None, tag=None, **kwargs):
		""" 
		Plots confidence contours corresponding to the parameters provided.
		
		Parameters
		----------
		"""
		if ax is None:
			ax = plt.gca()

		for sigmas in howmanysigma: 
			pos, Aalpha, Balpha, theta = self.ellipse_pars(p1, p2, sigmas)

			ellip = Ellipse(xy=pos, width=Aalpha, height=Balpha, angle=theta, alpha=1./sigmas, **kwargs)

			ax.add_artist(ellip)
			# sz = np.max(width, height)
			s1 = 1.5*nstd*self.sigma_marg(p1)
			s2 = 1.5*nstd*self.sigma_marg(p2)
			ax.set_xlim(pos[0] - s1, pos[0] + s1)
			ax.set_ylim(pos[1] - s2, pos[1] + s2)
			#ax.set_xlim(pos[0] - sz, pos[0] + sz)
			#ax.set_ylim(pos[1] - sz, pos[1] + sz)

		if labels is None:
			ax.set_xlabel(Labels[p1])
			ax.set_ylabel(Labels[p2])
		elif labels is False:
			pass
		else:
			ax.set_xlabel(labels[0], size=14)
			ax.set_ylabel(labels[1], size=14)

		if tag is not None:
			if ellip.fill:
				ax.plot(1e5,1e5, color=ellip.get_facecolor(), label=tag)
			else:
				ax.plot(1e5,1e5, color=ellip.get_edgecolor(), label=tag)
			# ax.legend([ellip], tag)

		# ax.plot(pos, 'w+', mew=2.)
		plt.draw()
		return ellip

class FisherBAO(Fisher):

	def __init__(self, fid_cosmo, 
					   fid_surv, 
					   params, 
					   obs=['TT','EE','TE'],  
					   priors={}, 
					   steps={},
					   margin_params=[],
					   inv_routine=np.linalg.inv,
					   verbose=False):
		"""
		Constructor
		* fid_cosmo : dictionary (can be composed by more params than the one to forecast/marginalize)
		* fid_survey : dictionary => {[z-range], }
		* params : list of Fisher analysis parameters
		"""

		super(FisherBAO, self).__init__(fid_cosmo, fid_surv, params, margin_params, priors, steps, inv_routine, verbose)

		# self.fid_cosmo = fid_cosmo.copy()
		# self.fid_surv = fid_surv.copy()
		# self.params = []
		# self.priors = {}
		# self.steps  = {}
		self.obs = obs

		# Few checks on survey params -> initialize to default value if not present
		for key, val in default_bao_survey_dict.iteritems():
			self.fid_surv.setdefault(key,val)
			setattr(self, key, self.fid_surv[key]) 
			# print key, self.fid_surv[key]

		# Check that the parameters provided are present in survey or cosmo
		for p in params:
			# First find the fiducial value for the parameter in question
			if p in self.fid_cosmo.keys():# or p in self.fid_surv.keys()
				self.params.append(p)
			else:
				print("Warning, unknown parameter in derivative :" + p)

		print self.params

		# Check and store priors
		for key, val in priors.iteritems():
			if key in self.params:
				self.priors[key] = val

		# Check and store steps
		for key, val in steps.iteritems():
			if key in self.params:
				self.steps[key] = val

		if self.verbose:
			for key, val in self.fid_surv.iteritems():
				print key, val

		# print ''
		# print self.fid_surv
		# print ''

		if '100theta' in self.params:
			fid_cosmo['H0'] = GuessH0(fid_cosmo['100theta'], params=fid_cosmo)

		# Create a Cosmo object with a copy of the fiducial cosmology
		self.cosmo = Cosmo(params=fid_cosmo)

		# Compute derivatives
		if dfdp is not None:
			self._dfdp = copy.copy(dfdp)
			print("...derivatives loaded...")
			# if dcldp.shape[0] == len(params+margin_params):
			# 	self._dcldp = dcldp
			# else:
			# 	print("...Computing derivatives...")
			# 	self._dcldp = self._computeDerivatives()
			# 	print("...done...")
		else:
			print("...Computing derivatives...")
			self._dfdp = self._computeDerivatives()
			print("...done...")


	def _computeCovariance(self, l):

		return Cov

	def _computeObservables(self, par_cosmo):
		if '100theta' in par_cosmo:
			par_cosmo['H0'] = GuessH0(par_cosmo['100theta'], par_cosmo)

		return Cosmo(params=par_cosmo).cmb_spectra(self.lmax, spec='lensed_scalar', dl=False)

	def _computeFullMatrix(self):
		if self._dfdp is None:
			print("...Computing derivatives...")
			self._dfdp = self._computeDerivatives()
			print("...done...")

		nparams = len(self._dfdp)#len(self.params)

		_fullMat = np.zeros((nparams,nparams))

		print("...Computing Full Fisher matrix...")
		# Computes the fisher Matrix
		for i in xrange(nparams):
			for j in xrange(i+1):
				# print i,j 
				tmp = 0
				for l in self.lrange:
					cov = self._computeCovariance(l)
					
					inv_cov = self.inv_routine(cov)
					df_i = np.array([self._dfdp[i][l,0], self._dfdp[i][l,1], self._dfdp[i][l,3]])
					df_j = np.array([self._dfdp[j][l,0], self._dfdp[j][l,1], self._dfdp[j][l,3]])
					tmp += np.nan_to_num(np.dot(df_i, np.dot(inv_cov, df_j)))	

				_fullMat[i,j] = tmp
				_fullMat[j,i] = tmp#_fullMat[i,j]
				del tmp

		print("...done...")

		# _Priors = np.zeros((nparams,nparams))
		# for p in self.params:
		# 	i = self.params.index(p)
		# 	try:
		# 		_Priors[i,i] = 1./self.priors[p]**2.
		# 		print '\t...Including prior for', p, str(self.priors[p])
		# 	except KeyError: 
		# 		pass
		# if (_Priors == 0).all():
		# 	print '\t...No priors included...'

		# Priors ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		_Priors = np.zeros((nparams,nparams))
		allparams = self.params + self.margin_params
		for p in allparams:
			i = allparams.index(p)
			try:
				_Priors[i,i] = 1./self.priors[p]**2.
				print '---Including prior for',p,str(self.priors[p])
			except KeyError:
				pass
		if (_Priors == 0).all():
			print '---No prior included'

		_fullMat += _Priors

		#print "prior matrix ",FisherPriors
		#print "fisher before adding prior",self.totFisher

		_fullMat += _Priors

		return _fullMat

	def _computeDerivatives(self):
		""" 
		Computes all the derivatives of the specified observable with
		respect to the parameters and nuisance parameters in the analysis
		"""
		
		# List the derivatives with respect to all the parameters
		dcldp = []

		# Computes all the derivatives with respect to the main parameters
		for p in self.params + self.margin_params:
			if self.verbose:
				print("varying :" + p)

			# Forward ~~~~~~~~~~~~~~~~~~~~~~
			par_cosmo = self.fid_cosmo.copy()				

			try:
				step = self.steps[p]
			except:
				step = par_cosmo[p] * self.step		
				if par_cosmo[p] == 0:
					step = self.step

			# par_cosmo[p] = par_cosmo[p] + step/2.
			par_cosmo[p] = par_cosmo[p] + step
			
			if self.verbose:
				print '\t %3.2e' %par_cosmo[p]

			clsp = self._computeObservables(par_cosmo)

			del par_cosmo

			# Backward ~~~~~~~~~~~~~~~~~~~~~~
			par_cosmo = self.fid_cosmo.copy()				

			try:
				step = self.steps[p]
			except:
				step = par_cosmo[p] * self.step	
				if par_cosmo[p] == 0:
					step = self.step
			# par_cosmo[p] = par_cosmo[p] - step/2.
			par_cosmo[p] = par_cosmo[p] - step
			
			if self.verbose:
				print '\t %3.2e' %par_cosmo[p]


			clsm = self._computeObservables(par_cosmo)

			if p == 'As':
				step = step * 1e9
			if p == 'H0':
				step = step / 100.

			# dcldp.append( (clsp - clsm)/ (step))		
			dcldp.append( (clsp - clsm)/ (2. * step))		

			del par_cosmo, clsp, clsm

		return dcldp

	def sub_matrix(self, subparams):
		"""
		Extracts a submatrix from the current fisher matrix using the
		parameters in params
		"""
		params = []
		for p in subparams:
			# Checks that the parameter exists in the orignal matrix
			if p in self.params:
				params.append(p)
			else:
				print("Warning, parameter not present in original \
					Fisher matrix, left ignored :" + p)
		newFisher = FisherCMB(self.fid_cosmo, self.fid_surv, params)

		# Fill in the fisher matrix from the precomputed matrix
		newFisher._mat = np.zeros((len(params), len(params)))

		for i in xrange(len(params)):
			indi = self.params.index(params[i])
			for j in xrange(len(params)):
				indj = self.params.index(params[j])
				newFisher._mat[i, j] = self.mat[indi, indj]

		newFisher._invmat = self.inv_routine(newFisher._mat)

		return newFisher

class FisherCMB(Fisher):

	def __init__(self, fid_cosmo, 
					   fid_surv, 
					   params, 
					   obs=['TT','EE','TE'],  
					   priors={}, 
					   steps={},
					   margin_params=[],
					   inv_routine=np.linalg.inv,
					   verbose=False,
					   path_file_NLKK=None,
					   cls=None,
					   dcldp=None):
		"""
		Constructor
		* fid_cosmo : dictionary (can be composed by more params than the one to forecast/marginalize)
		* fid_survey : dictionary => {\Delta_T, l_knee, fsky, lminT, lminP, lmaxT, lmaxP, lminK, lmaxK}
		* params : list of Fisher analysis parameters
		"""

		super(FisherCMB, self).__init__(fid_cosmo, fid_surv, params, margin_params, priors, steps, inv_routine, verbose)

		# self.fid_cosmo = fid_cosmo.copy()
		# self.fid_surv = fid_surv.copy()
		# self.params = []
		# self.priors = {}
		# self.steps  = {}
		self.obs = obs

		# Few checks on survey params -> initialize to default value if not present
		for key, val in default_cmb_survey_dict.iteritems():
			self.fid_surv.setdefault(key,val)
			setattr(self, key, self.fid_surv[key]) 
			# print key, self.fid_surv[key]

		# Check that the parameters provided are present in survey or cosmo
		for p in params:
			# First find the fiducial value for the parameter in question
			if p in self.fid_cosmo.keys():# or p in self.fid_surv.keys()
				self.params.append(p)
			else:
				print("Warning, unknown parameter in derivative :" + p)

		print self.params

		# Check and store priors
		for key, val in priors.iteritems():
			if key in self.params:
				self.priors[key] = val

		# Check and store steps
		for key, val in steps.iteritems():
			if key in self.params:
				self.steps[key] = val

		if self.verbose:
			for key, val in self.fid_surv.iteritems():
				print key, val

		# print ''
		# print self.fid_surv
		# print ''

		if '100theta' in self.params:
			fid_cosmo['H0'] = GuessH0(fid_cosmo['100theta'], params=fid_cosmo)

		# Create a Cosmo object with a copy of the fiducial cosmology
		self.cosmo = Cosmo(params=fid_cosmo)

		# Multipoles range
		self.lmax = np.max([self.lmaxT, self.lmaxP, self.lmaxK])
		self.lmin = np.min([self.lminT, self.lminP, self.lminK])
		self.lrange = np.arange(self.lmin, self.lmax+1)
		if self.verbose:
			print self.lmin, self.lmax

		# Compute noise power spectra
		self.NlTT = nl_cmb(self.DeltaT, self.fwhm, lmax=self.lmax, lknee=self.lknee, alpha=self.alpha)
		self.NlPP = nl_cmb(self.DeltaP, self.fwhm, lmax=self.lmax, lknee=self.lknee, alpha=self.alpha)
		self.NlTT[:self.lminT] = 1.e40
		self.NlTT[self.lmaxT+1:] = 1.e40
		self.NlPP[:self.lminP] = 1.e40
		self.NlPP[self.lmaxP+1:] = 1.e40
		self.NlKK = np.zeros_like(self.NlTT)

		if 'KK' in self.obs:
			# Assumes nlkk_ starts from l = 0
			if path_file_NLKK is None:
				l_, nlkk_ = np.loadtxt('../data/nlkk_cmb_s4_1muK_2fwhm_lmax2500.dat', unpack=True)
				nlkk_ = np.interp(np.arange(self.lmax+1), l_, nlkk_, left=0., right=0.)
				# l_, nlkk_ = np.loadtxt('../data/nlkk_planck2015.dat', unpack=True)
			else:
				l_, nlkk_ = np.loadtxt(path_file_NLKK, unpack=True)
			# self.NlKK[:nlkk_.size] = nlkk_
			self.NlKK = nlkk_
			self.NlKK[:self.lminK] = 1.e40
			self.NlKK[self.lmaxK+1:] = 1.e40

		# print self.NlTT
		# print self.NlPP
		# print self.NlKK

		# Compute fiducial CMB power spectra w/ fiducial cosmo + survey
		if cls is None:
			print("...Computing fiducial CMB power spectra...")
			self.cls = self.cosmo.cmb_spectra(self.lmax, spec='lensed_scalar', dl=False)
			print("...done...")
		else:
			self.cls = cls.copy()

		# Compute derivatives
		if dcldp is not None:
			self._dcldp = copy.copy(dcldp)
			print("...derivatives loaded...")
			# if dcldp.shape[0] == len(params+margin_params):
			# 	self._dcldp = dcldp
			# else:
			# 	print("...Computing derivatives...")
			# 	self._dcldp = self._computeDerivatives()
			# 	print("...done...")
		else:
			print("...Computing derivatives...")
			self._dcldp = self._computeDerivatives()
			print("...done...")


	def _computeCovariance(self, l):

		TT = self.cls[l,0] + self.NlTT[l]
		EE = self.cls[l,1] + self.NlPP[l]
		TE = self.cls[l,3]
		KK = self.cls[l,4] + self.NlKK[l]
		KT = self.cls[l,5]

		f = 0.5

		if set(self.obs) == set(['TT','TE','EE']):
			Cov = 2./((2*l+1)*self.fsky) * \
				  np.asarray([[TT**2., TE**2., TT*TE           ],
							  [TE**2., EE**2., EE*TE           ],
							  [TT*TE,  EE*TE,  f*(TE**2.+TT*EE)]])

		if set(self.obs) == set(['EE','TE']):
			Cov = 2./((2*l+1)*self.fsky) * \
				  np.asarray([[EE**2., EE*TE           ],
							  [EE*TE,  f*(TE**2.+TT*EE)]])

		if set(self.obs) == set(['TT','TE','EE', 'KK', 'KT']):
			Cov = 2./((2*l+1)*self.fsky) * \
					np.array([[TT**2., TE**2., TT*TE,            KT*KT],
			                  [TE**2., EE**2., EE*TE,            0.   ],
			                  [TT*TE,  EE*TE,  f*(TE**2.+TT*EE), 0.   ],
			                  [KT*KT,  0.,     0.,               KK*KK]])

		if set(self.obs) == set(['TT','TE','EE', 'KK']):
			Cov = 2./((2*l+1)*self.fsky) * \
					np.array([[TT**2., TE**2., TT*TE,            0.   ],
			                  [TE**2., EE**2., EE*TE,            0.   ],
			                  [TT*TE,  EE*TE,  f*(TE**2.+TT*EE), 0.   ],
			                  [0.   ,  0.,     0.,               KK*KK]])

		elif self.obs == ['TT']:
			Cov = 2./((2*l+1)*self.fsky) * TT**2

		elif self.obs == ['EE']:
			Cov = 2./((2*l+1)*self.fsky) * EE**2

		elif self.obs == ['TE']:
			Cov = 2./((2*l+1)*self.fsky) * f * (TE**2.+TT*EE)

		elif self.obs == ['KK']:
			Cov = 2./((2*l+1)*self.fsky) * KK**2

		return Cov

	def _computeObservables(self, par_cosmo):
		if '100theta' in par_cosmo:
			par_cosmo['H0'] = GuessH0(par_cosmo['100theta'], par_cosmo)
		return Cosmo(params=par_cosmo).cmb_spectra(self.lmax, spec='lensed_scalar', dl=False)

	def _computeFullMatrix(self):
		if self._dcldp is None:
			print("...Computing derivatives...")
			self._dcldp = self._computeDerivatives()
			print("...done...")

		# nparams = len(self._dvdp)
		nparams = len(self._dcldp)#len(self.params)

		# print self._dvdp
		# print self.inv_cov[0].shape

		_fullMat = np.zeros((nparams,nparams))

		print("...Computing Full Fisher matrix...")
		# Computes the fisher Matrix
		for i in xrange(nparams):
			for j in xrange(i+1):
				# print i,j 
				tmp = 0
				for l in self.lrange:
					cov = self._computeCovariance(l)
					
					if set(self.obs) == set(['TT','TE','EE']):
						inv_cov = self.inv_routine(cov)
						dcl_i = np.array([self._dcldp[i][l,0], self._dcldp[i][l,1], self._dcldp[i][l,3]])
						dcl_j = np.array([self._dcldp[j][l,0], self._dcldp[j][l,1], self._dcldp[j][l,3]])
						tmp += np.nan_to_num(np.dot(dcl_i, np.dot(inv_cov, dcl_j)))	

					if set(self.obs) == set(['EE','TE']):
						inv_cov = self.inv_routine(cov)
						dcl_i = np.array([self._dcldp[i][l,1], self._dcldp[i][l,3]])
						dcl_j = np.array([self._dcldp[j][l,1], self._dcldp[j][l,3]])
						tmp += np.nan_to_num(np.dot(dcl_i, np.dot(inv_cov, dcl_j)))	
					
					elif set(self.obs) == set(['TT','TE','EE', 'KK', 'KT']):
						inv_cov = self.inv_routine(cov)
						dcl_i = np.array([self._dcldp[i][l,0], self._dcldp[i][l,1], self._dcldp[i][l,3], self._dcldp[i][l,4]])
						dcl_j = np.array([self._dcldp[j][l,0], self._dcldp[j][l,1], self._dcldp[j][l,3], self._dcldp[j][l,4]])
						tmp += np.nan_to_num(np.dot(dcl_i, np.dot(inv_cov, dcl_j)))	

					elif set(self.obs) == set(['TT','TE','EE', 'KK']):
						inv_cov = self.inv_routine(cov)
						dcl_i = np.array([self._dcldp[i][l,0], self._dcldp[i][l,1], self._dcldp[i][l,3], self._dcldp[i][l,4]])
						dcl_j = np.array([self._dcldp[j][l,0], self._dcldp[j][l,1], self._dcldp[j][l,3], self._dcldp[j][l,4]])
						tmp += np.nan_to_num(np.dot(dcl_i, np.dot(inv_cov, dcl_j)))	

					elif self.obs == ['TT']:
						inv_cov = 1./cov
						dcl_i = self._dcldp[i][l,0]
						dcl_j = self._dcldp[j][l,0]
						tmp += np.nan_to_num(dcl_i * inv_cov * dcl_j)

					elif self.obs == ['EE']:
						inv_cov = 1./cov
						dcl_i = self._dcldp[i][l,1]
						dcl_j = self._dcldp[j][l,1]
						tmp += np.nan_to_num(dcl_i * inv_cov * dcl_j)

					elif self.obs == ['TE']:
						inv_cov = 1./cov
						dcl_i = self._dcldp[i][l,3]
						dcl_j = self._dcldp[j][l,3]
						tmp += np.nan_to_num(dcl_i * inv_cov * dcl_j)

					elif self.obs == ['KK']:
						inv_cov = 1./cov
						dcl_i = self._dcldp[i][l,4]
						dcl_j = self._dcldp[j][l,4]
						tmp += np.nan_to_num(dcl_i * inv_cov * dcl_j)

				_fullMat[i,j] = tmp
				_fullMat[j,i] = tmp#_fullMat[i,j]
				del tmp

		print("...done...")

		# _Priors = np.zeros((nparams,nparams))
		# for p in self.params:
		# 	i = self.params.index(p)
		# 	try:
		# 		_Priors[i,i] = 1./self.priors[p]**2.
		# 		print '\t...Including prior for', p, str(self.priors[p])
		# 	except KeyError: 
		# 		pass
		# if (_Priors == 0).all():
		# 	print '\t...No priors included...'

		# Priors ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		_Priors = np.zeros((nparams,nparams))
		allparams = self.params + self.margin_params
		for p in allparams:
			i = allparams.index(p)
			try:
				_Priors[i,i] = 1./self.priors[p]**2.
				print '---Including prior for',p,str(self.priors[p])
			except KeyError:
				pass
		if (_Priors == 0).all():
			print '---No prior included'

		_fullMat += _Priors

		#print "prior matrix ",FisherPriors
		#print "fisher before adding prior",self.totFisher

		_fullMat += _Priors

		return _fullMat

	def _computeDerivatives(self):
		""" 
		Computes all the derivatives of the specified observable with
		respect to the parameters and nuisance parameters in the analysis
		"""
		
		# List the derivatives with respect to all the parameters
		dcldp = []

		# Computes all the derivatives with respect to the main parameters
		for p in self.params + self.margin_params:
			if self.verbose:
				print("varying :" + p)

			# Forward ~~~~~~~~~~~~~~~~~~~~~~
			par_cosmo = self.fid_cosmo.copy()				

			try:
				step = self.steps[p]
			except:
				step = par_cosmo[p] * self.step		
				if par_cosmo[p] == 0:
					step = self.step

			# par_cosmo[p] = par_cosmo[p] + step/2.
			par_cosmo[p] = par_cosmo[p] + step
			
			if self.verbose:
				print '\t %3.2e' %par_cosmo[p]

			clsp = self._computeObservables(par_cosmo)

			del par_cosmo

			# Backward ~~~~~~~~~~~~~~~~~~~~~~
			par_cosmo = self.fid_cosmo.copy()				

			try:
				step = self.steps[p]
			except:
				step = par_cosmo[p] * self.step	
				if par_cosmo[p] == 0:
					step = self.step
			# par_cosmo[p] = par_cosmo[p] - step/2.
			par_cosmo[p] = par_cosmo[p] - step
			
			if self.verbose:
				print '\t %3.2e' %par_cosmo[p]


			clsm = self._computeObservables(par_cosmo)

			if p == 'As':
				step = step * 1e9
			if p == 'H0':
				step = step / 100.

			# dcldp.append( (clsp - clsm)/ (step))		
			dcldp.append( (clsp - clsm)/ (2. * step))		

			del par_cosmo, clsp, clsm

		return dcldp

	def sub_matrix(self, subparams):
		"""
		Extracts a submatrix from the current fisher matrix using the
		parameters in params
		"""
		params = []
		for p in subparams:
			# Checks that the parameter exists in the orignal matrix
			if p in self.params:
				params.append(p)
			else:
				print("Warning, parameter not present in original \
					Fisher matrix, left ignored :" + p)
		newFisher = FisherCMB(self.fid_cosmo, self.fid_surv, params)

		# Fill in the fisher matrix from the precomputed matrix
		newFisher._mat = np.zeros((len(params), len(params)))

		for i in xrange(len(params)):
			indi = self.params.index(params[i])
			for j in xrange(len(params)):
				indj = self.params.index(params[j])
				newFisher._mat[i, j] = self.mat[indi, indj]

		newFisher._invmat = self.inv_routine(newFisher._mat)

		return newFisher

class MultiFisherCMB(object):
	def __init__(self, fid_cosmo, 
					   params, 
					   exps,
					   margin_params=[],  
					   priors={}, 
					   steps={},
					   inv_routine=np.linalg.inv,
					   verbose=False,
					   dcldp=None):
		"""
		Constructor
		* fid_cosmo : dictionary (can be composed by more params than the one to forecast/marginalize)
		* exp : dictionary *OF DICTIONARIES* => {'EXP1':{fid_survey, obs}}
		* params : list of Fisher analysis parameters
		"""

		self.fid_cosmo = fid_cosmo.copy()
		self.params = []
		self.priors = {}
		self.steps  = {}
		self.step   = 0.003
		self.exps   = exps
		self.fisher = {}
		self.fid_surv = {} # FIXME: just a quick hack to be able to make plots
		self.inv_routine = inv_routine

		# Check that the parameters provided are present in survey or cosmo
		for p in params:
			# First find the fiducial value for the parameter in question
			if p in self.fid_cosmo.keys():# or p in self.fid_surv.keys()
				self.params.append(p)
			else:
				print("Warning, unknown parameter in derivative :" + p)

		self.D = np.diag(np.ones(len(self.params)))

		# Check and store priors
		for key, val in priors.iteritems():
			if key in self.params:
				self.priors[key] = val

		# Check and store steps
		for key, val in steps.iteritems():
			if key in self.params:
				self.steps[key] = val

		self.margin_params = []
		for p in margin_params:
			if p in self.fid_cosmo.keys() or p in self.fid_surv.keys():
				self.margin_params.append(p)
			else:
				print("Warning, requested marginalisation parameter " + p + " is not included in the analysis")

		if '100theta' in self.params:
			fid_cosmo['H0'] = GuessH0(fid_cosmo['100theta'], params=fid_cosmo)

		# Create a Cosmo object with a copy of the fiducial cosmology
		self.cosmo = Cosmo(params=fid_cosmo)

		self.lmax = 5000 # FIXME: this shouldn't be hardcoded 

		# Compute fiducial CMB power spectra w/ fiducial cosmo + survey
		print("...Computing fiducial CMB power spectra...")
		self.cls = self.cosmo.cmb_spectra(self.lmax, spec='lensed_scalar', dl=False)
		print("...done...")

		# Compute derivatives
		# Compute derivatives
		if dcldp is not None:
			self._dcldp = copy.copy(dcldp)
			print("...derivatives loaded...")
			# if dcldp.shape[0] == len(params+margin_params):
			# 	self._dcldp = dcldp
			# else:
			# 	print("...Computing derivatives...")
			# 	self._dcldp = self._computeDerivatives()
			# 	print("...done...")
		else:
			print("...Computing derivatives...")
			self._dcldp = self._computeDerivatives()
			print("...done...")

		# Initialize Fisher classes for each experiment
		print("...Initializing Fisher classes...")
		for exp in self.exps.keys():
			print('\t'+exp)
			# if exp != self.exps.keys()[0]:
			# 	dcldp = self.fisher[self.exps.keys()[0]]._dcldp
			# else:
			# 	dcldp = None
			print self.exps[exp]['fid_surv']['fwhm']
			self.fisher[exp] = FisherCMB(self.fid_cosmo.copy(),
										 self.exps[exp]['fid_surv'].copy(), 
										 copy.copy(self.params), 
										 obs=copy.copy(self.exps[exp]['obs']),  
										 priors={}, 
										 inv_routine=self.inv_routine, 
										 steps=self.steps.copy(), 
										 verbose=verbose, 
										 cls=copy.copy(self.cls), 
										 dcldp=copy.copy(self._dcldp),
										 path_file_NLKK=self.exps[exp]['path_file_NLKK'])
			self.fisher[exp].mat()
		print("...done...")

		# Precomputed Fisher matrix
		self._fullMat = None
		self._fullInvMat = None
		self._mat = None
		self._invmat = None

		self._fullMat_scaled = None
		self._fullInvMat_scaled = None
		self._mat_scaled = None
		self._invmat_scaled = None

	def _computeObservables(self, par_cosmo):
		if '100theta' in par_cosmo:
			par_cosmo['H0'] = GuessH0(par_cosmo['100theta'], par_cosmo)
		return Cosmo(params=par_cosmo).cmb_spectra(self.lmax, spec='lensed_scalar', dl=False)

	def _computeDerivatives(self):
		""" 
		Computes all the derivatives of the specified observable with
		respect to the parameters and nuisance parameters in the analysis
		"""
		
		# List the derivatives with respect to all the parameters
		dcldp = []

		# Computes all the derivatives with respect to the main parameters
		for p in self.params + self.margin_params:
			if False:
				print("varying :" + p)

			# Forward ~~~~~~~~~~~~~~~~~~~~~~
			par_cosmo = self.fid_cosmo.copy()				

			try:
				step = self.steps[p]
			except:
				step = par_cosmo[p] * self.step		
				if par_cosmo[p] == 0:
					step = self.step

			# par_cosmo[p] = par_cosmo[p] + step/2.
			par_cosmo[p] = par_cosmo[p] + step
			
			if False:
				print '\t %3.2e' %par_cosmo[p]

			clsp = self._computeObservables(par_cosmo)

			del par_cosmo

			# Backward ~~~~~~~~~~~~~~~~~~~~~~
			par_cosmo = self.fid_cosmo.copy()				

			try:
				step = self.steps[p]
			except:
				step = par_cosmo[p] * self.step	
				if par_cosmo[p] == 0:
					step = self.step
			# par_cosmo[p] = par_cosmo[p] - step/2.
			par_cosmo[p] = par_cosmo[p] - step
			
			if False:
				print '\t %3.2e' %par_cosmo[p]


			clsm = self._computeObservables(par_cosmo)

			if p == 'As':
				step = step * 1e9
			if p == 'H0':
				step = step / 100.

			# dcldp.append( (clsp - clsm)/ (step))		
			dcldp.append( (clsp - clsm)/ (2. * step))		

			del par_cosmo, clsp, clsm

		return dcldp

	def Fij(self, param_i, param_j, scaled=True):
		"""
		Returns the matrix element of the Fisher matrix for parameters
		param_i and param_j
		"""
		i = self.params.index(param_i)
		j = self.params.index(param_j)

		if scaled:
			return reduce(np.dot, [self.D, self.mat(scaled=True), self.D])[i, j]
		else:
			return self.mat()[i, j]

	def invFij(self, param_i, param_j, scaled=True):
		"""
		Returns the matrix element of the inverse Fisher matrix for
		parameters param_i and param_j
		"""
		i = self.params.index(param_i)
		j = self.params.index(param_j)

		if scaled:
			return reduce(np.dot, [self.D, self.invmat(scaled=True), self.D])[i, j]
		else:
			return self.invmat()[i, j]

	def sigma_fix(self, param):
		return 1.0 / np.sqrt(self.Fij(param, param))

	def sigma_marg(self, param, scaled=True):
		return np.sqrt(self.invFij(param, param, scaled=scaled))

	def _marginalise(self, params):
		""" 
		Marginalises the Fisher matrix over unwanted parameters.
		
		Parameters
		----------
		params: list
			List of parameters that should *not* be marginalised over.
		
		Returns
		-------
		(mat, invmat): ndarray
			Marginalised Fisher matrix and covariance
		"""
		# Builds covariance matrix
		marg_inv = np.zeros((len(params), len(params)))
		D_ = np.diag(np.ones(len(params)))#np.zeros((len(params), len(params)))
		for i in xrange(len(params)):
			indi = self.params.index(params[i])
			for j in xrange(len(params)):
				indj = self.params.index(params[j])
				marg_inv[i, j] = self.invmat(scaled=True)[indi, indj]
				D_[i, j] = self.D[indi, indj]

		marg_mat = reduce(np.dot, [D_, self.inv_routine(marg_inv), D_])
		marg_inv = reduce(np.dot, [D_, marg_inv, D_])

		return (marg_mat, marg_inv)

	@property
	def FoM_DETF(self):
		"""
			Computes the figure of merit from the Dark Energy Task Force
			Albrecht et al 2006
			FoM = 1/sqrt(det(F^-1_{w0,wa}))
		"""
		det = (self.invFij('w', 'w') * self.invFij('wa', 'wa') -
			   self.invFij('wa', 'w') * self.invFij('w', 'wa'))
		return 1.0 / np.sqrt(det)

	@property
	def FoM(self):
		"""
		Total figure of merit : ln (1/det(F^{-1}))
		"""
		return np.log(1.0 / abs(linalg.det(self.invmat)))

	# @property
	def invmat(self, scaled=False):
		"""
		Returns the inverse Fisher matrix (= covariance)
		"""
		if scaled:
			if self._invmat_scaled is None:
				self._invmat_scaled = self.inv_routine(self.mat(scaled=True))
			return self._invmat_scaled
		else:
			if self._invmat is None:
				self._invmat = self.inv_routine(self.mat)
			return self._invmat

	# @staticmethod
	def _rescaleFisherMatrix(self, F, simple=True, return_scaling_matrix=True):
		if simple:
			D = np.diag(1./np.sqrt(np.diag(F)))
			F_scaled = reduce(np.dot, [D, F, D]) # rescaled Fisher matrix
		else:
			# see https://www.cs.ubc.ca/~inutard/files/cs517-project.pdf
			T = np.tril(F)
			d = np.zeros(T.shape[0])
			for i in xrange(T.shape[0]):
			    # print np.sqrt(T[i,i]), np.dot(d[:i], T[i,:i])
			    d[i] = 1. / np.max( [np.sqrt(T[i,i]), np.dot(d[:i], T[i,:i])] )
			D = np.diag(d)
			F_scaled = reduce(np.dot, [D, F, D]) # rescaled Fisher matrix

		if return_scaling_matrix:
			return F_scaled, D
		else:
			return F_scaled

	# @property
	def mat(self, scaled=False, simple=True):
		"""
		Returns the Fisher matrix marginalised over nuisance parameters
		"""
		# If the Fisher matrix is not already computed, compute it
		if (self._mat is None) or (self._mat_scaled is None):
			self._fullMat = np.zeros((len(self.params),len(self.params)))
			for exp in self.exps.keys():
				print exp, '\n'
				# print self.fisher[exp].D, '\n'
				self._fullMat += self.fisher[exp].mat(scaled=False) #reduce(np.dot, [np.linalg.inv(self.fisher[exp].D), self.fisher[exp].mat, np.linalg.inv(self.fisher[exp].D)])

			_Priors = np.zeros((len(self.params),len(self.params)))
			for p in self.params:
				i = self.params.index(p)
				try:
					_Priors[i,i] = 1./self.priors[p]**2.
					print '\t...Including prior for', p, str(self.priors[p])
				except KeyError: 
					pass
			if (_Priors == 0).all():
				print '\t...No priors included...'

			# print _fullMat

			self._fullMat += _Priors

			self._fullMat_scaled, self.D_fullMat = self._rescaleFisherMatrix(self._fullMat, simple=simple, return_scaling_matrix=True) # rescaled Full (w/ margin params) Fisher matrix  
			self._fullInvMat_scaled = self.inv_routine(self._fullMat_scaled) # rescaled Full (w/ margin params) Covariance matrix 
			self._fullInvMat = reduce(np.dot, [self.D_fullMat, self._fullInvMat_scaled, self.D_fullMat]) # Full (w/ margin params) Covariance matrix 

			# Apply marginalisation over nuisance parameters 
			self._invmat = self._fullInvMat[0:len(self.params),0:len(self.params)]
			self._mat = self.inv_routine(self._invmat)
			self._mat_scaled, self.D = self._rescaleFisherMatrix(self._mat, simple=simple, return_scaling_matrix=True) # rescaled marginalizes Fisher matrix  
			self._invmat_scaled = self.inv_routine(self._mat_scaled)

		if scaled:
			return self._mat_scaled
		else:
			return self._mat

	def clearall(self):
		print("...!!! erasing Fisher matrix & covariance !!!...")
		self._fullMat = None
		self._fullInvMat = None
		self._mat = None
		self._invmat = None
		print("...done...")

	def corner_plot(self, nstd=1, labels=None, **kwargs):
		""" 
		Makes a corner plot including all the parameters in the Fisher analysis
		"""

		if labels is None:
			labels = self.params

		for i in xrange(len(self.params)):
			for j in range(i):
				ax = plt.subplot(len(self.params)-1, len(self.params)-1 , (i - 1)*(len(self.params)-1) + (j+1))

				self.plot(self.params[j], self.params[i], nstd=nstd, ax=ax, labels=False, **kwargs)

				if i == len(self.params) - 1:
					# ax.set_xlabel(labels[j])
					ax.set_xlabel(Labels[self.params[j]])
				else:
					ax.set_xticklabels([])
				if j == 0:
					# ax.set_ylabel(labels[i])
					ax.set_ylabel(Labels[self.params[i]])
				else:
					ax.set_yticklabels([])

		plt.subplots_adjust(wspace=0)
		plt.subplots_adjust(hspace=0)

	def ellipse_pars(self, p1, p2, howmanysigma=1):
		params = [p1, p2]

		def eigsorted(cov):
			vals, vecs = linalg.eigh(cov)
			order = vals.argsort()[::-1]
			return vals[order], vecs[:, order]

		mat, COV = self._marginalise(params)
		# print COV

		# First find the fiducial value for the parameter in question
		fid_param = None
		pos = [0, 0]
		for p in params:
			if p in dir(self.fid_surv):
				fid_param = self.fid_surv[p]
			else:
				fid_param = self.fid_cosmo[p]
				if p == 'As':
					fid_param *= 1e9 
				if p == 'H0':
					fid_param /= 100. 

			pos[params.index(p)] = fid_param

		vals, vecs = eigsorted(COV)
		theta = np.arctan2(*vecs[:, 0][::-1])
		theta = np.degrees(theta)

		assert COV.shape == (2,2)

		confsigma_dic = {1:2.3, 2:6.17, 3: 11.8}

		sig_x2, sig_y2 = COV[0,0], COV[1,1]
		sig_xy = COV[0,1]

		t1 = (sig_x2 + sig_y2)/2.
		t2 = np.sqrt( (sig_x2 - sig_y2)**2. /4. + sig_xy**2. )
		a = np.sqrt( abs(t1 + t2) )
		b = np.sqrt( abs(t1 - t2) )

		t1 = 2 * sig_xy
		t2 = sig_x2 - sig_y2

		theta = np.degrees(np.arctan2(t1,t2) / 2.)
		alpha = np.sqrt(confsigma_dic[howmanysigma])

		return pos, a*alpha, b*alpha, theta

	def plot(self, p1, p2, nstd=1, ax=None, howmanysigma=[1,2], labels=None, tag=None, **kwargs):
		""" 
		Plots confidence contours corresponding to the parameters provided.
		
		Parameters
		----------
		"""
		if ax is None:
			ax = plt.gca()

		for sigmas in howmanysigma: 
			pos, Aalpha, Balpha, theta = self.ellipse_pars(p1, p2, sigmas)

			ellip = Ellipse(xy=pos, width=Aalpha, height=Balpha, angle=theta, alpha=1./sigmas, **kwargs)

			ax.add_artist(ellip)
			# sz = np.max(width, height)
			s1 = 1.5*nstd*self.sigma_marg(p1)
			s2 = 1.5*nstd*self.sigma_marg(p2)
			ax.set_xlim(pos[0] - s1, pos[0] + s1)
			ax.set_ylim(pos[1] - s2, pos[1] + s2)
			#ax.set_xlim(pos[0] - sz, pos[0] + sz)
			#ax.set_ylim(pos[1] - sz, pos[1] + sz)

		if labels is None:
			ax.set_xlabel(Labels[p1])
			ax.set_ylabel(Labels[p2])
		elif labels is False:
			pass
		else:
			ax.set_xlabel(labels[0], size=14)
			ax.set_ylabel(labels[1], size=14)

		if tag is not None:
			ax.legend([ellip], tag)

		ax.plot(pos, 'w+', mew=2.)
		plt.draw()
		return ellip

class FisherPairwise(Fisher):

	def __init__(self, fid_cosmo, 
					   fid_surv, 
					   params, 
   					   margin_params=[],  
					   priors={}, 
					   steps={}, 
					   cv=None, 
					   cs=None, 
					   cm=None, 
					   cov=None,
					   planck_prior=False,
					   cmb_prior=None,
					   inv_routine=np.linalg.inv,
					   dvdp=None,
					   verbose=False):#, margin_params=[]):
		"""
		Constructor
		* fid_cosmo : dictionary (can be composed by more params than the one to forecast/marginalize)
		* fid_survey : dictionary => {M_min, fsky, sigma_v, zmin, zmax, Nz, rmin, rmax, Nr}
		* params : list of Fisher analysis parameters
		* cmb_prior: dictionary with CMB related stuff 
		"""
		super(FisherPairwise, self).__init__(fid_cosmo, fid_surv, params, margin_params, priors, steps, inv_routine, verbose)

		self.planck_prior = planck_prior
		# self.step = 0.003
		# self.fid_cosmo = fid_cosmo.copy()
		# self.fid_surv = fid_surv.copy()
		# self.params = []
		# self.priors = {}
		# self.steps  = {}

		# Few checks on survey params -> initialize to default value if not present
		for key, val in default_pw_survey_dict.iteritems():
			self.fid_surv.setdefault(key,val)
			setattr(self, key, self.fid_surv[key]) 
			# print key, self.fid_surv[key]

		if self.fid_surv['b_tau'] is None:
			for i in xrange(self.fid_surv['Nz']):
				self.fid_surv['b_tau_'+str(i)] = 1.

		# Check that the parameters provided are present in survey or cosmo
		for p in params:
			# First find the fiducial value for the parameter in question
			if p in self.fid_surv.keys() or p in self.fid_cosmo.keys():
				self.params.append(p)
			else:
				print("Warning, unknown parameter in derivative :" + p)

		# Check and store priors
		for key, val in priors.iteritems():
			if key in self.params or key in self.margin_params:
				if key != 'b_tau':
					self.priors[key] = val
				else:
					for i in xrange(self.fid_surv['Nz']):
						self.priors['b_tau_'+str(i)] = val

		# Check and store steps
		for key, val in steps.iteritems():
			if key in self.params or key in self.margin_params:
				if key != 'b_tau':
					self.steps[key] = val
				else:
					for i in xrange(self.fid_surv['Nz']):
						self.steps['b_tau_'+str(i)] = val

		if 'b_tau' in self.margin_params:
			del self.margin_params[self.margin_params.index('b_tau')]
			for i in xrange(self.fid_surv['Nz']):
				self.margin_params.append('b_tau_'+str(i))

		# print self.fid_surv
		# print self.params

		if '100theta' in self.params:
			fid_cosmo['H0'] = GuessH0(fid_cosmo['100theta'], params=fid_cosmo)

		# Create a Cosmo object with a copy of the fiducial cosmology
		self.cosmo = Cosmo(params=fid_cosmo)

		# Create a BinPairwise object initializied w/ fiducial cosmo + survey
		print("...Computing fiducial pairwise...")
		self.pw = BinPairwise(self.cosmo, self.zmin, self.zmax, self.Nz, self.rmin, self.rmax, self.Nr, fsky=self.fsky, M_min=self.M_min, M_max=self.M_max)
		print("...done...")

		# Precompute covariance matrices
		print("...Computing covariance matrix...")
		if cov is None:
			if cv is None:
				self.cov_cosmic = self.pw.Cov_cosmic()
			else:
				self.cov_cosmic = cv
			if cs is None:
				self.cov_gauss_shot = self.pw.Cov_gauss_shot()
			else:
				self.cov_gauss_shot = cs
			if cm is None:
				# self.cov_meas = self.pw.Cov_meas(self.sigma_v) # FIXME: make sigma_v dependent on other params such as tau
				self.cov_meas = {}
				for idz in xrange(self.pw.Nz):
					self.cov_meas[idz] = self.pw.Cov_meas(self.pw.GetSigmaV(self.fid_surv['noise_uK_arcmin'], self.fid_surv['fwhm_arcmin'], self.pw.zmean[idz], theta_R=self.fid_surv['fwhm_arcmin'], fg=self.fid_surv['fg']))[idz]

				# self.cov_meas = self.pw.Cov_meas(self.pw.GetSigmaV(self.noise_uK_arcmin, self.fwhm_arcmin, z, theta_R=self.fwhm_arcmin))[idz] # FIXME: make sigma_v dependent on other params such as tau
			else:
				self.cov_meas = cm

			self.cov = {i : self.cov_cosmic[i] + self.cov_gauss_shot[i] + self.cov_meas[i] for i in xrange(self.Nz)}
			print("...done...")
		else:
			self.cov = cov
			print("...covariance matrix loaded...")

		self.inv_cov = {i : inv_routine(self.cov[i]) for i in xrange(self.Nz)}
		# self.cov = linalg.block_diag(*cov.values())
		# self.inv_cov = linalg.pinv2(self.cov)

		# Deprecated !
		if self.planck_prior:
			print("...Calculating CMB priors...")
			self.params_cmb = copy.copy(self.params)
			bad = []
			for p in self.params:
				if (p in self.fid_surv.keys()) or (p in ['gamma0', 'gammaa']):
					bad.append(p)
					# self.params_cmb.remove(p)

			for b in bad:
				self.params_cmb.remove(b)

			if self.verbose:
				print self.params_cmb

			self.fishycmb = FisherCMB(fid_cosmo=self.fid_cosmo.copy(),\
			                     fid_surv={'fsky':.5, 
			                     		   'DeltaT':[28.6,45.], 
			                     		   'DeltaP':[40.,64.],
			                     		   'fwhm':[5.,7.0], 
			                     		   'lminT':30, 
			                     		   'lmaxT':2500,
			                     		   'lminP':30, 
			                     		   'lmaxP':2500},\
			                     steps=self.steps.copy(), \
			                     params=copy.copy(self.params_cmb))#,\
			                     # margin_params=) 

			self.fcmb = self.fishycmb.mat()#reduce(np.dot, [np.linalg.inv(self.fishycmb.D), self.fishycmb.mat, np.linalg.inv(self.fishycmb.D)]) 
			print("...done...") # TODO: make it general!
		else:
			self.fcmb = 0.

		# CMB priors ~~~~~~~~~~~~~~~~~~~~~~~~~~
		if cmb_prior:
			self.fcmb = cmb_prior
		else:
			self.fcmb = 0.

		# Compute derivatives
		if dvdp is not None:
			self._dvdp = dvdp
			print("...derivatives loaded...")
		else:
			print("...Computing derivatives...")
			self._dvdp = self._computeDerivatives()
			print("...done...")

		# Precomputed Fisher matrix
		self._fullMat = None
		self._fullInvMat = None
		self._mat = None
		self._invmat = None

	def _computeObservables(self, par_cosmo, par_sur):
		if '100theta' in par_cosmo:
			par_cosmo['H0'] = GuessH0(par_cosmo['100theta'], par_cosmo)

		if par_cosmo == self.fid_cosmo:
			_cosmo = copy.copy(self.cosmo)
		else:
			_cosmo = Cosmo(params=par_cosmo)

		_pw = BinPairwise(_cosmo, par_sur['zmin'], par_sur['zmax'], par_sur['Nz'], par_sur['rmin'], \
						  par_sur['rmax'], par_sur['Nr'], fsky=par_sur['fsky'], M_min=par_sur['M_min'])

		result = _pw.V_Delta.copy()

		for idz in xrange(self.fid_surv['Nz']):
			result[idz] = par_sur['b_tau_'+str(idz)] * result[idz]

		return result

	def _computeFullMatrix(self):
		if self._dvdp is None:
			print("...Computing derivatives...")
			self._dvdp = self._computeDerivatives()
			print("...done...")

		nparams = len(self._dvdp[0])#len(self.params)
		
		# print self._dvdp
		# print self.inv_cov[0].shape

		_fullMat = np.zeros((nparams,nparams))

		print("...Computing Full Fisher matrix...")
		# Computes the fisher Matrix ~~~~~~~~~~~~~
		for i in xrange(nparams):
			for j in xrange(i+1):
				tmp = 0
				for idz in xrange(self.Nz):
					tmp = tmp + np.dot(self._dvdp[idz][i], np.dot(self.inv_cov[idz], self._dvdp[idz][j]))	
				_fullMat[i,j] = tmp
				_fullMat[j,i] = _fullMat[i,j]
				del tmp

		# Priors ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		_Priors = np.zeros((nparams,nparams))
		allparams = self.params + self.margin_params
		for p in allparams:
			i = allparams.index(p)
			try:
				_Priors[i,i] = 1./self.priors[p]**2.
				print '---Including prior for',p,str(self.priors[p])
			except KeyError:
				pass
		if (_Priors == 0).all():
			print '---No prior included'

		_fullMat += _Priors

		# if self.planck_prior:
		# 	print '---Including Planck priors'
		# 	for i in xrange(nparams):
		# 		for j in xrange(i+1):
		# 			par_i = self.params[i]
		# 			par_j = self.params[j]  
		# 			if (par_i in self.params_cmb) and (par_j in self.params_cmb):
		# 				print par_i, par_j
		# 				_fullMat[i,j] += self.fcmb[self.params_cmb.index(par_i),self.params_cmb.index(par_j)] 
		# 				_fullMat[j,i] = _fullMat[i,j]

		# if self.fcmb != 0.:
		# 	print '---Including CMB priors'
		# 	for i in xrange(nparams):
		# 		for j in xrange(i+1):
		# 			par_i = self.params[i]
		# 			par_j = self.params[j]  
		# 			if (par_i in self.fcmb.params) and (par_j in self.fcmb.params):
		# 				# print par_i, par_j
		# 				_fullMat[i,j] += self.fcmb.mat(scaled=False)[self.fcmb.params.index(par_i),self.fcmb.params.index(par_j)] 
		# 				_fullMat[j,i] = _fullMat[i,j]

		# CMB priors ~~~~~~~~~~~~~~~~~~~~~~~~~~
		if self.fcmb != 0.: 
			print '---Including CMB priors'
			for i in xrange(len(allparams)):
				for j in xrange(i+1):
					par_i = allparams[i]
					par_j = allparams[j]  
					if (par_i in self.fcmb.params) and (par_j in self.fcmb.params):
						# print par_i, par_j
						_fullMat[i,j] += self.fcmb.mat(scaled=False)[self.fcmb.params.index(par_i),self.fcmb.params.index(par_j)] 
						_fullMat[j,i] = _fullMat[i,j]

		return _fullMat 

	def _computeDerivatives(self):
		""" 
		Computes all the derivatives of the specified observable with
		respect to the parameters and nuisance parameters in the analysis
		"""
		
		# List the derivatives with respect to all the parameters
		dvdp = {}
		for idz in xrange(self.Nz):
			dvdp[idz] = []

		# Computes all the derivatives with respect to the main parameters
		for p in self.params + self.margin_params:
			if self.verbose:
				print("varying :" + p)

			# Forward ~~~~~~~~~~~~~~~~~~~~~~
			par_sur = self.fid_surv.copy()				
			par_cosmo = self.fid_cosmo.copy()				

			if p in self.fid_surv.keys():
				try:
					step = self.steps[p]
				except:
					step = par_sur[p] * self.step		
					if par_sur[p] == 0:
						step = self.step
				par_sur[p] = par_sur[p] + step
				
				if self.verbose:
					print par_sur[p]

			elif p in self.fid_cosmo.keys():
				try:
					step = self.steps[p]
				except:
					step = par_cosmo[p] * self.step		
					if par_cosmo[p] == 0:
						step = self.step
				par_cosmo[p] = par_cosmo[p] + step

				if self.verbose:
					print par_cosmo[p]

			Vp = self._computeObservables(par_cosmo, par_sur)

			del par_sur, par_cosmo

			# Backward ~~~~~~~~~~~~~~~~~~~~~~
			par_sur = self.fid_surv.copy()				
			par_cosmo = self.fid_cosmo.copy()				

			if p in self.fid_surv.keys():
				try:
					step = self.steps[p]
				except:
					step = par_sur[p] * self.step		
					if par_sur[p] == 0:
						step = self.step
				par_sur[p] = par_sur[p] - step

			if self.verbose:
				print par_sur[p]

			elif p in self.fid_cosmo.keys():
				try:
					step = self.steps[p]
				except:
					step = par_cosmo[p] * self.step		
					if par_cosmo[p] == 0:
						step = self.step
				par_cosmo[p] = par_cosmo[p] - step
				
				if self.verbose:
					print par_cosmo[p]

			Vm = self._computeObservables(par_cosmo, par_sur)

			if p == 'As':
				step = step * 1e9

			if p == 'H0':
				step = step / 100.

			for idz in xrange(self.Nz):
				dvdp[idz].append( (Vp[idz] - Vm[idz])/ (2.*step) )

			del par_sur, par_cosmo, Vp, Vm

		return dvdp
