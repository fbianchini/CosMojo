import numpy as np
from scipy import linalg
from astropy import constants as const
from defaults import *
from utils import nl_cmb
from universe import Cosmo
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class FisherCMB(object):


	def __init__(self, fid_cosmo, fid_surv, params, obs=['TT','EE','TE'],  priors={}, steps={}):#, margin_params=[]):
		"""
		Constructor
		* fid_cosmo : dictionary (can be composed by more params than the one to forecast/marginalize)
		* fid_survey : dictionary => {\Delta_T, l_knee, fsky, lminT, lminP, lmaxT, lmaxP, lminK, lmaxK}
		* params : list of Fisher analysis parameters
		"""

		self.step = 0.003
		self.fid_cosmo = fid_cosmo.copy()
		self.fid_surv = fid_surv.copy()
		self.params = []
		self.priors = {}
		self.steps  = {}
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

		# Check and store priors
		for key, val in priors.iteritems():
			if key in self.params:
				self.priors[key] = val

		# Check and store steps
		for key, val in steps.iteritems():
			if key in self.params:
				self.steps[key] = val

		print self.fid_surv
		print self.params

		# Checks that the marginalisation parameters are actually considered
		# in the Fisher analysis
		# self.margin_params = []
		# for p in margin_params:
		# 	if p in self.fid_surv.keys() or p in self.fid_cosmo.keys():
		# 		self.margin_params.append(p)
		# 	else:
		# 		print("Warning, requested marginalisation parameter " + p + " is not included in the analysis")

		# Create a Cosmo object with a copy of the fiducial cosmology
		self.cosmo = Cosmo(params=fid_cosmo)

		# Multipoles range
		self.lmax = np.max([self.lmaxT, self.lmaxP, self.lmaxK])
		self.lmin = np.min([self.lminT, self.lminP, self.lminK])
		self.lrange = np.arange(self.lmin, self.lmax+1)
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
			l_, nlkk_ = np.loadtxt('nlkk_planck2015.dat', unpack=True)
			self.NlKK[:nlkk_.size] = nlkk_
			self.NlKK[:self.lminK] = 1.e40
			self.NlKK[self.lmaxK+1:] = 1.e40

		# print self.NlTT
		# print self.NlPP

		# Compute fiducial CMB power spectra w/ fiducial cosmo + survey
		print("...Computing fiducial CMB power spectra...")
		self.cls = self.cosmo.cmb_spectra(self.lmax, spec='lensed_scalar', dl=False)
		print("...done...")

		# Precomputed Fisher matrix
		self._fullMat = None
		self._fullInvMat = None
		self._mat = None
		self._invmat = None

	def _computeCovariance(self, l):

		TT = self.cls[l,0] + self.NlTT[l]
		EE = self.cls[l,1] + self.NlPP[l]
		TE = self.cls[l,3]
		KK = self.cls[l,4] + self.NlKK
		KT = self.cls[l,5]

		f = 0.5

		if set(self.obs) == set(['TT','TE','EE']):
			mat = 2./((2*l+1)*self.fsky) * \
				  np.asarray([[TT**2., TE**2., TT*TE           ],
							  [TE**2., EE**2., EE*TE           ],
							  [TT*TE,  EE*TE,  f*(TE**2.+TT*EE)]])

		if set(self.obs) == set(['TT','TE','EE', 'KK', 'KT']):
			mat = 2./((2*l+1)*self.fsky) * \
					np.array([[TT**2., TE**2., TT*TE,            KT*KT],
			                  [TE**2., EE**2., EE*TE,            0.   ],
			                  [TT*TE,  EE*TE,  f*(TE**2.+TT*EE), 0.   ],
			                  [KT*KT,  0.,     0.,               KK*KK]])

		elif self.obs == ['TT']:
			mat = 2./((2*l+1)*self.fsky) * TT**2

		elif self.obs == ['EE']:
			mat = 2./((2*l+1)*self.fsky) * EE**2

		elif self.obs == ['TE']:
			mat = 2./((2*l+1)*self.fsky) * f * (TE**2.+TT*EE)

		elif self.obs == ['KK']:
			mat = 2./((2*l+1)*self.fsky) * KK**2

		return mat

	def _computeObservables(self, par_cosmo):
		return Cosmo(params=par_cosmo).cmb_spectra(self.lmax, spec='lensed_scalar', dl=False)

	def _computeFullMatrix(self):
		print("...Computing derivatives...")
		self._dcldp = self._computeDerivatives()
		# nparams = len(self._dvdp)
		nparams = len(self.params)
		
		# print self._dvdp
		# print self.inv_cov[0].shape

		_fullMat = np.zeros((nparams,nparams))

		print("Computing Full Fisher matrix")
		# Computes the fisher Matrix
		for i in xrange(nparams):
			for j in xrange(i+1):
				tmp = 0
				for l in self.lrange:
					cov = self._computeCovariance(l)
					
					if set(self.obs) == set(['TT','TE','EE']):
						inv_cov = linalg.pinv2(cov)
						dcl_i = np.array([self._dcldp[i][l,0], self._dcldp[i][l,1], self._dcldp[i][l,3]])
						dcl_j = np.array([self._dcldp[j][l,0], self._dcldp[j][l,1], self._dcldp[j][l,3]])
						tmp += np.nan_to_num(np.dot(dcl_i, np.dot(inv_cov, dcl_j)))	
					
					elif set(self.obs) == set(['TT','TE','EE', 'KK', 'KT']):
						inv_cov = linalg.pinv2(cov)
						dcl_i = np.array([self._dcldp[i][l,0], self._dcldp[i][l,1], self._dcldp[i][l,3], self._dcldp[i][l,4], self._dcldp[i][l,5]])
						dcl_j = np.array([self._dcldp[j][l,0], self._dcldp[j][l,1], self._dcldp[j][l,3], self._dcldp[j][l,4], self._dcldp[j][l,5]])
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

				_fullMat[i,j] = tmp
				_fullMat[j,i] = _fullMat[i,j]
				del tmp

		_Priors = np.zeros((nparams,nparams))
		for p in self.params:
			i = self.params.index(p)
			try:
				_Priors[i,i] = 1./self.priors[p]**2.
				print '\t...Including prior for', p, str(self.priors[p])
			except KeyError: 
				pass
		if (_Priors == 0).all():
			print '\t...No priors included...'

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
		for p in self.params:
			print("varying :" + p)

			# Forward ~~~~~~~~~~~~~~~~~~~~~~
			par_cosmo = self.fid_cosmo.copy()				

			try:
				step = self.steps[p]
			except:
				step = par_cosmo[p] * self.step		
				if par_cosmo[p] == 0:
					step = self.step

			par_cosmo[p] = par_cosmo[p] + step
			print '\t %f' %par_cosmo[p]

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
			par_cosmo[p] = par_cosmo[p] - step
			print '\t %f' %par_cosmo[p]


			clsm = self._computeObservables(par_cosmo)

			if p == 'As':
				step = step * 1e9

			if p == 'w': 
				dcldp.append( (clsp - clsm)/ (2.0 * step))
			else:
				dcldp.append( (clsp - clsm)/ (2.0 * step))		

			del par_cosmo, clsp, clsm

		return dcldp

	def Fij(self, param_i, param_j):
		"""
		Returns the matrix element of the Fisher matrix for parameters
		param_i and param_j
		"""
		i = self.params.index(param_i)
		j = self.params.index(param_j)

		return self.mat[i, j]

	def invFij(self, param_i, param_j):
		"""
		Returns the matrix element of the inverse Fisher matrix for
		parameters param_i and param_j
		"""
		i = self.params.index(param_i)
		j = self.params.index(param_j)

		return self.invmat[i, j]

	def sigma_fix(self, param):
		return 1.0 / np.sqrt(self.Fij(param, param))

	def sigma_marg(self, param):
		return np.sqrt(self.invFij(param, param))

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

		newFisher._invmat = linalg.pinv2(newFisher._mat)

		return newFisher

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
			Marginalised Fisher matrix and its invers
		"""
		# Builds inverse matrix
		marg_inv = np.zeros((len(params), len(params)))
		for i in xrange(len(params)):
			indi = self.params.index(params[i])
			for j in xrange(len(params)):
				indj = self.params.index(params[j])
				marg_inv[i, j] = self.invmat[indi, indj]

		marg_mat = linalg.pinv2(marg_inv)

		return (marg_mat, marg_inv)

	# @property
	# def FoM_DETF(self):
	# 	"""
	# 		Computes the figure of merit from the Dark Energy Task Force
	# 		Albrecht et al 2006
	# 		FoM = 1/sqrt(det(F^-1_{w0,wa}))
	# 	"""
	# 	det = (self.invFij('w0', 'w0') * self.invFij('wa', 'wa') -
	# 		   self.invFij('wa', 'w0') * self.invFij('w0', 'wa'))
	# 	return 1.0 / sqrt(det)

	@property
	def FoM(self):
		"""
			Total figure of merit : ln (1/det(F^{-1}))
		"""
		return np.log(1.0 / abs(linalg.det(self.invmat)))

	@property
	def invmat(self):
		"""
		Returns the inverse fisher matrix
		"""
		if self._invmat is None:
			self._invmat = linalg.pinv2(self.mat)
		return self._invmat

	@property
	def mat(self):
		"""
		Returns the fisher matrix marginalised over nuisance parameters
		"""
		# If the matrix is not already computed, compute it
		if self._mat is None:
			self._fullMat = self._computeFullMatrix()
			self._fullInvMat = linalg.pinv2(self._fullMat)

			# Apply marginalisation over nuisance parameters ! ! ! FIXME: USELESS
			self._invmat = self._fullInvMat[0:len(self.params),0:len(self.params)]
			self._mat = linalg.pinv2(self._invmat)

		return self._mat
				
	def corner_plot(self, nstd=2, labels=None, **kwargs):
		""" 
		Makes a corner plot including all the parameters in the Fisher analysis
		"""

		if labels is None:
			labels = self.params

		for i in xrange(len(self.params)):
			for j in range(i):
				ax = plt.subplot(len(self.params)-1, len(self.params)-1 , (i - 1)*(len(self.params)-1) + (j+1))
				if i == len(self.params) - 1:
					ax.set_xlabel(labels[j])
				else:
					ax.set_xticklabels([])
				if j == 0:
					ax.set_ylabel(labels[i])
				else:
					ax.set_yticklabels([])

				self.plot(self.params[j], self.params[i], nstd=nstd, ax=ax, **kwargs)

		plt.subplots_adjust(wspace=0)
		plt.subplots_adjust(hspace=0)

	def ellipse_pars(self, p1, p2, howmanysigma=1):
		params = [p1, p2]

		def eigsorted(cov):
			vals, vecs = linalg.eigh(cov)
			order = vals.argsort()[::-1]
			return vals[order], vecs[:, order]

		mat, COV = self._marginalise(params)

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


	def plot(self, p1, p2, nstd=1, ax=None, howmanysigma=[1,2], labels=False, **kwargs):
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
			ax.set_xlabel(p1)
			ax.set_ylabel(p2)

		plt.plot(pos, 'r+', mew=2.)
		plt.draw()
		return ellip
