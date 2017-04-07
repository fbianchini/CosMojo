import numpy as np
from scipy import interpolate, integrate, linalg, special
from astropy import constants as const
from defaults import *
from utils import V_bin, W_Delta, nl_cmb
from mass_func import MassFunction
from corr_func import CorrFunction
from universe import Cosmo
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class Pairwise(object):
	"""
	Class to compute the pairwise velocity as function of redshift and comoving separation

	Attributes
	----------
	cosmo : Cosmo object (from universe.py)
	    Cosmology object

	"""
	def __init__(self, cosmo):
		self.cosmo = cosmo
		self.hmf = MassFunction(self.cosmo)


	def v12(self, r, z, b=None, M_min=None, M_max=None): #[km/s]
		"""
		Returns the linear theory pairwise velocity at redshift z and as function
		of the pair separation 

		Notes
		-----
		As given in Eq. 8 of Sheth et al. 2001. Given by
		
		v_{12}(R, z) = [-2/3 * f * H * a ] * b * r \bar{xi}(r,z) / (1 + b^2 xi(r,z) )

		Parameters
		----------
		r : {float, array_like}
			The separations to compute v12 at [units :math: `Mpc`]
		
		z : float
			The redshift to compute the function at

		b : float
			Halo bias (if None, it is computed from mass range)
		
		M_min : float
			The minimum halo mass used to compute the average halo bias
		
		M_max : float
			The maximum halo mass used to compute the average halo bias

		Returns
		-------
		v12 : {float, array_like}
			The pairwise velocity in km/s
		"""
		if b is None:
			b = self.hmf.bias_avg(z=z, M_min=M_min, M_max=M_max)
		H = self.cosmo.H_z(z)
		f = self.cosmo.f_z(z)
		a = 1./(1.+z)

		cf  = CorrFunction(self.cosmo, z=z, r_min=r.min(), r_max=r.max())
		xi  = cf.xi(r)
		xib = cf.xi_bar(r) 

		return -2./3. * a * H * f * r * (b*xib)/(1+b**2*xi) 

	def v12_LO(self, r, z, b=None, M_min=None, M_max=None): #[km/s]
		"""
		Returns the lowest-order mean pairwise velocity 

		Notes
		-----
		Given by:

		v_{12}(R, z) = [-2/3 * f * H * a ] * b * r \bar{xi}(r,z) 

		Parameters
		----------
		r : {float, array_like}
			The separations to compute v12 at [units :math: `Mpc`]
		
		z : float
			The redshift to compute the function at
		
		b : float
			Halo bias (if None, it is computed from mass range)

		M_min : float
			The minimum halo mass used to compute the average halo bias
		
		M_max : float
			The maximum halo mass used to compute the average halo bias

		Returns
		-------
		v12 : {float, array_like}
			The lowest-order pairwise velocity in km/s
		"""
		if b is None:
			b = self.hmf.bias_avg(z=z, M_min=M_min, M_max=M_max)
		H = self.cosmo.H_z(z)
		f = self.cosmo.f_z(z)
		a = 1./(1.+z)

		cf  = CorrFunction(self.cosmo, z=z, r_min=r.min(), r_max=r.max())
		xib = cf.xi_bar(r) 

		return -2./3. * a * H * f * r * (b*xib) 

	def T_pkSZ(self, r, z, b=None, tau=3e-3, sigma_r=None, sigma_dc=None, M_min=None, M_max=None): # [K]
		"""
		Pairwise kSZ signal as given by Eq. 11 of Soergel+16 (arXiv:1603.039042)

		Parameters
		----------
		r : {float, array_like}
			The separations to compute v12 at [units :math: `Mpc`]
		
		z : float
			The redshift to compute the function at
		
		b : float
			Halo bias (if None, it is computed from mass range)

		tau : float
			Effective optical depth

		sigma_r : float
			Pair separation smoothing scale (if None, checks if sigma_dc is not None, else 0)

		sigma_dc : float
			RMS uncertainty in the comoving distance due to photo-z errors ()

		M_min : float
			The minimum halo mass used to compute the average halo bias
		
		M_max : float
			The maximum halo mass used to compute the average halo bias

		Returns
		-------
		T_pkSZ : {float, array_like}
			Pairwise kSZ signal in [K]
		"""
		if sigma_r is None:
			if sigma_dc is None:
				sigma_r = 0.
			else:
				sigma_r = np.sqrt(2) * sigma_dc

		return tau * self.cosmo.pars.TCMB/const.c.to('km/s') * self.v12(r, z, b=b, M_min=M_min, M_max=M_max) * (1.-np.exp(-r**2/(2.*sigma_r**2)))


class BinPairwise(object):
	"""
	Class to encapsulate a binned observation of the pairwise velocity 

	Attributes
	----------
	cosmo : Cosmo object (from universe.py)
	    Cosmology object
	
	z_min : float
		Minimum redshift of LSS survey 

	z_max : float
		Maximum redshift of LSS survey 

	Nz : int
		Number of redshift bins

	rmin : float
		Minimum comoving separation

	rmax : float
		Maximum comoving separation
	
	Nr : int
		Number of separation bins

	fsky : float
		Sky fraction observed
	
	M_min : float
		The minimum halo mass used to compute the average halo bias
	
	M_max : float
		The maximum halo mass used to compute the average halo bias
	"""
	def __init__(self, cosmo, zmin, zmax, Nz, rmin, rmax, Nr, fsky=1., M_min=None, M_max=None, kmin=None, kmax=None, verbose=False):
		self.cosmo = cosmo
		self.fsky  = fsky
		self.M_min = M_min
		self.M_max = M_max
		self.verbose = verbose

		if kmin is None:
			kmin = self.cosmo.kmin
		self.kmin = kmin
		if kmax is None:
			kmax = self.cosmo.kmax
		self.kmax = kmax

		assert ( (zmin > 0) and (zmax > 0) and (zmin < zmax) and (Nz > 0))
		assert ( (rmin > 0) and (rmax > 0) and (rmin < rmax) and (Nr > 0))
		assert ( (kmin > 0) and (kmax > 0) and (kmin < kmax) )

		self.zmin = zmin
		self.zmax = zmax
		self.Nz = Nz
		self.zbin_edges = np.linspace(self.zmin, self.zmax, self.Nz+1)
		self.zmean = (self.zbin_edges[1:] + self.zbin_edges[:-1])/2.
		self.dz = self.zbin_edges[1] - self.zbin_edges[0]

		self.rmin = rmin
		self.rmax = rmax
		self.Nr = Nr
		self.rbin_edges = np.linspace(self.rmin, self.rmax, self.Nr+1)
		self.rmean = (self.rbin_edges[1:] + self.rbin_edges[:-1])/2.
		self.dr = self.rbin_edges[1] - self.rbin_edges[0]

		self.initialized_xis = False
		self.initialized_hmf = False
		self.initialized_v12 = False
		self.initialized_V_delta = False
		self.initialized_n_pair = False

		self._initialize_volumes()
		self._initialize_hmf()
		self._initialize_xis()
		self._initialize_v12()
		self._initialize_V_delta()
		self._initialize_n_pair()

		# super(BinPairwise, self).__init__(cosmo)

	def _initialize_volumes(self):
		"""
		Initializer of volume related quantities such as 
		- survey volume @ different z
		- separation bin spherical shell
		"""
		if self.verbose:
			print("...calcultating volumes...")
		
		# Calculate survey volumes at different redshifts
		self.V_sur = np.zeros(self.Nz)
		for idz in xrange(self.Nz):
			self.V_sur[idz] = self.cosmo.V_survey(self.zbin_edges[idz+1], zmin=self.zbin_edges[idz], fsky=self.fsky)
			# print idz, self.V_sur[idz]

		# Calculate separation bin volumes
		self.V_bin = np.zeros(self.Nr)
		for idr, r in enumerate(self.rmean):
			self.V_bin[idr] = V_bin(r, self.dr) # Mpc^3
			# print self.V_bin[idr]

		if self.verbose:
			print("...done...")

	def _initialize_hmf(self):
		"""
		Halo mass function initializer
		"""
		if self.verbose:
			print("...calcultating HMF quantities...")
		
		self.nhalo = np.zeros(self.Nz)
		self.bias  = np.zeros(self.Nz)

		hmf = MassFunction(self.cosmo)

		# Loop over z-bins
		for idz, z in enumerate(self.zmean):
			self.nhalo[idz] = hmf.n_cl(z=z, M_min=self.M_min, M_max=self.M_max) #* self.dz # FIXME: multiply by dz ~ integration over z
			self.bias[idz]  = hmf.bias_avg(z=z, M_min=self.M_min, M_max=self.M_max)
		
		self.initialized_hmf = True	
		
		if self.verbose:
			print("...done...")

	def _initialize_xis(self):
		"""
		Matter correlation functions initializer
		"""
		if self.verbose:
			print("...calcultating Correlation functions xi(r,z)...")
		
		self.xi = {}
		self.xi_bar = {}
		
		# Loop over z-bins
		for idz, z in enumerate(self.zmean):
			cf = CorrFunction(self.cosmo, z=z, r_max=400) # r_max HARDCODED
			self.xi[idz] = cf.xi
			self.xi_bar[idz] = cf.xi_bar
		
		self.initialized_xis = True
		
		if self.verbose:
			print("...done...")

	def _initialize_v12(self):
		"""
		Theoretical pairwise velocity initializer
		"""
		if not self.initialized_xis:
			self._initialize_xis()
		if not self.initialized_hmf:
			self._initialize_hmf()

		if self.verbose:
			print("...calcultating pairwise velocity v_12(r,z)...")

		self.v_12 = {}
		self.aHf  = {}

		# Loop over z-bins
		for idz, z in enumerate(self.zmean):
			H = self.cosmo.H_z(z)
			f = self.cosmo.f_z(z)
			a = 1./(1.+z)
			self.aHf[idz] = a * H * f
			self.v_12[idz] = lambda r, i=idz: -2./3. * self.aHf[i] * r * ( self.bias[i] * self.xi_bar[i](r) ) / ( 1. + self.bias[i]**2 * self.xi[i](r) ) 
		
		self.initialized_v12 = True
	
		if self.verbose:
			print("...done...")

	def _initialize_V_delta(self):
		"""
		Binned pairwise velocity initializer
		"""
		if self.verbose:
			print("...calcultating volume bin averaged V(r,z)...")

		if not self.initialized_v12:
			self._initialize_v12()
	
		self.V_Delta = {}

		# Loop over z-bins
		for idz in xrange(self.Nz):
			self.V_Delta[idz] = np.zeros(self.Nr)
			for idr in xrange(self.Nr):
				self.V_Delta[idz][idr] = 4.*np.pi / self.V_bin[idr] * integrate.quad(lambda r: self.v_12[idz](r) * r**2 , self.rbin_edges[idr], self.rbin_edges[idr+1], epsabs=0.0, epsrel=1e-5, limit=100)[0]				
		self.initialized_V_delta = True

		if self.verbose:
			print("...done...")

	def _initialize_n_pair(self):
		"""
		Calculates the number of pairs as function of the comoving separation
		"""
		if self.verbose:
			print("...calcultating N_pair...")

		if not self.initialized_V_delta:
			self._initialize_V_delta()
		self.n_pair = {}
		
		# Loop over z-bins
		for idz in xrange(self.Nz):
			self.n_pair[idz] = np.zeros(self.Nr)
			for idr, r in enumerate(self.rmean):
				# print self.nhalo[idz]
				# print self.V_sur[idz]
				# print self.V_Delta[idz][idr]
				# print self.rmean[idr]
				# print self.bias[idz]
				# print self.xi[idz](r)
				# print self.dr
				# print self.V_Delta[idz][idr] * self.nhalo[idz]
				# print 4.*np.pi * r**2 * self.nhalo[idz] * self.bias[idz]**2 * self.xi[idz](r) * self.dr

				self.n_pair[idz][idr] = self.nhalo[idz]**2 * self.V_sur[idz] * self.V_bin[idr] / 2. * \
									   (1. + self.bias[idz]**2 * self.xi[idz](r))

									   # (self.V_Delta[idz][idr] * self.nhalo[idz] + \
									   # 4.*np.pi * r**2 * self.nhalo[idz] * self.bias[idz]**2 * self.xi[idz](r) * self.dr)

		self.initialized_n_pair = True
		
		if self.verbose:
			print("...done...")

	def Cov_meas(self, sigma_v):
		"""
		Measurement error contribution to the covariance matrix.
		See Eq. 17 of Mueller+15a (arXiv:1408.6248)
		"""
		cov_meas = {}
		for idz in xrange(self.Nz):
			cov_meas[idz] = np.diag(2. * sigma_v**2 / self.n_pair[idz])

		return cov_meas

	def Cov_cosmic(self):
		"""
		Gaussian (cosmic variance) contribution to the covariance matrix.
		See Eq. 14 of Mueller+15a (arXiv:1408.6248)
		"""
		cov_cosmic = {}

		# Loop over redshift bins
		for idz, z in enumerate(self.zmean):
			cov_cosmic[idz] = np.zeros((self.Nr, self.Nr))
			fact = 4./(np.pi**2*self.V_sur[idz]) * ( self.cosmo.f_z(z) * self.cosmo.H_z(z) * self.bias[idz]/ (1+z))**2 # doesn't depend on separation r

			# Loop over separation bins
			for idr, r in enumerate(self.rmean):
				fact_ = fact / (1 + self.bias[idz]**2 * self.xi[idz](r))
				for idrp in xrange(self.Nr):
					fact__ = fact_ / (1 + self.bias[idz]**2 * self.xi[idz](self.rmean[idrp]))
					integral = integrate.quad(self.Cov_cosmic_integrand, self.kmin, self.kmax, 
											  args=(self.rbin_edges[idr], self.rbin_edges[idr+1], self.rbin_edges[idrp], self.rbin_edges[idrp+1], z, idz),  
						                      epsabs=0.0, epsrel=1e-5, limit=100)[0]
					cov_cosmic[idz][idr,idrp] = integral * fact__
					# cov_cosmic[idz][idrp,idr] = cov_cosmic[idz][idr,idrp]

		return cov_cosmic

	def Cov_gauss_shot(self):
		"""
		Gaussian shot-noise contribution to the covariance matrix.
		See Eq. 14 of Mueller+15a (arXiv:1408.6248)
		"""
		cov_gauss_shot = {}

		# Loop over redshift bins
		for idz, z in enumerate(self.zmean):
			cov_gauss_shot[idz] = np.zeros((self.Nr, self.Nr))
			fact = 4./(np.pi**2*self.V_sur[idz]) * ( self.cosmo.f_z(z) * self.cosmo.H_z(z)/((1+z) * self.nhalo[idz]))**2 # doesn't depend on separation r

			# Loop over separation bins
			for idr, r in enumerate(self.rmean):
				fact_ = fact / (1 + self.bias[idz]**2 * self.xi[idz](r))
				# for idrp in xrange(idr+1):
				for idrp in xrange(self.Nr):
					fact__ = fact_ / (1 + self.bias[idz]**2 * self.xi[idz](self.rmean[idrp]))
					integral = integrate.quad(self.Cov_gauss_shot_integrand, self.kmin, self.kmax, 
											  args=(self.rbin_edges[idr], self.rbin_edges[idr+1], self.rbin_edges[idrp], self.rbin_edges[idrp+1], idz), 
											  epsabs=0.0, epsrel=1e-5, limit=100)[0]
					cov_gauss_shot[idz][idr,idrp] = integral * fact__
					# cov_gauss_shot[idz][idrp,idr] = cov_gauss_shot[idz][idr,idrp]

		return cov_gauss_shot

	def Cov_poiss_shot(self):
		"""
		Poisson shot-noise contribution to the covariance matrix.
		See Eq. 14 of Mueller+15a (arXiv:1408.6248)
		!!! FIXME: not working yet !!!
		"""
		cov_poiss_shot = {}

		# Loop over redshift bins
		for idz, z in enumerate(self.zmean):
			cov_poiss_shot[idz] = np.zeros((self.Nr, self.Nr))
			fact = 4./(np.pi**2*self.V_sur[idz]) * ( 1/(1+z) * self.cosmo.f_z(z) * self.cosmo.H_z(z))**2 # doesn't depend on separation r

			# Loop over separation bins
			for idr, r in enumerate(self.rmean):
				fact_ = fact / (1 + self.bias[idz]**2 * self.xi[idz](r))**2 * self.dr
				for idrp in xrange(self.Nr):
				# for idrp in xrange(idr+1):
					fact__ = fact_ / self.V_Delta[idz][idrp]
					integral = integrate.quad(self.Cov_poiss_shot_integrand, self.kmin, self.kmax, 
						                      args=(self.rbin_edges[idr], self.rbin_edges[idr+1], z, idz), 
						                      epsabs=0.0, epsrel=1e-5, limit=200)
					# print integral[1]/integral[0], integral[0], idr, idrp
					cov_poiss_shot[idz][idr,idrp] = integral[0] * fact__
					# cov_poiss_shot[idz][idrp,idr] = cov_poiss_shot[idz][idr,idrp]

		return cov_poiss_shot

	def Cov_cosmic_integrand(self, k, Rmin, Rmax, Rpmin, Rpmax, z, idz):
		return self.cosmo.pkz.P(z, k)**2  * W_Delta(k, Rmin, Rmax) * W_Delta(k, Rpmin, Rpmax)

	def Cov_gauss_shot_integrand(self, k, Rmin, Rmax, Rpmin, Rpmax, idz):
		return W_Delta(k, Rmin, Rmax) * W_Delta(k, Rpmin, Rpmax)

	def Cov_poiss_shot_integrand(self, k, Rmin, Rmax, z, idz):
		return (k * self.cosmo.pkz.P(z, k) * self.bias[idz] /self.nhalo[idz]**2)  * W_Delta(k, Rmin, Rmax)

	def sigma_T_kSZ(self, theta_R, noise_uK_arcmin, fwhm_arcmin, lmax, lknee=1e-9, alpha=0.):
		theta_R = np.radians(theta_R/60.)
		cmb = self.cosmo.cmb_spectra(lmax=lmax)[2:,0]
		nl = nl_cmb(noise_uK_arcmin, fwhm_arcmin, lmax=lmax, lknee=lknee, alpha=alpha)[2:]
		C_ell = np.nan_to_num(cmb + nl) # + FG
		ells = np.arange(2, lmax+1)
		integral = integrate.simps(ells * C_ell * self.W_AP(ells, theta_R)**2, x=ells) 
		return np.sqrt( 2 * np.pi * theta_R**4 * integral)
		
	def W_AP(self, ell, theta_R):
		return (2.*special.jv(1, ell*theta_R) - np.sqrt(2)*special.jv(1, np.sqrt(2)*ell*theta_R)) / (ell * theta_R)

class FisherPairwise(object):


	def __init__(self, fid_cosmo, fid_surv, params, priors={}, steps={}, cv=None, cs=None, cm=None, cov=None):#, margin_params=[]):
		"""
		Constructor
		* fid_cosmo : dictionary (can be composed by more params than the one to forecast/marginalize)
		* fid_survey : dictionary => {M_min, fsky, sigma_v, zmin, zmax, Nz, rmin, rmax, Nr}
		* params : list of Fisher analysis parameters
		"""

		self.step = 0.003
		self.fid_cosmo = fid_cosmo.copy()
		self.fid_surv = fid_surv.copy()
		self.params = []
		self.priors = {}
		self.steps  = {}

		# Few checks on survey params -> initialize to default value if not present
		for key, val in default_pw_survey_dict.iteritems():
			self.fid_surv.setdefault(key,val)
			setattr(self, key, self.fid_surv[key]) 
			# print key, self.fid_surv[key]

		# Check that the parameters provided are present in survey or cosmo
		for p in params:
			# First find the fiducial value for the parameter in question
			if p in self.fid_surv.keys() or p in self.fid_cosmo.keys():
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

		# print self.fid_surv
		# print self.params

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

		# Create a BinPairwise object initializied w/ fiducial cosmo + survey
		print("...Computing fiducial pairwise...")
		self.pw = BinPairwise(self.cosmo, self.zmin, self.zmax, self.Nz, self.rmin, self.rmax, self.Nr, fsky=self.fsky, M_min=self.M_min)

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
				self.cov_meas = self.pw.Cov_meas(self.sigma_v) # FIXME: make sigma_v dependent on other params such as tau
			else:
				self.cov_meas = cm

			self.cov = {i : self.cov_cosmic[i] + self.cov_gauss_shot[i] + self.cov_meas[i] for i in xrange(self.Nz)}

		else:
			self.cov = cov

		self.inv_cov = {i : linalg.pinv2(self.cov[i]) for i in xrange(self.Nz)}
		# self.cov = linalg.block_diag(*cov.values())
		# self.inv_cov = linalg.pinv2(self.cov)

		# Precomputed Fisher matrix
		self._fullMat = None
		self._fullInvMat = None
		self._mat = None
		self._invmat = None

	def _computeObservables(self, par_cosmo, par_sur):
		if par_cosmo == self.fid_cosmo:
			_cosmo = copy.copy(self.cosmo)
		else:
			_cosmo = Cosmo(params=par_cosmo)

		_pw = BinPairwise(_cosmo, par_sur['zmin'], par_sur['zmax'], par_sur['Nz'], par_sur['rmin'], \
						  par_sur['rmax'],  par_sur['Nr'], fsky=par_sur['fsky'], M_min=par_sur['M_min'])

		return _pw.V_Delta

	def _computeFullMatrix(self):
		print("...Computing derivatives...")
		self._dvdp = self._computeDerivatives()
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
				for idz in xrange(self.Nz):
					tmp = tmp + np.dot(self._dvdp[idz][i], np.dot(self.inv_cov[idz], self._dvdp[idz][j]))	
				_fullMat[i,j] = tmp
				_fullMat[j,i] = _fullMat[i,j]
				del tmp

		_Priors = np.zeros((nparams,nparams))
		for p in self.params:
			i = self.params.index(p)
			try:
				_Priors[i,i] = 1./self.priors[p]**2.
				print '---Including prior for',p,str(self.priors[p])
			except KeyError:
				pass
		if (_Priors == 0).all():
			print '---No prior included'

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
		dvdp = {}
		for idz in xrange(self.Nz):
			dvdp[idz] = []

		# Computes all the derivatives with respect to the main parameters
		for p in self.params:
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
				print par_sur[p]

			elif p in self.fid_cosmo.keys():
				try:
					step = self.steps[p]
				except:
					step = par_cosmo[p] * self.step		
					if par_cosmo[p] == 0:
						step = self.step
				par_cosmo[p] = par_cosmo[p] + step
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
				print par_sur[p]

			elif p in self.fid_cosmo.keys():
				try:
					step = self.steps[p]
				except:
					step = par_cosmo[p] * self.step		
					if par_cosmo[p] == 0:
						step = self.step
				par_cosmo[p] = par_cosmo[p] - step
				print par_cosmo[p]

			Vm = self._computeObservables(par_cosmo, par_sur)

			if p == 'As':
				step = step * 1e9

			for idz in xrange(self.Nz):
				dvdp[idz].append( (Vp[idz] - Vm[idz])/ (2.0 * step) )

			del par_sur, par_cosmo, Vp, Vm

		return dvdp

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
		newFisher = FisherPairwise(self.fid_cosmo, self.fid_surv, params)

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
		r""" Marginalises the Fisher matrix over unwanted parameters.
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
		

	# # @property
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


