import numpy as np
from scipy import interpolate, integrate, linalg, special, optimize
from astropy import constants as const
from defaults import *
from utils import V_bin, W_Delta, nl_cmb, bl
from mass_func import MassFunction
from corr_func import CorrFunction
from universe import Cosmo
from profiles import ClusterOpticalDepth
import copy
import matplotlib.pyplot as plt
from scipy.io import readsav

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
			The minimum halo mass used to compute the average halo bias [units :math: `M_{\odot}`]
		
		M_max : float
			The maximum halo mass used to compute the average halo bias [units :math: `M_{\odot}`]

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
	def __init__(self, cosmo, zmin, zmax, Nz, rmin, rmax, Nr, fsky=1., M_min=None, M_max=None, kmin=None, kmax=None, verbose=False, fg_path='/Users/fbianchini/Downloads/george_plot_bestfit_line.sav'):
		self.cosmo = cosmo
		self.fsky  = fsky
		if M_min is None:
			M_min = 1e14
		self.M_min = M_min
		if M_max is None:
			M_max = 1e16
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

		self.lmax = 30000
		self.cls = self.cosmo.cmb_spectra(self.lmax)#(self.cosmo.lmax)
		self.george_fg_dic = readsav(fg_path)

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

		self.hmf = MassFunction(self.cosmo)

		# Loop over z-bins
		for idz, z in enumerate(self.zmean):
			self.nhalo[idz] = self.hmf.n_cl(z=z, M_min=self.M_min, M_max=self.M_max) #* self.dz # FIXME: multiply by dz ~ integration over z
			self.bias[idz]  = self.hmf.bias_avg(z=z, M_min=self.M_min, M_max=self.M_max)
		
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
		
	def W_AP(self, ell, theta_R):
		# theta_R in *radians*
		return (2.*special.jv(1, ell*theta_R) - np.sqrt(2)*special.jv(1, np.sqrt(2)*ell*theta_R)) / (ell * theta_R)

	def W_disk(self, ell, theta_R):
		return 2.*special.jv(1, ell * theta_R) / (ell * theta_R)

	def W_annulus(self, ell, theta_R):
		theta1 = theta_R
		theta2 = np.sqrt(2)*theta_R
		return 2.*(theta1 * special.jv(1,ell * theta1) - theta2 *
					 special.jv(1,ell * theta2)) / (ell * (theta1**2 - theta2**2))
		# return (np.sqrt(2) * theta_R * special.jv(1, np.sqrt(2)*ell*theta_R) - theta_R * special.jv(1, ell*theta_R)) / (ell * theta_R**2)

	def GetT_kSZ(self, M, z, theta_R, fwhm_arcmin=0., fid_v=300., profile='Battaglia', lowpass=False): # \muK
		"""
		Returns the expected aperture-photometry estimated  kSZ temperature shift 
		associated to an object of mass M (M_sun) and velocity fid_v at redshift z 
		when observed with a beam given by fwhm_arcmin.

		Parameters
		----------
		M : float
			Mass of the halo [units :math: `M_{\odot}`]
		
		z : float
			The redshift of the halo

		theta_R : float
			Aperture radius [units : arcmin]

		fwhm_arcmin : float
			Beam of the telescope [units : arcmin]

		fid_v : float
			Velocity of the halo [units :math: `km/s`]

		profile : string 
			Density profile of the halo [Battaglia, NFW or gNFW_Moore]

		Returns
		-------
		T_kSZ : float
			kSZ temperature shift [units :math: `\mu K`]
		"""
		return ClusterOpticalDepth(self.cosmo).GetApertureTau(M, z, theta_R, fwhm_arcmin=fwhm_arcmin, profile=profile, lowpass=lowpass) * fid_v / const.c.to('km/s').value * 2.725e6

	def sigma_T_kSZ(self, theta_R, noise_uK_arcmin, fwhm_arcmin, lmax=20000, lknee=1e-9, alpha=0., cmb=True, fg=None, lowpass=False):
		"""
		Returns the expected variance of the .

		Parameters
		----------
		theta_R : float
			Aperture radius [units : arcmin]

		noise_uK_arcmin : float
			White-noise level of experiment [units :math: `\mu K - arcmin`]

		fwhm_arcmin : float
			Beam of the telescope [units : arcmin]

		lmax : float
			Maximum multipole to compute residual calculation

		lknee : float
			
		alpha : float

		cmb : bool
			If True, includes primary (lensed) CMB in the residual calculation [default : True]

		fg : str
			Whether to include foregrounds (@ 150 GHz) or not, it can be 'kSZ' or 'all' [default : None]

		Returns
		-------
		sigma_T_kSZ : float
			residuals [units :math: `\mu K`]
		"""
		theta_R = np.radians(theta_R/60.)
		ells = np.arange(2, lmax+1)

		# Primary lensed CMB
		if cmb:
			# if self.cosmo.lmax < lmax:
			if self.lmax < lmax:
				cmb = self.cosmo.cmb_spectra(lmax=lmax)[2:,0]
			else:
				cmb = self.cls[2:lmax+1,0]
		else:
			cmb = 0.

		# Beam
		bl2 = bl(fwhm_arcmin, lmax=lmax)[2:]**2 

		# Instrumental noise (no beam)
		nl = nl_cmb(noise_uK_arcmin, 0., lmax=lmax, lknee=lknee, alpha=alpha)[2:]
		
		# Foregrounds
		if fg is not None: # 3 = (150x150)
			dl_fgs = np.zeros_like(self.george_fg_dic['ml_dls'][3][4])
			
			if fg == 'all':
				for i in xrange(2,9):
					dl_fgs += self.george_fg_dic['ml_dls'][3][i]
			elif fg == 'kSZ':
				dl_fgs += self.george_fg_dic['ml_dls'][3][4]
			
			l_fgs = np.arange(2,dl_fgs.size+2)
			cl_fgs = 2*np.pi/(l_fgs*(l_fgs+1.)) * dl_fgs

			fgs = np.zeros_like(nl)

			if lmax < cl_fgs.size:
				fgs += cl_fgs[:lmax-1]
			else:
				fgs[:cl_fgs.size] += cl_fgs
		else:
			fgs = 0.

		# Summing up the MW sky components
		C_ell = np.nan_to_num((cmb + fgs)* bl2 + nl)

		# Smoothly filter out noisy scales below the beam
		if lowpass:
			l = np.arange(2,C_ell.size+2)
			C_ell *= np.exp(-(l/(np.pi/np.radians(fwhm_arcmin/60.)))**4)

		# plt.loglog(C_ell)
		# plt.show()
		
		# integral = integrate.simps(ells * C_ell * self.W_AP(ells, theta_R)**2, x=ells) 
		integrand = ells * C_ell * (self.W_disk(ells, theta_R)**2 + \
									self.W_annulus(ells, theta_R)**2 - \
									2*self.W_disk(ells, theta_R) * self.W_annulus(ells,theta_R))
		# plt.loglog(ells, integrand)
		# plt.plot(ells, self.W_disk(ells, theta_R)**2 + self.W_annulus(ells, theta_R)**2 - 2*self.W_disk(ells, theta_R) * self.W_annulus(ells,theta_R))
		integral = (1./(2.*np.pi)) * integrate.simps(integrand, x=ells)

		# return integral
		return np.sqrt(integral)

	def GetSigmaV(self, noise_uK_arcmin, fwhm_arcmin, z, theta_R=3., lmax=20000,
						Delta_tau_over_tau=0.15, fid_v=300., lknee=1e-9, alpha=0., 
						fg=False, M=None, profile='Battaglia', lowpass=False):
		"""
		Returns the error on the measured velocity given the uncertainty on the optical depth and the CMB survey specs, i.e. noise 'n' beam
		We write the error on the measured velocity as
		\sigma_V = \sqrt{ \sigma_inst^2 + \sigma_tau^2}
		where 
		i.  \sigma_inst = \sigma_T_kSZ/T_kSZ * v
		ii. \sigma_tau  =  
		""" 
		if M is None: 
			# Mean
			# M_ = np.linspace(self.M_min,self.M_max)
			# M = integrate.simps(self.hmf.dndm(M_, z) * M_, x=M_)/integrate.simps(self.hmf.dndm(M_,z), x=M_)

			# Median
			dndm_temp = lambda m: self.hmf.dndm(m, z)
			dndm_int = integrate.romberg(dndm_temp, self.M_min, self.M_max)
			f = lambda x: integrate.romberg(dndm_temp, self.M_min, x) - dndm_int/2.
			M = optimize.brentq(f, self.M_min, self.M_max)
			# print M

		sigma_T_kSZ = self.sigma_T_kSZ(theta_R, noise_uK_arcmin, fwhm_arcmin, lmax=lmax, lknee=lknee, alpha=alpha, fg=fg, lowpass=lowpass) 
		T_kSZ = self.GetT_kSZ(M, theta_R, z, fwhm_arcmin=fwhm_arcmin, fid_v=fid_v, profile=profile, lowpass=lowpass)
		# print sigma_T_kSZ/T_kSZ*fid_v, sigma_T_kSZ, T_kSZ, 
		sigma_inst = (sigma_T_kSZ / T_kSZ) * fid_v
		sigma_tau  = Delta_tau_over_tau * fid_v

		return np.sqrt(sigma_inst**2 + sigma_tau**2)


	def GetS2N(self, noise_uK_arcmin, fwhm_arcmin, theta_R, 
		             Delta_tau_over_tau=0.15, fid_v=300., lknee=1e-9, alpha=0., fg=False, lmax=20000, 
		             cov_cosmic=None, cov_gauss_shot=None, M=None, profile='battaglia', lowpass=False):
		"""
		Returns the expected S/N of the kSZ detection.
		"""
		# Load or calculate covariance matrices
		# TODO: check if format of the passed covariances is valid
		if cov_cosmic is None:
			cov_cosmic = self.Cov_cosmic().copy()
		if cov_gauss_shot is None:
			cov_gauss_shot = self.Cov_gauss_shot().copy()

		S2N = 0.

		# Loop over redshift bins
		for idz in xrange(self.Nz):
			sigma_v = self.GetSigmaV(noise_uK_arcmin, fwhm_arcmin, self.zmean[idz], theta_R=theta_R, 
									 Delta_tau_over_tau=Delta_tau_over_tau, fid_v=fid_v, lmax=lmax, 
									 lknee=lknee, alpha=alpha, fg=fg, M=M, profile=profile, lowpass=lowpass)
			# print sigma_v
			# cov_meas = self.Cov_meas(500.)
			cov_meas = self.Cov_meas(sigma_v)
			# cov = {i : cov_cosmic[i] + cov_gauss_shot[i] + cov_meas[i] for i in xrange(self.Nz)}
			# inv_cov = linalg.pinv2(cov[idz])

			cov = cov_cosmic[idz] + cov_gauss_shot[idz] + cov_meas[idz]
			inv_cov = np.linalg.inv(cov)
			
			S2N = S2N + np.dot(self.V_Delta[idz], np.dot(inv_cov, self.V_Delta[idz]))	

			# plt.errorbar(self.rmean, self.V_Delta[idz], yerr=np.sqrt(np.diag(cov[idz])))
			# plt.errorbar(self.rmean, self.V_Delta[idz], yerr=np.sqrt(np.diag(cov)))
			# plt.axhline(ls='--')
		return np.sqrt(S2N)



