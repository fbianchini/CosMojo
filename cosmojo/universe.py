import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy import optimize
from astropy import constants as const
from astropy import units as u
import camb
from camb import model, initialpower
from defaults import *
from utils import *
from classy import Class
from IPython import embed
import copy
from scipy.interpolate import RectBivariateSpline


class Cosmo(object):
	""" 
	Class to encapsulate cosmological paramters and quantities. 
	
	This is built around the python version of CAMB, in fact most of the functions 
	are wrapper around CAMB outputs.

	Attributes
	----------
	params : dictionary
		Cosmological parameters to initialize the class

	TODO: add reionization parameters, add w_a
	"""
	def __init__(self, params=None, lmax=3000, **kwargs): 

		self.lmax = lmax

		# Setting cosmo params
		if params is None:
			params = default_cosmo_dict.copy()
		else:
			for key, val in default_cosmo_dict.iteritems():
				params.setdefault(key,val)

		self.params_dict = params.copy()

		# print self.params_dict

		# Apparently CAMB goes nuts if you pass a dictionary w/ params that it doesn't need
		# such as \gamma_0 and \gamma_a, so remove non-CAMB params from the dictionary 
		for par in params.keys():
			if par == 'gamma0':
				params.pop(par)
			if par == 'gammaa':
				params.pop(par)
			if par == 'w':
				params.pop(par)
			if par == 'wa':
				params.pop(par)
			if par == 'cs2':
				params.pop(par)


		# Initialize CAMB
		pars = camb.set_params(**params)

		if self.params_dict['wa'] == 0.:
			pars.set_dark_energy(w=self.params_dict['w'], cs2=self.params_dict['cs2'], wa=0, dark_energy_model='fluid')
		else:
			pars.set_dark_energy(w=self.params_dict['w'], cs2=self.params_dict['cs2'], wa=self.params_dict['wa'], dark_energy_model='ppf')
		
		# pars.set_dark_energy(w=self.params_dict['w'], cs2=self.params_dict['cs2'], wa=self.params_dict['wa'])

		pars.set_for_lmax(lmax=self.lmax, lens_potential_accuracy=1.0)
		# pars.set_accuracy(AccuracyBoost=3.0, lSampleBoost=3.0, lAccuracyBoost=3.0)

		if params['r'] != 0:
			pars.WantTensors = True

		self.pars = pars

		# Calculating (minimal) background quantities such as distances et al.
		self.bkd = camb.get_background(self.pars)

		# Derived params
		for parname, parval in self.bkd.get_derived_params().iteritems():
			setattr(self, parname, parval) 

		# Initializing Matter Power spectrum P(z,k) spline up to LSS
		self.kmin = default_limits['pk_kmin']
		self.kmax = default_limits['pk_kmax']
		nonlinear = False # Linear matter power spectrum as a defaults 
		# self.pars.NonLinear = NonLinear_none

		for kw in kwargs:
			if kw == 'pk_kmin':
				self.kmin = kwargs[kw]
			if kw == 'pk_kmax':
				self.kmax = kwargs[kw]
			if kw == 'NonLinear' or kw == 'nonlinear':
				nonlinear = True
				camb.set_halofit_version('takahashi') # FIXME: let the user choose which NL prescription

		self.pkz = camb.get_matter_power_interpolator(self.pars, nonlinear=nonlinear, hubble_units=False, k_hunit=False, kmax=self.kmax, zmax=self.zstar)

		# Shortcuts for few params 
		self.omegab = self.pars.omegab
		self.omegac = self.pars.omegac
		self.omegam = self.omegab + self.omegac
		self.omegav = self.pars.omegav
		self.omegak = self.pars.omegak
		self.H0     = self.pars.H0
		self.h      = self.H0/100.
		self.ns     = self.pars.InitPower.an[0]
		self.As     = self.pars.InitPower.ScalarPowerAmp[0]
		self.r      = self.pars.InitPower.rat[0]
		self.w      = self.params_dict['w']
		# self.wa     = self.params_dict['wa']
		self.gamma0 = self.params_dict['gamma0']
		self.gammaa = self.params_dict['gammaa']


	def rho_c(self, z): # [kg/m^3]
		"""
		Returns the critical density at redshift z [kg/m^3].
		"""
		return 3.*(self.H_z(z)*(u.km).to(u.Mpc))**2/(8.*np.pi*const.G.value)

	def rho_bar(self, z=0.): # [M_sun/Mpc^3]
		"""
		Returns the comoving matter density at redshift z [M_sun/Mpc^3].
		"""
		return self.rho_c(0.) * self.omegam * (1+z)**3 * u.kg.to('M_sun') / u.m.to('Mpc')**3

	def d_L(self, z): # [Mpc]
		""" 
		Returns the luminosity distance out to redshift z [Mpc]. 
		"""
		return  self.bkd.luminosity_distance(z)

	def d_A(self, z): # [Mpc]
		""" 
		Returns the angular diameter distance out to redshift z [Mpc].
		"""
		return  self.bkd.angular_diameter_distance(z)

	def f_K(self, z): # [Mpc]
		""" 
		Returns the transverse comoving radial distance out to redshift z [Mpc].
		FIXME: double check this definition
		"""
		return  self.bkd.comoving_radial_distance(z)

	def t_z(self, z): # [Gyr]
		""" 
		Returns the age of the Universe at redshift z [Gyr].
		"""
		if np.isscalar(z) or (np.size(z) == 1):
			return  self.bkd.physical_time(z)
		else:
			return np.asarray([ self.bkd.physical_time(tz) for tz in z ])

	def H_a(self, a): # [km/s/Mpc]
		""" 
		Returns the hubble factor at scale factor a=1/(1+z) [km/s/Mpc]. 
		"""
		z = np.nan_to_num(1./a - 1.)
		return self.H_z(z)

	def H_z(self, z): # [km/s/Mpc]
		""" 
		Returns the hubble factor at redshift z [km/s/Mpc]. 
		"""
		if np.isscalar(z) or (np.size(z) == 1):
			return  self.bkd.hubble_parameter(z)
		else:
			return np.asarray([ self.bkd.hubble_parameter(tz) for tz in z ])

	def H_x(self, x): # [km/s/Mpc] 
		""" 
		Returns the hubble factor at conformal distance x (in Mpc) [km/s/Mpc]. 
		"""
		return self.H_z( self.bkd.redshift_at_comoving_radial_distance(x) )

	def E_z(self, z): # [unitless]
		""" 
		Returns the unitless Hubble expansion rate at redshift z 
		"""
		return  self.H_z(z) / self.H0

	def D_z(self, z):
		""" 
		Returns the growth factor at redshift z (Eq. 7.77 of Dodelson). 
		"""
		if np.isscalar(z) or (np.size(z) == 1):
			return 2.5 * self.omegam * self.H_a(1./(1.+z)) / self.H0 * integrate.quad( lambda a : ( self.H0 / (a * self.H_a(a)) )**3, 0, 1./(1.+z) )[0] 
		else:
			return [ self.D_z(tz) for tz in z ]

	def f_z(self, z, gamma0=None, gammaa=None): # [unitless]
		"""
		Returns the growth rate a' la Linder

		f(z) = d\ln{D}/d\ln{a} = \Omega_m(z)^\gamma(z)
		
		where the growth index gamma can be expanded as

		\gamma(z) = \gamma_0 - z/(1+z)\gamma_a
		"""
		if gamma0 is None: 
			gamma0 = self.gamma0

		if gammaa is None: 
			gammaa = self.gammaa

		gamma = gamma0 - z/(1+z)*gammaa

		return (self.omegam*(1+z)**3/self.E_z(z)**2)**gamma

	def Wz(self, z):
		"""
		Some sort of lensing kernel needed to evaluate E_G
		"""
		return self.f_K(z) * (1 - self.f_K(z)/self.f_K(self.zstar))

	def E_G(self, z, gamma0=None, gammaa=None):
		"""
		Returns the E_G statistic in GR

		"""
		if gamma0 is None: 
			gamma0 = self.gamma0

		if gammaa is None: 
			gammaa = self.gammaa

		gamma = gamma0 - z/(1+z)*gammaa

		return self.omegam / self.f_z(z, gamma0=gamma0, gammaa=gammaa)
	def dVdz(self, z): 
		""" 
		The differential comoving volume element :math: `dV_c / / dz, 
		which has dimensions of volume per unit redshift  [Mpc^3]
		
		Notes
		-----
		- It is integrated over the *full-sky*, hence the 4\pi

		"""
		return 4.*np.pi*const.c.to('km/s').value * (1+z)**2 * self.d_A(z)**2. / self.H_z(z) 

	def V(self, zmax, zmin=0.): # [Mpc^3]
		"""
		Returns the comoving volume between redshift [zmin(=0 by default), zmax] [Mpc^3].
		"""
		# if np.isscalar(zmax) or (np.size(zmax) == 1):
		return integrate.quad(self.dVdz, zmin, zmax, epsabs=0.0, epsrel=10e-5, limit=100)[0]
		# else:
		# 	return np.asarray([ self.V(tz) for tz in zmax ])

	def V_survey(self, zmax, zmin=0., fsky=1.):
		"""
		Returns the volume of a survey covering fsky % of the sky between [zmin, zmax]
		"""
		return fsky * self.V(zmax, zmin=zmin)

	def sigma_Rz(self, R, z=0., nk=10000):
		""" 
		Computes the square root of the variance of matter fluctuations within a sphere of radius R [Mpc]

		\sigma^2(R)= \frac{1}{2 \pi^2} \int_0^\infty \frac{dk}{k} k^3 P(k,z) W^2(kR)
		
		where
		
		W(kR) = \\frac{3j_1(kR)}{kR}
		"""
		if np.isscalar(R) or (np.size(R) == 1):
			def int_sigma(logk):
				k  = np.exp(logk)
				kR = k * R
				W  = W_k_tophat(kR)#3.0 * (np.sin(kR) - kR * np.cos(kR)) / kR**3
				pk = self.pkz.P(z,k)

				return k**3 * W**2 * pk

			lnks   = np.linspace(np.log(self.kmin), np.log(self.kmax), nk)
			sigma2 = integrate.simps( int_sigma(lnks), x=lnks )
			sigma2 /= (2.*np.pi**2)

			return np.sqrt(sigma2)
			# return np.sqrt(1.0/(2.0*np.pi**2.0) * integrate.romberg(int_sigma, np.log(self.kmin), np.log(self.kmax)))
		else:
			return np.asarray([ self.sigma_Rz(Rs) for Rs in R ])

	def sigma_Mz(self, M, z=0., nk=10000):
		""" 		
		Computes the square root of the variance of matter fluctuations within a sphere of mass M [M_sun]
		"""
		R = self.M2R(M)#, z=z)
		return self.sigma_Rz(R, z=z, nk=nk)

	def M2R(self, M):#, z=None): # [Mpc]
		"""
		Lagrangian scale R [Mpc] of density of mass M [M_sun]
		FIXME: check h factor
		"""
		return ((3.* M)/(4.*np.pi * self.rho_bar(0.)))**(1./3.)

	def R2M(self, R):#, z=None): # [M_sun]
		"""
		FIXME: check h factor
		"""
		return 4./3.*np.pi * self.rho_bar(0.) * R**3

	def delta_c(self):
		# FIXME: valid for flat-only cosmology, see NFW97
		#return 0.15*(12.0*np.pi)**(2.0/3.0)
		return 1.686
		
	# def delta_v():
	# 	# FIXME: valid for flat-only cosmology, see NFW97
	# 	return 0.15*(12.0*np.pi)**(2.0/3.0)

	# def M2R_vir(self, M_Delta, Delta=200.):
	# 	return self.M2R(M_Delta) * Delta**(-1./3.)

	# def R2M_vir(self, R_Delta, Delta=200.):
	# 	return self.R2M(R_Delta) * Delta

	def nu_M(self, M, z=0.):
		"""
		Returns the normalized mass overdensity as function of mass [M_sun] at a given redshift (z=0 by default)
		"""
		return self.delta_c() / self.sigma_Mz(M, z=z)

	def nu_sigma(self, sigma):
		"""
		Input sigma instead of M if you already computed sigma(M).
		"""
		return self.delta_c()/sigma

	def cmb_spectra(self, lmax, spec='lensed_scalar', dl=False):
		"""
		Wrapper around the get_cmb_spectra routine of CAMB.
		It returns a 2d array w/
		cls [lmax+1, ] tt, ee, bb, te, kk, tk
		"""
		ls = np.arange(0,lmax+1)

		if dl:
			fact = 1.e12 * self.pars.TCMB**2.
		else:
			fact = 2.*np.pi/(ls*(ls+1)) * 1.e12 * self.pars.TCMB**2.

		results = camb.get_results(self.pars)
		spectra = results.get_cmb_power_spectra(self.pars, lmax=lmax)

		cls = spectra[spec]

		for i in xrange(cls.shape[1]):
			cls[:,i] *= fact

		# if self.pars.DoLensing != 0:
		clkk = spectra['lens_potential'][:,0] * np.pi / 2.#* (2.*np.pi/4.) * 4./(ls*(ls+1))**2
		cltk = spectra['lens_potential'][:,1] * np.pi * np.sqrt(ls*(ls+1)) * self.pars.TCMB*1.e6 #* (2.*np.pi/2.) / np.sqrt(ls*(ls+1.)) * self.pars.TCMB*1.e6
		clkk = np.nan_to_num(clkk)
		cltk = np.nan_to_num(cltk)
		cls  = np.column_stack((cls,clkk,cltk))

		# TODO: output cls as a dictionary?

		return np.nan_to_num(cls)

	def k_NL(self, z):
		"""
		Returns the non-linear as scale [Mpc] at a given redshift
		"""
		if np.isscalar(z) or (np.size(z) == 1):
			f = lambda k: self.pkz.P(z, k)*k**3/2./np.pi**2 - 1.
			return optimize.brentq(f, 1e-2, 100)
		else:
			return [ self.k_NL(tz) for tz in z ]


	def Dv_mz(self, z):
		""" 
		Returns the virial overdensity w.r.t. the mean matter density redshift z. 
		Based on:
	       * Bryan & Norman (1998) ApJ, 495, 80.
	       * Hu & Kravtsov (2002) astro-ph/0203169 Eq. C6. 
	    """
		den = self.oml + self.omm * (1.0+z)**3 + self.omr * (1.0+z)**4
		omm = self.omm * (1.0+z)**3 / den

		omr = self.omr * (1.0+z)**4 / den
		assert(omr < 1.e-2) # sanity check that omr is negligible at this redshift.

		return (18.*np.pi**2 + 82.*(omm - 1.) - 39*(omm - 1.)**2) / omm

	# def aeq_lm(self):
	# 	""" returns the scale factor at lambda - matter equality. """
	# 	return 1. / ( self.oml / self.omm )**(1./3.)

# class CosmoCLASS(object):
# 	""" 
# 	class to encapsulate a cosmology. 
	
# 	TODO: add reionization parameters 	
# 	"""
# 	def __init__(self, params=None, **kwargs): 

# 		# Setting cosmo params
# 		if params is None:
# 			params = default_cosmoCLASS_dict.copy()
# 		else:
# 			for key, val in default_cosmoCLASS_dict.iteritems():
# 				params.setdefault(key,val)

# 		params['z_max_pk'] = 1050

# 		if (params['w0_fld'] != 0) or (params['wa_fld'] != 0):
# 			params['Omega_Lambda'] = 0.

# 		self.params_dict = params.copy()

# 		# def values for matter PS
# 		self.kmin = default_limits['pk_kmin']
# 		self.kmax = default_limits['pk_kmax']
# 		params['P_k_max_1/Mpc'] = self.kmax

# 		# Apparently CLASS also goes nuts if you pass a dictionary w/ params that it doesn't need
# 		# such as \gamma_0 and \gamma_a, so remove non-CLASS params from the dictionary 
# 		for par in params.keys():
# 			if par == 'gamma0':
# 				params.pop(par)
# 			if par == 'gammaa':
# 				params.pop(par)


# 		# if params['r'] != 0:
# 		# 	params['modes'] = ['s', 't']

# 		# Initialize CLASS
# 		cosmo = Class()

# 		# Set the parameters to the cosmological code
# 		cosmo.set(params)
# 		cosmo.compute()

# 		self.cosmo = cosmo
# 		# # Derived params
# 		# for parname, parval in self.bkd.get_derived_params().iteritems():
# 		# 	setattr(self, parname, parval) 

# 		# Initializing Matter Power spectrum P(z,k) spline up to LSS
# 		self.pkz = self.GetMatterPK()
				
# 		# Shortcuts for few params 
# 		self.omegab = self.cosmo.Omega_b()
# 		# self.omegac = self.cosmo.Omega_c()
# 		self.omegam = self.cosmo.Omega_m()
# 		self.omegav = self.cosmo.get_current_derived_parameters(['Omega_Lambda'])
# 		# self.omegak = self.cosmo.get_current_derived_parameters(['O'])
# 		self.H0     = self.cosmo.get_current_derived_parameters(['H0'])
# 		self.h      = self.cosmo.h()
# 		self.ns     = self.cosmo.n_s()
# 		self.r      = self.cosmo.get_current_derived_parameters(['r'])
# 		self.gamma0 = self.params_dict['gamma0']
# 		self.gammaa = self.params_dict['gammaa']

# 	def GetMatterPK(self, nz=100, nk=200):
# 		"""
# 		"""
# 		zvec = np.exp(np.log(self.params_dict['z_max_pk'] + 1) * np.linspace(0, 1, nz)) - 1
# 		kvec = np.logspace(np.log10(self.kmin), np.log10(self.kmax), nk)

# 		pk = np.empty((nz,nk))

# 		for i in xrange(nz):
# 			pk[i,:] = np.array([self.cosmo.pk(k,zvec[i]) for k in kvec])

# 		class PKInterpolator(RectBivariateSpline):
# 			def P(self, z, kh, grid=None):
# 				if grid is None:
# 					grid = not np.isscalar(z) and not np.isscalar(kh)
# 				return self(z, kh, grid=grid)
		
# 		res = PKInterpolator(zvec, kvec, pk)

# 		return res

# 	def rho_c(self, z): # [kg/m^3]
# 		return 3.*(self.H_z(z)*(u.km).to(u.Mpc))**2/(8.*np.pi*const.G.value)

# 	def d_L(self, z): # [Mpc]
# 		""" returns the luminosity distance out to redshift z. """
# 		if np.isscalar(z) or (np.size(z) == 1):
# 			return self.cosmo.luminosity_distance(z)
# 		else:
# 			return np.asarray([ self.cosmo.luminosity_distance(tz) for tz in z ])

# 	def d_A(self, z): # [Mpc]
# 		""" returns the angular diameter distance out to redshift z """
# 		if np.isscalar(z) or (np.size(z) == 1):
# 			return self.cosmo.angular_distance(z)
# 		else:
# 			return np.asarray([ self.cosmo.angular_distance(tz) for tz in z ])

# 	# def f_K(self, z): # [Mpc]
# 	# 	""" returns the transverse comoving radial distance out to redshift z 
# 	# 		FIXME: double check this definition
# 	# 	"""
# 	# 	return self.bkd.comoving_radial_distance(z)

# 	def H_a(self, a): # [km/s/Mpc]
# 		""" returns the hubble factor at scale factor a=1/(1+z). """
# 		z = np.nan_to_num(1./a - 1.)
# 		return self.H_z(z)

# 	def H_z(self, z): # [km/s/Mpc]
# 		""" returns the hubble factor at redshift z. """
# 		if np.isscalar(z) or (np.size(z) == 1):
# 			return  self.cosmo.Hubble(z)*const.c.to('km/s').value
# 		else:
# 			return np.asarray([ self.cosmo.Hubble(tz)*const.c.to('km/s').value for tz in z ])

# 	def E_z(self, z): # [unitless]
# 		""" returns the unitless Hubble expansion rate at redshift z """
# 		return  self.H_z(z) / self.H0

# 	def D_z(self, z):
# 		""" returns the growth factor at redshift z (Eq. 7.77 of Dodelson). """
# 		if np.isscalar(z) or (np.size(z) == 1):
# 			return 2.5 * self.omegam * self.H_a(1./(1.+z)) / self.H0 * integrate.quad( lambda a : ( self.H0 / (a * self.H_a(a)) )**3, 0, 1./(1.+z) )[0] 
# 		else:
# 			return [ self.D_z(tz) for tz in z ]

# 	def f_z(self, z, gamma0=None, gammaa=None): # [unitless]
# 		"""
# 		Returns the growth rate a' la Linder

# 		f(z) = d\ln{D}/d\ln{a} = \Omega_m(z)^\gamma(z)
		
# 		where the growth index gamma can be expanded as

# 		\gamma(z) = \gamma_0 - z/(1+z)\gamma_a
# 		"""
# 		if gamma0 is None: 
# 			gamma0 = self.gamma0

# 		if gammaa is None: 
# 			gammaa = self.gammaa

# 		gamma = gamma0 - z/(1+z)*gammaa

# 		return (self.omegam*(1+z)**3/self.E_z(z)**2)**gamma

# 	def Wz(self, z):
# 		return self.f_K(z) * (1 - self.f_K(z)/self.f_K(self.zstar))

# 	def E_G(self, z, gamma0=None, gammaa=None):
# 		"""
# 		Returns the E_G statistic
# 		"""
# 		if gamma0 is None: 
# 			gamma0 = self.gamma0

# 		if gammaa is None: 
# 			gammaa = self.gammaa

# 		gamma = gamma0 - z/(1+z)*gammaa

# 		return self.omegam / self.f_z(z, gamma0=gamma0, gammaa=gammaa)

# 	def dVdz(self, z): 
# 		""" 
# 		The differential comoving volume element :math: `dV_c / / dz, 
# 		which has dimensions of volume per unit redshift  
		
# 		Notes
# 		-----
# 		"""
# 		return 4.*np.pi*const.c.to('km/s').value * (1+z)**2 * self.d_A(z)**2. / self.H_z(z) 

# 	def V(self, zmax, zmin=0.): # [Mpc^3]
# 		# if np.isscalar(zmax) or (np.size(zmax) == 1):
# 		return integrate.quad(self.dVdz, zmin, zmax, epsabs=0.0, epsrel=10e-5, limit=100)[0]
# 		# else:
# 		# 	return np.asarray([ self.V(tz) for tz in zmax ])

# 	def V_survey(self, zmax, zmin=0., fsky=1.):
# 		return fsky * self.V(zmax, zmin=zmin)

# 	def sigma_Rz(self, R, z=0., nk=10000):
# 		""" 
# 		FIXME: check h factors

# 		Computes the energy of the fluctuations within a sphere of R h^{-1} Mpc
# 		.. math::
# 		\\sigma^2(R)= \\frac{1}{2 \\pi^2} \\int_0^\\infty \\frac{dk}{k} k^3 P(k,z) W^2(kR)
# 		where
# 		.. math::
# 		W(kR) = \\frac{3j_1(kR)}{kR}
# 		"""
# 		if np.isscalar(R) or (np.size(R) == 1):
# 			def int_sigma(logk):
# 				k  = np.exp(logk)
# 				kR = k * R
# 				W  = W_k_tophat(kR)#3.0 * (np.sin(kR) - kR * np.cos(kR)) / kR**3
# 				pk = self.pkz.P(z,k)

# 				return k**3 * W**2 * pk

# 			lnks   = np.linspace(np.log(self.kmin), np.log(self.kmax), nk)
# 			sigma2 = integrate.simps( int_sigma(lnks), x=lnks )
# 			sigma2 /= (2.*np.pi**2)

# 			return np.sqrt(sigma2)
# 			# return np.sqrt(1.0/(2.0*np.pi**2.0) * integrate.romberg(int_sigma, np.log(self.kmin), np.log(self.kmax)))
# 		else:
# 			return np.asarray([ self.sigma_Rz(Rs) for Rs in R ])

# 	def sigma_Mz(self, M, z=0., nk=10000):
# 		""" 
# 		FIXME: check h factors
		
# 		Computes the energy of the fluctuations within a sphere of R h^{-1} Mpc
# 		.. math::
# 		\\sigma^2(R)= \\frac{1}{2 \\pi^2} \\int_0^\\infty \\frac{dk}{k} k^3 P(k,z) W^2(kR)
# 		where
# 		.. math::
# 		W(kR) = \\frac{3j_1(kR)}{kR}
# 		"""
# 		R = self.M2R(M)#, z=z)
# 		return self.sigma_Rz(R, z=z, nk=nk)

# 	def rho_bar(self, z=0.): # [M_sun/Mpc^3]
# 		return self.rho_c(0.) * self.omegam * (1+z)**3 * u.kg.to('M_sun') / u.m.to('Mpc')**3

# 	def M2R(self, M, z=None): # [Mpc]
# 		"""
# 		FIXME: check h factor
# 		Lagrangian scale of density of mass M
# 		Counts cdm and baryons towards mass budget, not massive neutrinos (!!??)
# 		- Mass units are *M_sun*.
# 		- The radius is in *Mpc*
# 		"""
# 		return ((3.* M)/(4.*np.pi * self.rho_bar(0.)))**(1./3.)

# 	def R2M(self, R, z=None): # [M_sun]
# 		"""
# 		FIXME: check h factor
# 		"""
# 		return 4./3.*np.pi * self.rho_bar(0.) * R**3

# 	def delta_c(self):
# 		# FIXME: valid for flat-only cosmology, see NFW97
# 		#return 0.15*(12.0*np.pi)**(2.0/3.0)
# 		return 1.686
	
# 	def nu_M(self, M, z=0.):
# 		return self.delta_c() / self.sigma_Mz(M, z=z)

# 	def nu_sigma(self, sigma):
# 		"""
# 		Input sigma instead of M if you already computed sigma(M).
# 		"""
# 		return self.delta_c()/sigma

# 	def cmb_spectra(self, lmax, dl=False, spec=None):
# 		"cls [lmax+1, ] tt, ee, bb, te, kk, tk"

# 		if dl:
# 			ls = np.arange(0,lmax+1)
# 			fact = (ls*(ls+1)) / (2.*np.pi) * 1.e12 * self.cosmo.T_cmb()**2.
# 		else:
# 			fact = 1.e12 * self.cosmo.T_cmb()**2.

# 		cls_ = self.cosmo.lensed_cl()

# 		cls = np.zeros((lmax+1,5))

# 		cls[:,0] = cls_['tt'][:lmax+1] * fact 
# 		cls[:,1] = cls_['ee'][:lmax+1] * fact
# 		cls[:,2] = cls_['bb'][:lmax+1] * fact
# 		cls[:,3] = cls_['te'][:lmax+1] * fact
# 		cls[:,4] = cls_['pp'][:lmax+1] * fact

# 		# if self.pars.DoLensing != 0:
# 		# clkk = spectra['lens_potential'][:,0] * (2.*np.pi/4.)
# 		# cltk = spectra['lens_potential'][:,1] * (2.*np.pi/2.) / np.sqrt(ls*(ls+1.)) * self.cosmo.T_cmb()*1.e6
# 		# clkk = np.nan_to_num(clkk)
# 		# cltk = np.nan_to_num(cltk)
# 		# cls  = np.column_stack((cls,clkk,cltk))

# 		# TODO: output cls as a dictionary?

# 		return np.nan_to_num(cls)


