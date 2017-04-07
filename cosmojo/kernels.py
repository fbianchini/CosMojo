import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy import constants as const
from astropy import units as u
from defaults import *


class Kernel(object):
	"""
	Base class for observables kernels.

	The idea is that a kernel object is a building block to be passed to the Limber integrator.
	A kernel is defined over the redshift range [z_min, z_max] and can be composed by more than one bin.
	When creating a new kernel, simply define a new .raw_W_z() method.

	Kernels can describe:
	- galaxy overdensity
	- lensing (galaxy and CMB)
	- 21 cm (tbi)
	- ISW (tbi)
	- ...

	Attributes
	----------
	z_min : float
		Minimum redshift defined by the kernel

	z_max : float
		Maximum redshift defined by the kernel

	nbins : float
		Number of z-bins
	"""
	def __init__(self, z_min, z_max, nbins=1):

		self.initialized_spline = False

		self.z_min = z_min
		self.z_max = z_max
		self.nbins = nbins

		self._z_array  = np.linspace(self.z_min, self.z_max, default_precision['kernel_npts'])
		self._wz_array = np.zeros((len(self._z_array),self.nbins), dtype='float128')

	def _initialize_spline(self):
		self._wz_spline = []

		for idb in xrange(self.nbins):
			for idz in xrange(self._z_array.size):
				self._wz_array[idz,idb] = self.raw_W_z(self._z_array[idz], i=idb)
			self._wz_spline.append(InterpolatedUnivariateSpline(self._z_array, self._wz_array[:,idb], k=1))
		
		self.initialized_spline = True

	def raw_W_z(self, z, i=0):
		"""
		Raw, possibly computationally intensive, window function.
		"""
		return 1.0

	def W_z(self, z, i=0):
		"""
		Wrapper for splined window function.
		"""
		if not self.initialized_spline:
			self._initialize_spline()

		return np.where(np.logical_and(z >= self.z_min, z <= self.z_max), self._wz_spline[i](z), 0.0)

	def plot():
		return NotImplemented

class LensCMB(Kernel):
	"""
	Redshift kernel for a CMB lensing

	Notes
	-----
	Don't need LensCMB.raw_W_z() because CMB lensing window function is pretty fast 
	and there's no need for the interpolation

	"""
	def __init__(self, cosmo):
		"""
		Attributes
		----------
	    cosmo : Cosmo object (from universe.py)
	        Cosmology object
		"""
		self.cosmo = cosmo
		self.xlss  = cosmo.f_K(cosmo.zstar)
		self.fac   = (1.5 * (self.cosmo.omegab+self.cosmo.omegac) * self.cosmo.H0**2.) / (const.c.to('km/s').value)  # TODO: change here after setting params in cosmo class

		super(LensCMB, self).__init__(0., cosmo.zstar)

	def W_z(self, z, i=0):
		"""
		Overwritten because we don't need to interpolate the CMB lensing window function
		
		Notes
		-----
		i = dummy argument 
		"""
		return self.fac / self.cosmo.H_z(z) * self.cosmo.f_K(z) * (1.+z) * ((self.xlss-self.cosmo.f_K(z))/self.xlss)
		# return (3.*(self.cosmo.bkd.Params.omegab+self.cosmo.bkd.Params.omegac)*self.cosmo.bkd.Params.H0**2 * (u.km).to(u.Mpc))/(2.*const.c.to('Mpc/s').value * self.cosmo.H_z(z)) * self.cosmo.f_K(z) * (1.+z) * ((self.xlss-self.cosmo.f_K(z))/self.xlss)

	# def W_chi(self, chi):
	# 	return self.fac * chi * (1.+self.cosmo.bkd.redshift_at_comoving_radial_distance(chi)) * ((self.xlss-chi/self.xlss))
		# return (3.*(self.cosmo.bkd.Params.omegab+self.cosmo.bkd.Params.omegac)*self.cosmo.bkd.Params.H0**2 * (u.km).to(u.Mpc))/(2.*const.c.to('Mpc/s').value * self.chi * (1.+self.cosmo.bkd.redshift_at_comoving_radial_distance(chi)) * ((self.xlss-chi)/self.xlss)

class Gals(Kernel):
	""" 
	Redshift kernel for a galaxy survey.
	!!! This is for the *Survey* class, i.e. the ones w/ just one z-bin !!!
	!!! The same results can be obtained if you use *Tomography* class w/ nbin = 1, I know it sucks !!!
	"""
	def __init__(self, cosmo, survey, b=1., alpha=1., z_max_lens=default_limits['lens_zmax'], npts=default_precision['magbias_npts']):
		"""
		Attributes
		----------
	    cosmo : Cosmo object (from universe.py)
	        Cosmology object
		
		survey : Survey object (from survey.py)
			Survey object

		b : float or callable
			Lineaer bias parameter, can be a number or a function such as b = lambda z: (1+z)^2

		alpha : float or callable
			Magnification bias parameter (i.e. slope of the integrated number counts as function
			of the flux, N(>S) \propto S^{-\alpha}), can be a number or a function such as alpha = lambda z: (1+z)^2
	
		z_max_lens : float
			Upper limit of magnification bias integral (performed with Simpson integration)	

		npts : int
			Number of points to sample the magnification bias integral (performed with Simpson integration)	
		"""
		self.cosmo = cosmo
		self.survey = survey
		self.b = b
		self.alpha = alpha
		self.z_max_lens = z_max_lens
		self.npts = npts

		self.survey.normalize()

		self.fac = (1.5 * (self.cosmo.omegab+self.cosmo.omegac) * self.cosmo.H0**2.) / (const.c.to('km/s').value)  

		super(Gals, self).__init__(1e-5, survey.z_max)

	def raw_W_z(self, z, i=0):
		"""
		Galaxy overdensity kernel decomposed as the sum of a clustering and magnification bias component.
		"""
		return ( self.W_cl(z, b=self.b) + self.W_mu(z, alpha=self.alpha) )

	def W_cl(self, z, b=1., i=0):
		"""
		Clustering part of the galaxy kernel.
		"""
		if callable(b):
			b = b(z)
		return b * self.survey.dndz(z)

	def W_mu(self, z, alpha=1.):
		"""
		Magnification bias part of the galaxy kernel.
		"""
		return self.fac * (1.+z) * self.cosmo.f_K(z) / self.cosmo.H_z(z) * self.MagBiasInt(z, alpha=alpha, zmax=self.z_max_lens, npts=self.npts)

	def MagBiasInt(self, z, alpha=1., zmax=default_limits['lens_zmax'], npts=default_precision['lens_npts']): # [unitless]
		"""
		Performs the integration of the magnification bias part of the kernel
		"""
		if np.isscalar(z) or (np.size(z) == 1):
			if not callable(alpha):
				alpha_ = lambda x: alpha

			lens_integrand = lambda zprime:  (1. - self.cosmo.f_K(z) / self.cosmo.f_K(zprime)) * (alpha_(z) - 1.) *self.survey.dndz(zprime)
			
			return integrate.quad(lens_integrand, z, zmax, epsabs=default_precision["global_precision"], epsrel=default_precision["lens_precision"])[0]

			# z_ = np.linspace(z, zmax, npts)

			# if callable(alpha):
			# 	alpha = alpha(z_)
			# elif alpha == None:
			# 	alpha = np.ones(z_.size)

			# return np.nan_to_num(integrate.simps(integrand(z_) * (alpha - 1.), x=z_)) # TODO: pass to quad integration?
		else:
			return np.asarray([ self.MagBiasInt(tz, alpha=alpha, zmax=zmax, npts=npts) for tz in z ])

class GalsTomo(Kernel):
	""" 
	Redshift kernel for a galaxy survey.
	"""
	def __init__(self, cosmo, tomo, b=1., alpha=1., z_max_lens=default_limits['lens_zmax'], npts=default_precision['magbias_npts']):
		"""
		Attributes
		----------
	    cosmo : Cosmo object (from universe.py)
	        Cosmology object
		
		tomo : Tomography object (from survey.py)
			Tomography object which allows for more z-bins

		b : float or callable
			Lineaer bias parameter, can be a number or a function such as b = lambda z: (1+z)^2

		alpha : float or callable
			Magnification bias parameter (i.e. slope of the integrated number counts as function
			of the flux, N(>S) \propto S^{-\alpha}), can be a number or a function such as alpha = lambda z: (1+z)^2
	
		z_max_lens : float
			Upper limit of magnification bias integral (performed with Simpson integration)	

		npts : int
			Number of points to sample the magnification bias integral (performed with Simpson integration)	
		"""
		self.cosmo = cosmo
		self.tomo  = tomo
		self.tomo.normalize() 

		if len(np.atleast_1d(b)) == 1: # just one bias value/function for all photo-z bins
			self.b = self.tomo.nbins * [b] 
		else:
			assert(len(b) == self.tomo.nbins)
			self.b = b

		if len(np.atleast_1d(alpha)) == 1: # just one magnification bias value/function for all photo-z bins
			self.alpha = self.tomo.nbins * [alpha] 
		else:
			assert(len(alpha) == self.tomo.nbins)
			self.alpha = alpha

		self.z_max_lens = z_max_lens
		self.npts = npts

		self.fac = (1.5 * (self.cosmo.omegab+self.cosmo.omegac) * self.cosmo.H0**2.) / (const.c.to('km/s').value)  

		super(GalsTomo, self).__init__(1e-5, tomo.z_max, tomo.nbins)

	def raw_W_z(self, z, i):
		"""
		Galaxy overdensity kernel decomposed as the sum of a clustering and magnification bias component.
		"""
		return ( self.W_cl(z, i, b=self.b[i]) + self.W_mu(z, i, alpha=self.alpha[i]) )

	def W_cl(self, z, i, b=1.):
		"""
		Clustering part of the galaxy kernel.
		"""
		if callable(b):
			b = b(z)
		return b * self.tomo.dndz_bin(z,i)

	def W_mu(self, z, i, alpha=1.):
		"""
		Magnification bias part of the galaxy kernel.
		"""
		return self.fac * (1.+z) * self.cosmo.f_K(z) / self.cosmo.H_z(z) * self.MagBiasInt(z, i, alpha=alpha, zmax=self.z_max_lens, npts=self.npts)

	def MagBiasInt(self, z, i, alpha=1., zmax=default_limits['lens_zmax'], npts=default_precision['lens_npts']): # [unitless]
		"""
		Performs the integration of the magnification bias part of the kernel
		"""
		if np.isscalar(z) or (np.size(z) == 1):
			if not callable(alpha):
				alpha_ = lambda x: alpha

			lens_integrand = lambda zprime:  (1. - self.cosmo.f_K(z) / self.cosmo.f_K(zprime)) * (alpha_(z) - 1.) *self.tomo.dndz_bin(zprime,i)
			
			return integrate.quad(lens_integrand, z, zmax, epsabs=default_precision["global_precision"], epsrel=default_precision["lens_precision"])[0]
			# z_ = np.linspace(z, zmax, npts)
			# integrand = lambda zprime:  (1. - self.cosmo.f_K(z) / self.cosmo.f_K(zprime)) * self.tomo.dndz_bin(zprime,i)
			
			# if callable(alpha):
			# 	alpha = alpha(z_)
			# elif alpha == None:
			# 	alpha = np.ones(z_.size)

			# return np.nan_to_num(integrate.simps(integrand(z_) * (alpha - 1.), x=z_)) # TODO: pass to quad integration?
		else:
			return np.asarray([ self.MagBiasInt(tz, i, alpha=alpha, zmax=zmax, npts=npts) for tz in z ])

class LensGal(Kernel):
	""" 
	Redshift kernel for a galaxy lensing survey.
	"""
	def __init__(self, cosmo, survey, z_max_lens=default_limits['lens_zmax'], npts=default_precision['lens_npts']):
		"""
		Attributes
		----------
	    cosmo : Cosmo object (from universe.py)
	        Cosmology object
		
		tomo : Survey object (from survey.py)
			Survey object

		b : float or callable
			Lineaer bias parameter, can be a number or a function such as b = lambda z: (1+z)^2

		alpha : float or callable
			Magnification bias parameter (i.e. slope of the integrated number counts as function
			of the flux, N(>S) \propto S^{-\alpha}), can be a number or a function such as alpha = lambda z: (1+z)^2
	
		z_max_lens : float
			Upper limit of magnification bias integral (performed with Simpson integration)	

		npts : int
			Number of points to sample the magnification bias integral (performed with Simpson integration)	
		"""
		self.cosmo = cosmo
		self.survey = survey
		self.z_max_lens = z_max_lens
		self.npts = npts
		self.survey.normalize()

		self.fac = (1.5 * (self.cosmo.omegab+self.cosmo.omegac) * self.cosmo.H0**2.) / (const.c.to('km/s').value)  

		super(LensGal, self).__init__(1e-5, survey.z_max)

	def raw_W_z(self, z, i=0):
		return self.fac / self.cosmo.H_z(z) * self.cosmo.f_K(z) * (1.+z) * self.W_lens(z, zmax=self.z_max_lens, npts=self.npts)

	def W_lens(self, z, zmax=default_limits['lens_zmax'], npts=default_precision['lens_npts']):
		"""
		Computes the 
		"""
		if np.isscalar(z) or (np.size(z) == 1):
			lens_integrand = lambda zprime:  (1.-self.cosmo.f_K(z)/self.cosmo.f_K(zprime)) * self.survey.dndz(zprime)
			return integrate.quad(lens_integrand, z, zmax, epsabs=default_precision["global_precision"], epsrel=default_precision["lens_precision"])[0]
			# returne integrate.romberg(lens_integrand, z, zmax, vec_func=True,
			#                          tol=default_precision["global_precision"],
			#                          rtol=default_precision["lens_precision"],
			#                          divmax=default_precision["divmax"])

			# z_ = np.linspace(z, zmax, npts)
			# return integrate.simps(lens_integrand(z_), x=z_)
		else:
			return np.asarray([ self.W_lens(tz) for tz in z ])

class LensGalTomo(Kernel):
	""" Redshift kernel for a galaxy lensing survey.
	       cosmo - quickspec.cosmo.lcdm object describing the cosmology.
	       dndz  - function dndz(z) which returns # galaxies per steradian per redshift.
	       b     - linear bias parameter.
	       alpha - magnification bias parameter.
	"""
	def __init__(self, cosmo, tomo, z_max_lens=default_limits['lens_zmax'], npts=default_precision['lens_npts']):
		self.cosmo = cosmo
		self.tomo = tomo
		self.z_max_lens = z_max_lens
		self.npts = npts
		self.tomo.normalize() 

		self.fac = (1.5 * (self.cosmo.omegab+self.cosmo.omegac) * self.cosmo.H0**2.) / (const.c.to('km/s').value)  

		super(LensGalTomo, self).__init__(1e-5, tomo.z_max, tomo.nbins)

	def raw_W_z(self, z, i):
		return self.fac / self.cosmo.H_z(z) * self.cosmo.f_K(z) * (1.+z) * self.W_lens(z, i, zmax=self.z_max_lens, npts=self.npts)

	def W_lens(self, z, i, zmax=default_limits['lens_zmax'], npts=default_precision['lens_npts']):
		"""
		"""
		if np.isscalar(z) or (np.size(z) == 1):
			lens_integrand = lambda zprime:  (1.-self.cosmo.f_K(z)/self.cosmo.f_K(zprime)) * self.tomo.dndz_bin(zprime, i)
			return integrate.quad(lens_integrand, z, zmax, epsabs=default_precision["global_precision"], epsrel=default_precision["lens_precision"])[0]
			# z_ = np.linspace(z, zmax, npts)
			# return integrate.simps(lens_integrand(z_), x=z_)
		else:
			return np.asarray([ self.W_lens(tz, i) for tz in z ])

# class Gals():
# 	def __init__(self, cosmo, dndz, b=1., alpha=1., 
# 				       zmax=default_limits['magbias_zmax'], 
# 				       npts=default_precision['magbias_npts']):
# 		""" Redshift kernel for a galaxy survey.
# 			cosmo - quickspec.cosmo.lcdm object describing the cosmology.
# 			dndz  - survey object
# 			b     - linear bias parameter, either float or callable, i.e. b = lambda z: (1+z)^2
# 			alpha - magnification bias parameter, either float or callable, i.e. alpha = lambda z: (1+z)^2
# 		"""
# 		self.cosmo = cosmo
# 		self.dndz  = dndz
# 		self.b     = b
# 		self.alpha = alpha
# 		self.zmax  = zmax
# 		self.npts  = npts
# 		self.fac   = (1.5 * (self.cosmo.omegab+self.cosmo.omegac) * self.cosmo.bkd.Params.H0**2.) / (const.c.to('km/s').value)  # TODO: change here after setting params in cosmo class

# 	def W_z(self, z):
# 		return ( self.W_cl(z, b=self.b) + self.W_mu(z, alpha=self.alpha) )

# 	def W_cl(self, z, b=None):
# 		if callable(b):
# 			b = b(z)
# 		elif b == None:
# 			b = np.ones(z.size)			
# 		return b * self.dndz.dndz(z)

# 	def W_mu(self, z, alpha=1.):
# 		return self.fac * (1.+z) * self.cosmo.f_K(z) / self.cosmo.H_z(z) * self.MagBiasInt(z, alpha=alpha, zmax=self.zmax, npts=self.npts)

# 	def MagBiasInt(self, z, alpha=1., zmax=default_limits['magbias_zmax'], npts=default_precision['magbias_npts']): # [unitless]
# 		"""
# 		"""
# 		if np.isscalar(z) or (np.size(z) == 1):
# 			z_ = np.linspace(z, zmax, npts)
# 			integrand = lambda zprime:  (1. - self.cosmo.f_K(z) / self.cosmo.f_K(zprime)) * self.dndz.dndz(zprime)
			
# 			if callable(alpha):
# 				alpha = alpha(z_)
# 			elif alpha == None:
# 				alpha = np.ones(z_.size)

# 			return np.nan_to_num(integrate.simps(integrand(z_) * (alpha - 1.), x=z_)) # TODO: pass to quad integration?
# 		else:
# 			return np.asarray([ self.MagBiasInt(tz, alpha=alpha, zmax=zmax, npts=npts) for tz in z ])

# 	# def PhotozErrs(self, zp, z, sigma=0.01, bias=0.):
# 	# 	return np.exp(-((zp-z-bias)**2./(2.*(sigma*(1+zp))**2)))/(2.*np.pi*(sigma*(1+zp))**2)**0.5

# 	# def ConvolvedNdz(self, z, sigma=0.26, bias=0., z_min=0., z_max=10., npts=1000):
# 	# 	if np.isscalar(z) or (np.size(z) == 1):
# 	# 		return integrate.quad(self.PhotozErrs, z_min, z_max, args=(z, sigma, bias))[0]
# 	# 	else:
# 	# 		return np.asarray([ self.ConvolvedNdz(tz, sigma=sigma, z_min=z_min, z_max=z_max) for tz in z ])

# class LensCMB():
# 	""" Redshift kernel for a galaxy survey.
# 	       cosmo - quickspec.cosmo.lcdm object describing the cosmology.
# 	"""
# 	def __init__(self, cosmo):
# 		self.cosmo = cosmo
# 		self.xlss  = cosmo.f_K(cosmo.zstar)
# 		self.fac   = (1.5 * (self.cosmo.bkd.Params.omegab+self.cosmo.bkd.Params.omegac) * self.cosmo.bkd.Params.H0**2.) / (const.c.to('km/s').value)  # TODO: change here after setting params in cosmo class

# 	def W_z(self, z):
# 		return self.fac / self.cosmo.H_z(z) * self.cosmo.f_K(z) * (1.+z) * ((self.xlss-self.cosmo.f_K(z))/self.xlss)
# 		# return (3.*(self.cosmo.bkd.Params.omegab+self.cosmo.bkd.Params.omegac)*self.cosmo.bkd.Params.H0**2 * (u.km).to(u.Mpc))/(2.*const.c.to('Mpc/s').value * self.cosmo.H_z(z)) * self.cosmo.f_K(z) * (1.+z) * ((self.xlss-self.cosmo.f_K(z))/self.xlss)

# 	# def W_chi(self, chi):
# 	# 	return self.fac * chi * (1.+self.cosmo.bkd.redshift_at_comoving_radial_distance(chi)) * ((self.xlss-chi/self.xlss))
# 		# return (3.*(self.cosmo.bkd.Params.omegab+self.cosmo.bkd.Params.omegac)*self.cosmo.bkd.Params.H0**2 * (u.km).to(u.Mpc))/(2.*const.c.to('Mpc/s').value * self.chi * (1.+self.cosmo.bkd.redshift_at_comoving_radial_distance(chi)) * ((self.xlss-chi)/self.xlss)

# class LensGal():
# 	""" Redshift kernel for a galaxy lensing survey.
# 	       cosmo - quickspec.cosmo.lcdm object describing the cosmology.
# 	       dndz  - function dndz(z) which returns # galaxies per steradian per redshift.
# 	       b     - linear bias parameter.
# 	       alpha - magnification bias parameter.
# 	"""
# 	def __init__(self, cosmo, dndz, 
# 					   zmax=default_limits['kappagal_zmax'], 
# 					   npts=default_precision['kappagal_npts']):
# 		self.cosmo = cosmo
# 		self.dndz  = dndz
# 		self.zmax  = zmax
# 		self.npts  = npts
# 		self.fac   = (1.5 * (self.cosmo.bkd.Params.omegab+self.cosmo.bkd.Params.omegac) * self.cosmo.bkd.Params.H0**2.) / (const.c.to('km/s').value)  # TODO: change here after setting params in cosmo class

# 		# if zmax is None: self.zmax = defaults.default_precision['magbias_zmax']
# 		# if npts is None: self.npts = defaults.default_precision['magbias_npts']

# 	def W_z(self, z):
# 		return self.fac / self.cosmo.H_z(z) * self.cosmo.f_K(z) * (1.+z) * self.W_lens(z, zmax=self.zmax, npts=self.npts)

# 	def W_lens(self, z, zmax=default_limits['kappagal_zmax'], npts=default_precision['kappagal_npts']):
# 		"""
# 		Computes 
# 		"""
# 		if np.isscalar(z) or (np.size(z) == 1):
# 			integrand = lambda zprime:  (1.-self.cosmo.f_K(z)/self.cosmo.f_K(zprime)) * self.dndz.dndz(zprime)
# 			z_ = np.linspace(z, zmax, npts)
# 			return integrate.simps(integrand(z_), x=z_)
# 		else:
# 			return np.asarray([ self.W_lens(tz) for tz in z ])

