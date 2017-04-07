import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy import special
from defaults import *
from utils import *
from mass_func import MassFunction

class CorrFunction(object):
	"""
	Class to compute the matter correlation function xi(r,z).

	Attributes
	----------
	cosmo : Cosmo object (from universe.py)
	    Cosmology object
	
	z : float
		Redshift

	r_min : float
		Minimum comoving separation

	r_max : float
		Maximum comoving separation

	r_npts : int
		Number of points to sample integration in configuration space

	R : float
		Top-hat filter size to smooth small-scale matter fluctuations [in Mpc]

	"""
	def __init__(self, cosmo, 
					   z=0.,
					   r_min=default_limits['r_min'],
					   r_max=default_limits['r_max'],
					   r_npts=default_precision['corr_npts'],
					   R=3.,
					   k_min=None,
					   k_max=None,
					   k_npts=1e4,
					   **kws):

		self.cosmo = cosmo
		self.r_min = r_min
		self.r_max = r_max
		self.r_npts = r_npts
		self.R = R/self.cosmo.h # Top-hat filter size to smooth small-scale fluctuations [in Mpc]
		self.z = z

		# Create array w/ distances r
		self.log_r_min = np.log10(r_min)
		self.log_r_max = np.log10(r_max)
		self.r_array = np.linspace(self.r_min, self.r_max, self.r_npts)
		if r_min == r_max:
			self.log_r_min = np.log10(r_min)
			self.log_r_max = np.log10(r_min)
			self.r_array = np.array([r_min])
		self.xi_array = np.zeros(self.r_array.size)
		self.xi_bar_array = np.zeros(self.r_array.size)
		
		# Create array w/ wavenumbers k
		if k_min is None:
		    k_min = self.cosmo.kmin
		self.k_min = k_min
		self.ln_k_min = np.log(k_min)
		if k_max is None:
		    k_max = self.cosmo.kmax
		self.k_max = k_max
		self.ln_k_max = np.log(k_max)
		self.lnks = np.linspace(self.ln_k_min, self.ln_k_max, k_npts)

		self.initialized_xi_spline = False
		self.initialized_xi_bar_spline = False

	def xi_log_integrand(self, lnk, r):
		"""
		not used
		"""
		k  = np.exp(lnk)
		kr = k * r
		W  = W_k_tophat(k*self.R/self.cosmo.h) #FIXME: divide by h???
		pk = self.cosmo.pkz.P(self.z, k)

		return k**3 * W**2 * pk * special.j0(kr)

	def xi_integrand(self, k, r):
		"""
		Correlation function integrand (as function of scale k and comoving separation r)
		"""
		return k / r * self.cosmo.pkz.P(self.z, k) * W_k_tophat(k*self.R/self.cosmo.h) #FIXME: divide by h???? Ah, this is because Soergel smooths the PS over a 3 Mpc/h scale 

	def xi_bar_integrand(self, rp):
		"""
		Spherical averaged correlation function integrand (as function of comoving separation r)
		"""
		return rp**2 * self.xi(rp)

	def compute_xi(self):
		"""
		Compute the value of the correlation over the range r_min - r_max
		"""
		# print('I AM COMPUTING XI')
		for idx, r in enumerate(self.r_array):
			self.xi_array[idx] = self.raw_xi(r)
		self._xi_spline = interpolate.InterpolatedUnivariateSpline(self.r_array, self.xi_array)
		self.initialized_xi_spline = True
		return None

	def compute_xi_bar(self):
		"""
		Compute the value of the spherical averaged correlation function over the range r_min - r_max
		"""
		for idx, r in enumerate(self.r_array):
			self.xi_bar_array[idx] = self.raw_xi_bar(r)
		self._xi_bar_spline = interpolate.InterpolatedUnivariateSpline(self.r_array, self.xi_bar_array)
		self.initialized_xi_bar_spline = True
		return None

	def raw_xi(self, r):
		"""
		Compute the value of the matter correlation function at array values r

        Parameters
        ----------   
		r : float or array  
		   Position values [Mpc]
		"""
		try:
			xi_out = np.empty(len(r))
			for idx, r_ in enumerate(r):
				# xi_out[idx] = integrate.simps( self.xi_log_integrand(self.lnks, r_), x=self.lnks ) / (2.*np.pi**2) 
				# print k_min, k_max
				xi_out = integrate.quad(self.xi_integrand, self.k_min, self.k_max, args=r_, epsabs=0.0,
						epsrel=default_precision['corr_precision'], limit=200, weight='sin', wvar=r_)[0]
				xi_out[idx] = xi_out/(2.0 * np.pi**2)
		except TypeError:
			# xi_out = integrate.simps( self.xi_log_integrand(self.lnks, r), x=self.lnks ) / (2.*np.pi**2) 
			# k_min = 1E-6 / r
			# k_max = 10.0 / 0.001 / r
			# print k_min, k_max
			xi_out = integrate.quad(self.xi_integrand, self.k_min, self.k_max, args=r, epsabs=0.0,
					epsrel=default_precision['corr_precision'], limit=200, weight='sin', wvar=r)[0]
			xi_out /= (2.0 * np.pi**2)

		return xi_out

	def raw_xi_bar(self, r): # [unitless]
		"""
		Compute the value of the matter correlation function averaged within a sphere of 
		comoving radius r 

        \bar{xi}(x, a) = 3 x^{-3} \int_0^x xi(y, a) y^2 dy

        Parameters
        ----------   
		   r: float array of position values in Mpc
		"""
		try:
			xi_out = np.empty(len(r))
			for idx, r_ in enumerate(r):
				# r_arr = np.linspace(1e-3, r_, 1000) 
				# xi_out[idx] = integrate.simps( self.xi_bar_integrand(r_arr), x=r_arr ) * 3. / r_**3 
				xi_out = integrate.quad(self.xi_bar_integrand, 1e-3, r_, epsabs=0.0,
						epsrel=default_precision['corr_precision'], limit=200)[0]
				xi_out[idx] = xi_out * 3. / r_**3

		except TypeError:
			# r_arr = np.linspace(1e-3, r, 1000) 
			# xi_out = integrate.simps( self.xi_bar_integrand(r_arr), x=r_arr ) * 3. / r**3 
			xi_out = integrate.quad(self.xi_bar_integrand, 1e-3, r, epsabs=0.0,
					epsrel=default_precision['corr_precision'], limit=200)[0] 
			xi_out = xi_out * 3. / r**3

		return xi_out

	def xi(self, r):
		"""
		Wrapper around correlation function spline
		"""
		if not self.initialized_xi_spline:
			self.compute_xi()
		# r_min = 10. ** self.log_r_min
		# r_max = 10. ** self.log_r_max
		return np.where(np.logical_and(r <= self.r_max, r > self.r_min), self._xi_spline(r), 0.0)

	def xi_bar(self, r):
		"""
		Wrapper around spherical averaged correlation function spline
		"""
		if not self.initialized_xi_bar_spline:
			self.compute_xi_bar()
		# r_min = 10. ** self.log_r_min
		# r_max = 10. ** self.log_r_max
		return np.where(np.logical_and(r <= self.r_max, r > self.r_min), self._xi_bar_spline(r), 0.0)
