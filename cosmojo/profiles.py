import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy.special import j0
from utils import btheta2D, Interpolate2D, bl, btheta
from astropy.convolution import convolve
import astropy.constants as const
import astropy.units as u
from photutils import CircularAperture, CircularAnnulus
from photutils import aperture_photometry
import matplotlib.pyplot as plt
from numba import jit

from mass_func import MassFunction

# @jit(nopython=True)
def rho_fit_Battaglia16(x, M, z, x_c=0.5, gamma=-0.2, sim='AGN'): # unitless
	"""
	Average density profile as function of halo mass and redshift from Battaglia's model, 
	see Eq.(A1) of arXiv:1607.02442
	! M_Delta in M_sun
	"""
	if sim == 'AGN' or sim == 'agn':
		rho_0 = scale_Battaglia16_params_AGN(M, z, 'rho_0')
		alpha = scale_Battaglia16_params_AGN(M, z, 'alpha')
		beta  = scale_Battaglia16_params_AGN(M, z, 'beta')
	elif sim == 'shock' or sim == 'SHOCK':
		rho_0 = scale_Battaglia16_params_Shock(M, z, 'rho_0')
		alpha = scale_Battaglia16_params_Shock(M, z, 'alpha')
		beta  = scale_Battaglia16_params_Shock(M, z, 'beta')

	# print rho_0, alpha, beta
	x_over_xc = x / x_c
	return rho_0 * x_over_xc**gamma * (1.+(x_over_xc)**alpha)**(-(beta-gamma)/alpha)
	# return np.exp(-x_over_xc**2)

def scale_Battaglia16_params_AGN(M, z, par):
	"""
	Scaling of fit parameters from Battaglia's model, see Eq.(A2) of arXiv:1607.02442
	! M_Delta in M_sun
	"""
	if par == 'rho_0':
		A_0 = 4e3
		alpha_m = 0.29
		alpha_z = -0.66
	elif par == 'alpha':
		A_0 = 0.88
		alpha_m = -0.03
		alpha_z = 0.19
	elif par == 'beta':
		A_0 = 3.83
		alpha_m = 0.04
		alpha_z = -0.025

	return A_0 * (M/1e14)**alpha_m * (1.+z)**alpha_z

def scale_Battaglia16_params_Shock(M, z, par):
	"""
	Scaling of fit parameters from Battaglia's model, see Eq.(A2) of arXiv:1607.02442
	! M_Delta in M_sun
	"""
	if par == 'rho_0':
		A_0 = 1.9e4
		alpha_m = 0.09
		alpha_z = -0.95
	elif par == 'alpha':
		A_0 = 0.7
		alpha_m = -0.017
		alpha_z = 0.27
	elif par == 'beta':
		A_0 = 4.43
		alpha_m = 0.005
		alpha_z = 0.037

	return A_0 * (M/1e14)**alpha_m * (1.+z)**alpha_z

def Tau(M, X_H=0.76, f_star=0.05, Omega_b=0.05, Omega_m=0.307, f_c=1.):
	# in m^-2
	x_e = (X_H+1)/(2.*X_H)
	mu = 4./(3.*X_H+1+X_H*x_e)
	f_b = Omega_b/Omega_m
	return const.sigma_T.value * x_e * X_H * (1-f_star) * f_b * f_c * M / (mu * const.m_p.to('M_sun').value)

# @jit(nopython=True)
def pressure_fit_Battaglia12(x, M, z, alpha=1.0, gamma=-0.3):
	"""
	Average pressure profile as function of halo mass and redshift from Battaglia's model, see Eq.(10) of arXiv:1109.3711
	! x = r / r_200
	! M_Delta in M_sun
	"""
	P_0 = 18.1 * ((M/1e14)**0.154 * (1.+z)**-0.758)
	x_c = 0.497 * ((M/1e14)**-0.00865 * (1.+z)**0.731)
	beta = 4.35 * ((M/1e14)**0.0393 * (1.+z)**0.415)

	x_over_xc = x / x_c

	return P_0 * x_over_xc**gamma * (1.+(x_over_xc)**alpha)**(-beta)

class ClusterOpticalDepth():

	def __init__(self, cosmo, mass_def=200.):

		self.epsrel = 1.49e-6
		self.epsabs = 0.
		self.cosmo = cosmo
		self.mass_def = mass_def
		# self.concentration = concentration

	def GetTauProfile(self, M, z, x_c=0.5, gamma=-0.2, reso=0.2, theta_max=10, chi=1., fwhm_arcmin=0., profile='battaglia', lowpass=False):
		"""
		Returns the 2D profile of the optical depth for a cluster of mass M (M_sun) at redshift z.
		"""
		theta_x,theta_y = np.meshgrid(np.arange(-theta_max,theta_max+reso,reso), np.arange(-theta_max,theta_max+reso,reso)) 
		theta = np.sqrt(theta_x**2+theta_y**2) # arcmin

		d_A = self.cosmo.d_A(z) # [Mpc]
		rho_c_z = self.cosmo.rho_c(z) # [kg/m^3]
		f_gas = self.cosmo.omegab/self.cosmo.omegam # we assume f_gas ~ f_baryon
		BAD_TWEAK = 1.4 # Battaglia profile seems lower in amplitude w.r.t. the plots in the paper
		# print f_gas

		R = np.radians(theta/60.) * d_A # [Mpc]
		R = R.flatten()
		rho_fit = np.zeros_like(R)

		# Radius such as M(< R_Delta) = Delta * rho_crit n stuff
		r_v = (((M*(u.Msun).to(u.kg)/(self.mass_def*4.*np.pi/3.))/rho_c_z)**(1./3.))*(u.m).to(u.Mpc) # [Mpc]

		if profile == 'battaglia' or profile == 'Battaglia':

			for ii in xrange(len(R)):
				def integrand(r):
					return 2. * rho_fit_Battaglia16(r/r_v, M, z, x_c=x_c, gamma=gamma) * ( r / np.sqrt(r**2. - R[ii]**2.) )
				rho_fit[ii] =  integrate.quad(integrand, R[ii], np.inf, epsabs=self.epsabs, epsrel=self.epsrel)[0]

			norm_for_int = BAD_TWEAK * rho_c_z
			# tau_theta = f_gas * rho_fit * u.Mpc.to('m') * rho_c_z * const.sigma_T.value * chi / (1.14*const.m_p.value)

		elif profile == 'NFW' or profile == 'nfw':
			concentration = self.c_Duffy(M,z)
			# print concentration

			#scale radius
			r_s = r_v/concentration # [Mpc]

			#NFW profile properties
			delta_c = (self.mass_def/3.)*(concentration**3.)/(np.log(1.+concentration)-concentration/(1.+concentration))

			#make sure that we get same delta_c using num. integration as well
			t1 = (self.mass_def/3.) * r_v**3. # [Mpc]
			t2 = 1./ r_s**3. / integrate.quad(lambda r : r / (r + r_s)**2., 0., r_v, epsabs=self.epsabs, epsrel=self.epsrel)[0]
			delta_c_2 = t1 * t2 #should be same as delta_c and it is same!
			assert round(delta_c,1) == round(delta_c_2,1)

			norm_for_int = 2. * delta_c * rho_c_z * r_s**3. #[kg/m^3 Mpc^3]

			for ii in xrange(len(R)):
				def integrand(r):
					return  1 / ( r * (r_s + r )**2. ) * ( r / np.sqrt(r**2. - R[ii]**2.))
				rho_fit[ii] =  integrate.quad(integrand, R[ii], np.inf, epsabs=self.epsabs, epsrel=self.epsrel)[0]

		elif profile == 'gNFW_Moore':
			concentration = self.c_Duffy(M,z)
			# print concentration

			#scale radius
			r_s = r_v/concentration
			alpha, beta, gamma = 1.5, 3., 1.5 #arXiv:1005.0411 (Eq. 27); https://arxiv.org/abs/astro-ph/9903164

			t1 = (self.mass_def/3.) * r_v**3. # [Mpc^3]
			t2 = 1. / integrate.quad(lambda r : r**2. * (2. / ( (r/r_s)**gamma * ( 1 + (r/r_s)**(3-gamma) ) ) ), 0., r_v, epsabs=self.epsabs, epsrel=self.epsrel)[0] # [Mpc^3]
			delta_c = t1 * t2 # [Mpc^6]

			norm_for_int = 2. * delta_c * rho_c_z #[Mpc^6 * kg/m^3]

			for ii in range(len(R)):
				rho_fit[ii] =  integrate.quad(lambda r : (2. / ( (r/r_s)**gamma * ( 1 + (r/r_s)**(3-gamma) ) ) ) * ( r / np.sqrt(r**2. - R[ii]**2.) ), R[ii], np.inf, epsabs=self.epsabs, epsrel=self.epsrel)[0]

		# rho -> tau
		tau_theta = f_gas * rho_fit * norm_for_int * u.Mpc.to('m') * const.sigma_T.value * chi / (1.14*const.m_p.value)

		# 1D -> 2D 
		tau_theta = tau_theta.reshape(theta.shape)

		# theta_R = R_Delta/d_A
		# print np.degrees(theta_R)*60.
		# tau_theta = (1.+theta**2/(np.degrees(theta_R)*60.)**2)**(-1.)

		# Apply beam smoothing
		if fwhm_arcmin != 0.:
			tau_theta = convolve(tau_theta, btheta2D(fwhm_arcmin, reso=reso, theta_max=theta_max), normalize_kernel=True)

		# Smoothly filter out scales below the beam
		if lowpass:
			l = np.arange(0,5e4)
			f_l =  np.exp(-(l/(np.pi/np.radians(fwhm_arcmin/60.)))**4)
			filt = Interpolate2D(tau_theta.shape[1], reso, l, f_l)
			tau_theta = np.fft.ifft2(np.fft.fft2(tau_theta)*filt).real
			# plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(tau_theta))))
		return tau_theta

	def GetApertureTau(self, M, z, theta_R, x_c=0.5, gamma=-0.2, reso=0.2, theta_max=10., chi=1., fwhm_arcmin=0., profile='Battaglia', lowpass=False):
		# if theta_R > theta_max:
		# 	theta_max = theta_R * 2.

		tau_profile = self.GetTauProfile(M, z, x_c=x_c, gamma=gamma, reso=reso, theta_max=theta_max, chi=chi, fwhm_arcmin=fwhm_arcmin, profile=profile, lowpass=lowpass)

		theta_R_pix = theta_R/reso

		apertures = CircularAperture([(tau_profile.shape[1]/2., tau_profile.shape[0]/2.)], r=theta_R_pix)
		annulus_apertures = CircularAnnulus([(tau_profile.shape[1]/2., tau_profile.shape[0]/2.)], r_in=theta_R_pix, r_out=theta_R_pix*np.sqrt(2))
		apers = [apertures, annulus_apertures]

		phot_table = aperture_photometry(tau_profile, apers)
		bkg_mean = phot_table['aperture_sum_1'] / annulus_apertures.area()
		bkg_sum = bkg_mean * apertures.area()
		final_sum = phot_table['aperture_sum_0'] - bkg_sum
		phot_table['residual_aperture_sum'] = final_sum

		return phot_table['residual_aperture_sum'][0]/apertures.area()

		# apertures = CircularAperture([(tau_profile.shape[1]/2., tau_profile.shape[0]/2.)], r=theta_R_pix)
		# phot_table = aperture_photometry(tau_profile, apertures)

		# return phot_table['aperture_sum'][0]/(np.pi*theta_R_pix**2)

	def c_Duffy(self, M, z):
	    """
	    Concentration from c(M) relation published in Duffy et al. (2008).
	    """

	    M_pivot = 2.e12/self.cosmo.h # [M_sun]
	    
	    A = 5.71
	    B = -0.084
	    C = -0.47

	    concentration = A * ((M / M_pivot)**B) * (1+z)**C

	    return concentration

class ClusterElectronicPressure():

	def __init__(self, cosmo, mass_def=200.):

		self.epsrel = 1.49e-6
		self.epsabs = 0.
		self.cosmo = cosmo
		self.mass_def = mass_def
		# self.concentration = concentration

	def GetProfile1D(self, M, z, alpha=1.0, gamma=-0.3, X=0.76):
		"""
		Returns the 2D profile of the optical depth for a cluster of mass M (M_sun) at redshift z.
		"""
		Pth2Pe = (2*X+2.)/(5*X+3)

		rho_c_z = self.cosmo.rho_c(z) # [kg/m^3]
		f_gas = self.cosmo.omegab/self.cosmo.omegam # we assume f_gas ~ f_baryon

		# Radius such as M(< R_Delta) = Delta * rho_crit n stuff
		r_v = (((M*(u.Msun).to(u.kg)/(self.mass_def*4.*np.pi/3.))/rho_c_z)**(1./3.))*(u.m).to(u.Mpc) # [Mpc]

		P_Delta = self.mass_def * rho_c_z * f_gas * const.G.value * M / 2. / r_v #[M_sun/Mpc/s^2]
		P_th = P_Delta * pressure_fit_Battaglia12(x, M, z, alpha=alpha, gamma=gamma)

		return P_th * Pth2Pe #[M_sun/Mpc/s^2]

	# @jit(nopython=True)
	def y_ell(self, ell, M, z, alpha=1.0, gamma=-0.3, X=0.76, npts=100):
		'''
		Fourier transform of the Compton-y 
		output: y_ell
		'''
		R200 = (((M*(u.Msun).to(u.kg)/(self.mass_def*4.*np.pi/3.))/self.cosmo.rho_c(z))**(1./3.))*(u.m).to(u.Mpc) # [Mpc]
		c200 = self.c_Duffy(M,z)
		R_s  = R200/c200 # [Mpc]
		ells = self.cosmo.d_A(z)/R_s

		xarr = np.logspace(-5,2,npts)

		Pth2Pe = (2*X+2.)/(5*X+3)
		rho_c_z = self.cosmo.rho_c(z) # [kg/m^3]
		f_gas = self.cosmo.omegab/self.cosmo.omegam # we assume f_gas ~ f_baryon
		r_v = (((M*(u.Msun).to(u.kg)/(self.mass_def*4.*np.pi/3.))/rho_c_z)**(1./3.))*(u.m).to(u.Mpc) # [Mpc]

		P_Delta = self.mass_def * rho_c_z * f_gas * const.G.value * M / 2. / r_v #[M_sun/Mpc/s^2]
		# P_th = Pth2Pe * P_Delta * pressure_fit_Battaglia12(x, M, z, alpha=alpha, gamma=gamma)
		pressure_profile = Pth2Pe * P_Delta * pressure_fit_Battaglia12(xarr/c200, M, z, alpha=alpha, gamma=gamma) #[M_sun/Mpc/s^2]
		arg = ((ell+0.5)*xarr/ells)
		yell = integrate.simps(xarr**2 * np.sin(arg)/arg * pressure_profile, xarr)
		yell *= 4*np.pi*(R_s) / ells**2 #[Mpc]
		yell *= u.M_sun.to(u.kg) * const.sigma_T.value / const.m_e.value / const.c.value**2 

		return yell

	# @np.vectorize
	def c_Duffy(self, M, z):
	    """
	    Concentration from c(M) relation published in Duffy et al. (2008).
	    """

	    M_pivot = 2.e12/self.cosmo.h # [M_sun]
	    
	    A = 5.71
	    B = -0.084
	    C = -0.47

	    concentration = A * ((M / M_pivot)**B) * (1+z)**C

	    return concentration

class ClusterLensing():

	def __init__(self, cosmo, mass_def=200.):

		self.epsrel = 1.49e-8
		self.epsabs = 0.
		self.cosmo = cosmo
		self.mass_def = mass_def
		self.hmf = MassFunction(self.cosmo)
		# self.concentration = concentration

	def SigmaCritical(self, z, z_source=1090):
		fact = const.c**2 / (4.*np.pi*const.G) # [kg/m]
		D_S = (self.cosmo.d_A(z_source) * u.Mpc).to(u.m) # [m]
		D_L = (self.cosmo.d_A(z) * u.Mpc).to(u.m) # [m]
		D_LS = (self.cosmo.d_A12(z,z_source) * u.Mpc).to(u.m) # [m]

		# D_S = (self.cosmo.f_K(z_source) * u.Mpc).to(u.m) / (1+z_source) # [Mpc]
		# D_L = (self.cosmo.f_K(z) * u.Mpc).to(u.m) / (1+z) # [Mpc]
		# D_LS = ((self.cosmo.f_K(z_source) - self.cosmo.f_K(z)) * u.Mpc).to(u.m) / (1+z_source)  # [Mpc]

		# return fact * D_S/(D_L*D_LS) # [kg/m^2]
		return (fact * D_S/(D_L*D_LS)).value # [kg/m^2]

	def GetKappa1hProfile(self, M, z, reso=0.2, theta_min=1e-2, theta_max=10, chi=1., fwhm_arcmin=0., profile='nfw', conc=None):
		"""
		!!! NOT YET WORKING
		Returns the 2D profile of the optical depth for a cluster of mass M (M_sun) at redshift z.
		"""
		theta = np.arange(theta_min,theta_max+reso,reso) # arcmin

		d_A = self.cosmo.d_A(z)  # [Mpc]
		rho_c_z = self.cosmo.rho_c(z)# * u.kg/u.m**3# [kg/m^3]

		R = np.radians(theta/60.) * d_A # [Mpc]
			
		rho_fit = np.zeros_like(R)

		# Radius such as   M(< R_Delta) = Delta * rho_crit n stuff
		r_v = (((M*(u.Msun).to(u.kg)/(self.mass_def*4.*np.pi/3.))/rho_c_z)**(1./3.))*(u.m).to(u.Mpc) # [Mpc]

		print 'r_v = %f Mpc' %r_v

		if conc is None:
			concentration = self.c_Duffy(M,z)
		else:
			concentration = conc

		if profile == 'NFW' or profile == 'nfw':
			# print  'c = %f' %concentration

			#scale radius
			r_s = r_v/concentration # [Mpc]
			print 'r_s = %f Mpc' %r_s

			#NFW profile properties
			delta_c = (self.mass_def/3.)*(concentration**3.)/(np.log(1.+concentration)-concentration/(1.+concentration))
			print 'delta_c = %f' %delta_c

			#make sure that we get same delta_c using num. integration as well
			t1 = (self.mass_def/3.) * r_v**3. # [Mpc]
			t2 = 1./ r_s**3. / integrate.quad(lambda r : r / (r + r_s)**2., 0., r_v, epsabs=self.epsabs, epsrel=self.epsrel)[0]
			delta_c_2 = t1 * t2 #should be same as delta_c and it is same!
			assert round(delta_c,1) == round(delta_c_2,1)

			norm_for_int = 2. * delta_c * rho_c_z * r_s**3. #[kg/m^3 Mpc^3]
			print 'norm_for_int = %e' %norm_for_int

			for ii in xrange(len(R)):
				def integrand(r):
					return  1 / ( r * (r_s + r )**2. ) * ( r / np.sqrt(r**2. - R[ii]**2.))
				rho_fit[ii] =  integrate.quad(integrand, R[ii], np.inf, epsabs=self.epsabs, epsrel=self.epsrel)[0] # 1 / Mpc^2
				# print R[ii]
				# print rho_fit[ii]* norm_for_int * u.Mpc.to('m')
				# quit()

		elif profile == 'gNFW_Moore':
			concentration = self.c_Duffy(M,z)
			# print concentration

			#scale radius
			r_s = r_v/concentration
			alpha, beta, gamma = 1.5, 3., 1.5 #arXiv:1005.0411 (Eq. 27); https://arxiv.org/abs/astro-ph/9903164

			t1 = (self.mass_def/3.) * r_v**3. # [Mpc^3]
			t2 = 1. / integrate.quad(lambda r : r**2. * (2. / ( (r/r_s)**gamma * ( 1 + (r/r_s)**(3-gamma) ) ) ), 0., r_v, epsabs=self.epsabs, epsrel=self.epsrel)[0] # [Mpc^3]
			delta_c = t1 * t2 # [Mpc^6]

			norm_for_int = 2. * delta_c * rho_c_z #[Mpc^6 * kg/m^3]

			for ii in range(len(R)):
				rho_fit[ii] =  integrate.quad(lambda r : (2. / ( (r/r_s)**gamma * ( 1 + (r/r_s)**(3-gamma) ) ) ) * ( r / np.sqrt(r**2. - R[ii]**2.) ), R[ii], np.inf, epsabs=self.epsabs, epsrel=self.epsrel)[0]

		# rho 
		rho = rho_fit * norm_for_int * u.Mpc.to('m')

		# # Apply beam smoothing
		# if fwhm_arcmin != 0.:
		# 	tau_theta = convolve(tau_theta, btheta2D(fwhm_arcmin, reso=reso, theta_max=theta_max), normalize_kernel=True)

		return rho/self.SigmaCritical(z)

	def GetKappa2hProfile(self, M, z, reso=0.2, theta_min=1e-2, theta_max=10, chi=1., fwhm_arcmin=0., profile='nfw', method='simps', lmax=10000):
		"""
		Returns the 2D profile of the optical depth for a cluster of mass M (M_sun) at redshift z.
		"""
		theta = np.arange(theta_min,theta_max+reso,reso) # arcmin

		d_A = self.cosmo.d_A(z)  # [Mpc]
		rho_m_z = self.cosmo.rho_bar(z=z) * u.M_sun.to('kg') / u.Mpc.to('m')**3 # [kg/m^3]
		bias = self.hmf.bias_M(M, z)			
		Sigma_crit = self.SigmaCritical(z) # [kg/m^2]
		fact = rho_m_z * bias  / (1+z)**3 / Sigma_crit / d_A**2 / 2./ np.pi / u.m.to('Mpc') # 1 / Mpc^3

		r_v = (((M*(u.Msun).to(u.kg)/(self.mass_def*4.*np.pi/3.))/self.cosmo.rho_c(z))**(1./3.))*(u.m).to(u.Mpc) # [Mpc]
		theta_vir = np.degrees(r_v/d_A) * 60.

		ell = np.arange(lmax+1)

		kappa_2h = np.zeros_like(theta)

		if method == 'quad':
			for idt in xrange(theta.size):
				# integrand = ell * j0(ell*np.radians(theta[idt]/60.)) * self.cosmo.pkz.P(z, ell/((1+z)*d_A), grid=False)
				# kappa_2h[idt] = integrate.simps(integrand, x=ell)
				def integrand(l):
					return l * j0(l*np.radians(theta[idt]/60.)) * self.cosmo.pkz.P(z, l/((1+z)*d_A), grid=False)
				kappa_2h[idt] =  integrate.quad(integrand, 0., np.inf, epsabs=self.epsabs, epsrel=self.epsrel)[0] # 1 / Mpc^2
		elif method == 'simps':
			for idt in xrange(theta.size):
				integrand = ell * j0(ell*np.radians(theta[idt]/60.)) * self.cosmo.pkz.P(z, ell/((1+z)*d_A), grid=False)# * bl(fwhm_arcmin, lmax=lmax)**2
				kappa_2h[idt] = integrate.simps(integrand, x=ell)

		if fwhm_arcmin != 0.:
			kappa_2h = convolve(kappa_2h, btheta (fwhm_arcmin, theta),)

			# plt.plot(ell, integrand)			

		return kappa_2h * fact, theta_vir

	def c_Duffy(self, M, z):
	    """
	    Concentration from c(M) relation published in Duffy et al. (2008).
	    """

	    M_pivot = 2.e12/self.cosmo.h # [M_sun]
	    
	    A = 5.71
	    B = -0.084
	    C = -0.47

	    concentration = A * ((M / M_pivot)**B) * (1+z)**C

	    return concentration



