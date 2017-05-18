import numpy as np
from scipy import interpolate
from scipy import integrate

from utils import btheta2D

from astropy.convolution import convolve
import astropy.constants as const
import astropy.units as u

from photutils import CircularAperture
from photutils import aperture_photometry

def rho_fit_Battaglia16(x, M, z, x_c=0.5, gamma=-0.2): # unitless
	"""
	Average density profile as function of halo mass and redshift from Battaglia's model, 
	see Eq.(A1) of arXiv:1607.02442
	! M_Delta in M_sun
	"""
	rho_0 = scale_Battaglia16_params(M, z, 'rho_0')
	alpha = scale_Battaglia16_params(M, z, 'alpha')
	beta  = scale_Battaglia16_params(M, z, 'beta')
	x_over_xc = x / x_c
	return rho_0 * x_over_xc**gamma * (1.+(x_over_xc)**alpha)**(-(beta-gamma)/alpha)

def scale_Battaglia16_params(M, z, par):
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

def Tau(M, X_H=0.76, f_star=0.05, Omega_b=0.05, Omega_m=0.307, f_c=1.):
	# in m^-2
	x_e = (X_H+1)/(2.*X_H)
	mu = 4./(3.*X_H+1+X_H*x_e)
	f_b = Omega_b/Omega_m
	return const.sigma_T.value * x_e * X_H * (1-f_star) * f_b * f_c * M / (mu * const.m_p.to('M_sun').value)

class ClusterOpticalDepth():

	def __init__(self, cosmo, mass_def=200.):

		self.epsrel = 1.49e-6
		self.epsabs = 0.
		self.cosmo = cosmo
		self.mass_def = mass_def

	def GetTauProfile(self, mass, z, x_c=0.5, gamma=-0.2, reso=0.2, theta_max=10, chi=1., fwhm_arcmin=0.):
		theta_x,theta_y = np.meshgrid(np.arange(-theta_max,theta_max+reso,reso), np.arange(-theta_max,theta_max+reso,reso)) 
		theta = np.sqrt(theta_x**2+theta_y**2) # arcmin

		d_A = self.cosmo.d_A(z) # [Mpc]
		rho_c_z = self.cosmo.rho_c(z) # [kg/m^3]

		R = np.radians(theta/60.) * d_A # [Mpc]
		R = R.flatten()
		tau_theta = np.zeros_like(R)

		R_Delta = (((mass*(u.Msun).to(u.kg)/(self.mass_def*4.*np.pi/3.))/rho_c_z)**(1./3.))*(u.m).to(u.Mpc) # [Mpc]

		for ii in xrange(len(R)):
			def integrand(r): # Sure about the factor 2?
				return 2. * rho_fit_Battaglia16(r/R_Delta, mass, z, x_c=x_c, gamma=gamma) * ( r / np.sqrt(r**2. - R[ii]**2.) )
			tau_theta[ii] =  integrate.quad(integrand, R[ii], np.inf, epsabs=self.epsabs, epsrel=self.epsrel)[0]
			# Sigma_int[ii] =  integrate.quad(lambda r : norm_for_int / ( r * (r_s.value + r )**2. ) * ( r / np.sqrt(r**2. - R[ii]**2.) ), R[ii], np.inf, epsabs=self.epsabs, epsrel=self.epsrel)[0]

		tau_theta *= u.Mpc.to('m') * rho_c_z * const.sigma_T.value * chi / (1.14*const.m_p.value)

		# 1D -> 2D 
		tau_theta = tau_theta.reshape(theta.shape)

		# Apply beam smoothing
		if fwhm_arcmin != 0.:
			tau_theta = convolve(tau_theta, btheta2D(fwhm_arcmin, reso=reso, theta_max=theta_max), normalize_kernel=True)

		return tau_theta

	def GetApertureTau(self, mass, z, theta_R, x_c=0.5, gamma=-0.2, reso=0.2, theta_max=10, chi=1., fwhm_arcmin=0.):
		tau_profile = self.GetTauProfile(mass, z, x_c=x_c, gamma=gamma, reso=reso, theta_max=theta_max, chi=chi, fwhm_arcmin=fwhm_arcmin)

		theta_R_pix = theta_R/reso

		apertures = CircularAperture([(tau_profile.shape[1]/2., tau_profile.shape[0]/2.)], r=theta_R_pix)
		phot_table = aperture_photometry(tau_profile, apertures)

		return phot_table['aperture_sum'][0]/(np.pi*theta_R_pix**2)







