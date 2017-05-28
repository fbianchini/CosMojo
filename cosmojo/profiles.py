import numpy as np
from scipy import interpolate
from scipy import integrate
from utils import btheta2D
from astropy.convolution import convolve
import astropy.constants as const
import astropy.units as u
from photutils import CircularAperture, CircularAnnulus
from photutils import aperture_photometry

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

class ClusterOpticalDepth():

	def __init__(self, cosmo, mass_def=200.):

		self.epsrel = 1.49e-6
		self.epsabs = 0.
		self.cosmo = cosmo
		self.mass_def = mass_def
		# self.concentration = concentration

	def GetTauProfile(self, M, z, x_c=0.5, gamma=-0.2, reso=0.2, theta_max=10, chi=1., fwhm_arcmin=0., profile='battaglia'):
		"""
		Returns the 2D profile of the optical depth for a cluster of mass M (M_sun) at redshift z.
		"""
		theta_x,theta_y = np.meshgrid(np.arange(-theta_max,theta_max+reso,reso), np.arange(-theta_max,theta_max+reso,reso)) 
		theta = np.sqrt(theta_x**2+theta_y**2) # arcmin

		d_A = self.cosmo.d_A(z) # [Mpc]
		rho_c_z = self.cosmo.rho_c(z) # [kg/m^3]
		f_gas = self.cosmo.omegab/self.cosmo.omegam # we assume f_gas ~ f_baryon
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

			norm_for_int = rho_c_z
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

		return tau_theta

	def GetApertureTau(self, M, z, theta_R, x_c=0.5, gamma=-0.2, reso=0.2, theta_max=10., chi=1., fwhm_arcmin=0., profile='Battaglia'):
		# if theta_R > theta_max:
		# 	theta_max = theta_R * 2.

		tau_profile = self.GetTauProfile(M, z, x_c=x_c, gamma=gamma, reso=reso, theta_max=theta_max, chi=chi, fwhm_arcmin=fwhm_arcmin, profile=profile)

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





