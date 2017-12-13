import numpy as np
from scipy import integrate, interpolate
from astropy import constants as const
from astropy import units as u
from defaults import *
from utils import W_k_tophat, V_sphere

class Bubbles(object):
	def __init__(self, cosmo, 
					   # mass_func,
					   lmin=default_limits['halo_lmin'], 
					   lmax=default_limits['halo_lmax'], 
					   zmin=0., 
					   zmax=1100, 
					   # kmin=default_limits['halo_Mmin'], 
					   # kmax=default_limits['halo_Mmax'], 
					   npts=default_precision['halo_npts'],
					   lrange=None):

		self.cosmo = cosmo
		# self.mass_func = mass_func
		self.lmin  = lmin
		self.lmax  = lmax
		self.npts  = npts
		self.zmin  = zmin

		if zmax is None:
			self.zmax = self.cosmo.zstar
		else:
			self.zmax = zmax

		# Multipoles interval
		if lrange is None:
			self.lrange = np.arange(self.lmin, self.lmax+1) 
		else:
			self.lrange = lrange

		# Integration interval linear in comoving distance \chi
		self.chis = np.linspace(self.zmin, self.cosmo.f_K(self.zmax), self.npts)
		self.zs   = self.cosmo.bkd.redshift_at_comoving_radial_distance(self.chis)
		self.dzs  = (self.zs[2:]-self.zs[:-2])/2
		self.zs   = self.zs[1:-1]
		self.chis = self.chis[1:-1]

		self.b = 1.
		self.R_bar = 5
		self.sigma_lnR = np.log(2)

		self.kappas = np.logspace(-5,3,100)

		self.spline_F_k = interpolate.interp1d(self.kappas, [self.F_k(k) for k in self.kappas])
		self.spline_I_k = interpolate.interp1d(self.kappas, [self.I_k(k) for k in self.kappas])

		# Geometrical factors
		self.fac = self.cosmo.H_z(self.zs) * (u.km).to(u.Mpc)/(self.cosmo.f_K(self.zs)**2 * const.c.to('Mpc/s').value)

	def P_R(self, R):#, R_bar=None, sigma_lnR=None):
		# if R_bar is None:
		# 	R_bar = self.R_bar
		# if sigma_lnR is None:
		# 	sigma_lnR = self.sigma_lnR
		return np.exp( -( np.log(R/self.R_bar) )**2 / ( 2*self.sigma_lnR**2 ) ) / (R * np.sqrt(2.*np.pi*self.sigma_lnR**2))

	def F_k(self, k):
		# def num(x,k):
			# return self.P_R(x) * V_sphere(x)**2 * W_k_tophat(k*x)**2
		num = lambda x,k: self.P_R(x) * V_sphere(x)**2 * W_k_tophat(k*x)**2
		den = lambda y: self.P_R(y) * V_sphere(y)

		Rarr = np.logspace(-5,2)

		return integrate.simps(num(Rarr,k),x=Rarr) / integrate.simps(den(Rarr),x=Rarr)


	def I_k(self, k):
		# def num(x,k):
			# return self.P_R(x) * V_sphere(x) * W_k_tophat(k*x)
		num = lambda x,k: self.P_R(x) * V_sphere(x) * W_k_tophat(k*x)
		den = lambda y: self.P_R(y) * V_sphere(y)

		Rarr = np.logspace(-5,2)

		return self.b*integrate.simps(num(Rarr,k),x=Rarr) / integrate.simps(den(Rarr),x=Rarr)

	def G_k(self, k, z):
		def G_k_integrand(mu,kprime,k_,z_):
			return 1./(2.*np.pi)**3 * self.cosmo.pkz.P(z_, np.sqrt(np.abs(k_**2+kprime**2-2*mu*k_*kprime)),grid=False) * self.spline_F_k(kprime)

		mus    = np.linspace(-1.,1,50)
		kappas = np.logspace(-5,2,50)

		integrand_k = np.empty(len(kappas))
		for idxk, kappa in enumerate(kappas):
			integrand_k_mu = np.empty(len(mus))
			for idxmu, mu in enumerate(mus):
				integrand_k_mu[idxmu] = G_k_integrand(mu, kappa, k, z)
			integrand_k[idxk] = integrate.simps(integrand_k_mu, x=mus)

		return integrate.simps(integrand_k, x=kappas)

	def P_k_bubble_2h(self, k, z,):

		if self.cosmo.x_e(z) >=1.:
			x_e = 0.999999
		else:
			x_e = self.cosmo.x_e(z)

		return ((1 - x_e) * np.log(1 - x_e) * self.I_k(k) - x_e)**2 * self.cosmo.pkz.P(z,k,grid=False)


	def P_k_bubble_1h(self, k, z):
		if self.cosmo.x_e(z) >=1.:
			x_e = 0.999999
		else:
			x_e = self.cosmo.x_e(z)


		return x_e * (1 - x_e) * (self.spline_F_k(k) + self.G_k(k,z))

	def P_k(self, k,z):
		return self.P_k_bubble_1h(k, z) + self.P_k_bubble_2h(k, z)


	def GetCl(self, k1, limb_fact=0.5):
		"""
		Returns the angular power spectrum C_l 

		Parameters
		----------
		"""

		kern = k1.W_z(self.zs, i) * k2.W_z(self.zs, j)
		w    = np.ones(self.zs.shape)
		Cl   = np.zeros(len(self.lrange))
		scal = k1.scal(self.lrange) * k2.scal(self.lrange)

		for ell, L in enumerate(self.lrange):
			# print i
			k = (L+limb_fact)/self.chis
			w[:] = 1
			w[k<1e-4] = 0
			w[k >= self.kmax] = 0
			pkin = self.cosmo.pkz.P(self.zs_pk, k, grid=False)
			common = (w*pkin) * self.fac
    
			Cl[ell] = np.dot(self.dzs, common * kern)
			# Cl[i] = integrate.simps(common * kern, x=self.zs)

		return Cl * scal
