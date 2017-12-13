import numpy as np
from scipy import integrate
from astropy import constants as const
from astropy import units as u
from defaults import *

class Limber(object):
	"""
	Class to compute Limber-approximated spectra by integrating the following:

	C_l = \int_0^{z_*} dz/c H(z)/(\chi(z))^2 W^X(z) W^Y(z) P(k=l/f_K(z), z)

	TODO: add method to convert to w(\theta)

	Attributes
	----------
	cosmo : cosmology class

	lmin : int
		Minimum power spectrum multipole

	lmax : int
		Maximum power spectrum multipole

	kmax : float
		Maximum P(k) scale considered in the Limber integration

	zmin : float
		Minimum redshift considered in the Limber integration

	zmax : float
		Maximum redshift considered in the Limber integration
		
	npts : int
		Number of points to sample Limber integral in redshift
	"""
	def __init__(self, cosmo, 
		               lmin=default_limits['limber_lmin'], 
		               lmax=default_limits['limber_lmax'], 
		               kmax=default_limits['limber_kmax'], 
		               zmin=default_limits['limber_zmin'], 
		               zmax=default_limits['limber_zmax'], 
		               npts=default_precision['limber_npts'],
		               compute_at_z0=False):
		
		# !! FIXME: zmin and zmax not used at the moment
		self.cosmo = cosmo
		self.lmin  = lmin
		self.lmax  = lmax
		self.kmax  = kmax
		self.npts  = npts
		self.zmin  = zmin

		if zmax is None:
			self.zmax = self.cosmo.zstar
		else:
			self.zmax = zmax
		
		# Multipoles interval
		self.lrange = np.arange(self.lmin, self.lmax+1) 

		# Integration interval linear in comoving distance \chi
		self.chis = np.linspace(self.zmin, self.cosmo.f_K(self.zmax), self.npts)
		self.zs   = self.cosmo.bkd.redshift_at_comoving_radial_distance(self.chis)
		self.dzs  = (self.zs[2:]-self.zs[:-2])/2
		self.zs   = self.zs[1:-1]
		self.chis = self.chis[1:-1]

		if compute_at_z0 is False:
			self.zs_pk = self.zs.copy()
		else:
			self.zs_pk = np.zeros_like(self.zs)

		# Geometrical factors
		self.fac = self.cosmo.H_z(self.zs) * (u.km).to(u.Mpc)/(self.cosmo.f_K(self.zs)**2 * const.c.to('Mpc/s').value)

	def GetCl(self, k1, k2=None, i=0, j=0, limb_fact=0.5):
		"""
		Returns the angular power spectrum C_l given the kernel k1 (and k2) objects

		Parameters
		----------
		k1 : kernel object from kernels.py
			Kernel of the first observable

		k2 : kernel object from kernels.py
			Kernel of the second observable (default: k1 = k2)

		i : int
			Redshift bin of first kernel (default = 0)

		j : int
			Redshift bin of second kernel (default = 0)
		"""
		if k2 is None:
			k2 = k1
			j  = i

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
