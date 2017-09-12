import numpy as np
from scipy import integrate
from astropy import constants as const
from astropy import units as u
from defaults import *

class HaloModel(object):
	def __init__(self, cosmo, 
					   mass_func,
					   lmin=default_limits['halo_lmin'], 
					   lmax=default_limits['halo_lmax'], 
					   zmin=default_limits['halo_zmin'], 
					   zmax=default_limits['halo_zmax'], 
					   Mmin=default_limits['halo_Mmin'], 
					   Mmax=default_limits['halo_Mmax'], 
					   npts=default_precision['halo_npts'],
					   lrange=None):

		self.cosmo = cosmo
		self.mass_func = mass_func
		self.lmin  = lmin
		self.lmax  = lmax
		self.npts  = npts
		self.zmin  = zmin
		self.Mmin  = Mmin
		self.Mmax  = Mmax

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
		# self.dzs  = (self.zs[2:]-self.zs[:-2])/2
		# self.zs   = self.zs[1:-1]
		# self.chis = self.chis[1:-1]

		# Integration interval linear in mass
		self.Ms = np.logspace(np.log10(self.Mmin), np.log10(self.Mmax), self.npts)

		# Factors
		self.dVdzdOmegas = const.c.to('km/s').value * (1+self.zs)**2 * self.cosmo.d_A(self.zs)**2. / self.cosmo.H_z(self.zs) 
		self.dndms = np.asarray([self.mass_func.dndm(self.Ms, z) for z in self.zs]) 
		self.bs = np.asarray([self.mass_func.bias_M(self.Ms, z) for z in self.zs]) # each row is bias at given M

	def GetCl(self, k1, k2=None, all_terms=False):
		"""
		Returns the angular power spectrum C_l given the kernel k1 (and k2) objects

		Parameters
		----------
		k1 : (ell, M, z)
			Kernel of the first observable

		k2 : kernel object from kernels.py
			Kernel of the second observable (default: k1 = k2)
		"""
		if k2 is None:
			k2 = k1

		cl1h = self.GetCl1Halo(k1, k2=k2)
		cl2h = self.GetCl2Halo(k1, k2=k2)

		if all_terms:
			return  cl1h + cl2h, cl1h, cl2h 
		else:
			return cl1h + cl2h

	def GetCl1Halo(self, k1, k2=None):
		Cl1h = np.zeros(len(self.lrange))

		for i, L in enumerate(self.lrange):    
			M_int = self.GetCl1HaloMassInt(L, k1, k2=k2)
			# Cl1h[i] = np.dot(self.dzs, M_int * self.dVdzdOmegas)
			Cl1h[i] = integrate.simps( M_int * self.dVdzdOmegas, x=self.zs)

		return Cl1h

	def GetCl1HaloMassInt(self, ell, k1, k2=None):
		M_integral = np.zeros(len(self.zs))

		if k2 is None:
			for idz, z in enumerate(self.zs):
				k1_int = np.asarray([k1(ell, M, z) for M in self.Ms])
				# M_integral[idz] = np.dot(self.Ms, k1_int * k2_int * self.dndms[idz,:])
				M_integral[idz] = integrate.simps( k1_int * k1_int * self.dndms[idz,:], x=self.Ms)
		else:
			for idz, z in enumerate(self.zs):
				k1_int = np.asarray([k1(ell, M, z) for M in self.Ms])
				k2_int = np.asarray([k2(ell, M, z) for M in self.Ms])
				# M_integral[idz] = np.dot(self.Ms, k1_int * k2_int * self.dndms[idz,:])
				M_integral[idz] = integrate.simps( k1_int * k2_int * self.dndms[idz,:], x=self.Ms)

		return M_integral

	def GetCl2Halo(self, k1, k2=None):
		if k2 is None:
			k2 = k1

		Cl2h = np.zeros(len(self.lrange))

		for i, L in enumerate(self.lrange):    
			M_int1 = self.GetCl2HaloMassInt(L, k1)
			if k1 == k2:
				M_int2 = M_int1.copy()
			else:
				M_int2 = self.GetCl2HaloMassInt(L, k2)
			k = (L+0.5)/self.chis
			pkin = self.cosmo.pkz.P(self.zs, k, grid=False)
			Cl2h[i] = integrate.simps( M_int1 * M_int2 * self.dVdzdOmegas * pkin, x=self.zs)
			# Cl2h[i] = np.dot(self.dzs, M_int1 * M_int2 * self.dVdzdOmegas * pkin)

		return Cl2h

	def GetCl2HaloMassInt(self, ell, k1):
		M_integral = np.zeros(len(self.zs))

		for idz, z in enumerate(self.zs):
			k1_int = np.asarray([k1(ell, M, z) for M in self.Ms])
			M_integral[idz] = integrate.simps( k1_int * self.bs[idz,:] * self.dndms[idz,:], x=self.Ms)
			# M_integral[idz] = np.dot(self.Ms, k1_int * self.bs[idz,:] * self.dndms[idz,:])

		return M_integral







