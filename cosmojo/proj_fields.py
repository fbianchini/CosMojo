import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy import special
from defaults import *
from utils import *
from mass_func import MassFunction
from universe import Cosmo
from bispectrum import Bispectrum
from scipy.io import readsav
import numba

import pylab as plt

@numba.jit
def Cl_T2T2_filt_integral(ttfiltpowerspec2d, dl, n, l, lmax):
	runningtotal = 0.
	usedvals = 0

	il = ltoi(l, n, dl)

	for ipx in range(n): 
		for ipy in range(n): 
			lp = itol((ipx,ipy), n, dl)
			l_min_lp = (l[0] - lp[0], l[1] - lp[1])
			il_min_ilp = ltoi(l_min_lp, n, dl)
		
			if not(circlecheck(lp, lmax, 0)): continue
			if not(circlecheck(l_min_lp, lmax, 0)): continue
				
			if (ipx < 0 \
				or ipx >= n \
				or il_min_ilp[0] < 0 \
				or il_min_ilp[0] >= n \
				or ipy < 0 \
				or ipy >= n \
				or il_min_ilp[1] < 0 \
				or il_min_ilp[1] >= n ):
				continue
			
			runningtotal += ttfiltpowerspec2d[ipx,ipy] * ttfiltpowerspec2d[il_min_ilp[0]][il_min_ilp[1]]
			usedvals += 1
  
	runningtotal *= 2. * dl * dl / (2*np.pi) / (2*np.pi)
	
	if runningtotal == 0.:
		return 0.
	else:
		return runningtotal

@numba.jit
def Cl_T2T2_filt_integrated(ttfiltpowerspec2d, dl, n, nwanted, lmax):
	output = np.zeros(nwanted)
	 
	for iell in range(1,nwanted):
		ell = (dl * iell, 0)
		output[iell] = Cl_T2T2_filt_integral(ttfiltpowerspec2d, dl, n, ell, lmax)

	return output

@numba.jit
def Cl_lens_correction_integral(ttunlpowerspec2d, filter2d, dl, n, l, lmax):
	runningtotal = 0.
	usedvals = 0

	il = ltoi(l, n, dl)

	for iLx in range(n): 
		for iLy in range(n): 
			L = itol((iLx,iLy), n, dl)
			L_min_l = (L[0] - l[0], L[1] - l[1])
			iL_min_il = ltoi(L_min_l, n, dl)
		
			if not(circlecheck(L, lmax, 0)): continue
			if not(circlecheck(L_min_l, lmax, 0)): continue
				
			if (iLx < 0 or iLx >= n or iL_min_il[0] < 0 or iL_min_il[0] >= n or iLy < 0 or iLy >= n or iL_min_il[1] < 0 or iL_min_il[1] >= n ):
				continue
			
			runningtotal += filter2d[iLx,iLy] * filter2d[iL_min_il[0],iL_min_il[1]] * ttunlpowerspec2d[iL_min_il[0]][iL_min_il[1]] * dotprod(l,L_min_l)
			usedvals += 1
  
	runningtotal *= dl * dl / (2*np.pi) / (2*np.pi)
	
	if runningtotal == 0.:
		return 0.
	else:
		return runningtotal

@numba.jit
def Cl_lens_correction_integrated(ttunlpowerspec2d, filter2d, dl, n, nwanted, lmax):
	output = np.zeros(nwanted)
	 
	for iell in range(1,nwanted):
		ell = (dl * iell, 0)
		output[iell] = Cl_lens_correction_integral(ttunlpowerspec2d, filter2d, dl, n, ell, lmax)

	return output


class ProjField(object):
	"""
	Class to compute the .

	Attributes
	----------
	cosmo : Cosmo object (from universe.py)
		Cosmology object
	
	"""
	def __init__(self, bispectrum, 
					   fwhm_arcmin=0.,
					   noise_uK_arcmin=0.,
					   fl_eq_1=False,
					   lmax=8000,
					   lknee=None,
					   alpha=None,
					   fg=None,
					   fg_path='../data/george_plot_bestfit_line.sav',
					   **kws):

		self.fwhm_arcmin = fwhm_arcmin
		self.noise_uK_arcmin = noise_uK_arcmin

		self.bispectrum = bispectrum

		self.lmax = lmax

		# Beam
		self.bl = bl(self.fwhm_arcmin, lmax=self.lmax)[2:]
		# self.bl2 = bl(self.fwhm_arcmin, lmax=self.lmax)[2:]

		# Instrumental noise (no beam)
		self.nl = nl_cmb(self.noise_uK_arcmin, 0., lmax=self.lmax, lknee=lknee, alpha=alpha)[2:]
		
		# Read in foregrounds templates from George+14 (@ 150 GHz !!!)
		self.george_fg_dic = readsav(fg_path)
		self.dl_ksz        = self.george_fg_dic['ml_dls'][3][4]
		self.dl_tsz        = self.george_fg_dic['ml_dls'][3][5]
		self.dl_dg_poiss   = self.george_fg_dic['ml_dls'][3][2]
		self.dl_dg_clust   = self.george_fg_dic['ml_dls'][3][3]

		self.dl_ksz[:200] = self.dl_ksz[200] # There's a strange bump in the kSZ template

		self.l_fgs = np.arange(2,self.dl_ksz.size+2)

		self.cl_ksz      = self.dl_ksz      * 2*np.pi/(self.l_fgs*(self.l_fgs+1.))  
		self.cl_tsz      = self.dl_tsz      * 2*np.pi/(self.l_fgs*(self.l_fgs+1.))  
		self.cl_dg_poiss = self.dl_dg_poiss * 2*np.pi/(self.l_fgs*(self.l_fgs+1.))  
		self.cl_dg_clust = self.dl_dg_clust * 2*np.pi/(self.l_fgs*(self.l_fgs+1.))  

		if self.lmax < self.l_fgs.size:
			self.cl_ksz      = self.cl_ksz[:self.lmax-1]        
			self.cl_tsz      = self.cl_tsz[:self.lmax-1]        
			self.cl_dg_poiss = self.cl_dg_poiss[:self.lmax-1]   
			self.cl_dg_clust = self.cl_dg_clust[:self.lmax-1]   

		# if fg == 'all':
		# 	for i in xrange(2,9):
		# 		self.dl_fgs += self.george_fg_dic['ml_dls'][3][i]
		# elif fg == 'kSZ':
		# 	self.dl_fgs += self.george_fg_dic['ml_dls'][3][4]
		
		# self.cl_fgs = 2*np.pi/(self.l_fgs*(self.l_fgs+1.)) * self.dl_fgs

		# fgs = np.zeros_like(nl)

		# if lmax < cl_fgs.size:
		# 	fgs += cl_fgs[:lmax-1]
		# else:
		# 	fgs[:cl_fgs.size] += cl_fgs

		self.cl_fgs = self.cl_ksz.copy() #only kSZ for now

		# Total CMB power spectrum
		self.cmbs   = self.bispectrum.cosmo_lin.cmb_spectra(lmax=self.lmax)[2:,:] 
		self.cl_tt  = self.cmbs[:,0].copy()
		self.cl_tot = self.cl_tt + self.cl_fgs + self.nl 

		self.cmbs_unl  = self.bispectrum.cosmo_lin.cmb_spectra(lmax=self.lmax, spec='unlensed_scalar')[2:,:] 
		self.cl_tt_unl = self.cmbs_unl[:,0].copy()
		self.l_cmb     = np.arange(2,self.lmax+1)

		# Wiener filter F
		self.F_l = self.cl_ksz / self.cl_tot

		# self.f_l = self.F_l * np.sqrt(self.bl2)
		self.f_l = self.F_l * self.bl

		if self.noise_uK_arcmin == 0. and self.fwhm_arcmin == 0:
			self.f_l_spline = interpolate.InterpolatedUnivariateSpline(np.arange(2,self.lmax+1), self.f_l, ext=3, k=4)
		else:
			self.f_l_spline = interpolate.InterpolatedUnivariateSpline(np.arange(2,self.lmax+1), self.f_l, ext=1, k=4)

		if fl_eq_1 is True:
			self.f_l_spline = interpolate.InterpolatedUnivariateSpline(np.arange(100000), np.ones(100000), ext=0)


		# Temperature power spectrum filtered
		# self.cl_T2_filt = self.F_l**2 * self.bl2 * (self.cl_tt + self.cl_fgs + self.nl/self.bl2)
		self.cl_T2_filt = self.F_l**2 * self.bl**2 * (self.cl_tt + self.cl_fgs + self.nl/self.bl**2)


		self.ks = np.logspace(-3,2,200)#np.logspace(np.log10(self.bispectrum.k_min),np.log10(self.bispectrum.k_max),100)
		self.zs = np.linspace(0,5,200)

		self.initialized_Tau_spline = False
		self.initialized_E3_spline = False

	def raw_E3(self, k, z, nq=1000):
		# def integrandA(logq):
		# 	q  = np.exp(logq)
		# 	a  = self.bispectrum.fit_a(z, q, grid=False)
		# 	fq = 1.#self.f_l_spline(q*self.bispectrum.cosmo_lin.f_K(z))
		# 	pq = self.bispectrum.cosmo_nl.pkz.P(z,q)

		# 	return a * fq * q**2 * pq / (2.*np.pi**2)

		# def integrandC(logq):
		# 	q  = np.exp(logq)
		# 	c  = self.bispectrum.fit_c(z, q, grid=False)
		# 	fq = 1.#self.f_l_spline(q*self.bispectrum.cosmo_lin.f_K(z))
		# 	pq = self.bispectrum.cosmo_nl.pkz.P(z,q)

		# 	return c * fq * q**2 * pq / (2.*np.pi**2)

		# lnqs = np.linspace(np.log(self.bispectrum.k_min), np.log(self.bispectrum.k_max), nk)
		# intA = integrate.simps( integrandA(lnqs), x=lnqs )
		# intC = integrate.simps( integrandC(lnqs), x=lnqs )

		# print intA
		# def integrandA(q):
		# 	return self.bispectrum.fit_a(z, q, grid=False) * self.bispectrum.cosmo_nl.pkz.P(z,q) * q / (2*np.pi**2)

		# def integrandC(q):
		# 	return self.bispectrum.fit_c(z, q, grid=False) * self.bispectrum.cosmo_nl.pkz.P(z,q) * q / (2*np.pi**2)

		# lnqs = np.logspace(np.log10(self.bispectrum.k_min), np.log10(self.bispectrum.k_max), nk)
		# intA = integrate.simps( integrandA(lnqs), x=lnqs )
		# intC = integrate.simps( integrandC(lnqs), x=lnqs )

		# plt.loglog(lnqs,integrandA(lnqs))

		# ak = self.bispectrum.fit_a(z, k, grid=False)
		# ck = self.bispectrum.fit_c(z, k, grid=False)
		# fk = 1.#self.f_l_spline(k*self.bispectrum.cosmo_lin.f_K(z))

		# print intA, intC, ak, ck, fk

		# return 6.*np.pi/7. * fk * ( 5.*ak*intA + ck*intC )

		def integrandA(q):
			return q * self.bispectrum.fit_a(z,q,grid=False) * self.bispectrum.cosmo_nl.pkz.P(z,q,grid=False) 

		def integrandC(q):
			return q * self.bispectrum.fit_c(z,q,grid=False) * self.bispectrum.cosmo_nl.pkz.P(z,q,grid=False) 

		qs   = np.logspace(-3, 2, nq)
		intA = integrate.simps( integrandA(qs), x=qs )
		intC = integrate.simps( integrandC(qs), x=qs )

		ak = self.bispectrum.fit_a(z, k, grid=False)
		ck = self.bispectrum.fit_c(z, k, grid=False)
		# fk = self.f_l_spline( k * self.bispectrum.cosmo_lin.f_K(z) )

		return 6*np.pi/7. * ( 5.*ak*intA + ck*intC )

	def raw_Tau(self, k, z, nq=1000):
		def integrandA(q):
			return q * self.f_l_spline(q*self.bispectrum.cosmo_lin.f_K(z)) * self.bispectrum.fit_a(z,q,grid=False) * self.bispectrum.cosmo_nl.pkz.P(z,q,grid=False) 

		def integrandC(q):
			return q * self.f_l_spline(q*self.bispectrum.cosmo_lin.f_K(z)) * self.bispectrum.fit_c(z,q,grid=False) * self.bispectrum.cosmo_nl.pkz.P(z,q,grid=False) 

		qs   = np.logspace(-3, 2, nq)
		intA = integrate.simps( integrandA(qs), x=qs )
		intC = integrate.simps( integrandC(qs), x=qs )

		ak = self.bispectrum.fit_a(z, k, grid=False)
		ck = self.bispectrum.fit_c(z, k, grid=False)
		fk = self.f_l_spline( k * self.bispectrum.cosmo_lin.f_K(z) )
		pk = self.bispectrum.cosmo_nl.pkz.P(z,k,grid=False) 

		# print intA, intC, ak, ck, fk

		return self.bispectrum.cosmo_lin.sigma_vz(z=z)**2 * fk * pk * ( 5.*ak*intA + ck*intC ) / 7. / (3e5)**2 # there should be a factor that gets divided out by the 1/3 * v_rms^2

	def B_dpp(self, z, k1, k2, k3):
		return self.bispectrum.cosmo_lin.sigma_vz(z=z)**2 * self.bispectrum.B_lss_3D(z,k1,k2,k3) / 3. 

	# def integrand_Tau(self, theta, q, k, z):
	# 	eta = self.bispectrum.cosmo_lin.f_K(z) # Mpc
	# 	k_plus_q_mod = np.sqrt(k**2+q**2-2*k*q*np.cos(theta))
	# 	# print k_plus_q_mod
	# 	# print self.B_dpp(z, [k,0,0],[q*np.cos(theta),q*np.sin(theta),0],[-k-q*np.cos(theta),-q*np.sin(theta),0])
	# 	# print self.f_l_spline(k_plus_q_mod*eta)
	# 	return q/(2.*np.pi)**2 * self.f_l_spline(k*eta) * self.f_l_spline(k_plus_q_mod*eta) * np.asanyarray(self.B_dpp(z, [k,0,0],[q*np.cos(theta),q*np.sin(theta),0],[-k-q*np.cos(theta),-q*np.sin(theta),0])[0][0])

	# @numba.jit()
	# def raw_Tau(self, k, z, npts_theta=10, npts_q=50): 
	# 	thetas = np.linspace(0,2.*np.pi,npts_theta)
	# 	qs     = np.logspace(-3,3,npts_q)

	# 	integrand_q = np.empty(len(qs))
	# 	for idxq, q in enumerate(qs):
	# 		integrand_q_theta = np.empty(len(thetas))
	# 		for idxtheta, theta in enumerate(thetas):
	# 			integrand_q_theta[idxtheta] = self.integrand_Tau(theta, q, k, z)
	# 		integrand_q[idxq] = integrate.simps(integrand_q_theta, x=thetas)

	# 	return integrate.simps(integrand_q, x=qs)

	def compute_Tau(self):
		"""
		"""
		# self.Tau_array = np.zeros((self.ks.size, self.bispectrum.cosmo_lin.zs.size))
		# for idk, k in enumerate(self.ks):
		# 	for idz, z in enumerate(self.bispectrum.cosmo_lin.zs):
		self.Tau_array = np.zeros((self.ks.size, self.zs.size))
		for idk, k in enumerate(self.ks):
			for idz, z in enumerate(self.zs):
				self.Tau_array[idk,idz] = self.raw_Tau(k,z)
		self._Tau_spline = interpolate.RectBivariateSpline(self.ks, self.zs, self.Tau_array, kx=5, ky=5, s=1)
		self.initialized_Tau_spline = True
		return None

	def Tau(self, k, z,grid=False):
		if not self.initialized_Tau_spline:
			self.compute_Tau()
		return self._Tau_spline(k,z,grid=grid)
		# return np.where(np.logical_and(k <= self.bispectrum.k_max, r > self.bispectrum.k_min), self._Tau_spline(k,z), 0.0)

	def compute_E3(self):
		self.E3_array = np.zeros((self.ks.size, self.bispectrum.zs.size))
		# for idk, k in enumerate(self.ks):
		# 	for idz, z in enumerate(self.bispectrum.cosmo_lin.zs):s
		for idk, k in enumerate(self.ks):
			for idz, z in enumerate(self.bispectrum.zs):
				self.E3_array[idk,idz] = self.raw_E3(k,z)
		self._E3_spline = interpolate.RectBivariateSpline(self.ks, self.bispectrum.zs, self.E3_array, s=2)
		# self._E3_spline = interpolate.RectBivariateSpline(self.ks, self.bispectrum.cosmo_lin.zs, self.E3_array, s=1)
		self.initialized_E3_spline = True
		return None

	def E3(self, k, z, nk=10000):
		""" 
		"""
		if not self.initialized_E3_spline:
			self.compute_E3()
		return self._E3_spline(k,z)

	def Cl_T2T2_filt(self, dl=16, n=1024, nwanted=500):
		lxgrid, lygrid  = np.meshgrid( np.arange(-n/2.,n/2.)*dl, np.arange(-n/2.,n/2.)*dl )
		lgrid = np.sqrt(lxgrid**2 + lygrid**2)
		L     = np.arange(0,nwanted)*dl    
				
		ttfiltpowerspec2d = np.interp(lgrid, np.arange(2,self.lmax+1), self.cl_T2_filt)
		
		cl_T2T2_filt = Cl_T2T2_filt_integrated(ttfiltpowerspec2d, dl, n, nwanted, self.lmax)
		
		return L, cl_T2T2_filt

	def Cl_lens_correction(self, dl=16, n=1024, nwanted=500):
		lxgrid, lygrid  = np.meshgrid( np.arange(-n/2.,n/2.)*dl, np.arange(-n/2.,n/2.)*dl )
		lgrid = np.sqrt(lxgrid**2 + lygrid**2)
		L     = np.arange(0,nwanted)*dl    
				
		ttunlpowerspec2d = np.interp(lgrid, np.arange(2,self.lmax+1), self.cl_tt_unl)
		filter2d         = np.interp(lgrid, np.arange(2,self.lmax+1), self.f_l)

		cl_lens_corr = Cl_lens_correction_integrated(ttunlpowerspec2d, filter2d, dl, n, nwanted, self.lmax)
		
		return L, -2*cl_lens_corr # Remember to multiply by the cross-spectrum lensing x (Galaxies,lensing,...)


	# def raw_Tau(self, k, z):
	# 	int1 = lambda q: q * self.f_l_spline(q*self.bispectrum.cosmo_lin.f_K(z)) * self.bispectrum.fit_a(z,q) * self.bispectrum.cosmo_nl.pkz.P(z, q, grid=False)
	# 	int2 = lambda q: q * self.f_l_spline(q*self.bispectrum.cosmo_lin.f_K(z)) * self.bispectrum.fit_c(z,q) * self.bispectrum.cosmo_nl.pkz.P(z, q, grid=False)
	# 	# int1 = lambda q,z: q * self.f_l_spline(q*self.bispectrum.cosmo_lin.f_K(z)) * self.bispectrum.fit_a(z,q) * self.bispectrum.cosmo_nl.pkz.P(z, q, grid=False)
	# 	# int2 = lambda q,z: q * self.f_l_spline(q*self.bispectrum.cosmo_lin.f_K(z)) * self.bispectrum.fit_c(z,q) * self.bispectrum.cosmo_nl.pkz.P(z, q, grid=False)

	# 	int1_integrated = integrate.simps(int1(self.bispectrum.ks), x=self.bispectrum.ks)
	# 	int2_integrated = integrate.simps(int2(self.bispectrum.ks), x=self.bispectrum.ks)
	# 	# int1_integrated = integrate.quad(int1, self.bispectrum.k_min, self.bispectrum.k_max, args=z, epsabs=0.0,
	# 	# 					epsrel=1e-5, limit=200, wvar=z)[0]
	# 	# int2_integrated = integrate.quad(int2, self.bispectrum.k_min, self.bispectrum.k_max, args=z, epsabs=0.0,
	# 	# 					epsrel=1e-5, limit=200, wvar=z)[0]

	# 	prefact = self.bispectrum.cosmo_lin.sigma_vz(z=z)**2 * self.bispectrum.cosmo_nl.pkz.P(z, k, grid=False) * self.f_l_spline(k*self.bispectrum.cosmo_lin.f_K(z)) / 7 / np.pi 

	# 	return prefact * ( 5.*self.bispectrum.fit_a(z,k)*int1_integrated + self.bispectrum.fit_c(z,k)*int2_integrated )

	# def raw_Tau(self, k, z):
	# 	return None

	# def compute_Tau(self):
	# 	"""
	# 	"""
	# 	# self.Tau_array = np.zeros((self.ks.size, self.bispectrum.cosmo_lin.zs.size))
	# 	# for idk, k in enumerate(self.ks):
	# 	# 	for idz, z in enumerate(self.bispectrum.cosmo_lin.zs):
	# 	self.Tau_array = np.zeros((self.ks.size, self.bispectrum.zs.size))
	# 	for idk, k in enumerate(self.ks):
	# 		for idz, z in enumerate(self.bispectrum.zs):

	# 			self.Tau_array[idk,idz] = self.raw_Tau(k,z)
	# 	self._Tau_spline = interpolate.RectBivariateSpline(self.ks, self.bispectrum.zs, self.Tau_array, s=1)
	# 	# self._Tau_spline = interpolate.RectBivariateSpline(self.ks, self.bispectrum.cosmo_lin.zs, self.Tau_array, s=1)
	# 	self.initialized_Tau_spline = True
	# 	return None

	def lens_correction_integrand(self, mu, L, l):
		return L**2 * self.f_l_spline(L) * np.interp(L,self.l_cmb,self.cl_tt_unl) * self.f_l_spline(np.sqrt(L**2 + l**2 + 2*mu*l*L)) * mu

	# @numba.jit()
	def lens_correction_integrand_cos(self, phi, L, l):
		return L**2 * self.f_l_spline(L) * np.interp(L,self.l_cmb,self.cl_tt_unl) * self.f_l_spline(np.sqrt(L**2+l**2+2*np.cos(phi)*l*L)) * np.cos(phi)

	@numba.jit()
	def Cl_lens_correction_cos(self, cl_phi_cross_tracer=None, npts=50):
		# lmax = len(cl_phi_cross_tracer) - 2
		lrange = np.linspace(2, self.lmax, npts, dtype=int)
		cl_lens_corr = np.zeros_like(lrange)

		phis = np.linspace(0., 2*np.pi, npts)
		Ls   = np.linspace(2, self.lmax, npts, dtype=int)

		for idl, l in enumerate(lrange):
			integrand_L = np.empty(len(Ls))
			for idL, L in enumerate(Ls):
				integrand_L_phi = np.empty(len(phis))
				for idphi, phi in enumerate(phis):
					integrand_L_phi[idphi] = self.lens_correction_integrand_cos(phi, L, l)
				integrand_L[idL] = integrate.simps(integrand_L_phi, x=phis)

			cl_lens_corr[idl] = integrate.simps(integrand_L, x=Ls)

		return cl_lens_corr * -2. * lrange / (2.*np.pi)**2 # * cl_phi_g 

	def Cl_lens_correction_cos_quad(self, cl_phi_cross_tracer=None, npts=50):

		lrange = np.linspace(2, self.lmax, npts, dtype=int)
		cl_lens_corr = np.zeros_like(lrange)

		for idl, l in enumerate(lrange):
			tmp = lambda phi, L : self.lens_correction_integrand_cos(phi,L,l)
			cl_lens_corr[idl] = integrate.nquad(tmp, [[0, 2*np.pi], [2, 8000]])[0]

		return cl_lens_corr * -2. * lrange / (2.*np.pi)**2 # * cl_phi_g

	def Cl_lens_correction(self, cl_phi_cross_tracer, npts=50):
		# lmax = len(cl_phi_cross_tracer) - 2
		lrange = np.linspace(2, self.lmax, npts, dtype=int)
		cl_lens_corr = np.zeros_like(lrange)

		# # thetas = np.linspace(0.,2*np.pi,npts)
		# mus = np.linspace(-1,1,npts)

		# for il, l in enumerate(lrange):
		# 	cl_lens_corr[il] = integrate.simps( [lp**2 * self.f_l_spline(lp) * np.interp(lp,self.l_cmb,self.cl_tt_unl) * integrate.simps(self.f_l_spline(np.sqrt(l**2+lp**2+2*l*lp*mus))*mus, x=mus ) for lp in lrange], x=lrange)
		# 	# cl_tmp = 0.
		# 	# for lp in lrange:
		# 	# 	int_ang = integrate.simps(self.f_l_spline(np.sqrt(l**2+lp**2+2*l*lp*mus))*mus, x=mus )
		# 	# 	# int_ang = np.simps(self.f_l_spline(np.sqrt(l**2+lp**2+2*l*lp*np.cos(thetas)))*np.cos(thetas), x=thetas )
		# 	# 	cl_tmp += 

		# return lrange, -2. * lrange * cl_lens_corr /2/np.pi 

		mus = np.linspace(-1.,1,npts)
		# Ls  = np.logspace(-3,2,npts)
		Ls = np.linspace(2, self.lmax, npts, dtype=int)

		for idl, l in enumerate(lrange):
			integrand_L = np.empty(len(Ls))
			for idL, L in enumerate(Ls):
				integrand_L_mu = np.empty(len(mus))
				for idmu, mu in enumerate(mus):
					integrand_L_mu[idmu] = self.lens_correction_integrand(mu, L, l)
				integrand_L[idL] = integrate.simps(integrand_L_mu, x=mus)

			cl_lens_corr[idl] = integrate.simps(integrand_L, x=Ls)

		return cl_lens_corr

	# def GetCl(self, tracer, ):



