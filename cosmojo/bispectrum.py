import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy import special
from defaults import *
from utils import *
from mass_func import MassFunction
from universe import Cosmo

class Bispectrum(object):
	"""
	Class to compute the matter bispectrum B(k1,k2,k3).

	Attributes
	----------
	cosmo : Cosmo object (from universe.py)
		Cosmology object

	"""
	def __init__(self, params=None, 
					   cosmo_lin=None,
					   cosmo_nl=None,
					   r_min=default_limits['r_min'],
					   r_max=default_limits['r_max'],
					   r_npts=default_precision['corr_npts'],
					   k_min=None,
					   k_max=None,
					   k_npts=500,
					   fit_type='SC',
					   **kws):


		self.zmax_lss = 5.2

		if (cosmo_lin is not None) and (cosmo_nl is not None):
			self.cosmo_lin = cosmo_lin
			self.cosmo_nl  = cosmo_nl
		else: 
			self.cosmo_lin = Cosmo(params=params)
			self.cosmo_nl  = Cosmo(params=params, nonlinear=True)

		# Create array w/ wavenumbers k
		self.k_min = k_min#self.cosmo_lin.kmin
		self.k_max = self.cosmo_lin.kmax
		self.ks    = np.logspace(np.log10(self.k_min),np.log10(self.k_max),k_npts)

		# De-wiggled d ln P/dln k 
		self.nk = self.cosmo_lin.pkz(0.1, np.log(self.ks), grid=False, dy=1)
		w  = np.ones_like(self.nk)
		w[self.ks<5e-3] = 100.
		w[self.ks>1]    = 10.
		self.nksp = interpolate.UnivariateSpline(np.log(self.ks), self.nk, s=30, w=w) 

		self.zs = np.exp(np.linspace(0, np.log(self.zmax_lss+1),50))-1#np.linspace(0,10,1000)

		if (fit_type is 'SC') or (fit_type is 'sc'):
			# a(n,k)
			# sc = np.zeros((self.cosmo_lin.zs.size,self.ks.size))
			# for i, z in enumerate(self.cosmo_lin.zs):
			sc = np.zeros((self.zs.size,self.ks.size))
			for i, z in enumerate(self.zs):
				sc[i,:] = self.SC_fit_a(z,self.ks)
			# sc[sc==np.inf] = 0.
			# sc = np.nan_to_num(sc)
			# print sc
			self.fit_a = interpolate.RectBivariateSpline(self.zs, self.ks, sc)
			# self.fit_a = interpolate.RectBivariateSpline(self.cosmo_lin.zs, self.ks, sc)
			
			# b(n,k)
			# sc = np.zeros((self.cosmo_lin.zs.size,self.ks.size))
			# for i, z in enumerate(self.cosmo_lin.zs):
			sc = np.zeros((self.zs.size,self.ks.size))
			for i, z in enumerate(self.zs):
				sc[i,:] = self.SC_fit_b(z,self.ks)
			# sc[sc==np.inf] = 0.
			# sc = np.nan_to_num(sc)
			# self.fit_b = interpolate.RectBivariateSpline(self.cosmo_lin.zs, self.ks, sc)
			self.fit_b = interpolate.RectBivariateSpline(self.zs, self.ks, sc)

			# c(n,k)
			# sc = np.zeros((self.cosmo_lin.zs.size,self.ks.size))
			# for i, z in enumerate(self.cosmo_lin.zs):
			sc = np.zeros((self.zs.size,self.ks.size))
			for i, z in enumerate(self.zs):

				sc[i,:] = self.SC_fit_c(z,self.ks)
			# sc[sc==np.inf] = 0.
			# sc = np.nan_to_num(sc)
			# self.fit_c = interpolate.RectBivariateSpline(self.cosmo_lin.zs, self.ks, sc)
			self.fit_c = interpolate.RectBivariateSpline(self.zs, self.ks, sc)


		elif (fit_type is 'GM') or (fit_type is 'gm'):
			# a(n,k)
			sc = np.zeros((self.cosmo_lin.zs.size,self.ks.size))
			for i, z in enumerate(self.cosmo_lin.zs):
				sc[i,:] = self.GM_fit_a(z,self.ks)
			self.fit_a = interpolate.RectBivariateSpline(self.cosmo_lin.zs, self.ks, sc)
			# b(n,k)
			sc = np.zeros((self.cosmo_lin.zs.size,self.ks.size))
			for i, z in enumerate(self.cosmo_lin.zs):
				sc[i,:] = self.GM_fit_b(z,self.ks)
			self.fit_b = interpolate.RectBivariateSpline(self.cosmo_lin.zs, self.ks, sc)
			# c(n,k)
			sc = np.zeros((self.cosmo_lin.zs.size,self.ks.size))
			for i, z in enumerate(self.cosmo_lin.zs):
				sc[i,:] = self.GM_fit_c(z,self.ks)
			self.fit_c = interpolate.RectBivariateSpline(self.cosmo_lin.zs, self.ks, sc)

	def SC_fit_a(self, z, k, a1=0.25, a2=3.5, a6=-0.2):
		 if z >= self.zmax_lss: return 1
		 q   = k/self.cosmo_lin.k_NL(z)
		 n   = self.nksp(np.log(k))    
		 Q3  = (4-2.**n)/(1+2.**(n+1))
		 fac = (q*a1)**(n+a2)
		 return (1 + self.cosmo_lin.sigma_Rz(8./self.cosmo_lin.h,z)**a6*np.sqrt(0.7*Q3)*fac)/(1+fac)

	def SC_fit_b(self, z, k, a3=2, a7=1, a8=0):
		 if z >= self.zmax_lss: return 1
		 q  = k/self.cosmo_lin.k_NL(z)
		 n   = self.nksp(np.log(k))    
		 qq = a7*q
		 return (1 + 0.2*a3*(n+3)*qq**(n+3+a8))/(1+qq**(n+3.5+a8))

	def SC_fit_c(self, z ,k, a4=1., a5=2, a9=0):
		 if z >= self.zmax_lss: return 1
		 q = k/self.cosmo_lin.k_NL(z)
		 n = self.nksp(np.log(k))    
		 return (1+4.5*a4/(1.5+(n+3)**4)*(q*a5)**(n+3+a9))/(1+(q*a5)**(n+3.5+a9))

	def GM_fit_a(self, z, k):
		return self.SC_fit_a(z, k, a1=0.484, a2=3.740, a6=-0.575)

	def GM_fit_b(self, z, k):
		return self.SC_fit_b(z, k, a3=-0.849, a7=0.128, a8=-0.722)

	def GM_fit_c(self, z, k):
		return self.SC_fit_c(z, k, a4=0.392, a5=1.013, a9=-0.926)

	def B_lss(self, zs, k1, k2, k3):
		cos12 = (k3**2-k1**2-k2**2)/2/k1/k2
		cos23 = (k1**2-k2**2-k3**2)/2/k2/k3
		cos31 = (k2**2-k3**2-k1**2)/2/k3/k1

		a1 = self.fit_a(zs, k1)
		a2 = self.fit_a(zs, k2)
		a3 = self.fit_a(zs, k3)
		b1 = self.fit_b(zs, k1)
		b2 = self.fit_b(zs, k2)
		b3 = self.fit_b(zs, k3)
		c1 = self.fit_c(zs, k1)
		c2 = self.fit_c(zs, k2)
		c3 = self.fit_c(zs, k3)

		PK1 = self.cosmo_lin.pkz.P(zs, k1, grid=False)
		PK2 = self.cosmo_lin.pkz.P(zs, k2, grid=False)
		PK3 = self.cosmo_lin.pkz.P(zs, k3, grid=False)

		F12 = 5./7*a1*a2 + b1*b2*0.5*(k1/k2 + k2/k1)*cos12 + c1*c2*2./7*cos12**2
		F23 = 5./7*a2*a3 + b2*b3*0.5*(k2/k3 + k3/k2)*cos23 + c2*c3*2./7*cos23**2
		F31 = 5./7*a1*a3 + b3*b1*0.5*(k3/k1 + k1/k3)*cos31 + c3*c1*2./7*cos31**2

		return 2*F12*PK1*PK2 + 2*F23*PK2*PK3 + 2*F31*PK3*PK1

	def B_lss_3D(self, zs, k1, k2, k3):
		k1 = np.asarray(k1)
		k2 = np.asarray(k2)
		k3 = np.asarray(k3)

		k1_norm = np.sqrt(k1.dot(k1))
		k2_norm = np.sqrt(k2.dot(k2))
		k3_norm = np.sqrt(k3.dot(k3))

		cos12 = k1.dot(k2)/k1_norm/k2_norm
		cos23 = k2.dot(k3)/k2_norm/k3_norm
		cos31 = k3.dot(k1)/k3_norm/k1_norm

		# print cos12, cos23, cos31

		a1 = self.fit_a(zs, k1_norm)
		a2 = self.fit_a(zs, k2_norm)
		a3 = self.fit_a(zs, k3_norm)
		b1 = self.fit_b(zs, k1_norm)
		b2 = self.fit_b(zs, k2_norm)
		b3 = self.fit_b(zs, k3_norm)
		c1 = self.fit_c(zs, k1_norm)
		c2 = self.fit_c(zs, k2_norm)
		c3 = self.fit_c(zs, k3_norm)

		PK1 = self.cosmo_nl.pkz.P(zs, k1_norm, grid=False)
		PK2 = self.cosmo_nl.pkz.P(zs, k2_norm, grid=False)
		PK3 = self.cosmo_nl.pkz.P(zs, k3_norm, grid=False)

		F12 = 5./7*a1*a2 + b1*b2*0.5*(k1_norm/k2_norm + k2_norm/k1_norm)*cos12 + c1*c2*2./7*cos12**2
		F23 = 5./7*a2*a3 + b2*b3*0.5*(k2_norm/k3_norm + k3_norm/k2_norm)*cos23 + c2*c3*2./7*cos23**2
		F31 = 5./7*a1*a3 + b3*b1*0.5*(k3_norm/k1_norm + k1_norm/k3_norm)*cos31 + c3*c1*2./7*cos31**2

		return 2*F12*PK1*PK2 + 2*F23*PK2*PK3 + 2*F31*PK3*PK1
		# return np.nan_to_num(2*F12*PK1*PK2 + 2*F23*PK2*PK3 + 2*F31*PK3*PK1)
