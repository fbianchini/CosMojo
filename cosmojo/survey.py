import numpy as np
from scipy import interpolate
from scipy import integrate
from defaults import *
from scipy.special import erf
from scipy.optimize import brentq

from IPython import embed

def p_zph_z(zph, z, sigma, bias): 
    """
    Returns the Gaussian photometric error distribution p(z_ph|z), Eq.(5) from astro-ph/0506614
    
    Parameters
    ----------
    zph : array-like (float)
        Photometric redshift

    z : array-like (float)
        True redshift

    sigma : float
        Photo-z error scatter

    bias : float
        Photo-z error bias
    """
    return np.nan_to_num(np.exp(-((zph-z-bias)**2./(2.*(sigma*(1+zph))**2)))/(2.*np.pi*(sigma*(1+zph))**2)**0.5)


class Survey(object):
    """
    Base class for a simple redshift distribution.
    Derived classes should be used for specific redshift distributions.

    Attributes
    ----------
    z_min : float 
        Minimum redshift considered in the survey

    z_max : float
        Maximum redshift considered in the survey    

    Attributes (derived)
    --------------------
    norm : float
        Normalization factor of the redshift distribution

    z_med : float
        Median redshift of the survey

    z_mean : float
        Mean redshift of the survey

    """
    def __init__(self, z_min=default_limits['dNdz_zmin'], z_max=default_limits['dNdz_zmax']):

        self.z_min   = z_min
        self.z_max   = z_max
        self.norm    = 1.0
        self._z_med  = None
        self._z_mean = None

        self.normalize()

    def normalize(self):
        """
        Compute the normalization factor for the redshift distribution in the range [z_min, z_max]
        """
        # norm = integrate.quad( self.raw_dndz, self.z_min, self.z_max )[0]

        # norm = integrate.romberg( self.raw_dndz, self.z_min, self.z_max, vec_func=True,
        #                           tol=default_precision["global_precision"],
        #                           rtol=default_precision["dNdz_precision"],
        #                           divmax=default_precision["divmax"])
        norm = integrate.quad( self.raw_dndz, self.z_min, self.z_max, epsabs=0., epsrel=1e-5)[0]

        self.norm = 1.0/norm

    def raw_dndz(self, z):
        """
        Raw definition of the redshift distribution (overwritten in the derived classes)
        """
        return 1.0

    def dndz(self, z):
        """
        Normalized dn/dz PDF
        """
        return np.where(np.logical_and(z <= self.z_max, z >= self.z_min), self.norm*self.raw_dndz(z), 0.0)

    @property
    def z_med(self):
        """ 
        Median of the redshift distribution 
        """
        if self._z_med is None:
            f = lambda x: integrate.romberg(self.dndz, self.z_min, x) - 0.5
            self._z_med = brentq(f, self.z_min, self.z_max)

        return self._z_med

    @property
    def z_mean(self):
        """ 
        Mean of the redshift distribution
        """
        if self._z_mean is None:
            self._z_mean = integrate.romberg(lambda z: z * self.dndz(z), self.z_min, self.z_max)
        return self._z_mean
   
class Tomography(Survey):
    """
    Class for a tomographic redshift distribution derived from the Survey class.

    Attributes
    ----------
    z_min : float 
        Minimum redshift considered in the survey

    z_max : float
        Maximum redshift considered in the survey    

    b_zph : float
        Photo-z error bias

    sigma_zph : float
        Photo-z error scatter (=> sigma_zph * (1+z))

    nbins : float
        Number of equally spaced redshift bins to consider

    bins : array-like
        List of redshift bins edges (=> [z_min, z_1, z_2, ..., z_n, ..., z_max])

    Notes
    -----
    * Either you input nbins *OR* bins

    """
    def __init__(self, z_min=default_limits['dNdz_zmin'], 
                       z_max=default_limits['dNdz_zmax'],
                       b_zph=0., 
                       sigma_zph=0.,
                       nbins=1,
                       bins=None):
            
        super(Tomography, self).__init__(z_min, z_max)

        self.b_zph     = b_zph
        self.sigma_zph = sigma_zph

        if bins is None:
            dz   = (z_max - z_min) / nbins
            bins =  np.asarray([z_min + dz*i for i in xrange(nbins+1)]) 
        else:
            nbins = len(bins) - 1

        self.bounds = [(bins[i],bins[i+1]) for i in xrange(nbins)]
        self.bins   = bins
        self.nbins  = nbins

        self._z_med_bin  = None
        self._z_mean_bin = None

        self.normalize_bins()

    def raw_dndz_bin(self, z, i):
        """
        Un-normalized redshift distribution within the photo-z bin i, see Eq.(6) and (7) from astro-ph/0506614
        """

        # if np.isscalar(z) or (np.size(z) == 1):
        #     f = lambda zph: self.raw_dndz(z) * p_zph_z(zph, z, self.sigma_zph, self.b_zph)
        #     return integrate.romberg( f, self.bounds[i][0], self.bounds[i][1], vec_func=True,
        #                               tol=default_precision["global_precision"],
        #                               rtol=default_precision["dNdz_precision"],
        #                               divmax=default_precision["divmax"])    
        # else:
        #     return np.asarray([ self.raw_dndz_bin(tz, i=i) for tz in z ])

        x_min = (self.bounds[i][0] - z + self.b_zph) / (np.sqrt(2.) * self.sigma_zph*(1+z))
        x_max = (self.bounds[i][1] - z + self.b_zph) / (np.sqrt(2.) * self.sigma_zph*(1+z))
        n_z   = self.raw_dndz(z)

        return np.nan_to_num( 0.5 * n_z * ( erf(x_max) - erf(x_min) ))

    def dndz_bin(self, z, i):
        """
        Normalized PDF for the photo-z bin i
        """
        return np.where(np.logical_and(z <= self.z_max, z >= self.z_min), self.norm_bin[i]*self.raw_dndz_bin(z,i), 0.0)

    def normalize_bins(self):
        """
        Compute the normalization factors for the photo-z bins the range [z_min, z_max]
        """
        self.norm_bin = np.ones(self.nbins)
        for i in xrange(self.nbins):
            f = lambda z: self.raw_dndz_bin(z, i)
            norm = integrate.quad(f, self.z_min, self.z_max, epsabs=default_precision["global_precision"], epsrel=default_precision["dNdz_precision"])[0]
            # norm = integrate.romberg( f, self.z_min, self.z_max, vec_func=True,
            #                       tol=default_precision["global_precision"],
            #                       rtol=default_precision["dNdz_precision"],
            #                       divmax=default_precision["divmax"])
            # print norm 
            self.norm_bin[i] = 1.0/norm

    # @property
    def z_med_bin(self, i):
        """ 
        Median of the redshift distribution for bin i
        """
        if self._z_med_bin is None:
            self._z_med_bin = np.zeros(self.nbins)
            for i in xrange(self.nbins):
                u = lambda y: self.dndz_bin(y, i)
                f = lambda x: integrate.romberg(u, self.bins[i][0], x) - 0.5
                self._z_med_bin[i] = brentq(f, self.bins[i][0], self.bins[i][1])

        return self._z_med_bin[i]

    # @property
    def z_mean_bin(self, i):
        """ 
        Mean of the redshift distribution for bin i
        """
        if self._z_mean_bin is None:
            self._z_mean_bin = np.zeros(self.nbins)
            for i in xrange(self.nbins):
                f = lambda z: z * self.dndz_bin(z,i)
                self._z_mean_bin[i] = integrate.quad(f, self.z_min, self.z_max)[0]

        return self._z_mean_bin[i]

class dNdzGaussian(Tomography):
    """
    Derived class (from Tomography class) for a Gaussian-shaped redshift distribution.
   
    dNdz ~ exp(-(z-z0)^2/sigma_z^2)
    
    Attributes
    ----------
    z0: float 
        Mean redshift of Gaussian

    sigma_z: float 
        Standard deviation of Gaussian

    z_min : float 
        Minimum redshift considered in the survey

    z_max : float
        Maximum redshift considered in the survey    

    b_zph : float
        Photo-z error bias

    sigma_zph : float
        Photo-z error scatter (=> sigma_zph * (1+z))

    nbins : float
        Number of equally spaced redshift bins to consider

    bins : array-like
        List of redshift bins edges (=> [z_min, z_1, z_2, ..., z_n, ..., z_max])

    Notes
    -----
    * Either you input nbins *OR* bins
    """
    def __init__(self, z0, sigma_z,
                       z_min=default_limits['dNdz_zmin'], 
                       z_max=default_limits['dNdz_zmax'],
                       b_zph=0., 
                       sigma_zph=0.,
                       nbins=1,
                       bins=None):
 
        if z_min < z0 - 8.0*sigma_z:
            z_min = z0 - 8.0*sigma_z
        if z_max > z0 + 8.0*sigma_z:
            z_max = z0 + 8.0*sigma_z
            
        self.z0      = z0
        self.sigma_z = sigma_z

        super(dNdzGaussian, self).__init__(z_min, z_max, b_zph, sigma_zph, nbins, bins)

    def raw_dndz(self, z):
        return np.exp(-1.0*(z-self.z0)*(z-self.z0)/(2.0*self.sigma_z*self.sigma_z))

class dNdzMagLim(Tomography):
    """
    Derived class (from Tomography class) for a magnitude-limited redshift distribution.

    dNdz ~ z^a*exp(-(z/z0)^b)

    Attributes
    ----------
    a  : float 
       Power law slope

    z0 : float 
       "Mean" redshift of dN/dz 

    b : float
        Exponential decay slope

    z_min : float 
        Minimum redshift considered in the survey

    z_max : float
        Maximum redshift considered in the survey    

    b_zph : float
        Photo-z error bias

    sigma_zph : float
        Photo-z error scatter (=> sigma_zph * (1+z))

    nbins : float
        Number of equally spaced redshift bins to consider

    bins : array-like
        List of redshift bins edges (=> [z_min, z_1, z_2, ..., z_n, ..., z_max])

    Notes
    -----
    * Either you input nbins *OR* bins
    """
    def __init__(self, a, z0, b,
                       z_min=default_limits['dNdz_zmin'], 
                       z_max=default_limits['dNdz_zmax'],
                       b_zph=0., 
                       sigma_zph=0.,
                       nbins=1,
                       bins=None):
        self.a  = a
        self.z0 = z0
        self.b  = b
        # tmp_zmax = (np.power(-1*np.log(default_precision['dNdz_precision']),1/b)*z0)
        # if tmp_zmax < z_max:
        #     print ("WARNING:: z_max requested could "
        #            "result in failed normalization...")
        #     print ("\tReseting z_max from %.2f to %.2f..." % (z_max, tmp_zmax))
        #     z_max = tmp_zmax
        
        super(dNdzMagLim, self).__init__(z_min, z_max, b_zph, sigma_zph, nbins, bins)

    def raw_dndz(self, z):
        return (np.power(z, self.a) * np.exp(-1.0*np.power(z/self.z0, self.b)))

class dNdzInterpolation(Tomography):
    """
    Derived class for a p(z) derived from real data assuming:
    - array of redshifts 
    - dN/dz (or probabilities) for each redshift

    Attributes
    ----------
    z_array   : float 
        Array of redshifts
    
    dndz_array: float 
        Array of weights

    z_min : float 
        Minimum redshift considered in the survey

    z_max : float
        Maximum redshift considered in the survey    

    b_zph : float
        Photo-z error bias

    sigma_zph : float
        Photo-z error scatter (=> sigma_zph * (1+z))

    nbins : float
        Number of equally spaced redshift bins to consider

    bins : array-like
        List of redshift bins edges (=> [z_min, z_1, z_2, ..., z_n, ..., z_max])

    Notes
    -----
    * Either you input nbins *OR* bins
    """

    def __init__(self, z_array, dndz_array,
                       z_min=default_limits['dNdz_zmin'], 
                       z_max=default_limits['dNdz_zmax'],
                       b_zph=0., 
                       sigma_zph=0.,
                       nbins=1,
                       bins=None):

        self.z_array    = z_array
        self.dndz_array = dndz_array

        self._p_of_z = interpolate.interp1d(z_array, dndz_array, bounds_error=False, fill_value=0.)

        super(dNdzInterpolation, self).__init__(z_min, z_max, b_zph, sigma_zph, nbins, bins)

    def raw_dndz(self, z):
        return self._p_of_z(z)



# def catalog2dndz(z_cat, zbins=(0., 10.), sigma=0.26, bias=0, nbins=30):
#     'Returns the interpolation object of a dN/dz given the array with the redshifts'
#     dNdz, edges  = np.histogram(z_cat, nbins, density=True)
#     bins_        = (edges[:-1] + edges[1:])/2. 
#     dNdz_cat     = get_dNdz_spline(dNdz, bins_)

#     z        = np.linspace(0., 10, 1000)
#     phi      = dNdz_cat(z) * convolve_window(z, sigma=sigma, bias=bias, z_min=zbins[0], z_max=zbins[1])
#     phi_norm = integrate.simps(phi, x=z)
#     phi     /= phi_norm

#     return interpolate.interp1d(z, phi, bounds_error=False, fill_value=0.)
