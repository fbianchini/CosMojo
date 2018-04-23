default_cosmo_dict = {
    "ombh2"  : 0.022, # baryon physical density at z=0
    "omch2"  : 0.12, # cold dark matter physical density at z=0
    "omk"    : 0.,    # Omega_K curvature paramter
    "mnu"    : 0.06,  # sum of neutrino masses [eV]
    "nnu"    : 3.046, # N_eff, # of effective relativistic dof
    "TCMB"   : 2.725, # temperature of the CMB in K at z=0
    "H0"     : 67.7,    # Hubble's constant at z=0 [km/s/Mpc]
    "w"      : -1.0,  # dark energy equation of state (fixed throughout cosmic history)
    "wa"     : 0.,  # dark energy equation of state (fixed throughout cosmic history)
    "cs2"    : 1.0,  # dark energy equation of state (fixed throughout cosmic history)
    "tau"    : 0.09,  # optical depth
    "YHe"    : None,  # Helium mass fraction
    "As"     : 2e-9,   # comoving curvature power at k=piveo_scalar
    "ns"     : 0.96,   # scalar spectral index
    "nrun"   : 0.,     # running of scalar spectral index
    "nrunrun": 0.,     # running of scalar spectral index
    "r"      : 0.,     # tensor to scalar ratio at pivot scale
    "nt"     : None,   # tensor spectral index
    "ntrun"  : 0.,     # running of tensor spectral index
    "pivot_scalar": 0.05, # pivot scale for scalar spectrum
    "pivot_tensor": 0.05, # pivot scale for tensor spectrum
    "meffsterile" : 0.,   # effective mass of sterile neutrinos
    "neutrino_hierarchy": 'degenerate', # degenerate', 'normal', or 'inverted' (1 or 2 eigenstate approximation)
    "num_massive_neutrinos" : 1, # number of massive neutrinos (ignored unless hierarchy == 'degenerate')
    "standard_neutrino_neff": 3.046,
    "gamma0"     : 0.55,   # growth rate index
    "gammaa"     : 0.,     # growth rate index (series expansion term)
    "deltazrei" : None,
    }

default_cosmoCLASS_dict = {
    "output"   : 'tCl pCl lCl mPk',
    "lensing"  : 'yes',
    "l_max_scalars" : 3000,
    "omega_b"  : 0.022, # baryon physical density at z=0
    "omega_cdm": 0.12, # cold dark matter physical density at z=0
    "Omega_k"  : 0.,    # Omega_K curvature paramter
    "T_cmb"    : 2.725, # temperature of the CMB in K at z=0
    "H0"       : 67.7,    # Hubble's constant at z=0 [km/s/Mpc]
    # "w"        : -1.0,  # dark energy equation of state (fixed throughout cosmic history)
    "tau_reio" : 0.09,  # optical depth
    "YHe"      : 'BBN',  # Helium mass fraction
    "A_s"      : 2e-9,   # comoving curvature power at k=piveo_scalar
    "n_s"      : 0.96,   # scalar spectral index
    "alpha_s"  : 0.,     # running of scalar spectral index
    # "r"        : 0.,     # tensor to scalar ratio at pivot scale
    # "n_t"      : 'scc',   # tensor spectral index
    # "alpha_t"  : 'scc',     # running of tensor spectral index
    "k_pivot": 0.05, # pivot scale for scalar spectrum
    # N_ur = 0.0
    # N_ncdm = 1
    # m_ncdm = 0.06
    # deg_ncdm = 3.0
    "N_ur"     : 3.046,
    'z_max_pk' : 1050.,
    "gamma0"     : 0.55,   # growth rate index
    "gammaa"     : 0.,     # growth rate index (series expansion term)
    }

default_pw_survey_dict = {
    'M_min'  : 1e14, # Minimum cluster mass in survey
    'M_max'  : 1e16, # Maximum cluster mass in survey
    'fsky'   : 1.,   # Fraction of the sky
    'sigma_v': 160., # Uncertainties on velcity [km/s] ... it can be an array though!
    'zmin'   : 0.1,  # Minimum redshift
    'zmax'   : 0.4,  # Maximum redshift
    'Nz'     : 1,    # # of z-bins
    'rmin'   : 30.,  # Minimum separation scale [Mpc]
    'rmax'   : 250., # Maximum separation scale [Mpc]
    'Nr'     : 80,   # of r-bins
    'b_tau'  : None, # optical depth bias
    'fg'     : None    }

default_cmb_survey_dict = {
    'fsky'   : 1.,      # Fraction of the sky
    'lmaxT'  : 3000,    # Maximum multipole Temperature
    'lmaxP'  : 3000,    # Maximum multipole Polarization
    'lmaxK'  : 1000,    # Maximum multipole CMB lensing
    'lminT'  : 2,       # Minimum multipole Temperature
    'lminP'  : 2,       # Minimum multipole Polarization
    'lminK'  : 2,       # Minimum multipole CMB lensing
    'DeltaT' : 3,       # Noise in Temperature [\muK-arcmin]
    'DeltaP' : 4.242,   # Noise in Polarization [\muK-arcmin]
    'fwhm'   : 3.,      # Beam [arcmin]
    'lknee'  : 1e-9,
    'alpha'  : 0.,
    }

default_bao_survey_dict = {
    'z': [.15, .25, .35, .45, .55, .65, .75, .85, .95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85],
    'sigma':[4.1,1.7,.88,.55,.38,.28,.21,.18,.18,.17,.16,.14,.15,.16,.19,.28,.41,.52]
}

default_limits = {
    "limber_zmin" : 0.,   ###
    "limber_zmax" : None, ### If None, zstar (~1100) will be used
    "limber_lmin" : 2,
    "limber_lmax" : 2000,
    "limber_kmax" : 40.,
    "dNdz_zmin"   : 0.,
    "dNdz_zmax"   : 5.,
    "lens_zmax"   : 5.,
    "pk_kmin"     : 1e-4, # 1/Mpc
    "pk_kmax"     : 40.,  # 1/Mpc
    "r_min"       : 1,    # Mpc
    "r_max"       : 300,  # Mpc
    "mass_min"    : -1,   #
    "mass_max"    : -1,    #
    "halo_zmin"   : 5e-3,   ###
    "halo_zmax"   : 10, ### If None, zstar (~1100) will be used
    "halo_lmin"   : 2,
    "halo_lmax"   : 2000,
#    "halo_Mmin"   : 5e5,  # M_sun
#    "halo_Mmax"   : 5e15, # M_sun
    "halo_Mmin"   : 5e5,  # M_sun
    "halo_Mmax"   : 5e20, # M_sun
}

default_precision = {
    "magbias_npts"  : 300,
    "lens_npts"     : 300,
    "limber_npts"   : 300,
    "halo_npts"     : 300,
    "kernel_npts"   : 50,
    "corr_npts"     : 300,
    "dNdz_precision": 1.48e-8,
    "lens_precision": 1.48e-6,
    "corr_precision": 1e-5,
    "global_precision": 1.48e-32, ### Since the code has large range of values
                                  ### from say 1e-10 to 1e10 we don't want to
                                  ### use absolute tolerances, instead using
                                  ### relative tolerances to define convergence
                                  ### of our integrands
    "divmax":20,                   ### Maximum number of subdivisions for
    "mass_npoints": 50,
    "mass_precision": 1.48e-8,
    "halo_npoints": 100,
    "halo_precision": 1.48e-5, ### The difference between e-4 and e-5 are at the
    #                                ### 0.1% level. Since this is the main slow down
    #                                ### in the calculation e-4 can be used to speed
    #                                ### up the code.

    }

### Default parameters specifying a halo.
default_halo_dict = {
    "stq"        :  0.3,
    "st_little_a":  0.707,
    "c0"         :  9.0,
    "beta"       : -0.13,
    "alpha"      : -1, ### Halo mass profile slope. [NFW = -1]
    "delta_v"    : -1.0 ### over-density for defining. -1 means default behavior of
                        ### redshift dependent over-density defined in NFW97
    }


# default_precision = {
#     "corr_npoints": 50,
#     "corr_precision": 1.48e-6,
#     "cosmo_npoints": 50,
#     "cosmo_precision": 1.48e-8,
#     "dNdz_precision": 1.48e-8,
#     "halo_npoints": 50,
#     "halo_precision": 1.48e-5, ### The difference between e-4 and e-5 are at the
#                                ### 0.1% level. Since this is the main slow down
#                                ### in the calculation e-4 can be used to speed
#                                ### up the code.
#     "halo_limit" : 100,
#     "kernel_npoints": 50,
#     "kernel_precision": 1.48e-6,
#     "kernel_limit": 100, ### If the variable force_quad is set in the Kernel
#                          ### class this value sets the limit for the quad
#                          ### integration
#     "kernel_bessel_limit": 8, ### Defines how many zeros before cutting off
#                               ### the Bessel function integration in kernel.py
#     "mass_npoints": 50,
#     "mass_precision": 1.48e-8,
#     "window_npoints": 100,
#     "window_precision": 1.48e-6,
#     "global_precision": 1.48e-32, ### Since the code has large range of values
#                                   ### from say 1e-10 to 1e10 we don't want to
#                                   ### use absolute tolerances, instead using
#                                   ### relative tolerances to define convergence
#                                   ### of our integrands
#     "divmax":20                   ### Maximum number of subdivisions for
#                                   ### the romberg integration.
#     }
