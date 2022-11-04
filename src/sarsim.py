"""
 Created By:   Andrew Player
 File Name:    sarsim.py
 Date Created: 07-2022
 Description:  Functions for simulated-deformation interferogram generation for R&D purposes.
 Credits:      Functions taken from https://github.com/matthew-gaddes/SyInterferoPy
"""

import math

import numpy as np
import matplotlib.pyplot as plt

from src.synthetic_interferogram import wrap_interferogram, add_noise


def I1(xi, eta, q, delta, nu, R, X, d_tild):

    """
    Compute the component I1 of the model (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return I1
    """

    if np.cos(delta) > 10E-8:
        return (1 - 2*nu)*(-xi/(np.cos(delta)*(R + d_tild))) - I5(xi, eta, q, delta, nu, R, X, d_tild)*np.sin(delta)/np.cos(delta)
    else:
        return -((1 - 2*nu)/2.)*(xi*q/((R + d_tild)**2))


def I2(xi, eta, q, delta, nu, R, y_tild, d_tild):

    """
    Compute the component I2 of the model (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return I2
    """

    return (1 - 2*nu)*(-np.log(R + eta)) - I3(xi, eta, q, delta, nu, R, y_tild, d_tild)


def I3(xi, eta, q, delta, nu, R, y_tild, d_tild):

    """
    Compute the component I3 of the model (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return I3
    """

    if np.cos(delta) > 10E-8:
        return (1 - 2*nu)*(y_tild/(np.cos(delta)*(R + d_tild)) - np.log(R + eta)) + I4(xi, eta, q, delta, nu, R, d_tild)*np.sin(delta)/np.cos(delta)
    else:
        return ((1 - 2*nu)/2.)*(eta/(R + d_tild) + y_tild*q/((R + d_tild)**2) - np.log(R + eta))
    

def I4(xi, eta, q, delta, nu, R, d_tild):

    """
    Compute the component I4 of the model (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return I4
    """

    if np.cos(delta) > 10E-8:
        return (1 - 2*nu)*(np.log(R + d_tild) - np.sin(delta)*np.log(R + eta))/np.cos(delta)
    else:
        return -(1 - 2*nu)*q/(R + d_tild)
    

def I5(xi, eta, q, delta, nu, R, X, d_tild):

    """
    Compute the component I5 of the model (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return I5
    """

    if np.cos(delta) > 10E-8:
        return (1 - 2*nu)*2*np.arctan((eta*(X + q*np.cos(delta)) + X*(R + X)*np.sin(delta))/(xi*(R + X)*np.cos(delta)))/np.cos(delta)
    else:
        return -(1 - 2*nu)*xi*np.sin(delta)/(R + d_tild)


def f_x_strike(xi, eta, q, delta, nu):

    """
    Fault strike component along the x axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    """

    R = np.sqrt(xi**2 + eta**2 + q**2)
    X = np.sqrt(xi**2 + q**2)
    d_tild = eta*np.sin(delta) - q*np.cos(delta)
    return xi*q/(R*(R + eta)) + np.arctan(xi*eta/(q*R)) + I1(xi, eta, q, delta, nu, R, X, d_tild)*np.sin(delta)


def f_x_dip(xi, eta, q, delta, nu):
    
    """
    Fault dip component along the x axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    """

    R = np.sqrt(xi**2 + eta**2 + q**2)
    y_tild = eta*np.cos(delta) + q*np.sin(delta)
    d_tild = eta*np.sin(delta) - q*np.cos(delta)
    return q/R - I3(xi, eta, q, delta, nu, R, y_tild, d_tild)*np.sin(delta)*np.cos(delta)


def f_x_tensile(xi, eta, q, delta, nu):
    
    """
    Fault tensile component along the x axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    """

    R = np.sqrt(xi**2 + eta**2 + q**2)
    y_tild = eta*np.cos(delta) + q*np.sin(delta)
    d_tild = eta*np.sin(delta) - q*np.cos(delta)
    return (q**2)/(R*(R + eta)) - I3(xi, eta, q, delta, nu, R, y_tild, d_tild)*np.sin(delta)**2


def f_y_strike(xi, eta, q, delta, nu):
    
    """
    Fault strike component along the y axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    """

    R = np.sqrt(xi**2 + eta**2 + q**2)
    y_tild = eta*np.cos(delta) + q*np.sin(delta)
    d_tild = eta*np.sin(delta) - q*np.cos(delta)
    return y_tild*q/(R*(R + eta)) + q*np.cos(delta)/(R + eta) + I2(xi, eta, q, delta, nu, R, y_tild, d_tild)*np.sin(delta)


def f_y_dip(xi, eta, q, delta, nu):
    
    """
    Fault dip component along the y axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    """

    R = np.sqrt(xi**2 + eta**2 + q**2)
    X = np.sqrt(xi**2 + q**2)
    y_tild = eta*np.cos(delta) + q*np.sin(delta)
    d_tild = eta*np.sin(delta) - q*np.cos(delta)
    return y_tild*q/(R*(R + xi)) + np.cos(delta)*np.arctan(xi*eta/(q*R)) - I1(xi, eta, q, delta, nu, R, X, d_tild)*np.sin(delta)*np.cos(delta)


def f_y_tensile(xi, eta, q, delta, nu):
   
    """
    Fault tensile component along the y axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    """

    R = np.sqrt(xi**2 + eta**2 + q**2)
    X = np.sqrt(xi**2 + q**2)
    d_tild = eta*np.sin(delta) - q*np.cos(delta)
    return -d_tild*q/(R*(R + xi)) - np.sin(delta)*(xi*q/(R*(R + eta)) - np.arctan(xi*eta/(q*R))) - I1(xi, eta, q, delta, nu, R, X, d_tild)*np.sin(delta)**2



def f_z_strike(xi, eta, q, delta, nu):
    
    """
    Fault strike component along the z axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    """
    
    R = np.sqrt(xi**2 + eta**2 + q**2)
    d_tild = eta*np.sin(delta) - q*np.cos(delta)
    return d_tild*q/(R*(R + eta)) + q*np.sin(delta)/(R + eta) + I4(xi, eta, q, delta, nu, R, d_tild)*np.sin(delta)


def f_z_dip(xi, eta, q, delta, nu):
    
    """
    Fault dip component along the z axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    """
    
    R = np.sqrt(xi**2 + eta**2 + q**2)
    X = np.sqrt(xi**2 + q**2)
    d_tild = eta*np.sin(delta) - q*np.cos(delta)
    return d_tild*q/(R*(R + xi)) + np.sin(delta)*np.arctan(xi*eta/(q*R)) - I5(xi, eta, q, delta, nu, R, X, d_tild)*np.sin(delta)*np.cos(delta)


def f_z_tensile(xi, eta, q, delta, nu):
    
    """
    Fault tensile component along the z axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    """
    
    R = np.sqrt(xi**2 + eta**2 + q**2)
    X = np.sqrt(xi**2 + q**2)
    y_tild = eta*np.cos(delta) + q*np.sin(delta)
    d_tild = eta*np.sin(delta) - q*np.cos(delta)
    return y_tild*q/(R*(R + xi)) + np.cos(delta)*(xi*q/(R*(R + eta)) - np.arctan(xi*eta/(q*R))) - I5(xi, eta, q, delta, nu, R, X, d_tild)*np.sin(delta)**2


def chinnerys_notation(f, x, p, q, L, W, delta, nu):
    
    """
    Formula to add the different fault components (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The combined components
    """
    
    return f(x, p, q, delta, nu)\
           - f(x, p - W, q, delta, nu)\
           - f(x - L, p, q, delta, nu)\
           + f(x - L, p - W, q, delta, nu)


def compute_okada_displacement(fault_centroid_x,
                               fault_centroid_y,
                               fault_centroid_depth,
                               fault_strike,
                               fault_dip,
                               fault_length,
                               fault_width,
                               fault_rake,
                               fault_slip,
                               fault_open,
                               poisson_ratio,
                               x_array,
                               y_array):
    '''
    Compute the surface displacements for a rectangular fault, based on
    Okada's model. For more information, see:
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154
    
    @param fault_centroid_x: x cooordinate for the fault's centroid
    @param fault_centroid_y: y cooordinate for the fault's centroid
    @param fault_centroid_depth: depth of the fault's centroid
    @param fault_strike: strike of the fault ([0 - 2pi], in radian)
    @param fault_dip: dip of the fault ([0 - pi/2], in radian)
    @param fault_length: length of the fault (same unit as x and y)
    @param fault_width: width of the fault (same unit as x and y)
    @param fault_rake: rake of the fault ([-pi - pi], in radian)
    @param fault_slip: slipe of the fault (same unit as x and y)
    @param fault_open: opening of the fault (same unit as x and y)
    @param poisson_ratio: Poisson's ratio
    @param x_array: x cooordinate for the domain within a NumPy array
    @param y_array: y cooordinate for the domain within a NumPy array
    
    @return The surface displacement field
    '''
    U1 = np.cos(fault_rake)*fault_slip
    U2 = np.sin(fault_rake)*fault_slip

    east_component = x_array - fault_centroid_x + np.cos(fault_strike)*np.cos(fault_dip)*fault_width/2.
    north_component = y_array - fault_centroid_y - np.sin(fault_strike)*np.cos(fault_dip)*fault_width/2.
    okada_x_array = np.cos(fault_strike)*north_component + np.sin(fault_strike)*east_component + fault_length/2.
    okada_y_array = np.sin(fault_strike)*north_component - np.cos(fault_strike)*east_component + np.cos(fault_dip)*fault_width
    
    d = fault_centroid_depth + np.sin(fault_dip)*fault_width/2.
    p = okada_y_array*np.cos(fault_dip) + d*np.sin(fault_dip)
    q = okada_y_array*np.sin(fault_dip) - d*np.cos(fault_dip)

    displacement_shape = [3] + list(x_array.shape)
    okada_displacement_array = np.zeros(displacement_shape)
    
    okada_displacement_array[0] = -U1*chinnerys_notation(f_x_strike, okada_x_array, p, q, fault_length, fault_width, fault_dip, poisson_ratio)/(2*np.pi)\
                                  - U2*chinnerys_notation(f_x_dip, okada_x_array, p, q, fault_length, fault_width, fault_dip, poisson_ratio)/(2*np.pi)\
                                  + fault_open*chinnerys_notation(f_x_tensile, okada_x_array, p, q, fault_length, fault_width, fault_dip, poisson_ratio)/(2*np.pi)
    okada_displacement_array[1] = -U1*chinnerys_notation(f_y_strike, okada_x_array, p, q, fault_length, fault_width, fault_dip, poisson_ratio)/(2*np.pi)\
                                  - U2*chinnerys_notation(f_y_dip, okada_x_array, p, q, fault_length, fault_width, fault_dip, poisson_ratio)/(2*np.pi)\
                                  + fault_open*chinnerys_notation(f_y_tensile, okada_x_array, p, q, fault_length, fault_width, fault_dip, poisson_ratio)/(2*np.pi)
    okada_displacement_array[2] = -U1*chinnerys_notation(f_z_strike, okada_x_array, p, q, fault_length, fault_width, fault_dip, poisson_ratio)/(2*np.pi)\
                                  - U2*chinnerys_notation(f_z_dip, okada_x_array, p, q, fault_length, fault_width, fault_dip, poisson_ratio)/(2*np.pi)\
                                  + fault_open*chinnerys_notation(f_z_tensile, okada_x_array, p, q, fault_length, fault_width, fault_dip, poisson_ratio)/(2*np.pi)

    displacement_array = np.zeros(displacement_shape)

    displacement_array[0] = np.sin(fault_strike)*okada_displacement_array[0] - np.cos(fault_strike)*okada_displacement_array[1]
    displacement_array[1] = np.cos(fault_strike)*okada_displacement_array[0] + np.sin(fault_strike)*okada_displacement_array[1]
    displacement_array[2] = okada_displacement_array[2]
            
    return displacement_array


def deformation_eq_dyke_sill(source, source_xy_m, xyz_m, **kwargs):    
    
    """
    A function to create deformation patterns for either an earthquake, dyke or sill.   Uses the Okada function from PyInSAR: https://github.com/MITeaps/pyinsar
    To aid in readability, different sources take different parameters (e.g. slip for a quake, opening for a dyke)
    are passed separately as kwargs, even if they ultimately go into the same field in the model parameters.  
    
    A quick recap on definitions:
        strike - measured clockwise from 0 at north, 180 at south.  fault dips to the right of this.  hanging adn fo
        dip - measured from horizontal, 0 for horizontal, 90 for vertical.  
        rake - direction the hanging wall moves during rupture, measured relative to strike, anticlockwise is positive, so:
            0 for left lateral ss 
            180 (or -180) for right lateral ss
            -90 for normal
            90 for thrust

    Inputs:
        source | string | quake or dyke or sill
        source_xy_m | tuple | x and y location of centre of source, in metres.  
        xyz_m | rank2 array | x and y locations of all points in metres.  0,0 is top left?  

        
        examples of kwargs:
            
        quake_normal = {'strike' : 0,
                        'dip' : 70,
                        'length' : 5000,
                        'rake' : -90,
                        'slip' : 1,
                        'top_depth' : 4000,
                        'bottom_depth' : 8000}
        
        quake_thrust = {'strike' : 0,
                        'dip' : 30,
                        'length' : 5000,
                        'rake' : 90,
                        'slip' : 1,
                        'top_depth' : 4000,
                        'bottom_depth' : 8000}
        
        quake_ss = {'strike' : 0,
                    'dip' : 80,
                    'length' : 5000,
                    'rake' : 0,
                    'slip' : 1,
                    'top_depth' : 4000,
                    'bottom_depth' : 8000}
        
        dyke = {'strike' : 0,
                'top_depth' : 1000,
                'bottom_depth' : 3000,
                'length' : 5000,
                'dip' : 80,
                'opening' : 0.5}
        
        sill = {'strike' : 0,
                'depth' : 3000,
                'width' : 5000,
                'length' : 5000,
                'dip' : 1,
                'opening' : 0.5}
        
    Returns:
        x_grid | rank 2 array | displacment in x direction for each point (pixel on Earth's surface)
        y_grid | rank 2 array | displacment in y direction for each point (pixel on Earth's surface)
        z_grid | rank 2 array | displacment in z direction for each point (pixel on Earth's surface)
        los_grid | rank 2 array | change in satellite - ground distance, in satellite look angle direction. Need to confirm if +ve is up or down.  
        
    History:
        2020/08/05 | MEG | Written
        2020/08/21 | MEG | Switch from disloc3d.m function to compute_okada_displacement.py functions.  
    """    
    
    # 1:  Setting for elastic parameters.  
    lame = {'lambda' : 2.3e10,                              # elastic modulus (Lame parameter, units are pascals)
            'mu'     : 2.3e10}                              # shear modulus (Lame parameter, units are pascals)
    v = lame['lambda'] / (2*(lame['lambda'] + lame['mu']))  #  calculate poisson's ration

    if source == 'quake':
        opening = 0
        slip = kwargs['slip']
        rake = kwargs['rake']
        width = kwargs['bottom_depth'] - kwargs['top_depth']
        # centroid_depth = np.mean((kwargs['bottom_depth'] - kwargs['top_depth']))
        centroid_depth = np.mean((kwargs['bottom_depth'], kwargs['top_depth']))
    elif source == 'dyke':                                                                               # ie dyke or sill
        opening = kwargs['opening']
        slip = 0
        rake = 0
        width = kwargs['bottom_depth'] - kwargs['top_depth']
        centroid_depth = np.mean((kwargs['bottom_depth'], kwargs['top_depth']))
    elif source == 'sill':                                                                               # ie dyke or sill
        opening = kwargs['opening']
        slip = 0
        rake = 0
        centroid_depth = kwargs['depth']
        width = kwargs['width']
    else:
        raise Exception(f"'Source' must be either 'quake', 'dyke', or 'sill', but is set to {source}.  Exiting.")
        
    # 3:  compute deformation using Okada function
    U = compute_okada_displacement(source_xy_m[0], source_xy_m[1],                    # x y location, in metres
                                   centroid_depth,                                    # fault_centroid_depth, guess metres?  
                                   np.deg2rad(kwargs['strike']),
                                   np.deg2rad(kwargs['dip']),
                                   kwargs['length'], width,                           # length and width, in metres
                                   np.deg2rad(rake),                                  # rake, in rads
                                   slip, opening,                                     # slip (if quake) or opening (if dyke or sill)
                                   v, xyz_m[0,], xyz_m[1,:])                          # poissons ratio, x and y coords of surface locations.

    return U


def atmosphere_turb(n_atms, lons_mg, lats_mg, method = 'fft', mean_m = 0.02, difference = False):
    
    """ A function to create synthetic turbulent atmospheres based on the  methods in Lohman Simons 2005, or using Andy Hooper and Lin Shen's fft method.  
    Note that due to memory issues, when using the covariance (Lohman) method, largers ones are made by interpolateing smaller ones.  
    Can return atmsopheres for an individual acquisition, or as the difference of two (as per an interferogram).  Units are in metres.  
    
    Inputs:
        n_atms | int | number of atmospheres to generate
        lons_mg | rank 2 array | longitudes of the bottom left corner of each pixel.  
        lats_mg | rank 2 array | latitudes of the bottom left corner of each pixel.  
        method | string | 'fft' or 'cov'.  Cov for the Lohmans Simons (sp?) method, fft for Andy Hooper/Lin Shen's fft method (which is much faster).  Currently no way to set length scale using fft method.  
        mean_m | float | average max or min value of atmospheres that are created.  e.g. if 3 atmospheres have max values of 0.02m, 0.03m, and 0.04m, their mean would be 0.03cm.  
        water_mask | rank 2 array | If supplied, this is applied to the atmospheres generated, convering them to masked arrays.  
        difference | boolean | If difference, two atmospheres are generated and subtracted from each other to make a single atmosphere.  
        verbose | boolean | Controls info printed to screen when running.  
        cov_Lc     | float | length scale of correlation, in metres.  If smaller, noise is patchier, and if larger, smoother.  
        cov_interpolate_threshold | int | if n_pixs is greater than this, images will be generated at size so that the total number of pixels doesn't exceed this.  
                                          e.g. if set to 1e4 (10000, the default) and images are 120*120, they will be generated at 100*100 then upsampled to 120*120.  
        
    
    Outputs:
        ph_turbs | r3 array | n_atms x n_pixs x n_pixs, UNITS ARE M.  Note that if a water_mask is provided, this is applied and a masked array is returned.  
        
    2019/09/13 | MEG | adapted extensively from a simple script
    2020/10/02 | MEG | Change so that a water mask is optional.  
    2020/10/05 | MEG | Change so that meshgrids of the longitudes and latitudes of each pixel are used to set resolution. 
                       Also fix a bug in how cov_Lc is handled, so this is now in meters.  
    2020/10/06 | MEG | Add support for rectangular atmospheres, fix some bugs.  
    2020_03_01 | MEG | Add option to use Lin Shen/Andy Hooper's fft method which is quicker than the covariance method.  
    """
    
    import numpy as np

    
    def lon_lat_to_ijk(lons_mg, lats_mg):
        
        """ Given a meshgrid of the lons and lats of the lower left corner of each pixel, 
        find their distances (in metres) from the lower left corner.  
        Inputs:
            lons_mg | rank 2 array | longitudes of the lower left of each pixel.  
            lats_mg | rank 2 array | latitudes of the lower left of each pixel.  
        Returns:
            ijk | rank 2 array | 3x lots.  The distance of each pixel from the lower left corner of the image in metres.  
            pixel_spacing | dict | size of each pixel (ie also the spacing between them) in 'x' and 'y' direction.  
        History:
            2020/10/01 | MEG | Written 
        """
        
        from geopy import distance
        import numpy as np
        
        ny, nx = lons_mg.shape
        pixel_spacing = {}
        pixel_spacing['x'] = distance.distance((lats_mg[-1,0], lons_mg[-1,0]), (lats_mg[-1,0], lons_mg[-1,1])).meters     # this should vary with latitude.  Usually around 90 (metres) for SRTM3 resolution.  
        pixel_spacing['y'] = distance.distance((lats_mg[-1,0], lons_mg[-1,0]), (lats_mg[-2,0], lons_mg[-1,0])).meters     # this should be constant at all latitudes.  
    
        X, Y = np.meshgrid(pixel_spacing['x'] * np.arange(0, nx), pixel_spacing['y'] * np.arange(0,ny))                   # make a meshgrid
        Y = np.flipud(Y)                                                                                                  # change 0 y cordiante from matrix style (top left) to axes style (bottom left)
        ij = np.vstack((np.ravel(X)[np.newaxis], np.ravel(Y)[np.newaxis]))                                                # pairs of coordinates of everywhere we have data   
        ijk = np.vstack((ij, np.zeros((1, ij.shape[1]))))                                                                 # xy and 0 depth, as 3xlots
        
        return ijk, pixel_spacing


    def generate_correlated_noise_fft(nx, ny, std_long, sp):
        
        """ A function to create synthetic turbulent troposphere delay using an FFT approach. 
        The power of the turbulence is tuned by the weather model at the longer wavelengths.
        
        Inputs:
            nx (int) -- width of troposphere 
            Ny (int) -- length of troposphere 
            std_long (float) -- standard deviation of the weather model at the longer wavelengths. Default = ?
            sp | int | pixel spacing in km
            
        Outputs:
            APS (float): 2D array, Ny * nx, units are m.
            
        History:
            2020_??_?? | LS | Adapted from code by Andy Hooper.  
            2021_03_01 | MEG | Small change to docs and inputs to work with SyInterferoPy
        """
                
        np.seterr(divide='ignore')
    
        cut_off_freq = 1/50                                                    # drop wavelengths above 50 km 
        
        x      = np.arange(0, int(nx / 2))                                     # positive frequencies only
        y      = np.arange(0, int(ny / 2))                                     # positive frequencies only
        freq_x = np.divide(x, nx * sp)
        freq_y = np.divide(y, ny * sp)
        Y, X   = np.meshgrid(freq_x, freq_y)
        freq   = np.sqrt((X * X + Y * Y) / 2)                                  # 2D positive frequencies
        
        log_power     = np.log10(freq) * -11/3                                 # -11/3 in 2D gives -8/3 in 1D
        ix            = np.where(freq < 2/3)
        log_power[ix] = np.log10(freq[ix]) * -8/3 -np.log10(2/3)               # change slope at 1.5 km (2/3 cycles per km)
        
        bin_power     = np.power(10, log_power)
        ix            = np.where(freq < cut_off_freq)
        bin_power[ix] = 0
        
        APS_power = np.zeros((ny,nx))                                         # mirror positive frequencies into other quadrants
        
        APS_power[0:int(ny / 2)      , 0:int(nx / 2)        ] = bin_power
        APS_power[0:int(ny / 2)      , int(np.ceil(nx / 2)):] = np.fliplr(bin_power)
        APS_power[int(np.ceil(ny/2)):, 0:int(nx / 2)        ] = np.flipud(bin_power)
        APS_power[int(np.ceil(ny/2)):, int(np.ceil(nx/2))  :] = np.fliplr(np.flipud(bin_power))
       
        APS_filt = np.sqrt(APS_power)
        
        x      = np.random.randn(ny, nx)      # white noise
        y_tmp  = np.fft.fft2(x)
        y_tmp2 = np.multiply(y_tmp, APS_filt) # convolve with filter
        y      = np.fft.ifft2(y_tmp2)
        
        APS = np.real(y)
        APS = APS / np.std(APS) * std_long  #  adjust the turbulence by the weather model at the longer wavelengths.
        APS = APS * 0.01                    # convert from cm to m
        
        return APS 


    def rescale_atmosphere(atm, atm_mean = 0.02, atm_sigma = 0.005):
        
        """ a function to rescale a 2d atmosphere with any scale to a mean centered
        one with a min and max value drawn from a normal distribution.  
        Inputs:
            atm | rank 2 array | a single atmosphere.  
            atm_mean | float | average max or min value of atmospheres that are created, in metres.  e.g. if 3 atmospheres have max values of 0.02m, 0.03m, and 0.04m, their mean would be 0.03m
            atm_sigma | float | standard deviation of Gaussian distribution used to generate atmosphere strengths.  
        Returns:
            atm | rank 2 array | a single atmosphere, rescaled to have a maximum signal of around that set by mean_m
        History:
            20YY/MM/DD | MEG | Written
            2020/10/02 | MEG | Standardise throughout to use metres for units.  
        """
        
        atm -= np.mean(atm)                                                         # mean centre
        atm_strength = (atm_sigma * np.random.randn(1)) + atm_mean                  # maximum strength of signal is drawn from a gaussian distribution, mean and sigma set in metres.  
        if np.abs(np.min(atm)) > np.abs(np.max(atm)):                               # if range of negative numbers is larger
            atm *= (atm_strength / np.abs(np.min(atm)))                             # strength is drawn from a normal distribution  with a mean set by mean_m (e.g. 0.02)
        else:
            atm *= (atm_strength / np.max(atm))                     # but if positive part is larger, rescale in the same way as above.  
        return atm


    # 0: Check inputs
    if method not in ['fft', 'cov']:
        raise Exception(f"'method' must be either 'fft' (for the fourier transform based method), "
                        f" or 'cov' (for the covariance based method).  {method} was supplied, so exiting.  ")

    #1: determine if linear interpolation is required
    ny, nx = lons_mg.shape

    nx_generate = nx      # if not interpolating, these don't change.  
    ny_generate = ny
    lons_mg_ds = lons_mg  # if not interpolating, don't need to downsample.  
    lats_mg_ds = lats_mg

    #2: calculate distance between points
    ph_turbs = np.zeros((n_atms, ny_generate, nx_generate))       # initiate output as a rank 3 (ie n_images x ny x nx)
    _, pixel_spacing = lon_lat_to_ijk(lons_mg_ds, lats_mg_ds) # get pixel positions in metres from origin in lower left corner (and also their size in x and y direction)      

    #3: generate atmospheres, using either of the two methods.  
    if difference == True:
        n_atms += 1   # if differencing atmospheres, create one extra so that when differencing we are left with the correct number

    for i in range(n_atms):
        ph_turbs[i,:,:] = generate_correlated_noise_fft(nx_generate, ny_generate,    std_long=1, 
                                                        sp = 0.001 * np.mean((pixel_spacing['x'], pixel_spacing['y'])) ) # generate noise using fft method.  pixel spacing is average in x and y direction (and m converted to km) 

    ph_turbs_output = ph_turbs # if we're not interpolating, no change needed
       
    # 4: rescale to correct range (i.e. a couple of cm)
    ph_turbs_m = np.zeros(ph_turbs_output.shape)
    for atm_n, atm in enumerate(ph_turbs_output):
        ph_turbs_m[atm_n,] = rescale_atmosphere(atm, mean_m)

    # 5: return back to the shape given, which can be a rectangle:
    ph_turbs_m = ph_turbs_m[:,:lons_mg.shape[0],:lons_mg.shape[1]]

    return ph_turbs_m


def gen_fake_topo(
    size:          int = 512,
    alt_scale_min: int = 0,
    alt_scale_max: int = 500
):

    """
    Generate fake topography (a dem in meters) for generating simulated atmospheric topographic error.

    Parameters:
    -----------
    size : int
        The size n for the (n, n) dimension array that is returned.
    alt_scale_min : int
        The minimum altitude scaling value for the generated perlin noise.
    alt_scale_max : int
        The maximum altitude scaling value for the generated perlin noise.

    Returns:
    --------
    dem : np.ndarray
        The array that is meant to be used as a simulated dem with values in meters.
    """

    from src.synthetic_interferogram import generate_perlin

    dem = np.zeros((size, size))    
    dem = generate_perlin(dem.shape[0]) * np.random.randint(alt_scale_min, alt_scale_max)

    return dem


def atm_topo_simulate(
    dem_m:         np.ndarray,
    strength_mean: float = 56.0,
    strength_var:  float = 2.0,
    difference:    bool  = True
):

    """
    Generate simulated topographic atmospheric error.

    Parameters:
    -----------
    dem_m : np.ndarray
        An array containing either real dem values (in meters) or simulated ones.
    strength_mean : float
        The mean strength/magnitude of the error.
    strength_var : float
        The maximum variation from strength_mean of the magnitude of the error.
    difference : bool
        Whether the error should come from the difference of 2 aquisitions or just 1.

    Returns:
    --------
    ph_turb : np.ndarray
        The array containing the turbulent atmospheric error.
    """

    import numpy as np
    import numpy.ma as ma

    envisat_lambda = 0.056                     # envisat/S1 wavelength in m
    dem = 0.001 * dem_m                        # convert from metres to km

    if difference is False:
        ph_topo = (strength_mean + strength_var * np.random.randn(1)) * dem
    elif difference is True:
        ph_topo_aq1 = (strength_mean + strength_var * np.random.randn(1)) * dem  # this is the delay for one acquisition
        ph_topo_aq2 = (strength_mean + strength_var * np.random.randn(1)) * dem  # and for another
        ph_topo = ph_topo_aq1 - ph_topo_aq2                                      # interferogram is the difference, still in rad
    else:
        print("'difference' must be either True or False.  Exiting...")
        import sys; sys.exit()

    # convert from rad to m
    ph_topo_m = (ph_topo / (4*np.pi)) * envisat_lambda # delay/elevation ratio is taken from a paper (pinel 2011) using Envisat data


    if np.max(ph_topo_m) < 0:                          # ensure that it always start from 0, either increasing or decreasing
        ph_topo_m -= np.max(ph_topo_m)
    else:
        ph_topo_m -= np.min(ph_topo_m)

    ph_topo_m  = ma.array(ph_topo_m, mask = ma.getmask(dem_m))
    ph_topo_m -= ma.mean(ph_topo_m)                   # mean centre the signal

    return ph_topo_m


def aps_simulate(
    size: int = 512
):

    """
    Generate simulated turbulent atmospheric error.

    Parameters:
    -----------
    size : int
        The size n for the (n, n) dimension array that is returned.

    Returns:
    --------
    ph_turb : np.ndarray
        The array containing the turbulent atmospheric error.
    """

    pixel_size_degs = 1/3600

    scaled_size = pixel_size_degs * size

    lons    = np.arange(0.0, 0.0 + scaled_size, pixel_size_degs)
    lats    = np.arange(0.0, 0.0 + scaled_size, pixel_size_degs)
    lons_mg = np.repeat(lons[np.newaxis,:], len(lats), axis = 0)
    lats_mg = np.repeat(lats[::-1, np.newaxis], len(lons), axis = 1)

    ph_turb = atmosphere_turb(1, lons_mg, lats_mg, mean_m = 0.02,
                                 method = 'fft')

    return ph_turb[0,]


def coherence_mask_simulate(
    size:      int   = 512,
    threshold: float = 0.3
):

    """
    Generate simulated incoherence to be masked out.

    Parameters:
    -----------
    size : int
        The size n for the (n, n) dimension array that is returned.
    threshold : float
        The maximum value of coherence to be masked to zeros.

    Returns:
    --------
    mask_coh : np.ndarray
        The masked coherence array.
    """

    pixel_size_degs = 1 / 3600
    
    lons = np.arange(0.0, 0.0 + (pixel_size_degs * size), pixel_size_degs)
    lats = np.arange(0.0, 0.0 + (pixel_size_degs * size), pixel_size_degs)
    lons_mg = np.repeat(lons[np.newaxis,:], len(lats), axis = 0)
    lats_mg = np.repeat(lats[::-1, np.newaxis], len(lons), axis = 1)

    mask_coh_values = atmosphere_turb(1, lons_mg, lats_mg, mean_m = 0.01,
                                 method = 'fft')

    mask_coh_values = (mask_coh_values - np.min(mask_coh_values)) / np.max(mask_coh_values - np.min(mask_coh_values))
    mask_coh = np.where(mask_coh_values > threshold, np.ones(lons_mg.shape), np.zeros(lons_mg.shape)) 

    return mask_coh


def gen_simulated_deformation(
    seed:              int   = 0,
    tile_size:         int   = 512,
    log:               bool  = False,
    atmosphere_scalar: float = 90 * np.pi,
    amplitude_scalar:  float = 1000 * np.pi
):

    """
    Generate a wrapped interferogram along with an event-mask from simulated deformation

    Parameters:
    -----------
    seed : int, Optional
        A seed for the random functions. For the same seed, with all other values the same
        as well, the interferogram generation will have the same results. If left at 0,
        the results will be different every time.
    tile_size : int, Optional
        The desired dimensional size of the interferogram pairs. This should match
        the input shape of the model.
    log : bool, Optional
        If true, the function will log various relevant values in the console. 
    atmosphere_scalar : float, Optional
        Scale factor for the intensity of atmospheric noise.
    amplitude_scalar : float, Optional
        Scale factor for the deformation.

    Returns:
    --------
    masked_grid : np.ndarray(shape=(tile_size, tile_size))
        An array representing a mask over the simulated deformation which simulates masking an event.
    wrapped_grid : np.ndarray(shape=(tile_size, tile_size)
        The wrapped interferogram.
    presence : [1] or [0]
        [1] if the image contains an event else [0]
    """

    if seed != 0: np.random.seed(seed)

    presence    = np.asarray([1])

    masked_grid = np.zeros((tile_size, tile_size)) 

    los_vector  = np.array(
        [
            [ 0.38213591],
            [-0.08150437],
            [ 0.92050485]
        ]
    )

    random_nums = np.random.rand(9)

    X, Y = np.meshgrid(np.arange(0, tile_size) * 90, np.arange(0, tile_size) * 90)
    Y    = np.flipud(Y)

    ij  = np.vstack((np.ravel(X)[np.newaxis], np.ravel(Y)[np.newaxis]))
    ijk = np.vstack((ij, np.zeros((1, ij.shape[1]))))   

    source_x = np.max(X) // ((random_nums[0] * 10) + 1)
    source_y = np.max(Y) // ((random_nums[1] * 10) + 1)

    length    = 1000 + ((1 + np.max(X) // 16) * random_nums[2])
    top_depth = 4000 + ((1 + np.max(X) // 8 ) * random_nums[3])

    kwargs = {
        'strike'      : 180 * random_nums[4],
        'dip'         :  90 * random_nums[5],
        'length'      : length,
        'rake'        : [90, -90][random_nums[6] < 0.5],
        'slip'        : 1,
        'top_depth'   : top_depth,
        'bottom_depth': top_depth * (1.1 + (5 * random_nums[7]))
    }

    U = deformation_eq_dyke_sill("quake", (source_x, source_y), ijk, **kwargs)

    x_grid   = np.reshape(U[0,], (X.shape[0], X.shape[1])) * los_vector[0,0]
    y_grid   = np.reshape(U[1,], (X.shape[0], X.shape[1])) * los_vector[1,0]
    z_grid   = np.reshape(U[2,], (X.shape[0], X.shape[1])) * los_vector[2,0]

    los_grid = (x_grid + y_grid + z_grid) * amplitude_scalar

    masked_indices = np.abs(los_grid) >= np.pi * 2 # Num of fringes to say yes to.
    fringes = np.floor(np.abs(los_grid[masked_indices]) / np.pi)
    masked_grid[masked_indices] = 1

    atmosphere_phase = aps_simulate(tile_size) * atmosphere_scalar
    interferogram    = los_grid + atmosphere_phase[0:tile_size, 0:tile_size]
    wrapped_grid     = wrap_interferogram(interferogram, noise = 0.0)

    coherence_mask                   = coherence_mask_simulate(tile_size, threshold=random_nums[8]*0.5)
    coh_masked_indices               = coherence_mask[0,0:tile_size, 0:tile_size] == 0
    wrapped_grid[coh_masked_indices] = 0

    if log:
        print("_______\n")
        print("Length         (meters)  ", length)
        print("Top Depth      (meters)  ", top_depth)
        print("Bottom Depth   (meters)  ", kwargs['bottom_depth'])
        print("")
        print("Max X Position (meters)  ", np.max(X))
        print("Max Y Position (meters)  ", np.max(Y))
        print("Src X Position (meters)  ", source_x)
        print("Src Y Position (meters)  ", source_y)
        print("")
        print("Slip           (0  or 1) ", kwargs['slip'])
        print("Dip            (degrees) ", kwargs['dip'])
        print("Rake           (degrees) ", kwargs['rake'])
        print("Strike         (degrees) ", kwargs['strike'])
        print("")
        print("Maximum Phase Value: ", np.max(np.abs(interferogram)))
        print("_______\n")

    return masked_grid, wrapped_grid, presence


def gen_gaussian_noise(
    seed:        int   = 0,
    tile_size:   int   = 512,
    noise_level: float = 90 * np.pi
):

    if seed != 0: np.random.seed(seed)

    masked_grid  = np.zeros((tile_size, tile_size))

    noise_grid1                 = np.random.uniform(-noise_level, noise_level, size=(tile_size, tile_size))
    inconsistancy1              = np.abs(noise_grid1) <  noise_level / 2
    noise_grid1[inconsistancy1] = 0

    noise_grid2                 = np.random.uniform(-noise_level, noise_level, size=(tile_size, tile_size))
    inconsistancy2              = np.abs(noise_grid2) >= noise_level / 2
    noise_grid2[inconsistancy2] = 0

    wrapped_grid = wrap_interferogram(noise_grid1 + noise_grid2, noise=0)

    threshold      = np.random.random() / 2
    coherence_mask = coherence_mask_simulate(tile_size, threshold=np.random.random() / 2)
    coh_indices    = coherence_mask[0, 0:tile_size, 0:tile_size] == 0        
    wrapped_grid[coh_indices] = 0

    return masked_grid, wrapped_grid

def gen_sim_noise(
    seed:              int   = 0,
    tile_size:         int   = 512,
    gaussian_only:     bool  = False,
    atmosphere_scalar: float = 90 * np.pi,
):

    """
    Generate a wrapped interferogram along with an event-mask simulating a noisy interferogram with no deformation.

    Parameters:
    -----------
    seed : int, Optional
        A seed for the random functions. For the same seed, with all other values the same
        as well, the interferogram generation will have the same results. If left at 0,
        the results will be different every time.
    tile_size : int, Optional
        The desired dimensional size of the interferogram pairs. This should match
        the input shape of the model.
    log : bool, Optional
        If true, the function will log various relevant values in the console.
    atmosphere_scalar : float, Optional
        Scale factor for the intensity of atmospheric noise.

    Returns:
    --------
    masked_grid : np.ndarray(shape=(tile_size, tile_size))
        An array representing a mask over the simulated deformation which simulates masking an event.
    wrapped_grid : np.ndarray(shape=(tile_size, tile_size)
        The wrapped interferogram.
    presence : [1] or [0]
        [1] if the image contains an event else [0]
    """

    if seed != 0: np.random.seed(seed)

    presence = np.asarray([0])

    if gaussian_only:

        masked_grid, wrapped_grid = gen_gaussian_noise(seed, tile_size)

    else:

        simulated_topography = gen_fake_topo(
            size          = tile_size,
            alt_scale_min = 100,
            alt_scale_max = 500
        )

        turb_phase = aps_simulate(tile_size) * atmosphere_scalar
        topo_phase = atm_topo_simulate(simulated_topography) * atmosphere_scalar * np.pi

        threshold      = np.random.random() / 2
        coherence_mask = coherence_mask_simulate(tile_size, threshold=threshold)
        coh_indices    = coherence_mask[0, 0:tile_size, 0:tile_size] == 0
        
        wrapped_grid = np.angle(np.exp(1j * (turb_phase + topo_phase)))
        wrapped_grid[coh_indices] = 0

        masked_grid = np.zeros((tile_size, tile_size))

    return masked_grid, wrapped_grid, presence