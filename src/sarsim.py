"""
 Created By:   Andrew Player
 File Name:    sarsim.py
 Date Created: 07-2022
 Description:  Functions for simulated-deformation interferogram generation for R&D purposes.
 Credits:      Functions taken from https://github.com/matthew-gaddes/SyInterferoPy
"""

import math
import random

import numpy as np
import matplotlib.pyplot as plt

from src.synthetic_interferogram import wrap_interferogram, add_noise


def I1(xi, eta, q, delta, nu, R, X, d_tild):
    '''
    Compute the component I1 of the model (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return I1
    '''
    if math.cos(delta) > 10E-8:
        return (1 - 2*nu)*(-xi/(math.cos(delta)*(R + d_tild))) - I5(xi, eta, q, delta, nu, R, X, d_tild)*math.sin(delta)/math.cos(delta)
    else:
        return -((1 - 2*nu)/2.)*(xi*q/((R + d_tild)**2))


def I2(xi, eta, q, delta, nu, R, y_tild, d_tild):
    '''
    Compute the component I2 of the model (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return I2
    '''
    return (1 - 2*nu)*(-np.log(R + eta)) - I3(xi, eta, q, delta, nu, R, y_tild, d_tild)


def I3(xi, eta, q, delta, nu, R, y_tild, d_tild):
    '''
    Compute the component I3 of the model (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return I3
    '''
    if math.cos(delta) > 10E-8:
        return (1 - 2*nu)*(y_tild/(math.cos(delta)*(R + d_tild)) - np.log(R + eta)) + I4(xi, eta, q, delta, nu, R, d_tild)*math.sin(delta)/math.cos(delta)
    else:
        return ((1 - 2*nu)/2.)*(eta/(R + d_tild) + y_tild*q/((R + d_tild)**2) - np.log(R + eta))
    

def I4(xi, eta, q, delta, nu, R, d_tild):
    '''
    Compute the component I4 of the model (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return I4
    '''
    if math.cos(delta) > 10E-8:
        return (1 - 2*nu)*(np.log(R + d_tild) - math.sin(delta)*np.log(R + eta))/math.cos(delta)
    else:
        return -(1 - 2*nu)*q/(R + d_tild)
    

def I5(xi, eta, q, delta, nu, R, X, d_tild):
    '''
    Compute the component I5 of the model (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return I5
    '''
    if math.cos(delta) > 10E-8:
        return (1 - 2*nu)*2*np.arctan((eta*(X + q*math.cos(delta)) + X*(R + X)*math.sin(delta))/(xi*(R + X)*math.cos(delta)))/math.cos(delta)
    else:
        return -(1 - 2*nu)*xi*math.sin(delta)/(R + d_tild)


def f_x_strike(xi, eta, q, delta, nu):
    '''
    Fault strike component along the x axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    '''
    R = np.sqrt(xi**2 + eta**2 + q**2)
    X = np.sqrt(xi**2 + q**2)
    d_tild = eta*math.sin(delta) - q*math.cos(delta)
    return xi*q/(R*(R + eta)) + np.arctan(xi*eta/(q*R)) + I1(xi, eta, q, delta, nu, R, X, d_tild)*math.sin(delta)


def f_x_dip(xi, eta, q, delta, nu):
    '''
    Fault dip component along the x axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    '''
    R = np.sqrt(xi**2 + eta**2 + q**2)
    y_tild = eta*math.cos(delta) + q*math.sin(delta)
    d_tild = eta*math.sin(delta) - q*math.cos(delta)
    return q/R - I3(xi, eta, q, delta, nu, R, y_tild, d_tild)*math.sin(delta)*math.cos(delta)


def f_x_tensile(xi, eta, q, delta, nu):
    '''
    Fault tensile component along the x axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    '''
    R = np.sqrt(xi**2 + eta**2 + q**2)
    y_tild = eta*math.cos(delta) + q*math.sin(delta)
    d_tild = eta*math.sin(delta) - q*math.cos(delta)
    return (q**2)/(R*(R + eta)) - I3(xi, eta, q, delta, nu, R, y_tild, d_tild)*math.sin(delta)**2


def f_y_strike(xi, eta, q, delta, nu):
    '''
    Fault strike component along the y axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    '''
    R = np.sqrt(xi**2 + eta**2 + q**2)
    y_tild = eta*math.cos(delta) + q*math.sin(delta)
    d_tild = eta*math.sin(delta) - q*math.cos(delta)
    return y_tild*q/(R*(R + eta)) + q*math.cos(delta)/(R + eta) + I2(xi, eta, q, delta, nu, R, y_tild, d_tild)*math.sin(delta)


def f_y_dip(xi, eta, q, delta, nu):
    '''
    Fault dip component along the y axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    '''
    R = np.sqrt(xi**2 + eta**2 + q**2)
    X = np.sqrt(xi**2 + q**2)
    y_tild = eta*math.cos(delta) + q*math.sin(delta)
    d_tild = eta*math.sin(delta) - q*math.cos(delta)
    return y_tild*q/(R*(R + xi)) + math.cos(delta)*np.arctan(xi*eta/(q*R)) - I1(xi, eta, q, delta, nu, R, X, d_tild)*math.sin(delta)*math.cos(delta)


def f_y_tensile(xi, eta, q, delta, nu):
    '''
    Fault tensile component along the y axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    '''
    R = np.sqrt(xi**2 + eta**2 + q**2)
    X = np.sqrt(xi**2 + q**2)
    d_tild = eta*math.sin(delta) - q*math.cos(delta)
    return -d_tild*q/(R*(R + xi)) - math.sin(delta)*(xi*q/(R*(R + eta)) - np.arctan(xi*eta/(q*R))) - I1(xi, eta, q, delta, nu, R, X, d_tild)*math.sin(delta)**2



def f_z_strike(xi, eta, q, delta, nu):
    '''
    Fault strike component along the z axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    '''
    R = np.sqrt(xi**2 + eta**2 + q**2)
    d_tild = eta*math.sin(delta) - q*math.cos(delta)
    return d_tild*q/(R*(R + eta)) + q*math.sin(delta)/(R + eta) + I4(xi, eta, q, delta, nu, R, d_tild)*math.sin(delta)


def f_z_dip(xi, eta, q, delta, nu):
    '''
    Fault dip component along the z axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    '''
    R = np.sqrt(xi**2 + eta**2 + q**2)
    X = np.sqrt(xi**2 + q**2)
    d_tild = eta*math.sin(delta) - q*math.cos(delta)
    return d_tild*q/(R*(R + xi)) + math.sin(delta)*np.arctan(xi*eta/(q*R)) - I5(xi, eta, q, delta, nu, R, X, d_tild)*math.sin(delta)*math.cos(delta)


def f_z_tensile(xi, eta, q, delta, nu):
    '''
    Fault tensile component along the z axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    '''
    R = np.sqrt(xi**2 + eta**2 + q**2)
    X = np.sqrt(xi**2 + q**2)
    y_tild = eta*math.cos(delta) + q*math.sin(delta)
    d_tild = eta*math.sin(delta) - q*math.cos(delta)
    return y_tild*q/(R*(R + xi)) + math.cos(delta)*(xi*q/(R*(R + eta)) - np.arctan(xi*eta/(q*R))) - I5(xi, eta, q, delta, nu, R, X, d_tild)*math.sin(delta)**2


def chinnerys_notation(f, x, p, q, L, W, delta, nu):
    '''
    Formula to add the different fault components (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The combined components
    '''
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
    U1 = math.cos(fault_rake)*fault_slip
    U2 = math.sin(fault_rake)*fault_slip

    east_component = x_array - fault_centroid_x + math.cos(fault_strike)*math.cos(fault_dip)*fault_width/2.
    north_component = y_array - fault_centroid_y - math.sin(fault_strike)*math.cos(fault_dip)*fault_width/2.
    okada_x_array = math.cos(fault_strike)*north_component + math.sin(fault_strike)*east_component + fault_length/2.
    okada_y_array = math.sin(fault_strike)*north_component - math.cos(fault_strike)*east_component + math.cos(fault_dip)*fault_width
    
    d = fault_centroid_depth + math.sin(fault_dip)*fault_width/2.
    p = okada_y_array*math.cos(fault_dip) + d*math.sin(fault_dip)
    q = okada_y_array*math.sin(fault_dip) - d*math.cos(fault_dip)

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

    displacement_array[0] = math.sin(fault_strike)*okada_displacement_array[0] - math.cos(fault_strike)*okada_displacement_array[1]
    displacement_array[1] = math.cos(fault_strike)*okada_displacement_array[0] + math.sin(fault_strike)*okada_displacement_array[1]
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
    lame = {'lambda' : 2.3e10,                                                         # elastic modulus (Lame parameter, units are pascals)
            'mu'     : 2.3e10}                                                         # shear modulus (Lame parameter, units are pascals)
    v = lame['lambda'] / (2*(lame['lambda'] + lame['mu']))                             #  calculate poisson's ration
      
    # import matplotlib.pyplot as plt
    # both_arrays = np.hstack((np.ravel(coords), np.ravel(xyz_m)))
    # f, axes = plt.subplots(1,2)
    # axes[0].imshow(coords, aspect = 'auto', vmin = np.min(both_arrays), vmax = np.max(both_arrays))                 # goes from -1e4 to 1e4
    # axes[1].imshow(xyz_m, aspect = 'auto', vmin = np.min(both_arrays), vmax = np.max(both_arrays))                  # goes from 0 to 2e4
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


def atmosphere_turb(n_atms, lons_mg, lats_mg, method = 'fft', mean_m = 0.02,
                    water_mask = None, difference = False, verbose = False,
                    cov_interpolate_threshold = 1e4, cov_Lc = 2000):
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
    import numpy.ma as ma
    from scipy.spatial import distance as sp_distance                                                # geopy also has a distance function.  Rename for safety.  
    from scipy import interpolate as scipy_interpolate
    
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

    def generate_correlated_noise_cov(pixel_distances, cov_Lc, shape):
        """ given a matrix of pixel distances (in meters) and a length scale for the noise (also in meters),
        generate some 2d spatially correlated noise.  
        Inputs:
            pixel_distances | rank 2 array | pixels x pixels, distance between each on in metres.  
            cov_Lc | float | Length scale over which the noise is correlated.  units are metres.  
            shape | tuple | (nx, ny)  NOTE X FIRST!
        Returns:
            y_2d | rank 2 array | spatially correlated noise.  
        History:
            2019/06/?? | MEG | Written
            2020/10/05 | MEG | Overhauled to be in metres and use scipy cholesky
            2020/10/06 | MEG | Add support for rectangular atmospheres.  
        """
        import scipy
        nx = shape[0]
        ny = shape[1]
        Cd = np.exp((-1 * pixel_distances)/cov_Lc)                                # from the matrix of distances, convert to covariances using exponential equation
        Cd_L = np.linalg.cholesky(Cd)                                             # ie Cd = CD_L @ CD_L.T      Worse error messages, so best called in a try/except form.  
        #Cd_L = scipy.linalg.cholesky(Cd, lower=True)                             # better error messages than the numpy versio, but can cause crashes on some machines
        x = np.random.randn((ny*nx))                                              # Parsons 2007 syntax - x for uncorrelated noise
        y = Cd_L @ x                                                              # y for correlated noise
        y_2d = np.reshape(y, (ny, nx))                                            # turn back to rank 2
        return y_2d
    
    
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
        
        import numpy as np
        import numpy.matlib as npm
        import math
        
        np.seterr(divide='ignore')
    
        cut_off_freq=1/50                                                   # drop wavelengths above 50 km 
        
        x=np.arange(0,int(nx/2))                                            # positive frequencies only
        y=np.arange(0,int(ny/2))                                            # positive frequencies only
        freq_x=np.divide(x,nx*sp)
        freq_y=np.divide(y,ny*sp)
        Y,X=npm.meshgrid(freq_x,freq_y)
        freq=np.sqrt((X*X+Y*Y)/2)                                           # 2D positive frequencies
        
        log_power=np.log10(freq)*-11/3                                      # -11/3 in 2D gives -8/3 in 1D
        ix=np.where(freq<2/3)
        log_power[ix]=np.log10(freq[ix])*-8/3-math.log10(2/3)               # change slope at 1.5 km (2/3 cycles per km)
        
        bin_power=np.power(10,log_power)
        ix=np.where(freq<cut_off_freq)
        bin_power[ix]=0
        
        APS_power=np.zeros((ny,nx))                                         # mirror positive frequencies into other quadrants
        APS_power[0:int(ny/2), 0:int(nx/2)]=bin_power
        # APS_power[0:int(ny/2), int(nx/2):nx]=npm.fliplr(bin_power)
        # APS_power[int(ny/2):ny, 0:int(nx/2)]=npm.flipud(bin_power)
        # APS_power[int(ny/2):ny, int(nx/2):nx]=npm.fliplr(npm.flipud(bin_power))
        APS_power[0:int(ny/2), int(np.ceil(nx/2)):]=npm.fliplr(bin_power)
        APS_power[int(np.ceil(ny/2)):, 0:int(nx/2)]=npm.flipud(bin_power)
        APS_power[int(np.ceil(ny/2)):, int(np.ceil(nx/2)):]=npm.fliplr(npm.flipud(bin_power))
        APS_filt=np.sqrt(APS_power)
        
        x=np.random.randn(ny,nx)                                            # white noise
        y_tmp=np.fft.fft2(x)
        y_tmp2=np.multiply(y_tmp,APS_filt)                                  # convolve with filter
        y=np.fft.ifft2(y_tmp2)
        APS=np.real(y)
    
        APS=APS/np.std(APS)*std_long                                        #  adjust the turbulence by the weather model at the longer wavelengths.
        APS=APS*0.01                                                        # convert from cm to m
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
    n_pixs = nx * ny
    
    if (n_pixs > cov_interpolate_threshold) and (method == 'cov'):
        if verbose:
            print(f"The number of pixels ({n_pixs}) is larger than 'cov_interpolate_threshold' ({int(cov_interpolate_threshold)}) so images will be created "
                  f"with {int(cov_interpolate_threshold)} pixels and interpolated to the full resolution.  ")
        interpolate = True                                                                                  # set boolean flag
        oversize_factor = n_pixs / cov_interpolate_threshold                                                # determine how many times too many pixels we have.  
        lons_ds = np.linspace(lons_mg[-1,0], lons_mg[-1,-1], int(nx * (1/np.sqrt(oversize_factor))))        # make a downsampled vector of just the longitudes (square root as number of pixels is a measure of area, and this is length)
        lats_ds = np.linspace(lats_mg[0,0], lats_mg[-1,0], int(ny * (1/np.sqrt(oversize_factor))))          # and for latitudes
        lons_mg_ds = np.repeat(lons_ds[np.newaxis, :], lats_ds.shape, axis = 0)                             # make rank 2 again
        lats_mg_ds = np.repeat(lats_ds[:, np.newaxis], lons_ds.shape, axis = 1)                             # and for latitudes
        ny_generate, nx_generate = lons_mg_ds.shape                                                         # get the size of the downsampled grid we'll be generating at
    else:
        interpolate = False                                                                                 # set boolean flag
        nx_generate = nx                                                                                    # if not interpolating, these don't change.  
        ny_generate = ny
        lons_mg_ds = lons_mg                                                                                # if not interpolating, don't need to downsample.  
        lats_mg_ds = lats_mg
    
    #2: calculate distance between points
    ph_turbs = np.zeros((n_atms, ny_generate, nx_generate))                                                 # initiate output as a rank 3 (ie n_images x ny x nx)
    xyz_m, pixel_spacing = lon_lat_to_ijk(lons_mg_ds, lats_mg_ds)                                           # get pixel positions in metres from origin in lower left corner (and also their size in x and y direction)
    xy = xyz_m[0:2].T                                                                                       # just get the x and y positions (ie discard z), and make lots x 2 (ie two columns)
      
    
    #3: generate atmospheres, using either of the two methods.  
    if difference == True:
        n_atms += 1                                                                                         # if differencing atmospheres, create one extra so that when differencing we are left with the correct number
    
    if method == 'fft':
        for i in range(n_atms):
            ph_turbs[i,:,:] = generate_correlated_noise_fft(nx_generate, ny_generate,    std_long=1, 
                                                           sp = 0.001 * np.mean((pixel_spacing['x'], pixel_spacing['y'])) )      # generate noise using fft method.  pixel spacing is average in x and y direction (and m converted to km) 
            if verbose:
                print(f'Generated {i+1} of {n_atms} single acquisition atmospheres.  ')
            
    else:
        pixel_distances = sp_distance.cdist(xy,xy, 'euclidean')                                                     # calcaulte all pixelwise pairs - slow as (pixels x pixels)       
        Cd = np.exp((-1 * pixel_distances)/cov_Lc)                                     # from the matrix of distances, convert to covariances using exponential equation
        success = False
        while not success:
            try:
                Cd_L = np.linalg.cholesky(Cd)                                             # ie Cd = CD_L @ CD_L.T      Worse error messages, so best called in a try/except form.  
                #Cd_L = scipy.linalg.cholesky(Cd, lower=True)                               # better error messages than the numpy versio, but can cause crashes on some machines
                success = True
            except:
                success = False
        for n_atm in range(n_atms):
            x = np.random.randn((ny_generate*nx_generate))                                               # Parsons 2007 syntax - x for uncorrelated noise
            y = Cd_L @ x                                                               # y for correlated noise
            ph_turb = np.reshape(y, (ny_generate, nx_generate))                                             # turn back to rank 2
            ph_turbs[n_atm,:,:] = ph_turb
            print(f'Generated {n_atm} of {n_atms} single acquisition atmospheres.  ')
        
        
        # nx = shape[0]
        # ny = shape[1]

        

        # return y_2d
        
        # success = 0
        # fail = 0
        # while success < n_atms:
        # #for i in range(n_atms):
        #     try:
        #         ph_turb = generate_correlated_noise_cov(pixel_distances, cov_Lc, (nx_generate,ny_generate))      # generate noise 
        #         ph_turbs[success,:,:] = ph_turb
        #         success += 1
        #         if verbose:
        #             print(f'Generated {success} of {n_atms} single acquisition atmospheres (with {fail} failures).  ')
        #     except:
        #         fail += 0
        #         if verbose:
        #             print(f"'generate_correlated_noise_cov' failed, which is usually due to errors in the cholesky decomposition that Numpy is performing.  The odd failure is normal.  ")
                
        #     # ph_turbs[i,:,:] = generate_correlated_noise_cov(pixel_distances, cov_Lc, (nx_generate,ny_generate))      # generate noise 
        #     # if verbose:
                
                

    #3: possibly interplate to bigger size
    if interpolate:
        if verbose:
            print('Interpolating to the larger size...', end = '')
        ph_turbs_output = np.zeros((n_atms, ny, nx))                                                                          # initiate output at the upscaled size (ie the same as the original lons_mg shape)
        for atm_n, atm in enumerate(ph_turbs):                                                                                # loop through the 1st dimension of the rank 3 atmospheres.  
            f = scipy_interpolate.interp2d(np.arange(0,nx_generate), np.arange(0,ny_generate), atm, kind='linear')           # and interpolate them to a larger size.  First we give it  meshgrids and values for each point
            ph_turbs_output[atm_n,:,:] = f(np.linspace(0, nx_generate, nx), np.linspace(0, ny_generate, ny))                  # then new meshgrids at the original (full) resolution.  
        if verbose:
            print('Done!')
    else:
        ph_turbs_output = ph_turbs                                                                                              # if we're not interpolating, no change needed
       
    # 4: rescale to correct range (i.e. a couple of cm)
    ph_turbs_m = np.zeros(ph_turbs_output.shape)
    for atm_n, atm in enumerate(ph_turbs_output):
        ph_turbs_m[atm_n,] = rescale_atmosphere(atm, mean_m)
                
    # 5: return back to the shape given, which can be a rectangle:
    ph_turbs_m = ph_turbs_m[:,:lons_mg.shape[0],:lons_mg.shape[1]]
    
    if water_mask is not None:
        water_mask_r3 = ma.repeat(water_mask[np.newaxis,], ph_turbs_m.shape[0], axis = 0)
        ph_turbs_m = ma.array(ph_turbs_m, mask = water_mask_r3)
    
    return ph_turbs_m


def aps_simulate():

    pixel_size_degs = 1/3600
    
    lons = np.arange(0.0, 0.0 + (pixel_size_degs * 512), pixel_size_degs)
    lats = np.arange(0.0, 0.0 + (pixel_size_degs * 512), pixel_size_degs)
    lons_mg = np.repeat(lons[np.newaxis,:], len(lats), axis = 0)
    lats_mg = np.repeat(lats[::-1, np.newaxis], len(lons), axis = 1)

    ph_turb = atmosphere_turb(1, lons_mg, lats_mg, verbose=True, mean_m = 0.02,
                                 method = 'fft')

    return ph_turb[0,]


def coherence_mask_simulate(threshold: float = 0.3):

    pixel_size_degs = 1/3600
    
    lons = np.arange(0.0, 0.0 + (pixel_size_degs * 512), pixel_size_degs)
    lats = np.arange(0.0, 0.0 + (pixel_size_degs * 512), pixel_size_degs)
    lons_mg = np.repeat(lons[np.newaxis,:], len(lats), axis = 0)
    lats_mg = np.repeat(lats[::-1, np.newaxis], len(lons), axis = 1)

    mask_coh_values = atmosphere_turb(1, lons_mg, lats_mg, verbose=True, mean_m = 0.01,
                                 method = 'fft')

    mask_coh_values = (mask_coh_values - np.min(mask_coh_values)) / np.max(mask_coh_values - np.min(mask_coh_values))
    mask_coh = np.where(mask_coh_values > threshold, np.ones(lons_mg.shape), np.zeros(lons_mg.shape)) 

    return mask_coh


def gen_simulated_deformation(
    seed:      int  = 0,
    tile_size: int  = 512,
    log:       bool = False
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

    Returns:
    --------
    masked_grid : np.ndarray(shape=(tile_size, tile_size))
        An array representing a mask over the simulated deformation which simulates masking an event.
    wrapped_grid : np.ndarray(shape=(tile_size, tile_size)
        The wrapped interferogram.
    """

    if seed != 0: random.seed = seed

    only_noise_dice_roll = random.randint(0, 9)

    los_vector  = np.array([[ 0.38213591],
                            [-0.08150437],
                            [ 0.92050485]])

    source      = "quake"
    nx          = ny = tile_size                                                  # ?m in each direction with 90m pixels
    X, Y        = np.meshgrid(90 * np.arange(0, nx),90 * np.arange(0,ny))         # make a meshgrid
    Y           = np.flipud(Y)                                                    # change 0 y cordiante from matrix style (top left) to axes style (bottom left)
    ij          = np.vstack((np.ravel(X)[np.newaxis], np.ravel(Y)[np.newaxis]))   # pairs of coordinates of everywhere we have data   
    ijk         = np.vstack((ij, np.zeros((1, ij.shape[1]))))   

    if only_noise_dice_roll != 0 and only_noise_dice_roll != 9 and only_noise_dice_roll != 10:

        source_x = np.max(X) / random.randint(1, 10)
        source_y = np.max(Y) / random.randint(1, 10)

        strike       = random.randint(0, 359)
        dip          = random.randint(0, 90)
        rake         = [0, 90, -90, 180][random.randint(0, 3)]
        slip         = 1

        length       = random.randint(1000, np.max(X) // 16)
        top_depth    = random.randint(1000, np.max(X) // 8)
        bottom_depth = top_depth * random.randint(2, 4)

        kwargs = {
            'strike'      : strike,
            'dip'         : dip,
            'length'      : length,
            'rake'        : rake,
            'slip'        : slip,
            'top_depth'   : top_depth,
            'bottom_depth': bottom_depth
        }

        U = deformation_eq_dyke_sill(source, (source_x, source_y), ijk, **kwargs)

        amplitude_adjustment = 1000 * np.pi

        x_grid   = np.reshape(U[0,], (X.shape[0], X.shape[1])) * amplitude_adjustment
        y_grid   = np.reshape(U[1,], (X.shape[0], X.shape[1])) * amplitude_adjustment
        z_grid   = np.reshape(U[2,], (X.shape[0], X.shape[1])) * amplitude_adjustment

        los_grid = ((x_grid * los_vector[0,0]) + (y_grid * los_vector[1,0]) + (z_grid * los_vector[2,0]))
    
        masked_grid = np.zeros((tile_size, tile_size))

        mask_one_indicies  = np.abs(los_grid) >= np.pi

        masked_grid[mask_one_indicies] = 1

        atmosphere_phase = aps_simulate() * 90 * np.pi

        coherence_mask = coherence_mask_simulate(0.3)
        coh_masked_indicies = coherence_mask[0,0:512, 0:512] == 0

        interferogram = los_grid + atmosphere_phase[0:512, 0:512]

        wrapped_grid = wrap_interferogram(interferogram, noise = 0.1)

        wrapped_grid[coh_masked_indicies] = 0

        if log:
            print("Max X Position (meters): ", np.max(X))
            print("Max Y Position (meters): ", np.max(Y))

            print("Source X Position (meters): ", source_x)
            print("Source Y Position (meters): ", source_y)

            print("Source X Position (meters): ", source_x)
            print("Source Y Position (meters): ", source_y)

            print("Source Parameters: ", kwargs)

            print("Maximum Phase Value: ", np.max(los_grid))
        
        return masked_grid, wrapped_grid

    elif only_noise_dice_roll == 0:

        noise = np.random.uniform(1.0, 50.0, size=(tile_size, tile_size))

        interferogram = add_noise(noise, tile_size)

        wrapped_grid = np.angle(np.exp(1j * (interferogram)))
        masked_grid  = np.zeros((tile_size, tile_size))

        return masked_grid, wrapped_grid

    else:

        atmosphere_turbulence = aps_simulate() * 90 * np.pi

        wrapped_grid = np.angle(np.exp(1j * (atmosphere_turbulence)))

        coherence_mask = coherence_mask_simulate(0.3)
        coh_masked_indicies = coherence_mask[0,0:512, 0:512] == 0

        wrapped_grid[coh_masked_indicies] = 0

        masked_grid = np.zeros((tile_size, tile_size))

        return masked_grid, wrapped_grid