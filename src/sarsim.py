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

    if only_noise_dice_roll != 0 and only_noise_dice_roll != 9:

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

        wrapped_grid = wrap_interferogram(add_noise(los_grid, tile_size), noise = 0.1)

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

        grid = np.zeros((tile_size, tile_size))

        interferogram = add_noise(grid, tile_size)

        wrapped_grid = np.angle(np.exp(1j * (interferogram)))
        masked_grid  = np.zeros((tile_size, tile_size))

        return masked_grid, wrapped_grid