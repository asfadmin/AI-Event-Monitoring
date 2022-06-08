"""
 Created By:   Jason Herning
 File Name:    synthetic_interferogram.py
 Date Created: 01-25-2021
 Description:  Functions for synthetic interferogram generation
"""

import random
from typing import Tuple

import numpy as np
from astropy.modeling.models import Gaussian2D
from src.processing import simulate_unet_cropping


def make_synthetic_interferogram(size: int, *gaussians: Gaussian2D) -> np.ndarray:

    """
    Returns an array that simulates an interferogram

    Parameters:
    -----------
    size : int
        The dimensional size of the desired interferogram.
    *gaussians : Gaussian2D
        An arbitrary number of guassians to be placed in the interferogram.

    Returns:
    --------
    interferogram : np.ndarray(shape=(size, size))
        The simulated interferogram array.
    """

    y, x = np.mgrid[0:size, 0:size]
    interferogram = np.zeros((size, size))
    for gaussian in gaussians:
        interferogram += (gaussian(x, y))

    return interferogram


def wrap_interferogram(interferogram: np.ndarray, seed: int = 0, noise: float = 0) -> np.ndarray:

    """
    Wrap the inputed array to values between -pi and pi.

    Parameters:
    -----------
    interferogram : np.ndarray
        The unwrapped interferogram which should be wrapped.
    seed : int, Optional
        A seed for the random functions. For the same seed, with all other values the same
        as well, the interferogram generation will have the same results. If left at 0,
        the results will be different every time.
    noise : float, Optional
        The maximum noise value to be added to the wrapped interferogram
        using a uniform distribution.

    Returns:
    --------
    wrapped_interferogram : np.ndarray(shape=interferogram.shape)
        The simulated wrapped interferogram array.
    """
    if seed:
        np.random.seed(seed)
    noise = np.random.uniform(-noise, noise, size=interferogram.shape)
    wrapped_interferogram = np.angle(np.exp(1j*interferogram))
    return wrapped_interferogram


def make_random_dataset(
                size:         int,
                seed:         int   = 0,
                min_g2d:      int   = 1,
                max_g2d:      int   = 1,
                crop_size:    int   = 0,
                max_noise:    float = 0.0,
                min_amp:      float = -44.0,
                max_amp:      float = 44.0,
                min_x_mean:   float = 64,
                max_x_mean:   float = 448,
                min_y_mean:   float = 64,
                max_y_mean:   float = 448,
                min_x_stddev:   float = 16,
                max_x_stddev:   float = 64,
                min_y_stddev:   float = 16,
                max_y_stddev:   float = 64,
                ) -> Tuple[np.ndarray, np.ndarray]:

    """
    Generate a psuedo-random unwrapped and wrapped interferogram pair.

    Parameters:
    -----------
    size : int
        The desired dimensional size of the interferogram pairs. This should match
        the input shape of the model.
    seed : int, Optional
        A seed for the random functions. For the same seed, with all other values the same
        as well, the interferogram generation will have the same results. If left at 0,
        the results will be different every time.
    min_g2d : int, Optional
        The minimum amount of gaussian peaks/events that the images should have.
    max_g2d : int, Optional
        The maximum amount of gaussian peaks/events that the images should have.
    crop_size : int, Optional.
        If the output shape of the model is different than the input shape, this should be
        set to be equal to the output shape. The unwrapped interferogram will be cropped to
        this.
    max_noise : float, Optional
        The amount of gaussian additive noise to add to the wrapped interferogram.
    min_amp : float, Optional
        The minimum value that the peak of the guassians can have.
    max_amp : float, Optional
        The maximum value that the peak of the guassians can have.
    min_x_mean : float, Optional
        The minimum x position of the peaks of gaussians.
    max_x_mean : float, Optional
        The maximum x position of the peaks of a gaussians.
    min_y_mean : float, Optional
        The minimum y position of the peaks of a gaussians.
    max_y_mean : float, Optional
        The maximum y position of the peaks of a gaussians.
    min_x_stddev : float, Optional
        The minimum standard deviation of gaussians in the x direction. This is
        how thin they can be in the x direction.
    max_x_stddev : float, Optional
        The maximum standard deviation of guassians in the x direction. This is
        how wide they can be in the x direction.
    min_y_stddev : float, Optional
        The minimum standard deviation of gaussians in the y direction. This is
        how thin they can be in the y direction.
    max_y_stddev : float, Optional
        The maximum standard deviation of guassians in the y direction. This is
        how wide they can be in the y direction

    Returns:
    --------
    interferogram : np.ndarray(shape=(size, size))
        The array of the generated simulated wrapped interferogram.
    unwrapped_interferogram : np.ndarray(shape=(size, size) or (crop_size, crop_size))
        The array of the generated simulated unwrapped interferogram.
    """

    if seed:
        random.seed(seed)

    g2d_count = random.randint(min_g2d, max_g2d)
    
    noise = random.uniform(0, max_noise)

    gaussians = []
    for g2d in range(g2d_count):
        
        amp      = random.uniform(min_amp, max_amp)
        x_mean   = random.uniform(min_x_mean, max_x_mean)
        y_mean   = random.uniform(min_y_mean, max_y_mean)
        x_stddev = random.uniform(min_x_stddev, max_x_stddev)
        y_stddev = random.uniform(min_y_stddev, max_y_stddev)
        theta    = random.uniform(-np.pi, np.pi)
        
        gaussians.append(Gaussian2D(amp, x_mean, y_mean, x_stddev, y_stddev, theta))

    interferogram = make_synthetic_interferogram(size, *gaussians)

    wrapped_interferogram = wrap_interferogram(interferogram, seed, noise)
    
    one_indices      = interferogram >= 1e-3
    neg_one_indicies = interferogram <= -1e-3

    interferogram[one_indices]      = 1.0
    interferogram[neg_one_indicies] = 1.0

    if crop_size > 0:
        interferogram = simulate_unet_cropping(interferogram, (crop_size, crop_size))

    return interferogram, wrapped_interferogram
