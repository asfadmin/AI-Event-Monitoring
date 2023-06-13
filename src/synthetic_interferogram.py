"""
 Summary
 -------
 Functions for synthetic interferogram generation for training.

 Notes
 -----
 Created by Andrew Player.
"""

import random
from typing import Tuple

import numpy as np
from astropy.modeling.models import Gaussian2D
from src.processing import simulate_unet_cropping


def make_synthetic_interferogram(size: int, *gaussians: Gaussian2D) -> np.ndarray:
    """
    Returns an array that simulates an unwrapped interferogram

    Parameters
    -----------
    size : int
        The dimensional size of the desired interferogram.
    *gaussians : Gaussian2D
        An arbitrary number of guassians to be placed in the interferogram.

    Returns
    --------
    interferogram : np.ndarray(shape=(size, size))
        The simulated unwrapped interferogram array.
    """

    y, x = np.mgrid[0:size, 0:size]

    gaussian_grid = np.zeros((size, size))

    for gaussian in gaussians:
        gaussian_grid += gaussian(x, y)

    interferogram = add_noise(np.copy(gaussian_grid), size)

    return interferogram, gaussian_grid


def add_noise(interferogram: np.ndarray, size: int) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]

    perlingram = generate_perlin(interferogram.shape[0])
    max_pelin = np.amax(perlingram)
    perlin_multplier = 25 / (np.pi * max_pelin)

    interferogram += perlingram * perlin_multplier

    m = np.random.uniform(0, 0.025, [2])
    C = np.random.randint(0, 1)

    interferogram = m[0] * x + m[1] * y + C + interferogram

    return interferogram


def generate_perlin_noise_2d(shape, resolution):
    delta = (resolution[0] / shape[0], resolution[1] / shape[1])
    delta2 = (shape[0] // resolution[0], shape[1] // resolution[1])

    grid = (
        np.mgrid[0 : resolution[0] : delta[0], 0 : resolution[1] : delta[1]].transpose(
            1, 2, 0
        )
        % 1
    )

    interpolated_grid = np.power(grid, 3) * (grid * (grid * 6 - 15) + 10)
    angles = 2 * np.pi * np.random.rand(resolution[0] + 1, resolution[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    gradients = gradients.repeat(delta2[0], 0).repeat(delta2[1], 1)

    gradients00 = gradients[: -delta2[0], : -delta2[1]]
    ramp00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * gradients00, 2)

    gradients10 = gradients[delta2[0] :, : -delta2[1]]
    ramp10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * gradients10, 2)

    gradients01 = gradients[: -delta2[0], delta2[1] :]
    ramp01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * gradients01, 2)
    gradients11 = gradients[delta2[0] :, delta2[1] :]
    ramp11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * gradients11, 2)

    ramps0 = (
        ramp00 * (1 - interpolated_grid[:, :, 0]) + ramp10 * interpolated_grid[:, :, 0]
    ) * (1 - interpolated_grid[:, :, 1])
    ramps1 = (
        ramp01 * (1 - interpolated_grid[:, :, 0]) + ramp11 * interpolated_grid[:, :, 0]
    ) * (interpolated_grid[:, :, 1])

    ramps = ramps0 + ramps1

    return np.sqrt(2) * ramps


def generate_perlin(size: int) -> np.ndarray:
    """
    Generate an array with perlin noise.

    Parameters
    -----------
    size : int
        The number of rows/cols.

    Returns
    --------
    perlin_array : np.ndarray
        The array containing the generated perlin noise.

    """

    perlin_array = np.zeros((size, size))
    for j in range(0, 3):
        for i in range(4 * (2**j), size):
            if size % i == 0:
                res = i
                perlin_array += generate_perlin_noise_2d((size, size), (res, res)) * (
                    2 ** (3 - j)
                )
                break
    min = np.amin(perlin_array)
    perlin_array -= min
    return perlin_array


def wrap_interferogram(
    interferogram: np.ndarray, seed: int = 0, noise: float = 0
) -> np.ndarray:
    """
    Wrap the inputed array to values between -pi and pi.

    Parameters
    -----------
    interferogram : np.ndarray
        The unwrapped interferogram which should be wrapped.
    seed : int, Optional
        A seed for the random functions. For the same seed, with all other values the
        same as well, the interferogram generation will have the same results. If left
        at 0, the results will be different every time.
    noise : float, Optional
        The maximum noise value to be added to the wrapped interferogram
        using a uniform distribution.

    Returns
    --------
    wrapped_interferogram : np.ndarray(shape=interferogram.shape)
        The simulated wrapped interferogram array.
    """

    if seed:
        np.random.seed(seed)

    if noise > 0.0:
        noise = np.random.uniform(-noise, noise, size=interferogram.shape)
        interferogram = noise + interferogram

    wrapped_interferogram = np.angle(np.exp(1j * (interferogram)))

    return wrapped_interferogram


def make_random_dataset(
    size: int,
    seed: int = 0,
    crop_size: int = 0,
    max_noise: float = 0.0,
    min_amp: float = -300.0,
    max_amp: float = 300.0,
    min_x_mean: float = 64,
    max_x_mean: float = 448,
    min_y_mean: float = 64,
    max_y_mean: float = 448,
    min_x_stddev: float = 16,
    max_x_stddev: float = 64,
    min_y_stddev: float = 16,
    max_y_stddev: float = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a simulated wrapped interferogram along with an event-mask

    Parameters
    -----------
    size : int
        The desired dimensional size of the interferogram pairs. This should match
        the input shape of the model.
    seed : int, Optional
        A seed for the random functions. For the same seed, with all other values the
        same as well, the interferogram generation will have the same results. If left
        at 0, the results will be different every time.
    min_g2d : int, Optional
        The minimum amount of gaussian peaks/events that the images should have.
    max_g2d : int, Optional
        The maximum amount of gaussian peaks/events that the images should have.
    crop_size : int, Optional.
        If the output shape of the model is different than the input shape, this should
        be set to be equal to the output shape. The unwrapped interferogram will be
        cropped to this.
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

    Returns
    --------
    interferogram : np.ndarray(shape=(size, size))
        The array of the generated simulated wrapped interferogram.
    masked_interferogram : np.ndarray(shape=(size, size) or (crop_size, crop_size))
        An array representing a mask over the gaussian which simulates masking an event.
    """

    if seed:
        random.seed(seed)

    g2d_count = random.randint(1, 3)

    noise = random.uniform(0, max_noise)

    gaussians = []
    for _g2d in range(g2d_count):
        amp = random.uniform(min_amp, max_amp)
        x_mean = random.uniform(min_x_mean, max_x_mean)
        y_mean = random.uniform(min_y_mean, max_y_mean)
        x_stddev = random.uniform(min_x_stddev, max_x_stddev)
        y_stddev = random.uniform(min_y_stddev, max_y_stddev)
        theta = random.uniform(-np.pi, np.pi)

        gaussians.append(Gaussian2D(amp, x_mean, y_mean, x_stddev, y_stddev, theta))

    interferogram, gaussian_only = make_synthetic_interferogram(size, *gaussians)

    wrapped_interferogram = wrap_interferogram(interferogram, seed, noise)

    masked_interferogram = np.copy(gaussian_only)

    one_indices = masked_interferogram >= 0.1
    neg_one_indicies = masked_interferogram <= -0.1
    zero_indicies = masked_interferogram < 0.1

    masked_interferogram[zero_indicies] = 0.0
    masked_interferogram[one_indices] = 1.0
    masked_interferogram[neg_one_indicies] = 1.0

    if crop_size > 0:
        interferogram = simulate_unet_cropping(interferogram, (crop_size, crop_size))

    return masked_interferogram, wrapped_interferogram
