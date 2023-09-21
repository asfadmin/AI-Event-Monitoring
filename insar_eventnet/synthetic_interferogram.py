"""
 Summary
 -------
 Functions for synthetic interferogram generation for training.

 Notes
 -----
 Created by Andrew Player.
"""

import numpy as np


def _generate_perlin_noise_2d(shape, resolution):
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


def _generate_perlin(size: int) -> np.ndarray:
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
                perlin_array += _generate_perlin_noise_2d((size, size), (res, res)) * (
                    2 ** (3 - j)
                )
                break
    min = np.amin(perlin_array)
    perlin_array -= min
    return perlin_array
