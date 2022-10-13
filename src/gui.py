"""
 Created By:  Jason Herning, Andrew Player, and Robert Lawton
 File Name:   gui.py
 Description: gui apps
"""


import matplotlib.pyplot as plt
import numpy as np

from src.io         import get_product_arrays
from src.processing import simulate_unet_cropping, tile, tiles_to_image


def show_dataset(
    masked:  np.ndarray, 
    wrapped: np.ndarray
) -> None:

    """
    Plot the masked and wrapped arrays.

    Parameters:
    -----------
    masked : np.ndarray
        The event-mask of the interferogram.
    wrapped : np.ndarray
        The wrapped interferogram.

    Returns:
    --------
    None
    """

    _, [axs_masked, axs_wrapped] = plt.subplots(1, 2)

    axs_masked.set_title("Masked")
    axs_masked.imshow(masked, origin='lower', cmap='jet', vmin=0, vmax=np.max(masked))

    axs_wrapped.set_title("Wrapped")
    axs_wrapped.imshow(wrapped, origin='lower', cmap='jet')

    plt.show()


def show_product(
    product_path: str    ,
    crop_size:    int = 0,
    tile_size:    int = 0
) -> None:

    """
    Plots the Wrapped, Unwrapped, and Correlation Images in the given product.

    Parameters:
    -----------
    product_path : str
        The path to the folder containing the ASF InSAR product to display.
        crop_size : int, Optional
    tile_size, Optional
        This is an optional value that simulates the padding and tiling that would
        happen if this was passed through a tiling model.
    crop_size : int, Optional
        This is an optional value that simulates the padding, tiling, and cropping
        that would happen if this was passed through a model with the give crop (output shape).

    Returns:
    --------
    None
    """

    arr_w, arr_uw, arr_c = get_product_arrays(product_path)

    tiled_arr_uw, tile_rows, tile_cols = tile(arr_uw, (1024, 1024), even_pad=True, crop_size=crop_size)

    cutoff_value = 0.2
    correlation_cutoff_indecies = arr_c < cutoff_value
    arr_c[correlation_cutoff_indecies] = np.NAN

    if crop_size:
        cropped_arr_uw = np.zeros((tile_rows * tile_cols, crop_size, crop_size))

        # Simulate UNET Cropping
        count = 0
        for tile_ in tiled_arr_uw:
            cropped_tile = simulate_unet_cropping(tile_, (crop_size, crop_size))
            cropped_arr_uw[count] = cropped_tile
            count += 1

        rebuilt_arr_uw = tiles_to_image(cropped_arr_uw, tile_rows, tile_cols, arr_uw.shape, (crop_size > 0), tile_size)

        _, [axs_wrapped, axs_correlation, axs_unwrapped, axs_tiled_unwrapped] = plt.subplots(1, 4)

    else:
        _, [axs_wrapped, axs_correlation, axs_unwrapped] = plt.subplots(1, 3)

    axs_wrapped.set_title("wrapped")
    axs_wrapped.imshow(arr_w, origin='lower', cmap='jet')

    axs_correlation.set_title("correlation")
    axs_correlation.imshow(arr_c, origin='lower', cmap='jet')

    axs_unwrapped.set_title("unwrapped")
    axs_unwrapped.imshow(arr_uw, origin='lower', cmap='jet')

    if crop_size:
        axs_tiled_unwrapped.set_title("tiled_unwrapped")
        axs_tiled_unwrapped.imshow(rebuilt_arr_uw, origin='lower', cmap='jet')

    plt.show()
