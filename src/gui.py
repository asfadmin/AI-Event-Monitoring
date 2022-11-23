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
    axs_masked.imshow(masked, origin='lower', cmap='jet')

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


def interactive_interferogram(event_type: str = "quake") -> None:

    """
    GUI interface for interactive interferogram simulation.
    """

    from matplotlib.widgets import Slider

    from src.sarsim import gen_simulated_deformation

    kwargs = {
        'source_x'    : 22000,
        'source_y'    : 22000,
        'strike'      : 180,
        'dip'         : 45,
        'length'      : 1000,
        'rake'        : 90,
        'slip'        : 1,
        'top_depth'   : 3000,
        'bottom_depth': 6000,
        'width'       : 3000,
        'depth'       : 3000,
        'opening'     : 0.5
    }

    fig, [axs_unwrapped, axs_wrapped] = plt.subplots(1, 2, sharex=True, sharey=True, tight_layout=True)

    axs_unwrapped.set_title("unwrapped")
    axs_unwrapped.set_position([0.05, 0.45, 0.5, 0.5])
    axs_wrapped.set_title("wrapped")
    axs_wrapped.set_position([0.5, 0.45, 0.5, 0.5])

    axs_slip            = plt.axes([0.375, 0.36, 0.25, 0.02])
    slider_slip         = Slider(axs_slip, 'slip', 0.0, 10.0, valinit=kwargs['slip'])

    axs_strike          = plt.axes([0.375, 0.33, 0.25, 0.02])
    slider_strike       = Slider(axs_strike, 'strike', 0.0, 180.0, valinit=kwargs['strike'])

    axs_dip             = plt.axes([0.375, 0.30, 0.25, 0.02])
    slider_dip          = Slider(axs_dip, 'dip', 0.0, 90.0, valinit=kwargs['dip'])

    axs_rake            = plt.axes([0.375, 0.27, 0.25, 0.02])
    slider_rake         = Slider(axs_rake, 'rake', -180.0, 180.0, valinit=kwargs['rake'])

    axs_opening         = plt.axes([0.375, 0.24, 0.25, 0.02])
    slider_opening      = Slider(axs_opening, 'opening', 0.0, 10.0, valinit=kwargs['opening'])

    axs_top_depth       = plt.axes([0.375, 0.21, 0.25, 0.02])
    slider_top_depth    = Slider(axs_top_depth, 'top_depth', 0.0, 45000.0, valinit=kwargs['top_depth'])

    axs_bottom_depth    = plt.axes([0.375, 0.18, 0.25, 0.02])
    slider_bottom_depth = Slider(axs_bottom_depth, 'bottom_depth', 0.0, 45000.0, valinit=kwargs['bottom_depth'])

    axs_width           = plt.axes([0.375, 0.15, 0.25, 0.02])
    slider_width        = Slider(axs_width, 'width', 100.0, 10000.0, valinit=kwargs['width'])

    axs_length          = plt.axes([0.375, 0.12, 0.25, 0.02])
    slider_length       = Slider(axs_length, 'length', 100.0, 10000.0, valinit=kwargs['length'])

    axs_source_x        = plt.axes([0.375, 0.09, 0.25, 0.02])
    slider_source_x     = Slider(axs_source_x, 'source_x', 0.0, 45000.0, valinit=kwargs['source_x'])

    axs_source_y        = plt.axes([0.375, 0.06, 0.25, 0.02])
    slider_source_y     = Slider(axs_source_y, 'source_y', 0.0, 45000.0, valinit=kwargs['source_y'])

    unwrapped, masked, wrapped, presence = gen_simulated_deformation(
        seed       = 100000,
        tile_size  = 512,
        event_type = event_type,
        **kwargs
    )

    axs_wrapped.imshow(wrapped, origin='lower', cmap='jet', vmin=-np.pi, vmax=np.pi)
    axs_unwrapped.imshow(masked, origin='lower', cmap='jet')

    def update(val):

        kwargs = {
            'source_x'    : slider_source_x.val,
            'source_y'    : slider_source_y.val,
            'strike'      : slider_strike.val,
            'dip'         : slider_dip.val,
            'length'      : slider_length.val,
            'rake'        : slider_rake.val,
            'slip'        : slider_slip.val,
            'top_depth'   : slider_top_depth.val,
            'bottom_depth': slider_bottom_depth.val,
            'width'       : slider_width.val,
            'depth'       : slider_top_depth.val,
            'opening'     : slider_opening.val        
        }

        unwrapped, masked, wrapped, presence = gen_simulated_deformation(
            seed       = 100000,
            tile_size  = 512,
            event_type = event_type,
            **kwargs
        )

        axs_wrapped.imshow(wrapped, origin='lower', cmap='jet', vmin=-np.pi, vmax=np.pi)
        axs_unwrapped.imshow(masked, origin='lower', cmap='jet')

        fig.canvas.draw()

    slider_source_x.on_changed(update)
    slider_source_y.on_changed(update)
    slider_strike.on_changed(update)
    slider_dip.on_changed(update)
    slider_length.on_changed(update)
    slider_rake.on_changed(update)
    slider_slip.on_changed(update)
    slider_top_depth.on_changed(update)
    slider_bottom_depth.on_changed(update)
    slider_width.on_changed(update)
    slider_opening.on_changed(update)

    plt.show()
