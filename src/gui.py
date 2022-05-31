"""
 Created By:  Jason Herning, Andrew Player, and Robert Lawton
 File Name:   gui.py
 Description: gui apps
"""


import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.models import Gaussian2D
from matplotlib.widgets import Slider
from src.io import get_product_arrays
from src.processing import simulate_unet_cropping, tile, tiles_to_image
from src.synthetic_interferogram import make_synthetic_interferogram, wrap_interferogram


def show_dataset(unwrapped, wrapped) -> None:

    """
    Plot the unwrapped and wrapped arrays.

    Parameters:
    -----------
    unwrapped : np.ndarray
        The unwrapped interferogram array.
    wrapped : np.ndarray
        The wrapped interferogram array.

    Returns:
    --------
    None
    """

    _, [axs_unwrapped, axs_wrapped] = plt.subplots(1, 2)

    axs_unwrapped.set_title("unwrapped")
    axs_unwrapped.imshow(unwrapped, origin='lower', cmap='jet')

    axs_wrapped.set_title("wrapped")
    axs_wrapped.imshow(wrapped, origin='lower', cmap='jet')

    plt.show()


def interactive_interferogram() -> None:

    """
    GUI interface for interactive interferogram simulation.
    """

    size = 256
    amp = 1.0
    noise = 0.0
    xmean = 200.0
    ymean = 200.0
    xstddev = 100.0
    ystddev = 100.0
    theta = 0.0

    fig, [axs_unwrapped, axs_wrapped] = plt.subplots(1, 2)

    axs_unwrapped.set_title("unwrapped")
    axs_unwrapped.set_position([0.05, 0.45, 0.5, 0.5])
    axs_wrapped.set_title("wrapped")
    axs_wrapped.set_position([0.5, 0.45, 0.5, 0.5])

    axs_amp = plt.axes([0.25, 0.35, 0.65, 0.03])
    slider_amp = Slider(axs_amp, 'amp', -250.0, 250.0, valinit=amp)

    axs_noise = plt.axes([0.25, 0.30, 0.65, 0.03])
    slider_noise = Slider(axs_noise, 'noise', 0.0, 1.0, valinit=noise)

    axs_xmean = plt.axes([0.25, 0.25, 0.65, 0.03])
    slider_xmean = Slider(axs_xmean, 'x mean', -250.0, 250.0, valinit=xmean)

    axs_ymean = plt.axes([0.25, 0.20, 0.65, 0.03])
    slider_ymean = Slider(axs_ymean, 'y mean', -250.0, 250.0, valinit=ymean)

    axs_xstddev = plt.axes([0.25, 0.15, 0.65, 0.03])
    slider_xstddev = Slider(axs_xstddev, 'x stddev',  0.0, 200.0, valinit=xstddev)

    axs_ystddev = plt.axes([0.25, 0.10, 0.65, 0.03])
    slider_ystddev = Slider(axs_ystddev, 'y stddev',  0.0, 200.0, valinit=ystddev)

    axs_theta = plt.axes([0.25, 0.05, 0.60, 0.03])
    slider_theta = Slider(axs_theta, 'theta', -np.pi, np.pi, valinit=theta)

    g2d_1 = Gaussian2D(amplitude=amp, x_mean=xmean, y_mean=ymean, x_stddev=xstddev, y_stddev=ystddev, theta=theta)
    gaussians = [g2d_1]
    unwrapped_img = make_synthetic_interferogram(size, *gaussians)
    wrapped_img = wrap_interferogram(unwrapped_img)

    axs_unwrapped.imshow(unwrapped_img, origin='lower', cmap='hsv')
    axs_wrapped.imshow(wrapped_img, origin='lower', cmap='hsv')

    def update(val):
        amp     = slider_amp.val
        noise   = slider_noise.val
        xmean   = slider_xmean.val
        ymean   = slider_ymean.val
        xstddev = slider_xstddev.val
        ystddev = slider_ystddev.val
        theta   = slider_theta.val

        g2d_1 = Gaussian2D(
            amplitude=amp,
            x_mean=xmean,
            y_mean=ymean,
            x_stddev=xstddev,
            y_stddev=ystddev,
            theta=theta
            )
        gaussians = [g2d_1]
        unwrapped_img = make_synthetic_interferogram(size, *gaussians)
        wrapped_img = wrap_interferogram(unwrapped_img, noise=noise)

        axs_wrapped.imshow(wrapped_img, origin='lower', cmap='hsv')
        axs_unwrapped.imshow(unwrapped_img, origin='lower', cmap='hsv')

        fig.canvas.draw()

    slider_amp.on_changed(update)
    slider_noise.on_changed(update)
    slider_xmean.on_changed(update)
    slider_ymean.on_changed(update)
    slider_xstddev.on_changed(update)
    slider_ystddev.on_changed(update)
    slider_theta.on_changed(update)

    plt.show()


def show_product(product_path: str, crop_size: int = 0, tile_size: int = 0) -> None:

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
