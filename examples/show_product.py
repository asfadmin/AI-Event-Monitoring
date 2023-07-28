# This example demonstrates plotting a products wrapped, unwrapped and correlation
# arrays

import matplotlib.pyplot as plt
import numpy as np

from insar_eventnet import io, processing

product_path = input("Product Path:")
crop_size = 0
tile_size = 0

arr_w, arr_uw, arr_c = io.get_product_arrays(product_path)

tiled_arr_uw, tile_rows, tile_cols = processing.tile(
    arr_uw, (1024, 1024), even_pad=True, crop_size=crop_size
)

cutoff_value = 0.2
correlation_cutoff_indecies = arr_c < cutoff_value
arr_c[correlation_cutoff_indecies] = np.NAN

if crop_size:
    cropped_arr_uw = np.zeros((tile_rows * tile_cols, crop_size, crop_size))

    # Simulate UNET Cropping
    for count, tile_ in enumerate(tiled_arr_uw):
        cropped_tile = processing.simulate_unet_cropping(tile_, (crop_size, crop_size))
        cropped_arr_uw[count] = cropped_tile

    rebuilt_arr_uw = processing.tiles_to_image(
        cropped_arr_uw,
        tile_rows,
        tile_cols,
        arr_uw.shape,
        (crop_size > 0),
        tile_size,
    )

    _, [
        axs_wrapped,
        axs_correlation,
        axs_unwrapped,
        axs_tiled_unwrapped,
    ] = plt.subplots(1, 4)

else:
    _, [axs_wrapped, axs_correlation, axs_unwrapped] = plt.subplots(1, 3)

axs_wrapped.set_title("wrapped")
axs_wrapped.imshow(arr_w, origin="lower", cmap="jet")

axs_correlation.set_title("correlation")
axs_correlation.imshow(arr_c, origin="lower", cmap="jet")

axs_unwrapped.set_title("unwrapped")
axs_unwrapped.imshow(arr_uw, origin="lower", cmap="jet")

if crop_size:
    axs_tiled_unwrapped.set_title("tiled_unwrapped")
    axs_tiled_unwrapped.imshow(rebuilt_arr_uw, origin="lower", cmap="jet")

plt.show()
