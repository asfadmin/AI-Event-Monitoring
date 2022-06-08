"""
 Created By:  Andrew Player
 File Name:   inference.py
 Description: Functions related to inference with the model
"""

import matplotlib.pyplot as plt
import numpy as np

from src.io           import get_image_array, get_product_arrays
from src.processing   import tile, tiles_to_image, tsrncp_merger, merge_offset
from src.synthetic_interferogram import make_random_dataset

from tensorflow.keras.models import load_model


def test_model(
    model_path: str,
    seed:       int,
    tile_size:  int,
    crop_size:  int = 0
) -> None:

    """
    Predicts on a wrapped/unwrapped pair and plots the results.

    Parameters:
    -----------
    model_path : str
        The path to the model.
    seed : int
        A seed for the random functions. For the same seed, with all other values the same
        as well, the interferogram generation will have the same results. If left at 0,
        the results will be different every time.
    tile_size : int
        The dimensional size of the simulated interferograms to generate, this must match the
        input shape of the model.
    crop_size : int, Optional
        If the models output shape is different than the input shape, this value needs to be
        equal to the output shape.

    Returns:
    --------
    None
    """

    model = load_model(model_path)

    if crop_size == 0:
        crop_size = tile_size

    y, x = make_random_dataset(size=tile_size, crop_size=crop_size, seed=seed)

    x  = x.reshape((1, tile_size, tile_size, 1))
    y  = y.reshape((1, crop_size, crop_size, 1))
    yp = model.predict(x)

    x  = x.reshape ((tile_size, tile_size))
    y  = y.reshape ((crop_size, crop_size))
    yp = yp.reshape((crop_size, crop_size))

    round_up   = yp >= 0.75
    round_down = yp < 0.75

    yp[round_up]   = 1
    yp[round_down] = 0

    _, [axs_wrapped, axs_unwrapped_true, axs_unwrapped_pred] = plt.subplots(1, 3)

    axs_wrapped.set_title("wrapped")
    axs_wrapped.imshow(x, origin='lower', cmap='jet')

    axs_unwrapped_true.set_title("unwrapped_true")
    axs_unwrapped_true.imshow(y, origin='lower', cmap='jet')

    axs_unwrapped_pred.set_title("unwrapped_predicted")
    axs_unwrapped_pred.imshow(yp, origin='lower', cmap='jet')

    nrmse = np.sqrt(np.nanmean(np.array(np.power(yp - y, 2))))
    mae   = np.mean(np.absolute(yp - y))

    print("*************************")
    print("nRMSE: ", nrmse)
    print("MAE: "  , mae  )
    print("*************************")

    plt.show()