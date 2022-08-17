"""
 Created By:  Andrew Player
 File Name:   inference.py
 Description: Functions related to inference with the model
"""

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from numpy import average

from src.io                      import get_product_arrays
from src.processing              import tile, tiles_to_image
from src.synthetic_interferogram import make_random_dataset

from tensorflow.keras.models import Model, load_model
from PIL                     import Image


def test_model(
    model_path: str,
    seed:       int,
    tile_size:  int,
    crop_size:  int = 0
) -> None:

    """
    Predicts the event-mask on a synthetic wrapped interferogram and plots the results.

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

    # round_up   = yp >= 0.7
    # round_down = yp < 0.7

    # yp[round_up]   = 1
    # yp[round_down] = 0

    tolerance1  = 0.75
    # tolerance2  = 1.95
    round_up1   = yp >= tolerance1
    # round_up2   = yp >= tolerance2
    round_down1 = yp <  tolerance1

    yp[round_up1 ] = 1
    # yp[round_up2 ] = 2
    yp[round_down1] = 0

    _, [axs_wrapped, axs_mask_true, axs_mask_pred] = plt.subplots(1, 3)

    axs_wrapped.set_title("wrapped")
    axs_wrapped.imshow(x, origin='lower', cmap='jet')

    axs_mask_true.set_title("true mask")
    axs_mask_true.imshow(y, origin='lower', cmap='jet')

    axs_mask_pred.set_title("predicted mask")
    axs_mask_pred.imshow(yp, origin='lower', cmap='jet')

    mse = np.mean(np.power(yp - y, 2))
    mae = np.mean(np.absolute(yp - y))

    print("Mean Squared Error   ", mse)
    print("Mean Absolute Error  ", mae)

    plt.show()


def mask(
    model_path: str,
    arr_w:      np.ndarray,
    tile_size:  int,
    crop_size:  int   = 0,
) -> np.ndarray:

    """
    Use a keras model prediction to mask events in a wrapped interferogram.

    Parameters:
    -----------
    model_path : str
        The path to the model to use for masking.
    arr_w : np.ndarray
        The wrapped interferogram array.
    tile_size : int
        The width and height of the tiles that the image will be broken into, this needs
        to match the input shape of the model.
    crop_size : int, Optional
        If the models output shape is different than the input shape, this value needs to be
        equal to the output shape.

    Returns:
    --------
    prediction : np.ndarray(shape=(tile_size, tile_size) or (crop_size, crop_size))
        The array containing the event-mask array as predicted by the model.
    """


    tiled_arr_w, w_rows, w_cols = tile(
        arr_w,
        (tile_size, tile_size),
        x_offset  = 0,
        y_offset  = 0,
        even_pad  = True,
        crop_size = crop_size
    )

    if crop_size == 0:
        crop_size = tile_size

    tile_predictions = np.zeros((tiled_arr_w.shape[0], crop_size, crop_size))

    model = load_model(model_path)

    count = 0
    for x in tiled_arr_w:
        y = model.predict(x.reshape((1, tile_size, tile_size, 1)))
        tile_predictions[count] = y.reshape((crop_size, crop_size))
        count += 1

    prediction = tiles_to_image(
        tile_predictions,
        w_rows,
        w_cols,
        arr_w.shape
    )

    return prediction


def mask_and_plot(
    model_path:   str,
    product_path: str,
    tile_size:    int,
    crop_size:    int   = 0,
) -> np.ndarray:

    """
    Generate a mask over potential events in a wrapped insar product and plot it.

    Parameters:
    -----------
    model_path : str
        The path to the model to use for generating the event-mask.
    product_path : str
        The path to the InSAR product from ASF that should be masked.
    tile_size : int
        The width and height of the tiles that the image will be broken into, this needs
        to match the input shape of the model.
    crop_size : int, Optional
        If the models output shape is different than the input shape, this value needs to be
        equal to the output shape.

    Returns:
    --------
    prediction : np.ndarray(shape=(tile_size, tile_size) or (crop_size, crop_size))
        The array containing the event-mask array as predicted by the model.
    """

    arr_w, arr_uw, coherence = get_product_arrays(product_path)

    zeros        = (arr_uw == 0)
    arr_w[zeros] = 0

    bad_coherence = coherence < 0.2
    arr_w[bad_coherence] = 0

    prediction = mask(
        model_path = model_path,
        arr_w      = arr_w,
        tile_size  = tile_size,
        crop_size  = crop_size
    )

    tolerance1  = 0.85
    # tolerance2  = 1.99
    round_up1   = prediction >= tolerance1
    # round_up2   = prediction >= tolerance2
    round_down1 = prediction <  tolerance1

    prediction[round_up1 ] = 1
    # prediction[round_up2 ] = 2
    prediction[round_down1] = 0

    prediction[bad_coherence] = 0

    _, [axs_wrapped, axs_mask] = plt.subplots(1, 2)

    axs_wrapped.set_title("wrapped")
    axs_wrapped.imshow(arr_w, origin='lower', cmap='jet')

    # axs_unwrapped_true.set_title("mask_true")
    # axs_unwrapped_true.imshow(arr_uw, origin='lower', cmap='jet')

    axs_mask.set_title("mask_predicted")
    axs_mask.imshow(prediction, origin='lower', cmap='jet')

    plt.show()

    average_val = np.mean(prediction)
    print("Average: ", average_val)

    return prediction


def visualize_layers(
    model_path: str,
    save_path:  str,
    seed:       int = 0
) -> None:

    """
    Visualize the layers in the model.

    Parameters:
    -----------
    model_path : str
        The path to the model to be visualized.
    save_path : str
        The path to the folder where the resulting tifs should be saved.
    seed : int
        An integer value to seed the random function
        (the same seed results in the same image, all else equal).

    Returns:
    --------
    None
    """

    model = load_model(model_path)
    model.summary()

    input_shape, output_shape = list(model.input.shape), list(model.output.shape)
    input_shape[0], output_shape[0] = 1, 1
    input_shape, output_shape = tuple(input_shape), tuple(output_shape)

    print(f"\nModel Input Shape: {input_shape}")
    print(f"Model Output Shape: {output_shape}")

    unwrapped, wrapped = make_random_dataset(size=1024, crop_size=883, seed=seed)

    unwrapped = unwrapped.reshape(output_shape)
    wrapped   = wrapped.reshape(input_shape)

    layer_names   = [layer.name   for layer in model.layers]
    layer_outputs = [layer.output for layer in model.layers]

    feature_map_model = Model(inputs=model.input, outputs=layer_outputs)
    
    feature_maps = feature_map_model.predict(wrapped)

    print(f"\nFeature Map Count: {len(feature_maps)}\n")

    num_feature_maps_filtered = len([feature_map for feature_map in feature_maps if len(feature_map.shape) == 4])

    index = 0
    for layer_name, feature_map in zip(layer_names, feature_maps):

        if len(feature_map.shape) == 4:

            k    = feature_map.shape[-1]
            size = feature_map.shape[ 1]

            image_belt = np.zeros((feature_map.shape[1], feature_map.shape[2] * feature_map.shape[3]))
            for i in range(k):
                feature_image = feature_map[0, :, :, i]
                image_belt[:, i * size : (i + 1) * size] = feature_image

            current_save_path = Path(save_path) / f"{layer_name}.tif"
            out = Image.fromarray(image_belt)
            out.save(current_save_path)

            print(f"Saved figure for layer: {layer_name}, {index} of {num_feature_maps_filtered}")
            index += 1

    print(f"\nImage Belt Length: {len(image_belt)}\n")