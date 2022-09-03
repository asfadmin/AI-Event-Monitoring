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
from src.processing              import tile, tiles_to_image, blur2d
from src.sarsim                  import gen_simulated_deformation
from src.synthetic_interferogram import make_random_dataset

from tensorflow.keras.models import Model, load_model
from PIL                     import Image


def test_model(
    model_path: str,
    seed:       int,
    tile_size:  int,
    crop_size:  int  = 0,
    count:      int  = 1,
    use_sim:    bool = False,
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
    count : int, Optional
        Predict on [count] simulated or synthetic interferograms and log the results. The
        default value of 1 simply plots the single prediction.
    use_sim : bool, Optional
        Use simulated interferograms rather than synthetic interferograms

    Returns:
    --------
    None
    """

    print(f'Running tests over {count} Simulated Interferograms\n_______\n')

    model = load_model(model_path)

    if crop_size == 0:
        crop_size = tile_size

    total_mae           = 0
    total_pos           = 0
    total_correct       = 0
    total_pos_incorrect = 0
    total_neg_incorrect = 0

    for i in range(count):

        if use_sim:
            y, x, presence = gen_simulated_deformation(seed=seed, tile_size=tile_size, log=False)
        else:
            y, x = make_random_dataset(size=tile_size, crop_size=crop_size, seed=seed)

        x  = x.reshape((1, tile_size, tile_size, 1))
        y  = y.reshape((1, crop_size, crop_size, 1))
        yp = model.predict(x)

        x  = x.reshape ((tile_size, tile_size))
        y  = y.reshape ((crop_size, crop_size))
        yp = yp.reshape((crop_size, crop_size))

        y_conv = blur2d(yp)
        for _ in range(64):
            y_conv = blur2d(y_conv)

        y_conv_r = np.zeros((tile_size, tile_size))

        tolerance2  = 0.4
        round_up2   = y_conv >= tolerance2
        round_down2 = y_conv <  tolerance2

        y_conv_r[round_up2  ] = 1
        y_conv_r[round_down2] = 0

        zeros           = x == 0
        y_conv[zeros]   = 0
        y_conv_r[zeros] = 0

        curr_mae    = np.mean(np.absolute(y_conv_r - y))
        total_mae  += curr_mae
        average_val = np.mean(y_conv_r)

        guess  = "Positive"   if average_val >= 2.25e-2 else "Negative"
        actual = "Positive"   if presence[0] == 1       else "Negative"
        result = "CORRECT!  " if guess == actual        else "INCORRECT!"

        print(f'{result} Guess: {guess}   Actual: {actual}   Count: {i}')

        if count > 1:

            correctness    = guess == actual 
            total_correct += int(correctness)
            
            total_pos  += presence[0]
            total_pos_incorrect += 1 if     presence[0] and not correctness else 0
            total_neg_incorrect += 1 if not presence[0] and not correctness else 0

    if count > 1:

        avg_mae = total_mae / count
        avg_cor = total_correct / count

        print("")
        print("Mean Absolute Error ", avg_mae)
        print("")
        print("Positives Missed: ", total_pos_incorrect, " of ", total_pos      , ".")
        print("Negatives Missed: ", total_neg_incorrect, " of ", count-total_pos, ".")
        print("")
        print("Overall Score ", avg_cor, "%")
        print("_______\n")

    else:

        print("_______\n")
        print("Maximum Value        ", np.max(y_conv))
        print("Minimum Value        ", np.min(y_conv))
        print("Mean Value           ", average_val)
        print("Mean Absolute Error  ", curr_mae)
        print("_______\n")

        y[zeros] = 0

        _, [[axs_wrapped, axs_mask], [axs_unwrapped, axs_mask_rounded]] = plt.subplots(2, 2, sharex=True, sharey=True, tight_layout=True)

        axs_wrapped.set_title("Wrapped")
        axs_wrapped.imshow(x, origin='lower', cmap='jet')

        axs_unwrapped.set_title("True Mask")
        axs_unwrapped.imshow(y, origin='lower', cmap='jet')

        axs_mask.set_title("Mask w/o Rounding")
        axs_mask.imshow(y_conv, origin='lower', cmap='jet', vmin=0.0, vmax=1.0)

        axs_mask_rounded.set_title("Mask w/ Rounding")
        axs_mask_rounded.imshow(y_conv_r, origin='lower', cmap='jet', vmin=0.0, vmax=1.0)

        plt.show()


def test_model_eval(
    model_path: str,
    seed:       int,
    tile_size:  int,
    crop_size:  int  = 0
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

    y, x, presence = gen_simulated_deformation(seed=seed, tile_size=tile_size, log=True)

    x  = x.reshape((1, tile_size, tile_size, 1))
    yp = model.predict(x)

    x = x.reshape((tile_size, tile_size))

    _, [axs_wrapped, axs_mask_true] = plt.subplots(1, 2)

    axs_wrapped.set_title("wrapped")
    axs_wrapped.imshow(x, origin='lower', cmap='jet')

    axs_mask_true.set_title("true mask")
    axs_mask_true.imshow(y, origin='lower', cmap='jet')

    print("Prediction  ", yp      )

    print("Presence    ", presence)

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

        yp = model.predict(x.reshape((1, tile_size, tile_size, 1))).reshape((crop_size, crop_size))

        y_conv = blur2d(yp)
        for _ in range(64):
            y_conv = blur2d(y_conv)

        tile_predictions[count] = y_conv
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
    crop_size:    int  = 0,
    mask_coh:     bool = True,
    round_pred:   bool = True
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

    if mask_coh:
        bad_coherence = coherence < 0.3
        arr_w[bad_coherence] = 0

    prediction = mask(
        model_path = model_path,
        arr_w      = arr_w,
        tile_size  = tile_size,
        crop_size  = crop_size
    )

    prediction[zeros] = 0

    if mask_coh:
        arr_uw[bad_coherence] = 0
        prediction[bad_coherence] = 0

    prediction_rounded = np.copy(prediction)
    if round_pred:

        tolerance1  = 0.4
        round_up1   = prediction_rounded >= tolerance1
        round_down1 = prediction_rounded <  tolerance1

        prediction_rounded[round_up1 ]  = 1
        prediction_rounded[round_down1] = 0

    average_val = np.mean(prediction_rounded)
    print("Average: ", average_val)

    if average_val >= 5e-3:
        print("Positive")
    else:
        print("Negative") 

    _, [[axs_wrapped, axs_unwrapped], [axs_mask, axs_mask_rounded]] = plt.subplots(2, 2, sharex=True, sharey=True, tight_layout=True)

    axs_wrapped.set_title("wrapped")
    axs_wrapped.imshow(arr_w, origin='lower', cmap='jet')

    axs_unwrapped.set_title("unwrapped_magnitude")
    axs_unwrapped.imshow(np.abs(arr_uw), origin='lower', cmap='jet')

    axs_mask.set_title("mask_unrounded")
    axs_mask.imshow(prediction, origin='lower', cmap='jet', vmin=0.0, vmax=1.0)

    axs_mask_rounded.set_title("mask_rounded")
    axs_mask_rounded.imshow(prediction_rounded, origin='lower', cmap='jet', vmin=0.0, vmax=1.0)

    plt.show()  

    return prediction


def visualize_layers(
    model_path: str,
    save_path:  str,
    tile_size:  int = 512,
    seed:       int = 0
) -> None:

    """
    Make a prediction on a simulated interferogram and save tifs of
    the outputs of each layer in the model.

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

    masked, wrapped, _ = gen_simulated_deformation(tile_size=tile_size, seed=seed, log=True)

    masked  = masked.reshape(output_shape)
    wrapped = wrapped.reshape(input_shape)

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

            current_save_path = Path(save_path) / f"{layer_name}.jpeg"
            out = Image.fromarray(image_belt)
            out.save(current_save_path, "JPEG")

            print(f"Saved figure for layer: {layer_name}, {index} of {num_feature_maps_filtered}")
            index += 1

    print(f"\nImage Belt Length: {len(image_belt)}\n")