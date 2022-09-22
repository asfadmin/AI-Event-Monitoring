"""
 Created By:  Andrew Player
 File Name:   inference.py
 Description: Functions related to inference with the model
"""

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from tensorflow.keras.models import Model, load_model

from src.io                      import get_product_arrays
from src.processing              import tile, tiles_to_image
from src.sarsim                  import gen_simulated_deformation
from src.synthetic_interferogram import make_random_dataset

from PIL                     import Image


def plot_imgs(x, y, y_conv, y_conv_r):

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


def test_masking(
    model_path: str,
    seed:       int,
    tile_size:  int,
    crop_size:  int  = 0,
    use_sim:    bool = False,
) -> None:

    """
    Predicts the event-mask on a synthetic wrapped interferogram and plots the results.

    Parameters:
    -----------
    model_path : str
        The path to the model that does the masking.
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

    print(f'\nGenerating Mask...\n_______\n')

    mask_model = load_model(model_path)

    if crop_size == 0:
        crop_size = tile_size

    if use_sim:
        y, x, presence = gen_simulated_deformation(seed=seed, tile_size=tile_size, log=False)
    else:
        y, x = make_random_dataset(size=tile_size, crop_size=crop_size, seed=seed)

    x  = x.reshape((1, tile_size, tile_size, 1))
    y_pred = np.abs(mask_model.predict(x).reshape((crop_size, crop_size)))
    x  = x.reshape((tile_size, tile_size))

    y_pred_rounded = np.zeros(y_pred.shape)

    tolerance  = 0.1
    round_up   = y_pred >= tolerance

    y_pred_rounded[round_up  ] = 1

    zeros    = x == 0
    y[zeros] = 0
    y_pred_rounded[zeros] = 0

    mae = np.mean(np.absolute(y_pred_rounded - y))
    average_val = np.mean(y_pred_rounded)

    guess  = "Positive"   if average_val >= 0.1  else "Negative"
    actual = "Positive"   if presence[0] == 1    else "Negative"
    result = "CORRECT!  " if guess == actual     else "INCORRECT!"

    print("")
    print(f'{result} Guess: {guess}   Actual: {actual}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Average Mask Value: {average_val}')
    print("")

    plot_imgs(x, y, y_pred, y_pred_rounded)


def test_binary_choice(
    model_path:      str,
    pres_model_path: str,
    seed:            int,
    tile_size:       int,
    crop_size:       int  = 0,
    count:           int  = 0,
    plot:            bool = False
) -> None:

    """
    Predicts the event-mask on a synthetic wrapped interferogram and plots the results.

    Parameters:
    -----------
    model_path : str
        The path to the model that masks images.
    pres_model_path : str
        The path to the model that predicts the presence of an event in a mask.
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
    plot : bool, Optional
        Plot the incorrect guesses during testing.

    Returns:
    --------
    None
    """

    print(f'Running tests over {count} Simulated Interferograms\n_______\n')

    mask_model = load_model(model_path)
    pres_model = load_model(pres_model_path)

    if crop_size == 0:
        crop_size = tile_size

    total_mae = 0
    total_pos = 0
    total_correct = 0
    total_pos_incorrect = 0
    total_neg_incorrect = 0

    for i in range(count):

        y, x, presence = gen_simulated_deformation(seed=seed, tile_size=tile_size, log=False)

        x  = x.reshape((1, tile_size, tile_size, 1))

        y_pred = np.abs(mask_model.predict(x)).reshape((crop_size, crop_size))

        x  = x.reshape ((tile_size, tile_size))

        y_pred_rounded = np.zeros((tile_size, tile_size))

        tolerance = 0.1
        round_up  = y_pred >= tolerance

        y_pred_rounded[round_up] = 1

        zeros          = x == 0
        y[zeros]       = 0
        y_pred[zeros]  = 0
        y_pred_rounded[zeros] = 0

        presence_pred = pres_model.predict(y_pred_rounded.reshape(1, tile_size, tile_size, 1))   

        curr_mae    = np.mean(np.absolute(y_pred_rounded - y))
        total_mae  += curr_mae
        average_val = np.mean(y_pred_rounded)

        guess  = "Positive"   if presence_pred[0] >= 0.5 else "Negative"
        actual = "Positive"   if presence[0] == 1        else "Negative"
        result = "CORRECT!  " if guess == actual         else "INCORRECT!"

        print(f'{result} Guess: {guess}   Actual: {actual}   Count: {i}')

        if count > 1:

            correctness    = guess == actual 
            total_correct += int(correctness)

            total_pos  += presence[0]

            if not correctness:
                if presence[0]:
                    total_pos_incorrect += 1
                else:
                    total_neg_incorrect += 1

                if plot:
                    plot_imgs(x, y, y_pred, y_pred_rounded)

                print("\nAverage Val: ", average_val, "\n")

    if count > 1:

        avg_mae = total_mae / count
        avg_cor = (total_correct / count) * 100.0

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
        print("Maximum Value        ", np.max(y_pred))
        print("Minimum Value        ", np.min(y_pred))
        print("Mean Value           ", average_val)
        print("Mean Absolute Error  ", curr_mae)
        print("_______\n")

        if plot:
            plot_imgs(x, y, y_pred, y_pred_rounded)


def mask(
    model_path:      str,
    pres_model_path: str,
    arr_w:           np.ndarray,
    tile_size:       int,
    crop_size:       int   = 0,
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

    pres_tiles = np.zeros((tiled_arr_w.shape[0], crop_size, crop_size))
    mask_tiles = np.zeros((tiled_arr_w.shape[0], crop_size, crop_size))

    mask_model = load_model(model_path)
    pres_model = load_model(pres_model_path)

    count = 0
    for x in tiled_arr_w:

        zeros = x == 0

        y_pred = np.abs(mask_model.predict(x.reshape((1, tile_size, tile_size, 1))).reshape((crop_size, crop_size)))

        y_pred[zeros] = 0

        y_pred_rounded = np.copy(y_pred)

        tolerance  = 0.1
        round_up   = y_pred_rounded >= tolerance
        round_down = y_pred_rounded <  tolerance

        y_pred_rounded[round_up ]  = 1
        y_pred_rounded[round_down] = 0

        presence = pres_model.predict(y_pred_rounded.reshape(1, tile_size, tile_size, 1))

        if presence[0] >= 0.99:
            pres_mask = np.ones((tile_size, tile_size))
        else:
            pres_mask = np.zeros((tile_size, tile_size))

        pres_tiles[count] = pres_mask
        mask_tiles[count] = y_pred_rounded
        count += 1

    mask = tiles_to_image(
        mask_tiles,
        w_rows,
        w_cols,
        arr_w.shape
    )

    pres_mask = tiles_to_image(
        pres_tiles,
        w_rows,
        w_cols,
        arr_w.shape
    )

    return mask, pres_mask


def mask_and_plot(
    model_path:   str,
    pres_model_path: str,
    product_path: str,
    tile_size:    int,
    crop_size:    int  = 0
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

    zeros = arr_uw == 0
    bad_coherence = coherence < 0.3
    
    arr_w[zeros] = 0
    arr_w[bad_coherence] = 0

    mask_pred, pres_mask = mask(
        model_path = model_path,
        pres_model_path = pres_model_path,
        arr_w      = arr_w,
        tile_size  = tile_size,
        crop_size  = crop_size
    )

    presence_guess = np.mean(pres_mask) > 0

    mask_pred[zeros] = 0
    mask_pred[bad_coherence] = 0

    if presence_guess:
        print("Positive")
    else:
        print("Negative") 

    plot_imgs(arr_w, arr_uw, mask_pred, pres_mask)

    return mask_pred


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