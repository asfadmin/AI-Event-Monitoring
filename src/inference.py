"""
 Created By:  Andrew Player
 File Name:   inference.py
 Description: Functions related to inference with the model
"""

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from pathlib import Path

from tensorflow.keras.models import Model, load_model

from src.io import get_product_arrays
from src.processing import tile, tiles_to_image
from src.sarsim import gen_simulated_deformation, gen_sim_noise


def mask_and_plot(
    mask_model_path: str,
    pres_model_path: str,
    product_path: str,
    tile_size: int = 0,
    crop_size: int = 0,
    mask_model: Model = None,
    pres_model: Model = None,
) -> np.ndarray:
    """
    Generate a mask over potential events in a wrapped insar product and plot it.

    Parameters:
    -----------
    model_path : str
        The path to the model to use for generating the event-mask.
    pres_model_path : str
        The path to the model that predicts the presence of an event in a mask.
    product_path : str
        The path to the InSAR product from ASF that should be masked.
    tile_size : int
        The width and height of the tiles that the image will be broken into, this needs
        to match the input shape of the model.
    crop_size : int, Optional
        If the models output shape is different than the input shape, this value needs
        to be equal to the output shape of the masking model and input shape of the
        presence model.

    Returns:
    --------
    mask_pred : np.ndarray(shape=(tile_size, tile_size) or (crop_size, crop_size))
        The array containing the event-mask array as predicted by the model.
    presence_guess : bool
        True if there is an event else False.
    """

    mask_model = load_model(mask_model_path)
    pres_model = load_model(pres_model_path)

    arr_w, arr_uw, corr, w_dataset = get_product_arrays(product_path)

    # zeros         = arr_uw == 0
    # bad_coherence = corr < 0.3

    # arr_w[zeros]         = 0
    # arr_w[bad_coherence] = 0

    mask_pred, pres_mask, pres_vals = mask_with_model(
        mask_model=mask_model,
        pres_model=pres_model,
        arr_w=arr_w,
        tile_size=tile_size,
        crop_size=crop_size,
    )

    presence_guess = np.mean(pres_mask) > 0.0

    # arr_uw[zeros]            = 0
    # arr_uw[bad_coherence]    = 0
    # mask_pred[zeros]         = 0
    # mask_pred[bad_coherence] = 0

    if presence_guess:
        print("Positive")
    else:
        print("Negative")

    plot_imgs(arr_w, arr_uw, mask_pred, pres_mask)

    filename = "output.tif"
    img = Image.fromarray(mask_pred)
    img.save(filename)

    from osgeo import gdal

    out_dataset = gdal.Open(filename, gdal.GA_Update)
    out_dataset.SetGeoTransform(w_dataset.GetGeoTransform())
    out_dataset.SetProjection(w_dataset.GetProjection())

    return mask_pred, presence_guess


def mask_with_model(
    mask_model, pres_model, arr_w: np.ndarray, tile_size: int, crop_size: int = 0
) -> np.ndarray:
    """
    Use a keras model prediction to mask events in a wrapped interferogram.

    Parameters:
    -----------
    model_path : str
        The path to the model to use for masking.
    pres_model_path : str
        The path to the model that predicts the presence of an event in a mask.
    arr_w : np.ndarray
        The wrapped interferogram array.
    tile_size : int
        The width and height of the tiles that the image will be broken into, this needs
        to match the input shape of the model.
    crop_size : int, Optional
        If the models output shape is different than the input shape, this value needs
        to be equal to the output shape of the masking model and input shape of the
        presence model.

    Returns:
    --------
    mask : np.ndarray(shape=(tile_size, tile_size) or (crop_size, crop_size))
        The array containing the event-mask array as predicted by the model.
    pres_mask : np.ndarray(shape=(tile_size, tile_size) or (crop_size, crop_size))
        An array containing tiles where the tile is all 1s if there is an event else 0s.
        If even a single tile has 1s that means an event has been identified.
    """

    tiled_arr_w, w_rows, w_cols = tile(
        arr_w,
        (tile_size, tile_size),
        x_offset=0,
        y_offset=0,
        even_pad=True,
        crop_size=crop_size,
    )

    zeros = tiled_arr_w == 0

    if crop_size == 0:
        crop_size = tile_size

    # tiled_arr_w += np.pi
    # tiled_arr_w /= (2*np.pi)
    # tiled_arr_w[zeros] = 0

    mask_tiles = mask_model.predict(tiled_arr_w, batch_size=1)

    mask_tiles[zeros] = 0

    rnd = mask_tiles >= 0.5
    trnc = mask_tiles < 0.5
    # # rnd2 = mask_tiles >= 0.5
    mask_tiles[trnc] = 0
    # # mask_tiles[rnd2]  = 0.5
    mask_tiles[rnd] = 1

    pres_vals = pres_model.predict(mask_tiles, batch_size=1)
    pres_tiles = np.zeros((w_rows * w_cols, tile_size, tile_size))

    index = 0
    for val in pres_vals:
        if val >= 0.75:
            pres_tiles[index] = 1
        index += 1

    mask_tiles = mask_tiles.reshape((w_rows * w_cols, tile_size, tile_size))

    mask = tiles_to_image(mask_tiles, w_rows, w_cols, arr_w.shape)

    mask[arr_w == 0] = 0

    pres_mask = tiles_to_image(pres_tiles, w_rows, w_cols, arr_w.shape)

    return mask, pres_mask, pres_vals


def plot_results(wrapped, mask, presence_mask):
    _, [axs_wrapped, axs_mask, axs_presence_mask] = plt.subplots(
        1, 3, sharex=True, sharey=True
    )

    axs_wrapped.set_title("Wrapped")
    axs_mask.set_title("Segmentation Mask")
    axs_presence_mask.set_title("Presence Mask")

    axs_wrapped.imshow(wrapped, origin="lower", cmap="jet")
    axs_mask.imshow(mask, origin="lower", cmap="jet")
    axs_presence_mask.imshow(presence_mask, origin="lower", cmap="jet")

    plt.show()


def test_images_in_dir(
    mask_model,
    pres_model,
    directory,
    tile_size,
    crop_size,
    save_images=False,
    output_dir=None,
):
    """
    Helper for test_model(). Evaluates EventNet Models over a directory of real
    interferograms.

    Parameters:
    -----------
    mask_model : Keras Model
        The model for masking
    pres_model : Keras Model
        The model for binary classification.
    directory : str
        A directory containing interferogram tifs.
    tile_size : int
        The width and height of the tiles that the image will be broken into, this needs
        to match the input shape of the model.
    crop_size : int, Optional
        If the models output shape is different than the input shape, this value needs
        to be equal to the output shape of the masking model and input shape of the
        presence model.

    Returns:
    --------
    None
    """

    from os import listdir, path
    from src.io import get_image_array

    positives = 0
    negatives = 0

    arr_uw = 0

    print("---------------------------------------")
    print("tag  | label    | guess    | confidence")
    print("---------------------------------------")

    for filename in listdir(directory):
        if "unw_phase" in filename:
            try:
                arr_uw, dataset = get_image_array(path.join(directory, filename))
                arr_w = np.angle(np.exp(1j * (arr_uw)))
            except:
                print(f"Failed to load unwrapped phase image: {filename}")
                continue
        elif "wrapped" in filename:
            try:
                arr_w, dataset = get_image_array(path.join(directory, filename))
            except:
                print(f"Failed to load wrapped phase image: {filename}")
                continue

        mask, pres_mask, pres_vals = mask_with_model(
            mask_model=mask_model,
            pres_model=pres_model,
            arr_w=arr_w,
            tile_size=tile_size,
            crop_size=crop_size,
        )

        presence_guess = np.any(np.max(pres_vals) > 0.75)

        tag = filename.split("_")[-3]
        label = "Positive" if "Positives" in directory else "Negative"
        guess = "Positive" if presence_guess else "Negative"

        print(f"{tag} | {label} | {guess} |{np.max(pres_vals): 0.8f}")

        plot_results(arr_w, mask, pres_mask)

        if presence_guess:
            positives += 1
        else:
            negatives += 1

        if save_images:
            filename = f"{output_dir}/{tag}_mask.tif"
            img = Image.fromarray(mask)
            img.save(filename)

            from osgeo import gdal

            out_dataset = gdal.Open(filename, gdal.GA_Update)
            out_dataset.SetGeoTransform(dataset.GetGeoTransform())
            out_dataset.SetProjection(dataset.GetProjection())
            out_dataset.FlushCache()

    return positives, negatives


def test_model(
    mask_model_path,
    pres_model_path,
    images_dir,
    tile_size,
    crop_size,
    save_images=False,
    output_dir=None,
):
    """
    Evaluate EventNet Models over a directory of real interferograms.

    Parameters:
    -----------
    model_path : str
        The path to the model to use for masking.
    pres_model_path : str
        The path to the model that predicts the presence of an event in a mask.
    images_dir : str
        A directory containing Positives and Negatives directories which have their
        respective tifs.
    tile_size : int
        The width and height of the tiles that the image will be broken into, this needs
        to match the input shape of the model.
    crop_size : int, Optional
        If the models output shape is different than the input shape, this value needs
        to be qual to the output shape of the masking model and input shape of the
        presence model.

    Returns:
    --------
    None
    """

    from os import path

    try:
        mask_model = load_model(mask_model_path)
        pres_model = load_model(pres_model_path)
    except Exception as e:
        print(f"Caught {type(e)}: {e}")
        return

    positive_dir = path.join(images_dir, "Positives")
    negative_dir = path.join(images_dir, "Negatives")

    true_positives, false_negatives = test_images_in_dir(
        mask_model,
        pres_model,
        positive_dir,
        tile_size,
        crop_size,
        save_images,
        output_dir,
    )
    false_positives, true_negatives = test_images_in_dir(
        mask_model,
        pres_model,
        negative_dir,
        tile_size,
        crop_size,
        save_images,
        output_dir,
    )

    total = true_positives + false_positives + true_negatives + false_negatives

    accuracy = 100 * (true_positives + true_negatives) / total

    print(f"Num True  Positives: {true_positives}")
    print(f"Num False Positives: {false_positives}")
    print(f"Num True  Negatives: {true_negatives}")
    print(f"Num False Negatives: {false_negatives}")
    print(f"Total Predictions:   {total}")
    print(f"Accuracy:            {accuracy}%")


def mask_simulated(
    mask_model,
    seed: int,
    tile_size: int,
    crop_size: int = 0,
    verbose: bool = False,
    noise_only: bool = False,
    gaussian_only: bool = False,
    zero_output: bool = False,
    event_type: str = "quake",
    tolerance: float = 0.7,
) -> None:
    """
    Predicts the event-mask on a synthetic wrapped interferogram and plots the results.

    Parameters:
    -----------
    model_path : str
        The path to the model that does the masking.
    seed : int
        A seed for the random functions. For the same seed, with all other values the
        same as well, the interferogram generation will have the same results. If left
        at 0, the results will be different every time.
    tile_size : int
        The dimensional size of the simulated interferograms to generate, this must
        match the input shape of the model.
    crop_size : int, Optional
        If the models output shape is different than the input shape, this value needs
        to be equal to the output shape.
    use_sim : bool, Optional
        Use simulated interferograms rather than synthetic interferograms

    Returns:
    --------
    None
    """

    if crop_size == 0:
        crop_size = tile_size

    if not noise_only:
        unwrapped, mask, wrapped, presence = gen_simulated_deformation(
            seed=seed, tile_size=tile_size, event_type=event_type, log=verbose
        )
    else:
        unwrapped, mask, wrapped, presence = gen_sim_noise(
            seed=seed,
            tile_size=tile_size,
            gaussian_only=gaussian_only,
        )

    wrapped = wrapped.reshape((1, tile_size, tile_size, 1))
    mask_pred = np.float32(mask_model.predict(wrapped).reshape((crop_size, crop_size)))
    wrapped = wrapped.reshape((tile_size, tile_size))

    rnd_indices = mask_pred >= tolerance

    mask_pred_rounded = np.zeros(mask_pred.shape)
    mask_pred_rounded[rnd_indices] = 1

    if zero_output:
        zeros = wrapped == 0
        mask[zeros] = 0
        mask_pred[zeros] = 0
        mask_pred_rounded[zeros] = 0

    return wrapped, mask, mask_pred, mask_pred_rounded, presence


def test_binary_choice(
    mask_model,
    pres_model,
    seed: int,
    tile_size: int,
    crop_size: int = 0,
    count: int = 0,
    plot: bool = False,
    use_rounded_mask: bool = False,
    positive_thresh: float = 0.5,
) -> None:
    """
    Evaluats a mask+binary model on the presence of events in simulated interferograms.

    Parameters:
    -----------
    model_path : str
        The path to the model that masks images.
    pres_model_path : str
        The path to the model that predicts the presence of an event in a mask.
    seed : int
        A seed for the random functions. For the same seed, with all other values the
        same as well, the interferogram generation will have the same results. If left
        at 0, the results will be different every time.
    tile_size : int
        The dimensional size of the simulated interferograms to generate, this must
        match the input shape of the model.
    crop_size : int, Optional
        If the models output shape is different than the input shape, this value needs
        to be equal to the output shape.
    count : int, Optional
        Predict on [count] simulated or synthetic interferograms and log the results.
        The default value of 1 simply plots the single prediction.
    plot : bool, Optional
        Plot the incorrect guesses during testing.
    use_rounded_mask : bool, Optional
        Run binary model on the rounded mask or the raw mask.
    positive_thresh : bool, Optional
        Threshold for the binary model to consider an image positive.

    Returns:
    --------
    None
    """

    print(f"\nRunning tests over {count} Simulated Interferograms\n_______\n")

    if crop_size == 0:
        crop_size = tile_size

    if count < 1:
        count = 1

    if seed != 0:
        np.random.seed(seed)

    seeds = np.random.randint(100000, 999999, count)

    total_pos = 0
    total_neg = 0

    total_pos_correct = 0
    total_neg_correct = 0
    total_pos_incorrect = 0
    total_neg_incorrect = 0

    quake_count = np.ceil(0.4 * count)
    dyke_count = quake_count + np.ceil(0.1 * count)
    sill_count = dyke_count + np.ceil(0.1 * count)
    mix_noise_count = sill_count + np.floor(0.3 * count)

    for i in range(count):
        event_type = ""
        noise_only = False
        gaussian_only = False

        if i < quake_count:
            event_type = "quake"
        elif i < dyke_count:
            event_type = "dyke"
        elif i < sill_count:
            event_type = "sill"
        else:
            noise_only = True
            gaussian_only = i >= mix_noise_count
            event_type = "gaussian_noise" if gaussian_only else "mixed_noise"

        wrapped, mask, mask_pred, mask_pred_rounded, presence = mask_simulated(
            mask_model,
            seed=seed,
            tile_size=tile_size,
            crop_size=crop_size,
            noise_only=noise_only,
            gaussian_only=gaussian_only,
            zero_output=False,
            event_type="quake",
            tolerance=0.7,
        )

        if use_rounded_mask:
            presence_pred = pres_model.predict(
                mask_pred_rounded.reshape(1, tile_size, tile_size, 1)
            )
        else:
            presence_pred = pres_model.predict(
                mask_pred.reshape(1, tile_size, tile_size, 1)
            )

        is_pos = presence[0] == 1

        if is_pos:
            total_pos += 1
        else:
            total_neg += 1

        is_pos_pred = presence_pred[0] >= positive_thresh

        actual = "Positive" if is_pos else "Negative"
        guess = "Positive" if is_pos_pred else "Negative"
        result = "CORRECT!  " if guess == actual else "INCORRECT!"

        print(f"{result} Guess: {guess}   Actual: {actual}   Count: {i}")

        b0 = (
            np.zeros((tile_size, tile_size))
            if guess == "Negative"
            else np.ones((tile_size, tile_size))
        )

        if count > 1:
            correctness = guess == actual

            total_pos += int(is_pos)
            total_neg += int(not is_pos)

            if not correctness:
                if is_pos:
                    total_pos_incorrect += 1
                else:
                    total_neg_incorrect += 1
                if plot:
                    plot_imgs(wrapped, mask, mask_pred, mask_pred_rounded)

                print(f"\nSeed:       {seeds[i]}")
                print(f"Event Type: {event_type}")
                print(f"Presence:   {presence_pred[0]}")
            else:
                if is_pos:
                    total_pos_correct += 1
                else:
                    total_neg_correct += 1

    if count > 1:
        acc = 100 * (total_pos_correct + total_neg_correct) / count

        print("")
        print(f"True  Positives:  {total_pos_correct}")
        print(f"False Positives:  {total_pos_incorrect}")
        print(f"True  Negatives:  {total_neg_correct}")
        print(f"False Negatives:  {total_neg_incorrect}")
        print("")
        print(f"Overall Accuracy: {acc: 0.2f}%")
        print("_______\n")

    else:
        if plot:
            plot_imgs(wrapped, mask, mask_pred, mask_pred_rounded)


def plot_imgs(wrapped, true_mask, pred_mask, pred_mask_rounded):
    """
    Helper for plotting the results of a mask prediction along with its corresponding
    truths.
    """

    rnd_indices = pred_mask < 0.4
    pred_mask_rounded = np.copy(wrapped)
    pred_mask_rounded[rnd_indices] = pred_mask_rounded[rnd_indices] - 10

    _, [[axs_wrapped, axs_mask], [axs_unwrapped, axs_mask_rounded]] = plt.subplots(
        2, 2, sharex=True, sharey=True, tight_layout=True
    )

    axs_wrapped.set_title("Wrapped")
    axs_wrapped.imshow(wrapped, origin="lower", cmap="jet")

    axs_unwrapped.set_title("True Mask")
    axs_unwrapped.imshow(true_mask, origin="lower", cmap="jet")

    axs_mask.set_title("Mask w/o Rounding")
    axs_mask.imshow(pred_mask_rounded, origin="lower", cmap="jet")

    axs_mask_rounded.set_title("Mask w/ Rounding")
    axs_mask_rounded.imshow(pred_mask, origin="lower", cmap="jet")

    plt.show()


def visualize_layers(
    model_path: str, save_path: str, tile_size: int = 512, seed: int = 0
) -> None:
    """
    Make a prediction on a simulated interferogram and save tifs of the outputs of each
    layer in the model.

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

    _, masked, wrapped, _ = gen_simulated_deformation(
        tile_size=tile_size, seed=seed, log=True
    )

    masked = masked.reshape(output_shape)
    wrapped = wrapped.reshape(input_shape)

    layer_names = [layer.name for layer in model.layers]
    layer_outputs = [layer.output for layer in model.layers]

    feature_map_model = Model(inputs=model.input, outputs=layer_outputs)

    feature_maps = feature_map_model.predict(wrapped)

    print(f"\nFeature Map Count: {len(feature_maps)}\n")

    num_feature_maps_filtered = len(
        [feature_map for feature_map in feature_maps if len(feature_map.shape) == 4]
    )

    index = 0
    for layer_name, feature_map in zip(layer_names, feature_maps):
        if len(feature_map.shape) == 4:
            k = feature_map.shape[-1]
            size = feature_map.shape[1]

            image_belt = np.zeros(
                (feature_map.shape[1], feature_map.shape[2] * feature_map.shape[3])
            )
            for i in range(k):
                feature_image = feature_map[0, :, :, i]
                image_belt[:, i * size : (i + 1) * size] = feature_image

            current_save_path = Path(save_path) / f"{layer_name}.jpeg"
            out = Image.fromarray(image_belt)
            out.save(current_save_path, "JPEG")

            print(
                f"Saved figure for layer: {layer_name}, {index} of {num_feature_maps_filtered}"
            )
            index += 1

    print(f"\nImage Belt Length: {len(image_belt)}\n")
