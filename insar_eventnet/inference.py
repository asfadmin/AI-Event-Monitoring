"""
 Summary
 -------
 Functions for performing inference, such as masking events, and binary classification
 of events

 Notes
 -----
 Created by Andrew Player.
"""

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from tensorflow.keras.models import load_model

from insar_eventnet.io import get_image_array
from insar_eventnet.processing import tile, tiles_to_image


def mask(
    mask_model_path: str,
    pres_model_path: str,
    image_path: str,
    output_image_path: str = None,
    tile_size: int = 0,
    crop_size: int = 0,
) -> np.ndarray:
    """
    Generate a mask over potential events in a wrapped insar product.

    Parameters:
    -----------
    mask_model_path : str
        The path to the model to use for generating the event-mask.
    pres_model_path : str
        The path to the model that predicts the presence of an event in a mask.
    image_path : str
        The path to the InSAR product from ASF that should be masked.
    output_image_path : str
        The output path for the inferred mask image
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
    image, gdal_dataset = get_image_array(image_path)

    mask_pred, pres_mask, pres_vals = mask_with_model(
        mask_model=mask_model,
        pres_model=pres_model,
        arr_w=image,
        tile_size=tile_size,
        crop_size=crop_size,
    )

    presence_guess = np.mean(pres_mask) > 0.0

    if output_image_path is not None:
        img = Image.fromarray(mask_pred)
        img.save(output_image_path)

        from osgeo import gdal

        out_dataset = gdal.Open(output_image_path, gdal.GA_Update)
        out_dataset.SetGeoTransform(gdal_dataset.GetGeoTransform())
        out_dataset.SetProjection(gdal_dataset.GetProjection())

    return mask_pred, presence_guess


def mask_with_model(
    mask_model, pres_model, arr_w: np.ndarray, tile_size: int, crop_size: int = 0
) -> np.ndarray:
    """
    Use a keras model prediction to mask events in a wrapped interferogram.

    Parameters
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

    Returns
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

    Parameters
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
    """

    from os import listdir, path
    from insar_eventnet.io import get_image_array

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
            except Exception as e:
                print(f"Failed to load unwrapped phase image: {filename} due to {e}")
                continue
        elif "wrapped" in filename:
            try:
                arr_w, dataset = get_image_array(path.join(directory, filename))
            except Exception as e:
                print(f"Failed to load unwrapped phase image: {filename} due to {e}")
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

    Parameters
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
