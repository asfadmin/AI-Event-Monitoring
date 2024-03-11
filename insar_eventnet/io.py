"""
 Summary
 -------
 Functions to handle file read/write.

 Notes
 -----
 Created by Andrew Player.
"""

import contextlib
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Tuple
from urllib import request
from zipfile import ZipFile

import numpy as np
from osgeo import gdal
from tensorflow.keras import models

from insar_eventnet import processing, sarsim
from insar_eventnet.config import (
    AOI_DIR,
    MASK_DIR,
    MODEL_DIR,
    PRODUCTS_DIR,
    REAL_DIR,
    SYNTHETIC_DIR,
    TENSORBOARD_DIR,
)


def _save_dataset(
    save_path: Path, mask: np.ndarray, wrapped: np.ndarray, presence: int
) -> None:
    """
    Saves event-mask and wrapped ndarrays to a single .npz file.

    Parameters
    ----------
    save_path : Path
        The path to save to.
    mask : np.ndarray
        The mask for the event.
    wrapped : np.ndarray
        The wrapped interferogram.
    presence : int
        The presence of an event in an interferogram. 1 if present, 0 if not.
    """

    np.savez(save_path, mask=mask, wrapped=wrapped, presence=presence)


def _save_time_series_dataset(
    save_path: Path, phases: list, mask: np.ndarray, presence: int
) -> None:
    """
    Saves event-mask and wrapped ndarrays to a single .npz file.
    """

    np.savez(save_path, phases=phases, mask=mask, presence=presence)


def _load_ts_dataset(load_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads event-mask and wrapped ndarrays from .npz file.

    Parameters
    -----------
    load_path : Path
        The path to the data example that should be loaded.

    Returns
    --------
    mask : np.ndarray
        The array of the event-mask loaded from the .npz.
    wrapped : np.ndarray
        The array of the wrapped interferogram loaded from the .npz.
    presence : [0] or [1]
        [1] if the image contains an event else [0]
    """

    dataset_file = np.load(load_path)
    return dataset_file["phases"], dataset_file["mask"], dataset_file["presence"]


def _load_dataset(load_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads event-mask and wrapped ndarrays from .npz file.

    Parameters
    -----------
    load_path : Path
        The path to the data example that should be loaded.

    Returns
    --------
    mask : np.ndarray
        The array of the event-mask loaded from the .npz.
    wrapped : np.ndarray
        The array of the wrapped interferogram loaded from the .npz.
    presence : [0] or [1]
        [1] if the image contains an event else [0]
    """

    dataset_file = np.load(load_path)
    return dataset_file["mask"], dataset_file["wrapped"], dataset_file["presence"]


def initialize() -> None:
    create_directories()
    if not (
        os.path.isdir("data/output/models/mask_model")
        and os.path.isdir("data/output/models/pres_model")
    ):
        print("Downloading model... this might take a bit.")
        _download_models("data/output")


def create_directories() -> None:
    """
    Creates the directories for storing our data.
    """

    directories = [
        PRODUCTS_DIR,
        AOI_DIR,
        SYNTHETIC_DIR,
        REAL_DIR,
        MODEL_DIR,
        MASK_DIR,
        TENSORBOARD_DIR,
    ]
    for directory in directories:
        try:
            directory.mkdir(parents=True)
        except OSError:
            print(directory.__str__() + " already exists.")


def _download_models(path: str) -> None:
    """
    Downloads pretrained UNet masking model and EvetNet presence prediction model

    Parameters
    ----------
    model_path: str
    """

    with request.urlopen(
        "https://eventnetmodels.s3.us-west-2.amazonaws.com/models.zip"
    ) as response, ZipFile(BytesIO(response.read())) as file:
        file.extractall(path)


def get_image_array(image_path: str) -> np.ndarray:
    """
    Load a interferogram .tif from storage into an array.

    Parameters
    -----------
    image_path : str
        The path to the interferogram .tif to be opened.

    Returns
    --------
    arr : np.ndarray
        The interferogram array.
    """

    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1)
    arr = band.ReadAsArray()

    return arr, dataset


def get_product_arrays(product_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load wrapped, unwrapped, and correlation .tifs from storage into arrays.

    Parameters
    -----------
    product_path : str
        The path to the InSAR product folder containing the images.

    Returns
    --------
    wrapped : np.ndarray
        The array of the wrapped interferogram loaded from the .tif.
    unwrapped : np.ndarray
        The array of the unwrapped interferogram loaded from the .tif.
    correlation : np.ndarray
        The correlation map array loaded from the .tif,
    """

    wrapped_path = ""
    correlation_path = ""
    unwrapped_path = ""

    for filename in os.listdir(product_path):
        if filename[-8:] == "corr.tif":
            correlation_path = product_path + "/" + filename
        elif filename[-17:] == "wrapped_phase.tif":
            wrapped_path = product_path + "/" + filename
        elif filename[-13:] == "unw_phase.tif":
            unwrapped_path = product_path + "/" + filename

    correlation, _ = get_image_array(correlation_path)
    unwrapped, dataset = get_image_array(unwrapped_path)

    if wrapped_path != "":
        wrapped, _ = get_image_array(wrapped_path)
    else:
        wrapped = np.angle(np.exp(1j * unwrapped))

    return wrapped, unwrapped, correlation, dataset


def _get_dataset_arrays(product_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load wrapped, unwrapped, and correlation .tifs from storage into arrays.

    Parameters
    -----------
    product_path : str
        The path to the InSAR product folder containing the images.

    Returns
    --------
    wrapped : np.ndarray
        The array of the wrapped interferogram loaded from the .tif.
    unwrapped : np.ndarray
        The array of the unwrapped interferogram loaded from the .tif.
    correlation : np.ndarray
        The correlation map array loaded from the .tif,
    """

    wrapped_path = ""
    masked_path = ""

    for filename in os.listdir(product_path):
        f_len = len(filename)
        if filename[f_len - 8 : f_len] == "sked.tif":
            masked_path = product_path + "/" + filename
        else:
            wrapped_path = product_path + "/" + filename

    masked = get_image_array(masked_path)
    wrapped = get_image_array(wrapped_path)

    unmasked_area = masked != 1
    masked[unmasked_area] = 0

    return wrapped, masked


def make_simulated_dataset(
    name: str,
    output_dir: str,
    amount: int,
    seed: int,
    tile_size: int,
    crop_size: int,
    model_path: str = "",
) -> Tuple[int, int, str]:
    """
    Generate a dataset containing pairs of wrapped interferograms from simulated
    deformation along with their event-masks

    Parameters
    -----------
    name : str
        The name of the dataset to be generate. The saved name will be formatted
        like <name>_amount<amount>_seed<seed>.
    output_dir : str
        The directory to save the generated dataset to.
    amount : int
        The amount of simulated interferogram pairs to be generated.
    seed : int
        A seed for the random functions. For the same seed, with all other values the
        same as well, the interferogram generation will have the same results. If left
        at 0, a seed will be generated and the results will be different every time.
    tile_size : int
        The size of the simulated interferograms, which should match the desired tile
        sizes of of the real interferograms. This also needs to match the input shape of
        the model.
    crop_size : int
        If the model's output shape does not match its input shape, this should be set
        to match the output shape. The unwrapped interferogram will be cropped to this.

    Returns
    --------
    seed : int
        The generated or inputed seed.
    count : int
        The number of samples that were generated.
    dir_name : str
        The generated name of the dataset directory.
    """

    if not seed:
        seed = np.random.randint(100000, 999999)

    np.random.seed(seed)

    seeds = np.random.randint(100000, 999999, size=amount)

    dir_name = f"{name}_amount{amount}_seed{seed}"

    if model_path != "":
        model = models.load_model(model_path)

    save_directory = Path(output_dir) / dir_name
    if not save_directory.is_dir():
        save_directory.mkdir()

    distribution = {
        "quake": 0,
        "dyke": 0,
        "sill": 0,
        "gaussian_noise": 0,
        "mixed_noise": 0,
    }

    quake_count = np.ceil(0.4 * amount)
    dyke_count = quake_count + np.ceil(0.1 * amount)
    sill_count = dyke_count + np.ceil(0.1 * amount)
    mix_noise_count = sill_count + np.floor(0.3 * amount)

    count = 0
    while count < amount:
        current_seed = seeds[count]

        event_type = ""
        gaussian_only = False

        if (count < quake_count) or (count < dyke_count):
            event_type = "quake"
        elif count < sill_count:
            event_type = "dyke"
        else:
            gaussian_only = count >= mix_noise_count
            event_type = "gaussian_noise" if gaussian_only else "mixed_noise"

        if count < sill_count:
            unwrapped, masked, wrapped, presence = sarsim.gen_simulated_deformation(
                seed=current_seed, tile_size=tile_size, event_type=event_type
            )
        else:
            unwrapped, masked, wrapped, presence = sarsim.gen_sim_noise(
                seed=current_seed, tile_size=tile_size, gaussian_only=gaussian_only
            )

        distribution[event_type] += 1

        if model_path != "":
            round_mask = True
            mask_zeros = True

            wrapped = wrapped.reshape((1, tile_size, tile_size, 1))
            masked_pred = model.predict(wrapped)

            wrapped = wrapped.reshape((tile_size, tile_size))
            masked_pred = np.abs(masked_pred.reshape((crop_size, crop_size)))

            if round_mask:
                tolerance = 0.7
                round_up = masked_pred >= tolerance
                round_down = masked_pred < tolerance

                masked_pred[round_up] = 1
                masked_pred[round_down] = 0

            if mask_zeros:
                zeros = wrapped == 0
                masked_pred[zeros] = 0

        if crop_size < tile_size:
            masked = processing.simulate_unet_cropping(masked, (crop_size, crop_size))

        if count % 10 == 0 and count != 0:
            print(f"Generated {count} of {amount} simulated interferogram pairs.")

        current_name = f"sim_seed{current_seed}_{count}_{event_type}"
        save_path = save_directory / current_name
        _save_time_series_dataset(
            save_path, mask=masked, wrapped=wrapped, presence=presence
        )

        count += 1

    dataset_info = (
        f"Name: {name}\n"
        + f"Size: {amount}\n"
        + f"Date: {datetime.utcnow()}\n"
        + f"Seed: {seed}\n"
        + f"Tile: {tile_size}\n"
        + f"Crop: {crop_size}\n"
        + f"\nDistribution:\n{distribution}\n"
        + f"\nSeed List:\n{seeds}\n"
    )

    print(f"Generated {count} of {amount} simulated interferogram pairs.")
    return seed, count, dir_name, distribution, dataset_info


def _make_simulated_time_series_dataset(
    name: str,
    output_dir: str,
    amount: int,
    seed: int,
    tile_size: int,
    crop_size: int,
) -> Tuple[int, int, str]:
    """
    Generate a dataset containing pairs of wrapped interferograms from simulated
    deformation along with their event-masks

    Parameters
    -----------
    name : str
        The name of the dataset to be generate. The saved name will be formatted
        like <name>_amount<amount>_seed<seed>.
    output_dir : str
        The directory to save the generated dataset to.
    amount : int
        The amount of simulated interferogram pairs to be generated.
    seed : int
        A seed for the random functions. For the same seed, with all other values the
        same as well, the interferogram generation will have the same results. If left
        at 0, a seed will be generated and the results will be different every time.
    tile_size : int
        The size of the simulated interferograms, which should match the desired tile
        sizes of of the real interferograms. This also needs to match the input shape of
        the model.
    crop_size : int
        If the model's output shape does not match its input shape, this should be set
        to match the output shape. The unwrapped interferogram will be cropped to this.

    Returns
    --------
    seed : int
        The generated or inputed seed.
    count : int
        The number of samples that were generated.
    dir_name : str
        The generated name of the dataset directory.
    """

    if not seed:
        seed = np.random.randint(100000, 999999)

    np.random.seed(seed)

    seeds = np.random.randint(100000, 999999, size=amount)

    dir_name = f"{name}_amount{amount}_seed{seed}"

    save_directory = Path(output_dir) / dir_name
    if not save_directory.is_dir():
        save_directory.mkdir()

    distribution = {"quake": 0, "mixed_noise": 0}

    quake_count = np.ceil(0.5 * amount)

    count = 0
    while count < amount:
        current_seed = seeds[count]

        if count < quake_count:
            phases, mask = sarsim.gen_simulated_time_series(
                seed=current_seed, tile_size=tile_size
            )

            distribution["quake"] += 1

            presence = 1
        else:
            phases, mask = sarsim.gen_simulated_time_series(
                seed=current_seed, tile_size=tile_size, noise_only=True
            )

            distribution["mixed_noise"] += 1

            presence = 0

        if count % 10 == 0 and count != 0:
            print(f"Generated {count} of {amount} simulated interferogram pairs.")

        current_name = f"sim_seed{current_seed}_{count}"
        save_path = save_directory / current_name
        _save_time_series_dataset(
            save_path, phases=phases[:, 0, :, :], mask=mask, presence=presence
        )

        count += 1

    dataset_info = (
        f"Name: {name}\n"
        + f"Size: {amount}\n"
        + f"Date: {datetime.utcnow()}\n"
        + f"Seed: {seed}\n"
        + f"Tile: {tile_size}\n"
        + f"Crop: {crop_size}\n"
        + f"\nDistribution:\n{distribution}\n"
        + f"\nSeed List:\n{seeds}\n"
    )

    print(f"Generated {count} of {amount} simulated interferogram pairs.")
    return seed, count, dir_name, distribution, dataset_info


def split_dataset(dataset_path: str, split: float) -> Tuple[int, int]:
    """
    Split the dataset into train and test folders

    Parameters
    -----------
    dataset_path : str
        The path to the dataset to be split
    split : float
        The train/test split, 0 < Split < 1, size(validation) <= split

    Returns
    --------
    num_train : int
        The number of elements that went to the training set.
    num_validation : int
        The number of elements that went to the validation set.
    """

    train_dir = Path(dataset_path) / "train"
    validation_dir = Path(dataset_path) / "validation"

    try:
        train_dir.mkdir()
        validation_dir.mkdir()
    except OSError:
        print("\nTrain or Validation Dir already exists -- skipping.\n")

    num_train = 0
    num_validation = 0
    for _, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            old_path = Path(dataset_path) / filename

            split_value = np.random.uniform(0, 1)
            if split_value <= split:
                num_validation += 1
                new_path = validation_dir / filename
            else:
                num_train += 1
                new_path = train_dir / filename

            with contextlib.suppress(OSError):
                os.rename(old_path, new_path)
        break

    return num_train, num_validation


def _dataset_from_products(
    dataset_name: str, product_path: str, save_path: str, tile_size: int, crop_size: int
) -> int:
    """
    Creates a dataset from a folder containing real interferogram products.

    Parameters
    -----------
    dataset_name : str
        The name for the folder that will contain the dataset.
    product_path : str
        The path to the folder containing sar product folders
    save_path : str
        The path to the folder where the dataset should be saved.
    tile_size : int
        The width and height of the tiles that the image will be broken into, this needs
        to match the input shape of the model.
    crop_size : int
        If the models output shape is different than the input shape, this value needs
        to be equal to the output shape.
    cutoff : int
        Set point in wrapped array to 0 if the correlation is below this value at that
        point.

    Returns
    --------
    dataset_size : int
        The size of the dataset that was created.
    """

    save_directory = Path(save_path) / dataset_name
    if not save_directory.is_dir():
        save_directory.mkdir()

    dataset_size = 0
    for _, products, _ in os.walk(product_path):
        product_count = len(products)

        for progress, product in enumerate(products):
            print(f"{progress}/{product_count} | {product_path + '/' + product}")

            wrapped, masked = _get_dataset_arrays(product_path + "/" + product)

            tiled_wrapped, w_rows, w_cols = processing.tile(
                wrapped, (tile_size, tile_size), even_pad=True, crop_size=crop_size
            )

            tiled_masked, _, _ = processing.tile(
                masked, (tile_size, tile_size), even_pad=True, crop_size=crop_size
            )

            for index in range(w_rows * w_cols):
                dataset_size += 1

                product_id = product[-4:]
                current_name = f"real_{product_id}_{index}"
                save_path = save_directory / current_name

                _save_time_series_dataset(
                    save_path, mask=tiled_masked[index], wrapped=tiled_wrapped[index]
                )

    return dataset_size
