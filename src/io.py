"""
 Created By:   Jason Herning and Andrew Player
 File Name:    io.py
 Date Created: 01-25-2021
 Description:  Functions to handle file read/write.
"""

import random
import sys
from os import rename, walk
from pathlib import Path
from typing import Tuple

import numpy as np
from osgeo import gdal
from src.config import AOI_DIR, MASK_DIR, MODEL_DIR, PRODUCTS_DIR, REAL_DIR, SYNTHETIC_DIR, TENSORBOARD_DIR
from src.processing import correlation_mask, tile
from src.synthetic_interferogram import make_random_dataset, simulate_unet_cropping


def save_dataset(save_path: Path, unwrapped: np.ndarray, wrapped: np.ndarray) -> None:

    """
    Saves wrapped & unwrapped ndarrays to a single .npz file.
    """

    np.savez(save_path, unwrapped=unwrapped, wrapped=wrapped)


def load_dataset(load_path: Path) -> Tuple[np.ndarray, np.ndarray]:

    """
    Loads wrapped & unwrapped ndarrays from .npz file.

    Parameters:
    -----------
    load_path : Path
        The path to the data example that should be loaded.

    Returns:
    --------
    unwrapped : np.ndarray
        The array of the unwrapped interferogram loaded from the .npz.
    wrapped : np.ndarray
        The array of the wrapped interferogram loaded from the .npz.
    """

    dataset_file = np.load(load_path)
    return dataset_file['unwrapped'], dataset_file['wrapped']


def create_directories() -> None:

    """
    Creates the directories for storing our data.
    """

    directories = [PRODUCTS_DIR, AOI_DIR, SYNTHETIC_DIR, REAL_DIR, MODEL_DIR, MASK_DIR, TENSORBOARD_DIR]
    for directory in directories:
        try:
            directory.mkdir(parents=True)
        except OSError:
            print(directory.__str__() + " already exists.")


def get_image_array(image_path: str) -> np.ndarray:

    """
    Load a interferogram .tif from storage into an array.

    Parameters:
    -----------
    image_path : str
        The path to the interferogram .tif to be opened.

    Returns:
    --------
    arr : np.ndarray
        The interferogram array.
    """

    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1)
    arr = band.ReadAsArray()

    return arr


def get_product_arrays(product_path: str,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Load wrapped, unwrapped, and correlation .tifs from storage into arrays.

    Parameters:
    -----------
    product_path : str
        The path to the InSAR product folder containing the images.

    Returns:
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

    for dir_path, _, filenames in walk(product_path):
        for file in filenames:
            f_len = len(file)
            if file[f_len - 8:f_len] == 'corr.tif':
                correlation_path = dir_path + '/' + file
            elif file[f_len - 17:f_len] == 'wrapped_phase.tif':
                wrapped_path = dir_path + '/' + file
            elif file[f_len - 13:f_len] == 'unw_phase.tif':
                unwrapped_path = dir_path + '/' + file

    dataset = gdal.Open(wrapped_path, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1)
    wrapped = band.ReadAsArray()

    coherence_image = gdal.Open(correlation_path, gdal.GA_ReadOnly)
    band = coherence_image.GetRasterBand(1)
    correlation = band.ReadAsArray()

    dataset = gdal.Open(unwrapped_path, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1)
    unwrapped = band.ReadAsArray()

    return wrapped, unwrapped, correlation


def dataset_from_products(
                    dataset_name: str,
                    product_path: str,
                    save_path: str,
                    tile_size: int,
                    crop_size: int,
                    cutoff: int
                    ) -> int:

    """
    Creates a dataset from a folder containing real interferogram products.

    Parameters:
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
        If the models output shape is different than the input shape, this value needs to be
        equal to the output shape.
    cutoff : int
        Set point in wrapped array to 0 if the correlation is below this value at that point.

    Returns:
    --------
    dataset_size : int
        The size of the dataset that was created.
    """

    save_directory = Path(save_path) / dataset_name
    if not save_directory.is_dir():
        save_directory.mkdir()

    dataset_size = 0
    for _, products, _ in walk(product_path):

        progress = 0
        product_count = len(products)

        for product in products:

            progress += 1
            print(f"{progress}/{product_count} | {product_path + '/' + product}")

            wrapped, unwrapped, correlation = get_product_arrays(product_path + '/' + product)

            if cutoff > 0:
                wrapped = correlation_mask(wrapped, correlation, cutoff, 0.0)

            tiled_wrapped, w_rows, w_cols = tile(
                wrapped,
                (tile_size, tile_size),
                even_pad=True,
                crop_size=crop_size
                )

            tiled_unwrapped, _, _ = tile(
                unwrapped,
                (tile_size, tile_size),
                even_pad=True,
                crop_size=crop_size
            )

            for index in range(w_rows * w_cols):

                dataset_size += 1

                cropped_unwrapped = simulate_unet_cropping(tiled_unwrapped[index], crop_shape=(crop_size, crop_size))

                product_id = product[-4:]
                current_name = f"real_{product_id}_{index}"
                save_path = save_directory / current_name

                save_dataset(save_path, unwrapped=cropped_unwrapped, wrapped=tiled_wrapped[index])

    return dataset_size


def make_synthetic_dataset(
                    name:         str,
                    output_dir:   str,
                    amount:       int,
                    seed:         int,
                    tile_size:    int,
                    crop_size:    int,
                    min_amp:      float,
                    max_amp:      float,
                    min_x_mean:   float,
                    max_x_mean:   float,
                    min_y_mean:   float,
                    max_y_mean:   float,
                    min_x_stddev: float,
                    max_x_stddev: float,
                    min_y_stddev: float,
                    max_y_stddev: float,
                    ) -> Tuple[int, int, str]:

    """
    Generate a dataset of synthetic interferograms

    Parameters:
    -----------
    name : str
        The name of the dataset to be generate. The saved name will be formatted
        like <name>_amount<amount>_seed<seed>.
    output_dir : str
        The directory to save the generated dataset to.
    amount : int
        The amount of simulated interferogram pairs to be generated.
    seed : int
        A seed for the random functions. For the same seed, with all other values the same
        as well, the interferogram generation will have the same results. If left at 0,
        a seed will be generated and the results will be different every time.
    tile_size : int
        The size of the simulated interferograms, which should match the desired tile sizes of
        of the real interferograms. This also needs to match the input shape of the model.
    crop_size : int
        If the model's output shape does not match its input shape, this should be set to match
        the output shape. The unwrapped interferogram will be cropped to this.

    Returns:
    --------
    seed : int
        The generated or inputed seed.
    count : int
        The number of samples that were generated.
    dir_name : str
        The generated name of the dataset directory.
    """

    def new_seed():
        seed_value = random.randrange(sys.maxsize)
        random.seed(seed_value)
        return random.randint(100000, 999999)

    if not seed:
        seed = random.randint(100000, 999999)

    dir_name = f"{name}_amount{amount}_seed{seed}"

    save_directory = Path(output_dir) / dir_name
    if not save_directory.is_dir():
        save_directory.mkdir()

    count = 1
    while count != amount:

        current_seed = new_seed()

        interferogram, wrapped_interferogram = make_random_dataset(
            size         = tile_size,
            seed         = current_seed,
            crop_size    = crop_size,
            min_amp      = min_amp,
            max_amp      = max_amp,
            min_x_mean   = min_x_mean,
            max_x_mean   = max_x_mean,
            min_y_mean   = min_y_mean,
            max_y_mean   = max_y_mean,
            min_x_stddev = min_x_stddev,
            max_x_stddev = max_x_stddev,
            min_y_stddev = min_y_stddev,
            max_y_stddev = max_y_stddev
        )

        current_name = f"synth_seed{current_seed}_{count}"
        save_path = save_directory / current_name
        save_dataset(save_path, unwrapped=interferogram, wrapped=wrapped_interferogram)

        count += 1

    return seed, count, dir_name


def split_dataset(dataset_path: str, split: float) -> Tuple[int, int]:

    """
        Split the dataset into train and test folders

        Parameters:
        -----------
        dataset_path : str
            The path to the dataset to be split
        split : float
            The train/test split, 0 < Split < 1, size(validation) <= split

        Returns:
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
    for _, _, filenames in walk(dataset_path):
        for filename in filenames:

            old_path = Path(dataset_path) / filename

            split_value = random.uniform(0, 1)
            if split_value <= split:
                num_validation += 1
                new_path = validation_dir / filename
            else:
                num_train += 1
                new_path = train_dir / filename

            try:
                rename(old_path, new_path)
            except OSError:
                pass
        break

    return num_train, num_validation
