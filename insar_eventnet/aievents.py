"""
 Created By:  Andrew Player
 File Name:   aievents.py
 Description: CLI Interface
"""
import json
import os
import time
from json import dumps
from os import listdir, mkdir, path, system
from pathlib import Path
from shutil import copyfile

import click
import matplotlib.pyplot as plt
import numpy as np
import PIL
import requests
from matplotlib import widgets
from osgeo import gdal
from PIL import Image
from tensorflow.keras import models

from insar_eventnet import gui, inference, io, processing, sarsim, training
from insar_eventnet.config import MASK_DIR, SYNTHETIC_DIR

# ------------- #
# Help Strings  #
# ------------- #

outputdir_help = "Directory to save to."
seed_help = "Seed integer for reproducible data."
cropsize_help = (
    "Crop unwrapped image to: (crop_size, crop_size). This has to match the models"
    "output shape."
)
split_help = (
    "Float between 0 and 1 representing the percentage of items to go to the training"
    "set."
)
tilesize_help = (
    "The width/height of the tiles. This should match the input shape of the model."
)
inputshape_help = "The input shape of the model. (input_shape, input_shape)"
epochs_help = (
    "The number of times that the network should train over the given training dataset."
)
batchsize_help = (
    "The number of data samples to go through before the model's weights are updated."
    " Lower = Slower = More Accurate."
)
filters_help = "The base number of filters for the convolutions in the model."
dropout_help = (
    "The amount of network nodes to randomly dropout throughout training to decrease"
    " over fitting. 0 <= Dropout <= 1"
)
learningrate_help = (
    "The rate which affects how much the model changes in response to estimated error."
)
learningrate_help += "0.0005 <= learning_rate <= 0.005"
amplitude_help = "Max amplitude of the gaussians (minimum is the negative of this)."
usesim_help = (
    "Flag to use simulated interferograms rather than synthetic"
    " interferograms. Default is False."
)
numtrials_help = "Test the model over this many images. If 1, the images are plotted."
silent_help = "Don't print to stdout"
no_plot_help = "Don't plot images"
copy_image_help = "Copy original positive interferogram tifs alongside output masks"

# ------------------ #
# CLI Implementation #
# ------------------ #


@click.group()
def cli():
    pass


@cli.command("setup")
def setup_wrapper():
    """
    Create data directory subtree and download models. This should be run before
    make-dataset.

    \b
    data/
      └──input/
        └──products/
        └──aoi/
      └──working/
        └──real/
        └──synthetic/
      └──output/
        └──models/
        └──mask/
        └──tensorboard/
    """

    print("")
    io.create_directories()

    print("")
    click.echo("Data directory created")
    print("")

    print("Downloading models... this may take a second")
    io.download_models("data/output")


@cli.command("download-models")
def download_models_wrapper():
    """
    Download models to data/output/models
    """

    print("Downloading... this may take a second")
    io.download_models("data/output")


@cli.command("make-simulated-dataset")
@click.argument("name", type=str)
@click.argument("amount", type=int, default=1)
@click.option("-t", "--tile_size", type=int, default=512, help=tilesize_help)
@click.option("-c", "--crop_size", type=int, default=512, help=cropsize_help)
@click.option(
    "-d",
    "--output_dir",
    type=click.Path(file_okay=False),
    default=SYNTHETIC_DIR,
    help=outputdir_help,
)
@click.option("-s", "--seed", type=int, default=None, help=seed_help)
@click.option("-s", "--split", type=float, default=0.0, help=split_help)
def make_simulated_dataset_wrapper(
    name, amount, tile_size, crop_size, output_dir, seed, split
):
    """
    Create a randomly generated simulated dataset of wrapped interferograms and their
    corresponding event-masks.

    ARGS:\n
    name        Name of dataset. Seed is appended.\n
                <name>_seed<seed>\n
    amount      Number of simulated interferograms created.\n
    """

    print("")

    name, count, dir_name, distribution, dataset_info = io.make_simulated_dataset(
        name, output_dir, amount, seed, tile_size, crop_size
    )

    num_train, num_validation = io.split_dataset(
        output_dir.__str__() + "/" + dir_name, split
    )

    with open(
        output_dir.__str__() + "/" + dir_name + "/parameters.txt", "w"
    ) as log_file:
        log_file.write(dataset_info)

    print("")
    print(f"Data Type Distribution: {distribution}")
    print("")
    print(
        f"Created simulated dataset with seed: {seed}, and {count} entries. Saved to {dir_name}"
    )
    print(
        f"Dataset was split into train and validation sets of size {num_train} and {num_validation}.\n"
    )


@cli.command("make-simulated-ts-dataset")
@click.argument("name", type=str)
@click.argument("amount", type=int, default=1)
@click.option("-t", "--tile_size", type=int, default=512, help=tilesize_help)
@click.option("-c", "--crop_size", type=int, default=512, help=cropsize_help)
@click.option(
    "-d",
    "--output_dir",
    type=click.Path(file_okay=False),
    default=SYNTHETIC_DIR,
    help=outputdir_help,
)
@click.option("-s", "--seed", type=int, default=None, help=seed_help)
@click.option("-s", "--split", type=float, default=0.0, help=split_help)
def make_simulated_time_series_dataset_wrapper(
    name, amount, tile_size, crop_size, output_dir, seed, split
):
    """
    Create a randomly generated simulated time-series dataset of unwrapped
    interferograms and their corresponding presences.

    ARGS:\n
    name        Name of dataset. Seed is appended.\n
                <name>_seed<seed>\n
    amount      Number of simulated interferograms created.\n
    """

    print("")

    (
        name,
        count,
        dir_name,
        distribution,
        dataset_info,
    ) = io.make_simulated_time_series_dataset(
        name, output_dir, amount, seed, tile_size, crop_size
    )

    num_train, num_validation = io.split_dataset(
        output_dir.__str__() + "/" + dir_name, split
    )

    try:
        log_file = open(output_dir.__str__() + "/" + dir_name + "/parameters.txt", "w")
        log_file.write(dataset_info)
    except FileNotFoundError:
        print("logfile does not exist")
    except Exception as e:
        print(f"{type(e)}: {e}")

    print("")
    print(f"Data Type Distribution: {distribution}")
    print("")
    print(
        f"Created simulated dataset with seed: {seed}, and {count} entries. Saved to {dir_name}"
    )
    print(
        f"Dataset was split into train and validation sets of size {num_train} and {num_validation}.\n"
    )


@cli.command("make-simulated-binary-dataset")
@click.argument("name", type=str)
@click.argument("model_path", type=str)
@click.argument("amount", type=int, default=1)
@click.option("-t", "--tile_size", type=int, default=512, help=tilesize_help)
@click.option("-c", "--crop_size", type=int, default=512, help=cropsize_help)
@click.option(
    "-d",
    "--output_dir",
    type=click.Path(file_okay=False),
    default=SYNTHETIC_DIR,
    help=outputdir_help,
)
@click.option("-s", "--seed", type=int, default=None, help=seed_help)
@click.option("-s", "--split", type=float, default=0.0, help=split_help)
def make_simulated_binary_dataset_wrapper(
    name, model_path, amount, tile_size, crop_size, output_dir, seed, split
):
    """
    Create a randomly generated simulated dataset of wrapped interferograms and their
    corresponding event-masks.

    ARGS:\n
    name             Name of dataset. Seed is appended.\n
                     <name>_seed<seed>\n
    pres_model_path  path to model that predicts whether there is an event.\n
    amount           Number of simulated interferograms created.\n
    """

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    seed, count, dir_name, _, _ = io.make_simulated_dataset(
        name, output_dir, amount, seed, tile_size, crop_size, model_path=model_path
    )

    num_train, num_validation = io.split_dataset(
        output_dir.__str__() + "/" + dir_name, split
    )

    print("")
    print(
        f"Dataset was split into train and validation sets of size {num_train} and {num_validation}.\n"
    )


@cli.command("split-dataset")
@click.argument("dataset_path", type=str)
@click.argument("split", type=float, default=0.2)
def split_dataset_wrapper(dataset_path, split):
    """
    Split the dataset into train and test sets

    ARGS:\n
    dataset_path  path to the dataset that should be split\n
    split         decimal percent of the dataset for validation\n
    """

    num_train, num_validation = io.split_dataset(dataset_path, split)

    print(
        f"\nSplit {dataset_path} into train and validation sets of size {num_train} and {num_validation}.\n"
    )


@cli.command("train-model")
@click.argument("model_name", type=str)
@click.argument("model_type", type=str)
@click.argument("dataset_path", type=str)
@click.option("-e", "--epochs", type=int, default=10, help=epochs_help)
@click.option("-t", "--input_shape", type=int, default=512, help=inputshape_help)
@click.option("-f", "--filters", type=int, default=64, help=filters_help)
@click.option("-b", "--batch_size", type=int, default=1, help=batchsize_help)
@click.option("-l", "--learning_rate", type=float, default=1e-4, help=learningrate_help)
@click.option("-a", "--using_aws", type=bool, default=False, help="")
def train_model_wrapper(
    model_name,
    model_type,
    dataset_path,
    epochs,
    input_shape,
    filters,
    batch_size,
    learning_rate,
    using_aws,
):
    """
    Train a U-Net or ResNet style model.

    ARGS:\n
    model_name      name of the model to be trained.\n
    model_type      type of model to train: eventnet, unet, or resnet.\n
    train_path      path to training data.\n
    test_path       path to validation data.\n
    """

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    if model_type not in ["eventnet", "unet", "unet3d", "resnet", "resnet_classifier"]:
        print(
            "\nBad model type. Should be 'eventnet', 'unet', 'uent3d' 'resnet_classifier', or 'resnet'."
        )
        return

    training.train(
        model_name,
        dataset_path,
        model_type,
        input_shape,
        epochs,
        filters,
        batch_size,
        learning_rate,
        using_aws,
    )


@cli.command("test-model")
@click.argument("model_path", type=str)
@click.argument("pres_model_path", type=str)
@click.argument("images_dir", type=str)
@click.option("-t", "--tile_size", type=int, default=512, help=tilesize_help)
@click.option("-c", "--crop_size", type=int, default=0, help=cropsize_help)
@click.option("-s", "--save_images", type=bool, default=False, help="")
@click.option("-o", "--output_dir", type=str, default=MASK_DIR, help="")
def test_model_wrapper(
    model_path,
    pres_model_path,
    images_dir,
    tile_size,
    crop_size,
    save_images,
    output_dir,
):
    """
    Predicts on a wrapped interferogram & event-mask pair and plots the results

    ARGS:\n
    model_path       path to model that does the masking.\n
    pres_model_path  path to model that predicts whether there is an event.\n
    images_dir       path to directory containing 'Positives' and 'Negatives' dirs which
                     contain the images.\n
    """

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    inference.test_model(
        model_path,
        pres_model_path,
        images_dir,
        tile_size,
        crop_size,
        save_images,
        output_dir,
    )


@cli.command("mask")
@click.argument("model_path", type=str)
@click.argument("pres_model_path", type=str)
@click.argument("image_path", type=str)
@click.option("-c", "--crop_size", type=int, default=0, help=cropsize_help)
@click.option("-t", "--tile_size", type=int, default=512, help=tilesize_help)
@click.option("-d", "--dest_path", type=str, default="", help=outputdir_help)
def mask_wrapper(
    model_path, pres_model_path, image_path, crop_size, tile_size, dest_path
):
    """
    Masks events in the given wrapped interferogram using a tensorflow model and plots
    it, with the option to save.

    ARGS:\n
    model_path       path to model to mask with.\n
    pres_model_path  path to model that predicts whether there is an event.\n
    image_path       path to wrapped interferogram to mask.\n
    """

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    mask_pred, _ = inference.mask_and_plot(
        model_path, pres_model_path, image_path, tile_size, crop_size
    )

    if dest_path != "":
        out = PIL.Image.fromarray(mask_pred)
        out.save(dest_path)


@cli.command("mask-image")
@click.argument("mask_model_path", type=str)
@click.argument("pres_model_path", type=str)
@click.argument("image_path", type=str)
@click.option("-c", "--crop_size", type=int, default=0, help=cropsize_help)
@click.option("-t", "--tile_size", type=int, default=512, help=tilesize_help)
@click.option("-d", "--dest_path", type=str, default=None, help=outputdir_help)
@click.option("-s", "--silent", type=bool, default=False, help=silent_help)
@click.option("-p", "--no_plot", type=bool, default=False, help=no_plot_help)
def mask_image_wrapper(
    mask_model_path,
    pres_model_path,
    image_path,
    crop_size,
    tile_size,
    dest_path,
    silent,
    no_plot,
):
    """
    Mask events in the given interferogram tif using a model and plot it, with the
    option to save.

    ARGS:\n
    mask_model_path     path to masking model.\n
    pres_model_path     path to binary presence model.\n
    image_path          path to interferogram tif to mask.\n
    """

    mask, presence = inference.mask(
        mask_model_path=mask_model_path,
        pres_model_path=pres_model_path,
        image_path=image_path,
        output_image_path=dest_path,
        tile_size=tile_size,
        crop_size=crop_size,
    )

    if not silent:
        if presence > 0.7:
            print("Positive")
        else:
            print("Negative")

    if not no_plot:
        image, _ = io.get_image_array(image_path)

        _, [axs_wrapped, axs_mask] = plt.subplots(1, 2, sharex=True, sharey=True)

        axs_wrapped.set_title("Wrapped")
        axs_mask.set_title("Segmentation Mask")

        axs_wrapped.imshow(image, origin="lower", cmap="jet")
        axs_mask.imshow(mask, origin="lower", cmap="jet")

        plt.show()


@cli.command("mask-directory")
@click.argument("mask_model_path", type=str)
@click.argument("pres_model_path", type=str)
@click.argument("directory", type=str)
@click.argument("output_directory", type=str)
@click.option("-c", "--crop_size", type=int, default=0, help=cropsize_help)
@click.option("-t", "--tile_size", type=int, default=512, help=tilesize_help)
@click.option("-s", "--silent", is_flag=True, default=False, help=silent_help)
@click.option("-c", "--copy_image", is_flag=True, default=False, help=copy_image_help)
def mask_directory_wrapper(
    mask_model_path,
    pres_model_path,
    directory,
    output_directory,
    crop_size,
    tile_size,
    silent,
    copy_image,
):
    """
    Generate masks for a directory of tifs and output to a directory using a model.

    ARGS:\n
    mask_model_path     path to masking model.\n
    pres_model_path     path to binary presence model.\n
    directory           path to directory of interferogram tifs to mask.\n
    """

    mask_model = models.load_model(mask_model_path)
    pres_model = models.load_model(pres_model_path)

    if not output_directory.endswith("/"):
        output_directory += "/"

    if not directory.endswith("/"):
        directory += "/"

    if path.isfile(directory):
        raise click.ClickException(
            f"Expected {directory} to be a directory, however it is a file"
        )

    if path.isfile(output_directory):
        raise click.ClickException(
            f"Expected {output_directory} to be a directory, not a file"
        )

    if path.isdir(output_directory):
        click.confirm(
            f"{output_directory} already exists, do you wish to continue and possibly overwrite files in this directory?"
        )
    else:
        mkdir(output_directory)

    if not path.isdir(f"{output_directory}positive"):
        mkdir(f"{output_directory}positive")

    if not path.isdir(f"{output_directory}negative"):
        mkdir(f"{output_directory}negative")

    output = []

    for image_name in listdir(directory):
        if not image_name.endswith(".tif"):
            continue

        image_path = directory + image_name
        image, gdal_dataset = io.get_image_array(image_path)

        mask_pred, pres_mask, pres_vals = inference.mask_with_model(
            mask_model=mask_model,
            pres_model=pres_model,
            arr_w=image,
            tile_size=tile_size,
            crop_size=crop_size,
        )

        presence = np.mean(pres_mask) > 0.0

        presence_string = "negative"
        if presence:
            presence_string = "positive"

        image_output_dir = (
            f"{output_directory}{presence_string}/{path.splitext(image_name)[0]}/"
        )

        if not path.isdir(image_output_dir):
            mkdir(image_output_dir)

        output_image_path = f"{image_output_dir}mask.tif"

        img = Image.fromarray(mask_pred)
        img.save(output_image_path)

        out_dataset = gdal.Open(output_image_path, gdal.GA_Update)
        out_dataset.SetGeoTransform(gdal_dataset.GetGeoTransform())
        out_dataset.SetProjection(gdal_dataset.GetProjection())

        if not silent:
            print(f"{image_name}: {presence_string}")

        if copy_image:
            copyfile(image_path, f"{image_output_dir}wrapped.tif")

        metadata = {
            "image_name": image_name,
            "presence": presence.item(),
            "image_path": image_path,
            "mask_path": f"{image_output_dir}mask.tif",
        }

        output.append(metadata)

        with open(f"{image_output_dir}metadata.json", "w") as metadata_file:
            metadata_file.write(dumps(metadata))

    with open(f"{output_directory}output.json", "w") as output_file:
        output_file.write(dumps(output))


@cli.command("interactive")
@click.option("-e", "--event_type", type=str, default="quake", help="")
def interactive_wrapper(event_type):
    """
    Show a randomly generated wrapped interferogram with simulated deformation,
    atmospheric turbulence, atmospheric topographic error, and incoherence masking.
    """

    kwargs = {
        "source_x": 22000,
        "source_y": 22000,
        "strike": 180,
        "dip": 45,
        "length": 1000,
        "rake": 90,
        "slip": 1,
        "top_depth": 3000,
        "bottom_depth": 6000,
        "width": 3000,
        "depth": 3000,
        "opening": 0.5,
    }

    fig, [axs_unwrapped, axs_wrapped] = plt.subplots(
        1, 2, sharex=True, sharey=True, tight_layout=True
    )

    axs_unwrapped.set_title("unwrapped")
    axs_unwrapped.set_position([0.05, 0.45, 0.5, 0.5])
    axs_wrapped.set_title("wrapped")
    axs_wrapped.set_position([0.5, 0.45, 0.5, 0.5])

    axs_slip = plt.axes([0.375, 0.36, 0.25, 0.02])
    slider_slip = widgets.Slider(axs_slip, "slip", 0.0, 10.0, valinit=kwargs["slip"])

    axs_strike = plt.axes([0.375, 0.33, 0.25, 0.02])
    slider_strike = widgets.Slider(
        axs_strike, "strike", 0.0, 180.0, valinit=kwargs["strike"]
    )

    axs_dip = plt.axes([0.375, 0.30, 0.25, 0.02])
    slider_dip = widgets.Slider(axs_dip, "dip", 0.0, 90.0, valinit=kwargs["dip"])

    axs_rake = plt.axes([0.375, 0.27, 0.25, 0.02])
    slider_rake = widgets.Slider(
        axs_rake, "rake", -180.0, 180.0, valinit=kwargs["rake"]
    )

    axs_opening = plt.axes([0.375, 0.24, 0.25, 0.02])
    slider_opening = widgets.Slider(
        axs_opening, "opening", 0.0, 10.0, valinit=kwargs["opening"]
    )

    axs_top_depth = plt.axes([0.375, 0.21, 0.25, 0.02])
    slider_top_depth = widgets.Slider(
        axs_top_depth, "top_depth", 0.0, 45000.0, valinit=kwargs["top_depth"]
    )

    axs_bottom_depth = plt.axes([0.375, 0.18, 0.25, 0.02])
    slider_bottom_depth = widgets.Slider(
        axs_bottom_depth, "bottom_depth", 0.0, 45000.0, valinit=kwargs["bottom_depth"]
    )

    axs_width = plt.axes([0.375, 0.15, 0.25, 0.02])
    slider_width = widgets.Slider(
        axs_width, "width", 100.0, 10000.0, valinit=kwargs["width"]
    )

    axs_length = plt.axes([0.375, 0.12, 0.25, 0.02])
    slider_length = widgets.Slider(
        axs_length, "length", 100.0, 10000.0, valinit=kwargs["length"]
    )

    axs_source_x = plt.axes([0.375, 0.09, 0.25, 0.02])
    slider_source_x = widgets.Slider(
        axs_source_x, "source_x", 0.0, 45000.0, valinit=kwargs["source_x"]
    )

    axs_source_y = plt.axes([0.375, 0.06, 0.25, 0.02])
    slider_source_y = widgets.Slider(
        axs_source_y, "source_y", 0.0, 45000.0, valinit=kwargs["source_y"]
    )

    unwrapped, masked, wrapped, presence = sarsim.gen_simulated_deformation(
        seed=100000, tile_size=512, event_type=event_type, **kwargs
    )

    axs_wrapped.imshow(wrapped, origin="lower", cmap="jet")
    axs_unwrapped.imshow(unwrapped, origin="lower", cmap="jet")

    def update():
        kwargs = {
            "source_x": slider_source_x.val,
            "source_y": slider_source_y.val,
            "strike": slider_strike.val,
            "dip": slider_dip.val,
            "length": slider_length.val,
            "rake": slider_rake.val,
            "slip": slider_slip.val,
            "top_depth": slider_top_depth.val,
            "bottom_depth": slider_bottom_depth.val,
            "width": slider_width.val,
            "depth": slider_top_depth.val,
            "opening": slider_opening.val,
        }

        unwrapped, masked, wrapped, presence = sarsim.gen_simulated_deformation(
            seed=100000, tile_size=512, event_type=event_type, **kwargs
        )

        axs_wrapped.imshow(wrapped, origin="lower", cmap="jet")
        axs_unwrapped.imshow(unwrapped, origin="lower", cmap="jet")

        fig.canvas.draw()

    slider_source_x.on_changed(update)
    slider_source_y.on_changed(update)
    slider_strike.on_changed(update)
    slider_dip.on_changed(update)
    slider_length.on_changed(update)
    slider_rake.on_changed(update)
    slider_slip.on_changed(update)
    slider_top_depth.on_changed(update)
    slider_bottom_depth.on_changed(update)
    slider_width.on_changed(update)
    slider_opening.on_changed(update)

    plt.show()


@cli.command("simulate")
@click.option("-s", "--seed", type=int, default=0, help=seed_help)
@click.option("-t", "--tile_size", type=int, default=512, help=tilesize_help)
@click.option("-c", "--crop_size", type=int, default=0, help=cropsize_help)
@click.option("-e", "--event_type", type=str, default="quake", help="")
@click.option("-n", "--noise_only", type=bool, default=False, help="")
@click.option("-g", "--gauss_only", type=bool, default=False, help="")
@click.option("-v", "--verbose", type=bool, default=False, help="")
def simulate_wrapper(
    seed, tile_size, crop_size, event_type, noise_only, gauss_only, verbose
):
    """
    Show a randomly generated wrapped interferogram with simulated deformation,
    atmospheric turbulence, atmospheric topographic error, and incoherence masking.
    """

    if not noise_only:
        _, masked, wrapped, event_is_present = processing.gen_simulated_deformation(
            seed, tile_size, verbose, event_type=event_type
        )
    else:
        _, masked, wrapped, event_is_present = sarsim.gen_sim_noise(
            seed, tile_size, gaussian_only=gauss_only
        )

    if crop_size < tile_size and crop_size != 0:
        masked = processing.simulate_unet_cropping(masked, (crop_size, crop_size))

    gui.show_dataset(masked, wrapped)


@cli.command("show")
@click.argument("file_path", type=click.Path())
def show_dataset_wrapper(file_path):
    """
    Show the wrapped interferograms and event-masks from a given dataset directory.

    ARGS:\n
    file_path       path to the .npz files to show.\n
    """

    filenames = os.listdir(file_path)

    def filename_check(x):
        "synth" in x or "sim" in x or "real" in x

    data_filenames = [item for item in filenames if filename_check(item)]

    for filename in data_filenames:
        mask, wrapped, presence = io.load_dataset(file_path + "/" + filename)
        print(f"Showing dataset {filename}")
        print(f"Presence:       {presence}")
        gui.show_dataset(mask, wrapped)


@cli.command("show-product")
@click.argument("product_path", type=str)
@click.option("-t", "--tile_size", type=int, default=0, help=tilesize_help)
@click.option("-c", "--crop_size", type=int, default=0, help=cropsize_help)
def show_product_wrapper(product_path, crop_size, tile_size):
    """
    Plots the wrapped, unwrapped, and correlation images in an InSAR Product.

    ARGS:\n
    product_path        path to folder containing the elements of the InSAR product
    from search.asf.alaska.edu.\n
    """

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
            cropped_tile = processing.simulate_unet_cropping(
                tile_, (crop_size, crop_size)
            )
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


@cli.command("sort-images-by-size")
@click.argument("images_path", type=str)
def sort_images_wrapper(images_path):
    """
    View images in a directory for manual labeling.

    ARGS:\n
    images_path        path to folder containing the wrapped or unwrapped GeoTiffs\n
    """

    Path("{images_path}/Small").mkdir(parents=True, exist_ok=True)
    Path("{images_path}/Medium").mkdir(parents=True, exist_ok=True)
    Path("{images_path}/Large").mkdir(parents=True, exist_ok=True)

    for filename in listdir(images_path):
        if filename.endswith(".tif"):
            image, _ = io.get_image_array(f"{images_path}/{filename}")

            image = np.angle(np.exp(1j * (image)))

            print(f"\n{filename}\n")

            plt.imshow(image, cmap="jet", vmin=-np.pi, vmax=np.pi)
            plt.show()

            size = input("Size? (S/M/L): ").lower()

            try:
                if size[0] == "s":
                    system(f"mv {images_path}/{filename} {images_path}/Small")
                elif size[0] == "m":
                    system(f"mv {images_path}/{filename} {images_path}/Medium")
                elif size[0] == "l":
                    system(f"mv {images_path}/{filename} {images_path}/Large")
            except Exception as e:
                print("Could not move file. Error: ", e)


@cli.command("check-image")
@click.argument("image_path", type=str)
def check_image_wrapper(image_path):
    """
    View images in a directory for manual labeling.

    ARGS:\n
    images_path        path to folder containing the wrapped or unwrapped GeoTiffs\n
    """

    image, _ = io.get_image_array(image_path)

    image = np.angle(np.exp(1j * (image)))

    plt.imshow(image, cmap="jet", vmin=-np.pi, vmax=np.pi)
    plt.show()


@cli.command("check-images")
@click.argument("images_path", type=str)
def check_images_wrapper(images_path):
    """
    View images in a directory for manual labeling.

    ARGS:\n
    images_path        path to folder containing the wrapped or unwrapped GeoTiffs\n
    """

    for filename in os.listdir(images_path):
        if filename.endswith(".tif"):
            image, _ = io.get_image_array(f"{images_path}/{filename}")

            image = np.angle(np.exp(1j * (image)))

            print(f"\n{filename}\n")

            plt.imshow(image, cmap="jet", vmin=-np.pi, vmax=np.pi)
            plt.show()


@cli.command("create-model")
@click.argument("model_name", type=str)
@click.argument("dataset_size", type=int)
@click.option("-e", "--epochs", type=int, default=10, help=epochs_help)
@click.option("-t", "--input_shape", type=int, default=512, help=inputshape_help)
@click.option("-f", "--filters", type=int, default=64, help=filters_help)
@click.option("-b", "--batch_size", type=int, default=1, help=batchsize_help)
@click.option("-s", "--val_split", type=float, default=0.1, help=split_help)
@click.option("-l", "--learning_rate", type=float, default=1e-4, help=learningrate_help)
@click.option("-a", "--using_aws", type=bool, default=False, help="")
@click.option("-r", "--seed", type=int, default=0, help=seed_help)
def create_model_wrapper(
    model_name,
    dataset_size,
    epochs,
    input_shape,
    filters,
    batch_size,
    val_split,
    learning_rate,
    using_aws,
    seed,
):
    """
    Batteries-Included EventNet Creation

    ARGS:\n
    model_name      name to give the model.\n
    dataset_size    number of samples in dataset.\n
    """

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    print("___________________________\n")
    print("Creating Mask Model Dataset")
    print("___________________________\n")

    mask_dataset_name = "tmp_dataset_masking_" + str(time.time()).strip(".")[0]

    output_dir = SYNTHETIC_DIR

    _, count, dir_name = io.make_simulated_dataset(
        mask_dataset_name, output_dir, dataset_size, seed, input_shape, input_shape
    )

    dataset_path = output_dir.__str__() + "/" + dir_name

    io.split_dataset(dataset_path, val_split)

    print("______________________\n")
    print("Training Masking Model")
    print("______________________")

    mask_model_type = "unet"
    pres_model_type = "eventnet"
    using_wandb = False
    using_jupyter = False

    training.train(
        model_name + "_masking",
        dataset_path,
        mask_model_type,
        input_shape,
        epochs,
        filters,
        batch_size,
        learning_rate,
        using_wandb,
        using_aws,
        using_jupyter,
    )

    print("__________________________________________\n")
    print("Creating Presence Prediction Model Dataset")
    print("__________________________________________\n")

    pres_dataset_name = "tmp_dataset_presence_" + str(time.time()).strip(".")[0]
    mask_model_path = "models/checkpoints/" + model_name + "_masking"

    _, count, dir_name = io.make_simulated_dataset(
        pres_dataset_name,
        output_dir,
        dataset_size,
        seed,
        input_shape,
        input_shape,
        model_path=mask_model_path,
    )

    dataset_path = output_dir.__str__() + "/" + dir_name

    io.split_dataset(dataset_path, val_split)

    print("_________________\n")
    print("Training EventNet")
    print("_________________")
    training.train(
        model_name + "_presence",
        dataset_path,
        pres_model_type,
        input_shape,
        5,
        32,
        1,
        0.005,
        using_wandb,
        using_aws,
        using_jupyter,
    )

    print("")
    print("Done...\n")


# ---------------------------------- #
# Commands for SageMaker Entrypoints #
# ---------------------------------- #


@cli.command("train")
@click.option("-n", "--model_name", type=str, default="aws_model", help="Name of model")
@click.option("-t", "--model_type", type=str, default="unet", help="Type of model")
@click.option("-e", "--epochs", type=int, default=1, help=epochs_help)
@click.option("-t", "--input_shape", type=int, default=512, help=inputshape_help)
@click.option("-f", "--filters", type=int, default=16, help=filters_help)
@click.option("-b", "--batch_size", type=int, default=8, help=batchsize_help)
@click.option(
    "-l", "--learning_rate", type=float, default=0.001, help=learningrate_help
)
def sagemaker_train_wrapper(
    model_name, model_type, epochs, input_shape, filters, batch_size, learning_rate
):
    """
    SageMaker compatible function to train a U-Net, ResNet, or Classification EventNet
    model.
    """

    root_dir = "/opt/ml"  # SageMaker expects things to happen here.

    input_dir = root_dir + "/input"  # SageMaker uploads things here.
    dataset_dir = input_dir + "/data/training"
    config_dir = input_dir + "/config"

    output_dir = root_dir + "/output/data"  # SageMaker takes optional extras from here.
    logs_dir = (
        output_dir + "/logs"
    )  # A file called failure must hold failing errors in /output
    checkpoint_dir = output_dir + "/best_checkpoint"

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        print("directory already exists")
    try:
        os.mkdir(checkpoint_dir)
    except FileExistsError:
        print("directory already exists")
    try:
        os.mkdir(logs_dir)
    except FileExistsError:
        print("directory already exists")

    try:
        with open(config_dir + "/hyperparameters.json", "r") as json_file:
            hyperparameters = json.load(json_file)

        print(f"Read hyperparameters: {hyperparameters}")

        model_type = str(hyperparameters["model_type"])
        epochs = int(hyperparameters["epochs"])
        filters = int(hyperparameters["filters"])
        batch_size = int(hyperparameters["batch_size"])
        learning_rate = float(hyperparameters["learning_rate"])

        os.system(f"cp -r {config_dir} {output_dir}")

    except FileNotFoundError:
        print("hyperparameter file does not exist")
    except Exception as e:
        print(f"Caught {type(e)}: {e}\n Using default hyperparameters.")

    training.train(
        model_name,
        dataset_dir,
        model_type,
        input_shape,
        epochs,
        filters,
        batch_size,
        learning_rate,
        using_aws=True,
        logs_dir=logs_dir,
    )


@cli.command("predict-event")
@click.argument("usgs-event-id", type=str)
@click.argument("product-name", type=str)
def model_inference(usgs_event_id, product_name):
    """
    Run event prediction in the cloud utilizing AWS api
    """
    url = "https://aevrv4z4vf.execute-api.us-west-2.amazonaws.com/test-3/predict-event"

    r = requests.post(
        url, json={"usgs_event_id": usgs_event_id, "product_name": product_name}
    )
    print(r.json())
