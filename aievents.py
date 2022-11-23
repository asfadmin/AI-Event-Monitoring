"""
 Created By:  Andrew Player
 File Name:   aievents.py
 Description: CLI Interface
"""

import click
from src.config import SYNTHETIC_DIR


# ------------- #
# Help Strings  #
# ------------- #

outputdir_help     = "Directory to save to."
seed_help          = "Seed integer for reproducible data."
cropsize_help      = "Crop unwrapped image to: (crop_size, crop_size). This has to match the models output shape."
split_help         = "Float between 0 and 1 representing the percentage of items to go to the training set."
tilesize_help      = "The width/height of the tiles. This should match the input shape of the model."
inputshape_help    = "The input shape of the model. (input_shape, input_shape)"
epochs_help        = "The number of times that the network should train over the given training dataset."
batchsize_help     = "The number of data samples to go through before the model's weights are updated."
batchsize_help    += " Lower = Slower = More Accurate."
filters_help       = "The base number of filters for the convolutions in the model."
dropout_help       = "The amount of network nodes to randomly dropout throughout training to decrease over fitting."
dropout_help      += " 0 <= Dropout <= 1"
learningrate_help  = "The rate which affects how much the model changes in response to estimated error."
learningrate_help += "0.0005 <= learning_rate <= 0.005"
amplitude_help     = "Max amplitude of the gaussians (minimum is the negative of this)."
usesim_help        = "Flag to use simulated interferograms rather than synthetic interferograms. Default is False."
numtrials_help     = "Test the model over this many images. If 1, the images are plotted."


# ------------------ #
# CLI Implementation #
# ------------------ #

@click.group()
def cli():
    pass


@cli.command('setup')
def setup():

    """
    Create data directory subtree. This should be run before make-dataset.

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

    from src.io import create_directories

    print("")
    create_directories()

    print("")
    click.echo("Data directory created")
    print("")


@cli.command   ('make-simulated-dataset')
@click.argument('name'              , type=str                                                                    )
@click.argument('amount'            , type=int                        , default=1                                 )
@click.option  ('-t', '--tile_size' , type=int                        , default=512          , help=tilesize_help )
@click.option  ('-c', '--crop_size' , type=int                        , default=512          , help=cropsize_help )
@click.option  ('-d', '--output_dir', type=click.Path(file_okay=False), default=SYNTHETIC_DIR, help=outputdir_help)
@click.option  ('-s', '--seed'      , type=int                        , default=None         , help=seed_help     )
@click.option  ('-s', '--split'     , type=float                      , default=0.0          , help=split_help    )
def make_simulated_dataset_wrapper(name, amount, tile_size, crop_size, output_dir, seed, split):

    """
    Create a randomly generated simulated dataset of wrapped interferograms and their corresponding event-masks.

    ARGS:\n
    name        Name of dataset. Seed is appended.\n
                <name>_seed<seed>\n
    amount      Number of simulated interferograms created.\n
    """

    from src.io import split_dataset, make_simulated_dataset

    print("")

    name, count, dir_name, distribution, dataset_info = make_simulated_dataset(
        name,
        output_dir,
        amount,
        seed,
        tile_size,
        crop_size
    )

    num_train, num_validation = split_dataset(output_dir.__str__() + '/' + dir_name, split)

    try:
        log_file = open(output_dir.__str__() + '/' + dir_name + '/parameters.txt', 'w')
        log_file.write(dataset_info)
    except Exception as e:
        print(f'{type(e)}: {e}')

    print("")
    print(f"Data Type Distribution: {distribution}")
    print("")
    print(f"Created simulated dataset with seed: {seed}, and {count} entries. Saved to {dir_name}")
    print(f"Dataset was split into train and validation sets of size {num_train} and {num_validation}.\n")


@cli.command   ('make-simulated-binary-dataset')
@click.argument('name'              , type=str                                                                    )
@click.argument('model_path'        , type=str                                                                    )
@click.argument('amount'            , type=int                        , default=1                                 )
@click.option  ('-t', '--tile_size' , type=int                        , default=512          , help=tilesize_help )
@click.option  ('-c', '--crop_size' , type=int                        , default=512          , help=cropsize_help )
@click.option  ('-d', '--output_dir', type=click.Path(file_okay=False), default=SYNTHETIC_DIR, help=outputdir_help)
@click.option  ('-s', '--seed'      , type=int                        , default=None         , help=seed_help     )
@click.option  ('-s', '--split'     , type=float                      , default=0.0          , help=split_help    )
def make_simulated_binary_dataset_wrapper(name, model_path, amount, tile_size, crop_size, output_dir, seed, split):

    """
    Create a randomly generated simulated dataset of wrapped interferograms and their corresponding event-masks.

    ARGS:\n
    name             Name of dataset. Seed is appended.\n
                     <name>_seed<seed>\n
    pres_model_path  path to model that predicts whether there is an event.\n
    amount           Number of simulated interferograms created.\n
    """

    from os import environ
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    from src.io import split_dataset, make_simulated_dataset

    name, count, dir_name, _, _ = make_simulated_dataset(
        name,
        output_dir,
        amount,
        seed,
        tile_size,
        crop_size,
        model_path = model_path
    )

    num_train, num_validation = split_dataset(output_dir.__str__() + '/' + dir_name, split)

    print("")
    print(f"Created binary dataset of size {count} at {output_dir.__str__() + '/' + name}.")
    print(f"Dataset was split into train and validation sets of size {num_train} and {num_validation}.\n")


@cli.command('split-dataset')
@click.argument('dataset_path', type=str               )
@click.argument('split'       , type=float, default=0.2)
def split_dataset_wrapper(dataset_path, split):

    """
    Split the dataset into train and test sets

    ARGS:\n
    dataset_path  path to the dataset that should be split\n
    split         decimal percent of the dataset for validation\n
    """

    from src.io import split_dataset

    num_train, num_validation = split_dataset(dataset_path, split)

    print(f"\nSplit {dataset_path} into train and validation sets of size {num_train} and {num_validation}.\n")


@cli.command   ('train-model')
@click.argument('model_name'           , type=str                                         )
@click.argument('model_type'           , type=str                                         )
@click.argument('dataset_path'         , type=str                                         )
@click.option  ('-e', '--epochs'       , type=int  , default=10   , help=epochs_help      )
@click.option  ('-t', '--input_shape'  , type=int  , default=512  , help=inputshape_help  )
@click.option  ('-f', '--filters'      , type=int  , default=64   , help=filters_help     )
@click.option  ('-b', '--batch_size'   , type=int  , default=1    , help=batchsize_help   )
@click.option  ('-l', '--learning_rate', type=float, default=1e-4 , help=learningrate_help)
@click.option  ('-a', '--using_aws'    , type=bool , default=False, help=""               )
def train_model_wrapper(
    model_name,
    model_type,
    dataset_path,
    epochs,
    input_shape,
    filters,
    batch_size,
    learning_rate,
    using_aws
):

    """
    Train a U-Net or ResNet style model.

    ARGS:\n
    model_name      name of the model to be trained.\n
    model_type      type of model to train: eventnet, unet, or resnet.\n
    train_path      path to training data.\n
    test_path       path to validation data.\n
    """

    from os import environ
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    from src.training import train

    if model_type not in ['eventnet', 'unet', 'resnet', 'resnet_classifier']:
        print("\nBad model type. Should be \'eventnet\', \'unet\', \'resnet_classifier\', or \'resnet\'.")
        return

    train(
        model_name,
        dataset_path,
        model_type,
        input_shape,
        epochs,
        filters,
        batch_size,
        learning_rate,
        using_aws
    )


@cli.command   ('test-masking')
@click.argument('mask_model_path'      , type=str                                     )
@click.option  ('-e', '--event_type'   , type=str , default=""   , help=""            )
@click.option  ('-n', '--noise_only'   , type=bool, default=False, help=""            )
@click.option  ('-g', '--gaussian_only', type=bool, default=False, help=""            )
@click.option  ('-v', '--verbose'      , type=bool, default=False, help=""            )
@click.option  ('-s', '--seed'         , type=int , default=0    , help=seed_help     )
@click.option  ('-t', '--tile_size'    , type=int , default=1024 , help=tilesize_help )
@click.option  ('-c', '--crop_size'    , type=int , default=0    , help=cropsize_help )
def test_masking_wrapper(mask_model_path, event_type, noise_only, gaussian_only, verbose, seed, tile_size, crop_size):

    """
    Predicts on a wrapped interferogram & event-mask pair and plots the results

    ARGS:\n
    model_path      path to model that should be tested.\n
    """


    from numpy import mean, abs
    from os    import environ    
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    from tensorflow.keras.models import load_model
    from src.inference           import mask_simulated, plot_imgs

    mask_model = load_model(mask_model_path)

    wrapped, mask, mask_pred, mask_pred_rounded, presence = mask_simulated(
        mask_model,
        seed,
        tile_size,
        crop_size,
        verbose       = verbose,
        noise_only    = noise_only,
        gaussian_only = gaussian_only,
        event_type    = event_type
    )

    mean_absolute_error = mean(abs(mask - mask_pred_rounded))

    print(f"Mean Absolute Error: {mean_absolute_error}\n")

    plot_imgs(wrapped, mask, mask_pred, mask_pred_rounded)


@cli.command   ('test-binary-choice')
@click.argument('mask_model_path'    , type=str                                      )
@click.argument('pres_model_path'    , type=str                                      )
@click.option  ('-s', '--seed'       , type=int  , default=0    , help=seed_help     )
@click.option  ('-n', '--num_trials' , type=int  , default=0    , help=numtrials_help)
@click.option  ('-t', '--tile_size'  , type=int  , default=512  , help=tilesize_help )
@click.option  ('-c', '--crop_size'  , type=int  , default=0    , help=cropsize_help )
@click.option  ('-p', '--plot'       , type=bool , default=False, help=""            )
@click.option  ('-c', '--use_rounded', type=bool , default=False, help=""            )
@click.option  ('-r', '--threshold'  , type=float, default=0.5  , help=""            )
def test_binary_choice_wrapper(mask_model_path, pres_model_path, seed, num_trials, tile_size, crop_size, plot, use_rounded, threshold):

    """
    Predicts on a wrapped interferogram & event-mask pair and plots the results

    ARGS:\n
    model_path       path to model that does the masking.\n
    pres_model_path  path to model that predicts whether there is an event.\n
    """

    from os import environ
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    from tensorflow.keras.models import load_model
    from src.inference           import test_binary_choice

    mask_model = load_model(mask_model_path)
    pres_model = load_model(pres_model_path)

    test_binary_choice(
        mask_model,
        pres_model,
        seed,
        tile_size,
        crop_size,
        count            = num_trials,
        plot             = plot,
        use_rounded_mask = use_rounded,
        positive_thresh  = threshold
    )


@cli.command   ('test-model')
@click.argument('model_path'           , type=str                                  )
@click.argument('pres_model_path'      , type=str                                  )
@click.argument('images_dir'           , type=str                                  )
@click.option  ('-t', '--tile_size'    , type=int , default=512, help=tilesize_help)
@click.option  ('-c', '--crop_size'    , type=int , default=0  , help=cropsize_help)
def test_model_wrapper(model_path, pres_model_path, images_dir, tile_size, crop_size):

    """
    Predicts on a wrapped interferogram & event-mask pair and plots the results

    ARGS:\n
    model_path       path to model that does the masking.\n
    pres_model_path  path to model that predicts whether there is an event.\n
    images_dir       path to directory containing 'Positives' and 'Negatives' dirs which contain the images.\n
    """

    from os import environ
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    from src.inference import test_model

    test_model(model_path, pres_model_path, images_dir, tile_size, crop_size)


@cli.command   ('model-summary')
@click.argument('model_path', type = str)
def model_summary_wrapper(model_path):

    """
    Prints the model summary.

    ARGS:\n
    model_path      path to model.\n
    """

    from os import environ
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    from tensorflow.keras.models import load_model
    
    model = load_model(model_path)
    model.summary()


@cli.command('visualize-layers')
@click.argument('model_path'   , type=str                           )
@click.argument('save_path'    , type=str                           )
@click.option  ('-s', '--seed' , type=int, default=0, help=seed_help)
def visualize_layers_wrapper(model_path, save_path, seed):

    """
    Visualize the feature maps of the model for a random synthetic interferogram.

    ARGS:\n
    model_path          path to model.\n
    save_path           path to folder to save the tifs to.\n
    """

    from os import environ
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    from src.inference import visualize_layers

    visualize_layers(model_path, save_path, seed)


@cli.command   ('mask')
@click.argument('model_path'          , type=str                                    )
@click.argument('pres_model_path'     , type=str                                    )
@click.argument('image_path'          , type=str                                    )
@click.option  ('-c', '--crop_size'   , type=int  , default=0  , help=cropsize_help )
@click.option  ('-t', '--tile_size'   , type=int  , default=512, help=tilesize_help )
@click.option  ('-d', '--dest_path'   , type=str  , default="" , help=outputdir_help)
def mask_wrapper(
    model_path,
    pres_model_path,
    image_path, 
    crop_size, 
    tile_size,
    dest_path
):

    """
    Masks events in the given wrapped interferogram using a tensorflow model and plots it, with the option to save.

    ARGS:\n
    model_path       path to model to mask with.\n
    pres_model_path  path to model that predicts whether there is an event.\n
    image_path       path to wrapped interferogram to mask.\n
    """

    from os import environ

    environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    from src.inference import mask_and_plot

    mask_pred, _ = mask_and_plot(
        model_path,
        pres_model_path,
        image_path,
        tile_size,
        crop_size
    )

    if dest_path != "":

        from PIL import Image

        out = Image.fromarray(mask_pred)
        out.save(dest_path)


@cli.command ('interactive')
@click.option('-e', '--event_type', type=str, default="quake", help="")
def interactive_wrapper(event_type):

    """
    Show a randomly generated wrapped interferogram with simulated deformation, atmospheric turbulence, atmospheric topographic error, and incoherence masking.
    """

    from src.gui import interactive_interferogram

    interactive_interferogram(event_type)


@cli.command ('simulate')
@click.option('-s', '--seed'      , type=int , default=0      , help=seed_help    )
@click.option('-t', '--tile_size' , type=int , default=512    , help=tilesize_help)
@click.option('-c', '--crop_size' , type=int , default=0      , help=cropsize_help)
@click.option('-e', '--event_type', type=str , default="quake", help=""           )
@click.option('-n', '--noise_only', type=bool, default=False  , help=""           )
@click.option('-g', '--gauss_only', type=bool, default=False  , help=""           )
@click.option('-v', '--verbose'   , type=bool, default=False  , help=""           )
def simulate_wrapper(seed, tile_size, crop_size, event_type, noise_only, gauss_only, verbose):

    """
    Show a randomly generated wrapped interferogram with simulated deformation, atmospheric turbulence, atmospheric topographic error, and incoherence masking.
    """
    
    from src.gui import show_dataset
    from src.sarsim import gen_simulated_deformation, gen_sim_noise
    from src.processing import simulate_unet_cropping

    if not noise_only:
        _, masked, wrapped, event_is_present = gen_simulated_deformation(
            seed,
            tile_size,
            verbose,
            event_type = event_type
        )
    else:
        _, masked, wrapped, event_is_present = gen_sim_noise(
            seed,
            tile_size,
            gaussian_only=gauss_only
        )        

    if crop_size < tile_size and crop_size != 0:
        masked = simulate_unet_cropping(masked, (crop_size, crop_size))

    show_dataset(masked, wrapped)


@cli.command   ('show')
@click.argument('file_path', type=click.Path())
def show_dataset_wrapper(file_path):

    """
    Show the wrapped interferograms and event-masks from a given dataset directory.

    ARGS:\n
    file_path       path to the .npz files to show.\n
    """

    from os import listdir

    from src.gui import show_dataset
    from src.io import load_dataset

    filenames      = listdir(file_path)
    filename_check = lambda x: "synth" in x or "sim" in x or "real" in x
    data_filenames = [item for item in filenames if filename_check(item)]

    for filename in data_filenames:
        mask, wrapped, presence = load_dataset(file_path + "/" + filename)
        print(f"Showing dataset {filename}")
        print(f"Presence:       {presence}")
        show_dataset(mask, wrapped)


@cli.command   ('show-product')
@click.argument('product_path'     , type=str                               )
@click.option  ('-t', '--tile_size', type=int, default=0, help=tilesize_help)
@click.option  ('-c', '--crop_size', type=int, default=0, help=cropsize_help)
def show_product_wrapper(product_path, crop_size, tile_size):

    """
    Plots the wrapped, unwrapped, and correlation images in an InSAR Product.

    ARGS:\n
    product_path        path to folder containing the elements of the InSAR product from search.asf.alaska.edu.\n
    """

    from src.gui import show_product

    show_product(product_path, crop_size, tile_size)


@cli.command   ('check-images')
@click.argument('images_path', type=str)
def check_images_wrapper(images_path):

    """
    View images in a directory for manual labeling.

    ARGS:\n
    images_path        path to folder containing the wrapped or unwrapped GeoTiffs\n
    """

    import matplotlib.pyplot as plt

    from os     import listdir
    from src.io import get_image_array

    for filename in listdir(images_path):        
        
        if filename.endswith(".tif"):
            
            image = get_image_array(f"{images_path}/{filename}")
            print(f"\n{filename}\n")
            
            plt.imshow(image)
            plt.show()


@cli.command   ('create-model')
@click.argument('model_name'           , type=str                                         )
@click.argument('dataset_size'         , type=int                                         )
@click.option  ('-e', '--epochs'       , type=int  , default=10   , help=epochs_help      )
@click.option  ('-t', '--input_shape'  , type=int  , default=512  , help=inputshape_help  )
@click.option  ('-f', '--filters'      , type=int  , default=64   , help=filters_help     )
@click.option  ('-b', '--batch_size'   , type=int  , default=1    , help=batchsize_help   )
@click.option  ('-s', '--val_split'    , type=float, default=0.1  , help=split_help       )
@click.option  ('-l', '--learning_rate', type=float, default=1e-4 , help=learningrate_help)
@click.option  ('-a', '--using_aws'    , type=bool , default=False, help=""               )
@click.option  ('-r', '--seed'         , type=int  , default=0    , help=seed_help        )
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
    seed
):

    """
    Batteries-Included EventNet Creation

    ARGS:\n
    model_name      name to give the model.\n
    dataset_size    number of samples in dataset.\n
    """

    import time

    from os import environ
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    from src.training import train
    from src.io       import split_dataset, make_simulated_dataset


    print("___________________________\n")
    print("Creating Mask Model Dataset")
    print("___________________________\n")

    mask_dataset_name = 'tmp_dataset_masking_' + str(time.time()).strip('.')[0]

    output_dir = SYNTHETIC_DIR

    _, count, dir_name = make_simulated_dataset(
        mask_dataset_name,
        output_dir,
        dataset_size,
        seed,
        input_shape,
        input_shape
    )

    dataset_path = output_dir.__str__() + "/" + dir_name

    split_dataset(dataset_path, val_split)


    print("______________________\n") 
    print("Training Masking Model")
    print("______________________") 

    mask_model_type = 'unet'
    pres_model_type = 'eventnet'
    using_wandb     = False
    using_jupyter   = False

    train(
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
        using_jupyter
    )

    
    print("__________________________________________\n") 
    print("Creating Presence Prediction Model Dataset")
    print("__________________________________________\n") 

    pres_dataset_name = 'tmp_dataset_presence_' + str(time.time()).strip('.')[0]
    mask_model_path = 'models/checkpoints/' + model_name + '_masking'

    _, count, dir_name = make_simulated_dataset(
        pres_dataset_name,
        output_dir,
        dataset_size,
        seed,
        input_shape,
        input_shape,
        model_path = mask_model_path
    )

    dataset_path = output_dir.__str__() + "/" + dir_name

    split_dataset(dataset_path, val_split)


    print("_________________\n")
    print("Training EventNet")
    print("_________________") 
    train(
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
        using_jupyter
    )

    print("")
    print("Done...\n")


# ---------------------------------- #
# Commands for SageMaker Entrypoints #
# ---------------------------------- #

@cli.command   ('train')
@click.option  ('-n', '--model_name'   , type=str  , default="aws_model" , help="Name of model"  )
@click.option  ('-t', '--model_type'   , type=str  , default="unet"      , help="Type of model"  )
@click.option  ('-e', '--epochs'       , type=int  , default=1           , help=epochs_help      )
@click.option  ('-t', '--input_shape'  , type=int  , default=512         , help=inputshape_help  )
@click.option  ('-f', '--filters'      , type=int  , default=16          , help=filters_help     )
@click.option  ('-b', '--batch_size'   , type=int  , default=8           , help=batchsize_help   )
@click.option  ('-l', '--learning_rate', type=float, default=0.001       , help=learningrate_help)
def sagemaker_train_wrapper(
    model_name,
    model_type,
    epochs,
    input_shape,
    filters,
    batch_size,
    learning_rate
):

    """
    SageMaker compatible function to train a U-Net, ResNet, or Classification EventNet model.
    """

    import json

    from os import system

    from src.training import train

    root_dir    = "/opt/ml"                          # SageMaker expects things to happen here.

    input_dir   = root_dir   + "/input"              # SageMaker uploads things here.
    dataset_dir = input_dir  + "/data/training"
    config_dir  = input_dir  + "/config"

    output_dir     = root_dir   + "/output/data"     # SageMaker takes optional extras from here.
    logs_dir       = output_dir + "/logs"            # A file called failure must hold failing errors in /output
    checkpoint_dir = output_dir + "/best_checkpoint"

    try:
        system(f'mkdir {output_dir} {checkpoint_dir} {logs_dir}')
    except Exception as e: 
        print(f'Caught {type(e)}: {e}. Continuing Anyway...')

    try:
        json_file = open(config_dir + "/hyperparameters.json", "r")
        hyperparameters = json.load(json_file)

        print(f'Read hyperparameters: {hyperparameters}')

        model_type    = str(hyperparameters['model_type'])
        epochs        = int(hyperparameters['epochs'])
        filters       = int(hyperparameters['filters'])
        batch_size    = int(hyperparameters['batch_size'])
        learning_rate = float(hyperparameters['learning_rate'])

        system(f'cp -r {config_dir} {output_dir}')

    except Exception as e:
        print(f'Caught {type(e)}: {e}\n Using default hyperparameters.')

    train(
        model_name,
        dataset_dir,
        model_type,
        input_shape,
        epochs,
        filters,
        batch_size,
        learning_rate,
        using_aws = True,
        logs_dir  = logs_dir 
    )


# TODO: Will require implementing a basic server to SageMaker's spec
@cli.command('serve')
def sagemaker_server_wrapper():

    """
    Masks events in the given wrapped interferogram using a tensorflow model and plots it, with the option to save.

    ARGS:\n
    model_path       path to model to mask with.\n
    pres_model_path  path to model that predicts whether there is an event.\n
    image_path       path to wrapped interferogram to mask.\n
    """

    import os
    import json

    from io import BytesIO

    import numpy as np
    import flask
    from   flask import request
    from tensorflow.keras.models import load_model

    from src.inference import mask_with_model
    from src.io import get_image_array

    ping_test_image = 'tests/test_image.tif'
    mask_model_path = '/opt/ml/models/mask_model'
    pres_model_path = '/opt/ml/models/pres_model'
            
    mask_model = load_model(mask_model_path)
    pres_model = load_model(pres_model_path)

    app = flask.Flask(__name__)

    @app.route('/ping', methods=['GET'])
    def ping():    

        """
        SageMaker expects a ping endpoint to test the server.
        This confirms that the container is working.
        """

        try:
            
            image = get_image_array(ping_test_image)

            masked, presence_mask, pres_vals = mask_with_model(mask_model, pres_model, image, tile_size=512)

            if np.mean(presence_mask) > 0.0:
                presense = True
            else:
                presense = False

            response = {
                "presense": presense,
            }

            return flask.Response(response=json.dumps(response), status=200, mimetype='application/json')
        
        except Exception as e:
            print(f'Caught {type(e)}: {e}')
            
            return flask.Response(response=json.dumps({'error': str(e)}), status=500, mimetype='application/json')    


    @app.route('/invocations', methods=['POST'])
    def invocations():

        """
        Main route for inference through SageMaker.
        """

        status = 200

        try:
            byteImg = BytesIO(request.get_data())
            with open("image.tif", "wb") as f:
                f.write(byteImg.getbuffer())

            image = get_image_array("image.tif")

            masked, presence_mask, presence_vals = mask_with_model(mask_model_path, pres_model_path, image, tile_size=512)

            if np.mean(presence_mask) > 0.0:
                presense = True
            else:
                presense = False

            result = {
                "presense": presense,
            }
            
        except Exception as err:

            status = 500

            result = {
                "error": str(err),
                "presense": 2
            }
                    
            print("invocations() err: {}".format(err))

        return flask.Response(response=json.dumps(result), status=status, mimetype='application/json')


    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)     


if __name__ == '__main__':
    cli()

