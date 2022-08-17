"""
 Created By:  Andrew Player
 File Name:   aievents.py
 Description: CLI Interface
"""

import click
from src.config import REAL_DIR, SYNTHETIC_DIR


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


@cli.command   ('make-synthetic-dataset')
@click.argument('name'              , type=str                                                                    )
@click.argument('amount'            , type=int                        , default=1                                 )
@click.option  ('-t', '--tile_size' , type=int                        , default=1024         , help=tilesize_help )
@click.option  ('-d', '--output_dir', type=click.Path(file_okay=False), default=SYNTHETIC_DIR, help=outputdir_help)
@click.option  ('-s', '--seed'      , type=int                        , default=None         , help=seed_help     )
@click.option  ('-c', '--crop_size' , type=int                        , default=0            , help=cropsize_help )
@click.option  ('-s', '--split'     , type=float                      , default=0.0          , help=split_help    )
@click.option  ('-a', '--amplitude' , type=float                      , default=128.0        , help=amplitude_help)
def make_synthetic_dataset_wrapper(name, amount, tile_size, output_dir, seed, crop_size, split, amplitude):

    """
    Create a randomly generated synthetic dataset of wrapped interferograms and their corresponding event-masks.

    ARGS:\n
    name        Name of dataset. Seed is appended.\n
                <name>_seed<seed>\n
    amount      Number of synthetic interferograms created.\n
    """

    from src.io import make_synthetic_dataset, split_dataset


    max_amp      =  amplitude
    min_amp      = -max_amp
    min_x_stddev =  tile_size / 32
    max_x_stddev =  tile_size / 4
    min_y_stddev =  tile_size / 32
    max_y_stddev =  tile_size / 4
    min_x_mean   =  max_x_stddev
    max_x_mean   =  tile_size - max_x_stddev
    min_y_mean   =  max_y_stddev
    max_y_mean   =  tile_size - max_y_stddev

    name, count, dir_name = make_synthetic_dataset(
        name,
        output_dir,
        amount,
        seed,
        tile_size,
        crop_size,
        min_amp,
        max_amp,
        min_x_mean,
        max_x_mean,
        min_y_mean,
        max_y_mean,
        min_x_stddev,
        max_x_stddev,
        min_y_stddev,
        max_y_stddev,
    )

    num_train, num_validation = split_dataset(output_dir.__str__() + '/' + dir_name, split)

    print(f"\nCreated dataset with seed: {seed}, and {count} entries. Saved to {dir_name}\n")
    print(f"Dataset was split into train and validation sets of size {num_train} and {num_validation}.\n")


@cli.command   ('make-simulated-dataset')
@click.argument('name'              , type=str                                                                    )
@click.argument('amount'            , type=int                        , default=1                                 )
@click.option  ('-t', '--tile_size' , type=int                        , default=1024         , help=tilesize_help )
@click.option  ('-d', '--output_dir', type=click.Path(file_okay=False), default=SYNTHETIC_DIR, help=outputdir_help)
@click.option  ('-s', '--seed'      , type=int                        , default=None         , help=seed_help     )
@click.option  ('-s', '--split'     , type=float                      , default=0.0          , help=split_help    )
def make_simulated_dataset_wrapper(name, amount, tile_size, output_dir, seed, split):

    """
    Create a randomly generated simulated dataset of wrapped interferograms and their corresponding event-masks.

    ARGS:\n
    name        Name of dataset. Seed is appended.\n
                <name>_seed<seed>\n
    amount      Number of simulated interferograms created.\n
    """

    from src.io import make_simulated_dataset, split_dataset

    name, count, dir_name = make_simulated_dataset(
        name,
        output_dir,
        amount,
        seed,
        tile_size
    )

    num_train, num_validation = split_dataset(output_dir.__str__() + '/' + dir_name, split)

    print(f"\nCreated dataset with seed: {seed}, and {count} entries. Saved to {dir_name}\n")
    print(f"Dataset was split into train and validation sets of size {num_train} and {num_validation}.\n")


@cli.command   ('make-real-dataset')
@click.argument('name'               , type=str                                         )
@click.argument('dataset-path'       , type=str                                         )
@click.option  ('-t', '--tile_size'  , type=int  , default=1024    , help=tilesize_help )
@click.option  ('-c', '--crop_size'  , type=int  , default=0       , help=cropsize_help )
@click.option  ('-d', '--output_dir' , type=str  , default=REAL_DIR, help=outputdir_help)
@click.option  ('-s', '--split'      , type=float, default=0.0     , help=split_help    )
def dataset_from_products_wrapper(name, dataset_path, tile_size, output_dir, crop_size, split):

    """
    Create a dataset from real interferogran products.

    ARGS:\n
    name              Name of dataset.\n
    path-to-products  Path to the folder that contains the product folders.\n
    """

    from src.io import dataset_from_products, split_dataset

    size = dataset_from_products(
        name,
        dataset_path,
        output_dir,
        tile_size,
        crop_size,
    )

    num_train, num_validation = split_dataset(output_dir.__str__() + '/' + name, split)

    print(f"\nCreated dataset of size {size} from {dataset_path} at {output_dir.__str__() + '/' + name}.\n")
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


@cli.command   ('show')
@click.argument('file_path', type=click.Path())
def show_dataset_wrapper(file_path):

    """
    Show the wrapped interferogram and event-mask from a given dataset file (.npz).

    ARGS:\n
    file_path       path to the .npz file to show.\n
    """

    from src.gui import show_dataset
    from src.io import load_dataset

    mask, wrapped = load_dataset(file_path)
    show_dataset(mask, wrapped)


@cli.command ('show-random')
@click.option('-s', '--seed'     , type=int, default=0   , help=seed_help    )
@click.option('-t', '--tile_size', type=int, default=1024, help=tilesize_help)
@click.option('-c', '--crop_size', type=int, default=0   , help=cropsize_help)
def show_random_wrapper(seed, tile_size, crop_size):

    """
    Show a randomly generated synthetic wrapped interferogram along with an event-mask.
    """

    from src.gui import show_dataset
    from src.synthetic_interferogram import make_random_dataset

    mask, wrapped = make_random_dataset(size=tile_size, crop_size=crop_size, seed=seed)
    show_dataset(mask, wrapped)


@cli.command   ('train-model')
@click.argument('model_name'           , type=str                                         )
@click.argument('dataset_path'         , type=str                                         )
@click.option  ('-e', '--epochs'       , type=int  , default=10   , help=epochs_help      )
@click.option  ('-t', '--input_shape'  , type=int  , default=1024 , help=inputshape_help  )
@click.option  ('-f', '--filters'      , type=int  , default=16   , help=filters_help     )
@click.option  ('-b', '--batch_size'   , type=int  , default=32   , help=batchsize_help   )
@click.option  ('-d', '--dropout'      , type=float, default=0.2  , help=dropout_help     )
@click.option  ('-l', '--learning_rate', type=float, default=0.001, help=learningrate_help)
def train_model_wrapper(
    model_name,
    dataset_path,
    epochs,
    input_shape,
    filters,
    batch_size,
    dropout,
    learning_rate
):

    """
    Train a U-Net or ResNet style model.

    ARGS:\n
    model_name      name of the model to be trained.\n
    train_path      path to training data.\n
    test_path       path to validation data.\n
    """

    from src.training import train

    train(
        model_name,
        dataset_path,
        input_shape,
        epochs,
        filters,
        batch_size,
        learning_rate,
        dropout
    )


@cli.command   ('test-model')
@click.argument('model_path'       , type=str                                  )
@click.option  ('-s', '--seed'     , type=int, default=0   , help=seed_help    )
@click.option  ('-t', '--tile_size', type=int, default=1024, help=tilesize_help)
@click.option  ('-c', '--crop_size', type=int, default=0   , help=cropsize_help)
def test_model_wrapper(model_path, seed, tile_size, crop_size):

    """
    Predicts on a wrapped interferogram & event-mask pair and plots the results

    ARGS:\n
    model_path      path to model that should be tested.\n
    """

    from os import environ
    
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    from src.inference import test_model

    test_model(model_path, seed, tile_size, crop_size)


@cli.command   ('model-summary')
@click.argument('model_path', type=str)
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


@cli.command   ('mask')
@click.argument('model_path'          , type=str                                    )
@click.argument('image_path'          , type=str                                    )
@click.option  ('-c', '--crop_size'   , type=int  , default=0  , help=cropsize_help )
@click.option  ('-t', '--tile_size'   , type=int  , default=512, help=tilesize_help )
@click.option  ('-d', '--dest_path'   , type=str  , default="" , help=outputdir_help)
def mask(
    model_path,
    image_path, 
    crop_size, 
    tile_size,
    dest_path
):

    """
    Masks events in the given wrapped interferogram using a tensorflow model and plots it, with the option to save.

    ARGS:\n
    model_path      path to model to mask with.\n
    image_path      path to wrapped interferogram to mask.\n
    """

    from os import environ
    
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    from src.inference import mask_and_plot

    mask = mask_and_plot(
        model_path,
        image_path,
        tile_size,
        crop_size
    )

    if dest_path != "":
        
        from PIL import Image
        
        out = Image.fromarray(mask)
        out.save(dest_path)        


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


@cli.command ('simulate')
@click.option('-s', '--seed'     , type=int , default=0    , help=seed_help    )
@click.option('-t', '--tile_size', type=int , default=1024 , help=tilesize_help)
@click.option('-v', '--verbose'  , type=bool, default=False, help=""           )
def simulate_wrapper(seed, tile_size, verbose):

    """
    Show a randomly generated wrapped interferogram with simulated deformation, atmospheric turbulence, atmospheric topographic error, and incoherence masking.
    """

    from src.gui    import show_dataset
    from src.sarsim import gen_simulated_deformation

    masked, wrapped = gen_simulated_deformation(
        seed,
        tile_size,
        verbose
    )

    show_dataset(masked, wrapped)


if __name__ == '__main__':
    cli()
