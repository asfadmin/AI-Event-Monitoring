"""
 Created By:   Andrew Player
 File Name:    training.py
 Date Created: 01-25-2021
 Description:  Contains the code for training models
"""

import os

from math   import ceil
from typing import Any

import numpy as np

from src.architectures.unet     import create_unet
from src.architectures.resnet   import create_resnet
from src.architectures.eventnet import create_eventnet

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils     import Sequence


class DataGenerator(Sequence):

    """
    Dataset Generator for sequencially passing files from storange into the model.
    """

    def __init__(self, file_list, path, tile_size, crop_size, train_feature, output_feature):
        
        self.file_list = file_list
        self.path = path
        self.tile_size = tile_size
        self.crop_size = crop_size
        self.train_feature = train_feature
        self.output_feature = output_feature
        self.output_shape = (1, 1) if train_feature == 'mask' else (1, crop_size, crop_size, 1)
        self.on_epoch_end()


    def __len__(self):
        
        """
        The amount of files in the dataset.
        """

        return int(len(self.file_list))


    def __getitem__(self, index):
        
        """
        Returns the set of inputs and their corresponding truths.
        """

        # Generate indexes of the batch
        indexes = self.indexes[index:(index + 1)]

        # single file
        file_list_temp = [self.file_list[k] for k in indexes]

        # Set of X_train and y_train
        X, y = self.__data_generation(file_list_temp)

        return X, y


    def on_epoch_end(self):
        
        self.indexes = np.arange(len(self.file_list))


    def __data_generation(self, file_list_temp):
        
        """
        Returns individual pairs of inputs and their corresponding truths.
        """

        # Generate data
        for ID in file_list_temp:
            data_loc = os.path.join(self.path, ID)
            X = np.load(data_loc)
            x = X[self.train_feature].reshape((1, self.tile_size, self.tile_size, 1))
            y = X[self.output_feature].reshape(self.output_shape)

        return x, y


def train(
    model_name:    str,
    dataset_path:  str,
    model_type:    str,
    input_shape:   int   = 1024,
    num_epochs:    int   = 10,
    num_filters:   int   = 16,
    batch_size:    int   = 64,
    learning_rate: float = 0.001,
    use_wandb:     bool  = False
) -> Any:

    """
    Trains a model.

    Parameters:
    -----------
    model_name : str
        The name for the saved model.
    train_path : str
        The path to the training dataset.
    test_path : str
        The path to the validation dataset.
    input_shape : int, Optional
        The input shape of the model: (1, input_shape, input_shape, 1).
    num_epochs : int, Optional
        The number of epochs. This is the number of times the model trains over
        the training dataset.
    num_filters : int, Optional
        The base number of filters for the convolutional layers.
    batch_size : int, Optional
        The number of samples to train on before updating the model weights. For
        the best accuracy, this should be 0; however, higher values will lead to
        much quicker training.
    learning_rate : float, Optional
        The rate at which the model will update weights in response to estimated error.
    dropout : float, Optional
        The percentage of network nodes that will be randomly dropped out during the
        training process. This helps to mitigate overfitting.

    Returns:
    --------
    history : any
        A history object containing the loss at each epoch of training.
    """

    if use_wandb:

        import wandb
        from wandb.keras import WandbCallback

        wandb.init(
            project="InSAR Event Monitor",
            config = {
                "learning_rate": learning_rate,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "filters": num_filters,
                "tile_size": input_shape,
                "dataset": dataset_path
            }
        )

    train_feature = 'mask' if model_type == 'eventnet' else 'wrapped'
    output_feature = 'presence' if model_type == 'eventnet' else 'mask'

    train_path = dataset_path + '/train'
    test_path  = dataset_path + '/validation'

    all_training_files   = os.listdir(train_path)
    all_validation_files = os.listdir(test_path)

    training_partition   = [item for item in all_training_files   if "synth" in item or "sim" in item or "real" in item]
    validation_partition = [item for item in all_validation_files if "synth" in item or "sim" in item or "real" in item]

    training_generator = DataGenerator(training_partition, train_path, input_shape, input_shape, train_feature, output_feature)
    val_generator      = DataGenerator(validation_partition, test_path, input_shape, input_shape, train_feature, output_feature)

    if model_type == 'eventnet':
        model = create_eventnet(
            model_name  = model_name,
            tile_size   = input_shape,
            num_filters = num_filters,
            label_count = 1
        )
    elif model_type == 'unet':
        model = create_unet(
            model_name    = model_name,
            tile_size     = input_shape,
            num_filters   = num_filters,
            learning_rate = learning_rate
        )
    elif model_type == 'resnet':
        model = create_resnet(
            model_name    = model_name,
            tile_size     = input_shape,
            num_filters   = num_filters,
            learning_rate = learning_rate
        )
    else:
        print("Invalid Model Type!")
        return

    model.summary()

    early_stopping = EarlyStopping(
        monitor  = 'loss',
        patience = 2,
        verbose  = 1
    )

    checkpoint = ModelCheckpoint(
        filepath       = 'models/checkpoints/' + model_name,
        monitor        = 'val_loss',
        mode           = 'min',
        verbose        = 1,
        save_best_only = True
    )

    training_samples   = len(training_partition)
    validation_samples = len(validation_partition)

    training_steps     = ceil(training_samples / batch_size)
    validation_steps   = ceil(validation_samples / batch_size)

    if use_wandb:
        history = model.fit(
            training_generator,
            epochs           = num_epochs,
            validation_data  = val_generator,
            batch_size       = batch_size,
            steps_per_epoch  = training_steps,
            validation_steps = validation_steps,
            callbacks        = [checkpoint, early_stopping, WandbCallback()]
        )

    else:
        history = model.fit(
            training_generator,
            epochs           = num_epochs,
            validation_data  = val_generator,
            batch_size       = batch_size,
            steps_per_epoch  = training_steps,
            validation_steps = validation_steps,
            callbacks        = [checkpoint, early_stopping]
        )

    model.save("models/" + model_name)

    return history
