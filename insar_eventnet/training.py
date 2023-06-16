"""
 Summary
 -------
 Functions for training models.

 Notes
 -----
 Created by Andrew Player.
"""

import os
import sys

from math import ceil
from typing import Any
from datetime import datetime

import numpy as np

from insar_eventnet.architectures.unet import create_unet
from insar_eventnet.architectures.unet3d import create_unet3d
from insar_eventnet.architectures.resnet import create_resnet
from insar_eventnet.architectures.eventnet import create_eventnet

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):

    """
    Dataset Generator for sequencially passing files from storange into the model.
    """

    def __init__(
        self, file_list, path, tile_size, crop_size, train_feature, output_feature
    ):
        self.file_list = file_list
        self.path = path
        self.tile_size = tile_size
        self.crop_size = crop_size
        self.train_feature = train_feature
        self.output_feature = output_feature
        self.output_shape = (
            (1, 1) if train_feature == "mask" else (1, crop_size, crop_size, 1)
        )
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
        indexes = self.indexes[index : (index + 1)]

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
            # x = X[self.train_feature].transpose().swapaxes(0, 1)
            # x = x.reshape((1, *x.shape, 1))
            x = X[self.train_feature].reshape(1, self.tile_size, self.tile_size, 1)
            y = X[self.output_feature].reshape(self.output_shape)

        return x, y


def print_model_info(
    model_name,
    model,
    history,
    input_shape,
    model_type,
    num_epochs,
    num_filters,
    batch_size,
    learning_rate,
):
    losses = history.history["loss"]
    val_losses = history.history["val_loss"]

    min_loss = min(losses)
    min_val_loss = min(val_losses)

    min_loss_at = losses.index(min_loss)
    min_val_loss_at = val_losses.index(min_val_loss)

    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(" " + x))
    summary_string = "\n".join(summary_list[1:])

    print(
        "\n",
        f"Model Name:    {model_name}\n",
        f"Model Type:    {model_type}\n",
        f"Date and Time: {datetime.utcnow()}\n",
        "\n",
        f"Input Shape:   {input_shape}\n",
        f"Epochs:        {num_epochs}\n",
        f"Filters:       {num_filters}\n",
        f"Batch Size:    {batch_size}\n",
        f"Learning Rate: {learning_rate}\n",
        "\n",
        f"Minimum Training Loss:   {min_loss: 0.15f} at epoch {min_loss_at}.\n",
        f"Minimum Validation Loss: {min_val_loss: 0.15f} at epoch {min_val_loss_at}.\n"
        "\n",
        f"All Training Losses:\n {losses}\n",
        "\n",
        f"All Validation Losses:\n {val_losses}\n",
        "\n",
        f"Model Parameters from History:\n {history.params}\n",
        "\n",
        f"Model Summary:\n{summary_string}\n",
    )


def train(
    model_name: str,
    dataset_path: str,
    model_type: str,
    input_shape: int = 1024,
    num_epochs: int = 10,
    num_filters: int = 16,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    use_wandb: bool = False,
    using_aws: bool = False,
    using_jupyter: bool = False,
    logs_dir: str = "",
) -> Any:
    """
    Trains a model.

    Parameters
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

    Returns
    --------
    history : any
        A history object containing the loss at each epoch of training.
    """

    failure_file = "failure"
    if using_aws:
        losses_file = logs_dir + "/losses.npz"
        failure_file = "/opt/ml/output/" + failure_file
        model_path = "/opt/ml/model"
        checkpoint_path = "/opt/ml/output/data/best_checkpoint/" + model_name + "_best"
    else:
        model_path = "./data/output/models/" + model_name
        checkpoint_path = "./data/output/models/checkpoints/" + model_name

    results_file = model_name + "_results.txt"
    if logs_dir != "":
        results_file = logs_dir + "/" + results_file

    # try:

    if use_wandb:
        import wandb
        from wandb.keras import WandbCallback

        wandb.init(
            project="InSAR Event Monitor",
            config={
                "learning_rate": learning_rate,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "filters": num_filters,
                "tile_size": input_shape,
                "dataset": dataset_path,
            },
        )

    train_feature = "mask" if model_type == "eventnet" else "wrapped"
    output_feature = "presence" if model_type == "eventnet" else "mask"

    if model_type == "unet3d":
        train_feature = "phases"
        output_feature = "mask"

    train_path = dataset_path + "/train"
    test_path = dataset_path + "/validation"

    all_training_files = os.listdir(train_path)
    all_validation_files = os.listdir(test_path)

    def filename_check(x):
        "synth" in x or "sim" in x or "real" in x

    training_partition = [item for item in all_training_files if filename_check(item)]
    validation_partition = [
        item for item in all_validation_files if filename_check(item)
    ]

    training_samples = len(training_partition)
    validation_samples = len(validation_partition)

    training_steps = ceil(training_samples / batch_size)
    validation_steps = ceil(validation_samples / batch_size)

    training_generator = DataGenerator(
        training_partition,
        train_path,
        input_shape,
        input_shape,
        train_feature,
        output_feature,
    )
    val_generator = DataGenerator(
        validation_partition,
        test_path,
        input_shape,
        input_shape,
        train_feature,
        output_feature,
    )

    if model_type == "eventnet":
        model = create_eventnet(
            model_name=model_name,
            tile_size=input_shape,
            num_filters=num_filters,
            label_count=1,
        )
    elif model_type == "unet":
        model = create_unet(
            model_name=model_name,
            tile_size=input_shape,
            num_filters=num_filters,
            learning_rate=learning_rate,
        )
    elif model_type == "unet3d":
        model = create_unet3d()
    elif model_type == "resnet":
        model = create_resnet(
            model_name=model_name,
            tile_size=input_shape,
            num_filters=num_filters,
            learning_rate=learning_rate,
        )
    else:
        SystemExit(
            f'Invalid model type! Expected "unet", "resnet", or "eventnet" but got {model_type}.'
        )

    early_stopping = EarlyStopping(monitor="loss", patience=2, verbose=1)

    best_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
    )

    model.summary()

    callbacks = [best_checkpoint, early_stopping]

    if use_wandb:
        callbacks.append(WandbCallback())

    history = model.fit(
        training_generator,
        epochs=num_epochs,
        validation_data=val_generator,
        batch_size=batch_size,
        steps_per_epoch=training_steps,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    if using_aws:
        np.savez(
            losses_file,
            loss=history.history["loss"],
            val_loss=history.history["val_loss"],
        )

    model.save(model_path)

    # except Exception as e:

    #     print(f'Error training model:\nCaught {type(e)}: {e}')

    #     # if not using_jupyter:

    #     #     with open(failure_file, "w") as sys.stdout:

    #     #         print(f'Caught {type(e)}: {e}')

    #     #     sys.exit(1)

    if not using_jupyter:
        try:
            with open(results_file, "w") as sys.stdout:
                print_model_info(
                    model_name,
                    model,
                    history,
                    input_shape,
                    model_type,
                    num_epochs,
                    num_filters,
                    batch_size,
                    learning_rate,
                )

        except Exception as e:
            print(f"Error creating results log file:\nCaught {type(e)}: {e}")

    return model, history
