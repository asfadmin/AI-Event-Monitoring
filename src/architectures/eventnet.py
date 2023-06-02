"""
 Created By:   Andrew Player
 File Name:    eventnet.py
 Date Created: September 2022
 Description:  Basic convolutional model, for now, for classifying the images.
"""

from tensorflow import Tensor
from tensorflow.keras.layers import (
    Conv2D,
    Input,
    LeakyReLU,
    Flatten,
    Dense,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import mixed_precision


policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)


def conv2d_block(
    input_tensor: Tensor, num_filters: int, kernel_size: int = 3, strides: int = 1
) -> Tensor:
    """
    2D-Convolution Block for encoding / generating feature maps.
    """

    x = Conv2D(
        filters=num_filters,
        kernel_size=(kernel_size, kernel_size),
        strides=(strides, strides),
        kernel_initializer="random_normal",
        padding="same",
    )(input_tensor)

    x = LeakyReLU()(x)

    return x


def create_eventnet(
    model_name: str = "model",
    tile_size: int = 512,
    num_filters: int = 32,
    label_count: int = 1,
    learning_rate: float = 0.005,
) -> Model:
    """
    Creates a basic convolutional network
    """

    input = Input(shape=(tile_size, tile_size, 1))

    # # --------------------------------- #
    # # Feature Map Generation            #
    # # --------------------------------- #

    c1 = conv2d_block(input, num_filters, kernel_size=7, strides=2)
    c2 = conv2d_block(c1, num_filters, kernel_size=3, strides=2)
    c3 = conv2d_block(c2, num_filters=1, kernel_size=1, strides=1)

    # # --------------------------------- #
    # # Dense Hidden Layer                #
    # # --------------------------------- #

    # TODO: Try Global Average Pooling

    g1 = GlobalAveragePooling2D(keepdims=True, data_format="channels_last")(c3)

    # f0 = Flatten()(c3)
    # d0 = Dense(512, activation='relu')(f0)

    # --------------------------------- #
    # Output Layer                      #
    # --------------------------------- #

    output = Dense(label_count, activation="sigmoid")(g1)

    # --------------------------------- #
    # Model Creation and Compilation    #
    # --------------------------------- #

    model = Model(inputs=[input], outputs=[output], name=model_name)

    model.compile(
        optimizer=SGD(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["acc", "mean_absolute_error"],
    )

    return model
