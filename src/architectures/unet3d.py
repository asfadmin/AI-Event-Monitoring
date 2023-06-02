"""
 Created By:   Andrew Player
 File Name:    model.py
 Date Created: 01-25-2021
 Description:  Contains a network for sar classification
"""


from tensorflow import Tensor
from tensorflow.keras.layers import (
    Input,
    concatenate,
    Activation,
    Dropout,
    Conv3D,
    Conv3DTranspose,
    MaxPooling3D,
    AveragePooling3D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)


def conv3d_block(
    input_tensor: Tensor,
    num_filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    strides_up: int = 1,
) -> Tensor:
    """
    UNET style 3D-Convolution Block for encoding / generating feature maps.
    """

    x = Conv3D(
        filters=num_filters,
        kernel_size=(kernel_size, kernel_size, kernel_size),
        strides=(strides, strides, strides),
        kernel_initializer="he_normal",
        padding="same",
        activation="relu",
        data_format="channels_last",
    )(input_tensor)

    x = Conv3D(
        filters=num_filters,
        kernel_size=(kernel_size, kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
        activation="relu",
        data_format="channels_last",
    )(x)

    return x


def transpose_block(
    input_tensor: Tensor,
    concat_tensor: Tensor,
    num_filters: int,
    kernel_size: int = 3,
    strides_up: int = 2,
) -> Tensor:
    """
    Learned Upscaling for decoding
    """

    x = Conv3DTranspose(
        filters=num_filters,
        kernel_size=(2, 2, 2),
        strides=(2, 2, 2),
        padding="same",
        data_format="channels_last",
    )(input_tensor)

    x = conv3d_block(x, num_filters)

    y = concatenate([x, concat_tensor], axis=-1)

    return y


def create_unet3d(
    model_name: str = "model",
    tile_size: int = 512,
    temporal_steps: int = 16,
    num_filters: int = 16,
    learning_rate: float = 1e-4,
) -> Model:
    """
    Creates a 3D U-Net style model
    """

    input = Input(shape=(tile_size, tile_size, temporal_steps, 1))

    # --------------------------------- #
    # Feature Map Generation            #
    # --------------------------------- #

    c1 = conv3d_block(input, num_filters * 1, strides=2, kernel_size=3)
    c2 = conv3d_block(c1, num_filters * 2, strides=2, kernel_size=3)
    c3 = conv3d_block(c2, num_filters * 4, strides=2, kernel_size=3)
    c4 = conv3d_block(c3, num_filters * 8, strides=2, kernel_size=3)
    c5 = conv3d_block(c4, num_filters * 16, strides=1, kernel_size=3)

    # --------------------------------- #
    # Learned Upscaling                 #
    # --------------------------------- #

    u8 = transpose_block(c5, c3, num_filters * 8, strides_up=2)
    u9 = transpose_block(u8, c2, num_filters * 4, strides_up=2)
    u10 = transpose_block(u9, c1, num_filters * 2, strides_up=2)
    u11 = transpose_block(u10, input, num_filters * 1, strides_up=2)

    # u12 = Conv3DTranspose(
    #     filters      = num_filters,
    #     kernel_size  = (2, 2, 2),
    #     strides      = (2, 2, 2),
    #     padding      = 'same',
    #     data_format  = "channels_last"
    # )(u11)

    # --------------------------------- #
    # Output Layer                      #
    # --------------------------------- #

    output = Conv3D(
        name="last_layer",
        kernel_size=(1, 1, 1),
        strides=(1, 1, 1),
        filters=1,
        padding="same",
        data_format="channels_last",
    )(u11)

    output = Activation("linear", dtype="float32")(output)

    # --------------------------------- #
    # Model Creation and Compilation    #
    # --------------------------------- #

    model = Model(inputs=[input], outputs=[output], name=model_name)

    # TODO: Test huber loss, and maybe some others as well, and binary_crossentropy for the classifier. Maybe test with SGD instead of Adam, as well.
    model.compile(
        loss="huber",
        metrics=["mean_squared_error", "mean_absolute_error"],
        optimizer=Adam(learning_rate=learning_rate),
    )

    return model
