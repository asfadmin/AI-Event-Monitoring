"""
 Summary
 -------
 Contains a network for sar classification
 
 Notes
 -----
 Created By: Andrew Player
"""


from tensorflow import Tensor
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Input,
    concatenate,
    Activation,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)


def conv2d_block(
    input_tensor: Tensor, num_filters: int, kernel_size: int = 3, strides: int = 1
) -> Tensor:
    """
    UNET style 2D-Convolution Block for encoding / generating feature maps.
    """

    x = Conv2D(
        filters=num_filters,
        kernel_size=(kernel_size, kernel_size),
        strides=(strides, strides),
        kernel_initializer="he_normal",
        padding="same",
        activation="relu",
    )(input_tensor)

    x = Conv2D(
        filters=num_filters,
        kernel_size=(kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
        activation="relu",
    )(x)

    return x


def transpose_block(
    input_tensor: Tensor,
    concat_tensor: Tensor,
    num_filters: int,
    kernel_size: int = 3,
) -> Tensor:
    """
    Learned Upscaling for decoding
    """

    x = Conv2DTranspose(
        filters=num_filters, kernel_size=(2, 2), strides=(2, 2), padding="same"
    )(input_tensor)

    x = conv2d_block(x, num_filters)

    y = concatenate([x, concat_tensor])

    return y


def create_unet(
    model_name: str = "model",
    tile_size: int = 512,
    num_filters: int = 64,
    learning_rate: float = 1e-4,
) -> Model:
    """
    Creates a U-Net style model
    """

    input = Input(shape=(tile_size, tile_size, 1))

    # --------------------------------- #
    # Feature Map Generation            #
    # --------------------------------- #

    c1 = conv2d_block(input, num_filters * 1, strides=2, kernel_size=3)
    # c1 = Dropout(0.1)(c1)
    c2 = conv2d_block(c1, num_filters * 2, strides=2, kernel_size=3)
    # c2 = Dropout(0.1)(c2)
    c3 = conv2d_block(c2, num_filters * 4, strides=2, kernel_size=3)
    # c3 = Dropout(0.1)(c3)
    c4 = conv2d_block(c3, num_filters * 8, strides=2, kernel_size=3)
    # c4 = Dropout(0.1)(c4)
    c5 = conv2d_block(c4, num_filters * 16, strides=2, kernel_size=1)
    # c5 = Dropout(0.1)(c5)

    # --------------------------------- #
    # Learned Upscaling                 #
    # --------------------------------- #

    u8 = transpose_block(c5, c4, num_filters * 8)
    u9 = transpose_block(u8, c3, num_filters * 4)
    u10 = transpose_block(u9, c2, num_filters * 2)
    u11 = transpose_block(u10, c1, num_filters * 1)
    u12 = transpose_block(u11, input, num_filters * 1)

    # u12 = Conv2DTranspose(
    #     filters      = num_filters,
    #     kernel_size  = (2, 2),
    #     strides      = (2, 2),
    #     padding      = 'same'
    # )(u11)

    # --------------------------------- #
    # Output Layer                      #
    # --------------------------------- #

    output = Conv2D(name="last_layer", kernel_size=(1, 1), filters=1, padding="same")(
        u12
    )

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
