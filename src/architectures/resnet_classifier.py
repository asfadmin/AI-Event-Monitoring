"""
 Created By:   Andrew Player
 Summary
 -------
 Contains a network for sar classification
 References
 ----------
 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7041455/
"""


from tensorflow import Tensor
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Input,
    concatenate,
    BatchNormalization,
    Dense,
    Activation,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)


def conv_block(input_tensor: Tensor, num_filters: int, mul: int = 4) -> Tensor:
    """
    2D-Convolution Block with a connecting convolution for short-term memory.
    """

    c = Conv2D(
        filters=num_filters,
        kernel_size=(1, 1),
        kernel_initializer="he_normal",
        padding="same",
    )(input_tensor)
    c = BatchNormalization()(c)
    c1 = Activation("relu")(c)

    c2 = Conv2D(
        filters=num_filters,
        kernel_size=(3, 3),
        kernel_initializer="he_normal",
        padding="same",
    )(c1)
    c2 = BatchNormalization()(c2)
    c2 = Activation("relu")(c2)

    c3 = Conv2D(
        filters=num_filters * 4,
        kernel_size=(1, 1),
        kernel_initializer="he_normal",
        padding="same",
    )(c2)
    c3 = BatchNormalization()(c3)
    c3 = Activation("relu")(c3)

    return concatenate([c3, c])


def identity_block(input_tensor: Tensor, num_filters: int):
    c = Conv2D(
        filters=num_filters,
        kernel_size=(1, 1),
        kernel_initializer="he_normal",
        padding="same",
    )(input_tensor)
    c = BatchNormalization()(c)
    c1 = Activation("relu")(c)

    c2 = Conv2D(
        filters=num_filters,
        kernel_size=(3, 3),
        kernel_initializer="he_normal",
        padding="same",
    )(c1)
    c2 = BatchNormalization()(c2)
    c2 = Activation("relu")(c2)

    c3 = Conv2D(
        filters=num_filters,
        kernel_size=(1, 1),
        kernel_initializer="he_normal",
        padding="same",
    )(c2)
    c3 = BatchNormalization()(c3)
    c3 = Activation("relu")(c3)

    return concatenate([c3, input_tensor])


def create_resnetclassifier(
    model_name: str = "model",
    tile_size: int = 512,
    num_filters: int = 4,
    learning_rate: float = 1e-4,
) -> Model:
    """
    Creates a model for unwrapping 2D wrapped phase images.

    References
    -----------
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7041455/
    """

    input = Input(shape=(tile_size, tile_size, 1))

    # --------------------------------- #
    # Res Blocks and Convolutions       #
    # --------------------------------- #

    c1 = Conv2D(
        filters=num_filters,
        kernel_size=(7, 7),
        strides=(2, 2),
        kernel_initializer="he_normal",
        padding="same",
    )(input)
    a1 = Activation("relu")(c1)
    p1 = MaxPooling2D((3, 3), strides=(2, 2))(a1)

    c2 = conv_block(p1, num_filters)

    i1 = identity_block(c2, num_filters * 2)
    i2 = identity_block(i1, num_filters * 2)

    c3 = conv_block(i2, num_filters * 2)

    i3 = identity_block(c3, num_filters * 4)
    i4 = identity_block(i3, num_filters * 4)
    i5 = identity_block(i4, num_filters * 4)

    c4 = conv_block(i5, num_filters * 4)

    i6 = identity_block(c4, num_filters * 8)
    i7 = identity_block(i6, num_filters * 8)
    i8 = identity_block(i7, num_filters * 8)
    i9 = identity_block(i8, num_filters * 8)
    i10 = identity_block(i9, num_filters * 8)

    c5 = conv_block(i10, num_filters * 8)

    i11 = identity_block(c5, num_filters * 16)
    i12 = identity_block(i11, num_filters * 16)

    # --------------------------------- #
    # Output Layer                      #
    # --------------------------------- #

    d0 = Dense(1000, activation="relu")(i12)
    output = Dense(1, activation="sigmoid")(d0)

    # --------------------------------- #
    # Mode Creation and Compilation     #
    # --------------------------------- #

    model = Model(inputs=[input], outputs=[output], name=model_name)

    model.compile(
        loss="mean_squared_error",
        optimizer=SGD(learning_rate=learning_rate),
        metrics=["acc", "mean_absolute_error"],
    )

    return model
