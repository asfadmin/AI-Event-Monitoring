"""
 Summary
 -------
 Contains a network for sar classification

 References
 ----------
 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7041455/

 Notes
 -----
 Created By: Andrew Player
"""


from tensorflow import Tensor, keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

policy = keras.mixed_precision.Policy("mixed_float16")
keras.mixed_precision.set_global_policy(policy)


def res_block(input_tensor: Tensor, num_filters: int) -> Tensor:
    """
    2D-Convolution Block with a connecting convolution for short-term memory.
    """

    c1 = layers.Conv2D(
        filters=num_filters,
        kernel_size=(1, 1),
        kernel_initializer="he_normal",
        padding="same",
    )(input_tensor)

    c2 = layers.Conv2D(
        filters=num_filters,
        kernel_size=(3, 3),
        kernel_initializer="he_normal",
        padding="same",
    )(input_tensor)
    c2 = layers.LeakyReLU()(c2)

    c3 = layers.Conv2D(
        filters=num_filters,
        kernel_size=(3, 3),
        kernel_initializer="he_normal",
        padding="same",
    )(c2)
    c3 = layers.LeakyReLU()(c3)

    return layers.concatenate([c3, c1])


def res4_block(input_tensor: Tensor, num_filters: int) -> Tensor:
    """
    Sequential block of 4 Residual Convolution Blocks
    """

    r1 = res_block(input_tensor=input_tensor, num_filters=num_filters)

    r2 = res_block(input_tensor=r1, num_filters=num_filters)

    r3 = res_block(input_tensor=r2, num_filters=num_filters)

    r4 = res_block(input_tensor=r3, num_filters=num_filters)

    return r4


def full_block(input_tensor: Tensor, num_filters: int) -> Tensor:
    """
    Sequential Block with a 2D-Convolution into MaxPooling, followed by
    4 Residual Convolution Blocks.
    """

    c1 = layers.Conv2D(
        filters=num_filters * 2,
        kernel_size=(3, 3),
        kernel_initializer="he_normal",
        padding="same",
    )(input_tensor)

    m1 = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(c1)

    r1 = res4_block(input_tensor=m1, num_filters=num_filters * 2)

    return r1


def create_resnet(
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

    input = layers.Input(shape=(tile_size, tile_size, 1))

    # --------------------------------- #
    # Res Blocks and Convolutions       #
    # --------------------------------- #

    c1 = layers.Conv2D(
        filters=num_filters,
        kernel_size=(3, 3),
        kernel_initializer="he_normal",
        padding="same",
    )(input)
    r1 = res4_block(c1, num_filters)

    c2 = layers.Conv2D(
        filters=num_filters * 2,
        kernel_size=(3, 3),
        kernel_initializer="he_normal",
        padding="same",
    )(r1)
    c2 = layers.MaxPooling2D((2, 2), strides=2)(c2)
    r2 = res4_block(c2, num_filters * 2)

    c3 = layers.Conv2D(
        filters=num_filters * 4,
        kernel_size=(3, 3),
        kernel_initializer="he_normal",
        padding="same",
    )(r2)
    c3 = layers.MaxPooling2D((2, 2), strides=2)(c3)
    r3 = res4_block(c3, num_filters * 4)

    c4 = layers.Conv2D(
        filters=num_filters * 8,
        kernel_size=(3, 3),
        kernel_initializer="he_normal",
        padding="same",
    )(r3)
    c4 = layers.MaxPooling2D((2, 2), strides=2)(c4)
    r4 = res4_block(c4, num_filters * 8)

    # --------------------------------- #
    # Deconvolutions                    #
    # --------------------------------- #

    c5 = layers.Conv2D(
        filters=num_filters * 16,
        kernel_size=(3, 3),
        kernel_initializer="he_normal",
        padding="same",
    )(r4)
    c5 = layers.LeakyReLU()(c5)
    dc1 = layers.Conv2DTranspose(
        filters=num_filters * 4, kernel_size=(2, 2), strides=(2, 2), padding="same"
    )(c5)

    c6 = layers.Conv2D(
        filters=num_filters * 8,
        kernel_size=(3, 3),
        kernel_initializer="he_normal",
        padding="same",
    )(dc1)
    c6 = layers.LeakyReLU()(c6)
    dc2 = layers.Conv2DTranspose(
        filters=num_filters * 2, kernel_size=(2, 2), strides=(2, 2), padding="same"
    )(c6)

    c7 = layers.Conv2D(
        filters=num_filters * 4,
        kernel_size=(3, 3),
        kernel_initializer="he_normal",
        padding="same",
    )(dc2)
    c7 = layers.LeakyReLU()(c7)
    dc3 = layers.Conv2DTranspose(
        filters=num_filters, kernel_size=(2, 2), strides=(2, 2), padding="same"
    )(c7)

    # ---------------------------------- #
    # Final Convolutions and Concat      #
    # ---------------------------------- #

    c8 = layers.Conv2D(
        filters=num_filters,
        kernel_size=(3, 3),
        kernel_initializer="he_normal",
        padding="same",
    )(dc3)

    c9 = layers.Conv2D(
        filters=num_filters,
        kernel_size=(1, 1),
        kernel_initializer="he_normal",
        padding="same",
    )(r1)

    c10 = layers.concatenate([c8, c9])

    output = layers.Conv2D(
        filters=1,
        kernel_size=(1, 1),
        kernel_initializer="he_normal",
        activation="linear",
        padding="same",
    )(c10)

    # --------------------------------- #
    # Mode Creation and Compilation     #
    # --------------------------------- #

    model = Model(inputs=[input], outputs=[output], name=model_name)

    model.compile(
        loss="mean_squared_error",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["mean_absolute_error"],
    )

    return model
