"""
 Created By:   Andrew Player
 File Name:    eventnet.py
 Date Created: September 2022
 Description:  Basic convolutional model, for now, for classifying the images.
"""

from tensorflow                  import Tensor
from tensorflow.keras.layers     import Conv2D, Input, LeakyReLU, Flatten, Dense
from tensorflow.keras.models     import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras            import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


def conv2d_block(
    input_tensor: Tensor ,
    num_filters:  int
) -> Tensor:

    """
    2D-Convolution Block for encoding / generating feature maps.
    """

    x = Conv2D(
        filters            = num_filters,
        kernel_size        = (3, 3)     ,
        kernel_initializer = 'random_normal',
        padding            = 'same'     ,
    )(input_tensor)

    x = LeakyReLU()(x)

    return x


def create_eventnet(
    model_name:    str   = 'model',
    tile_size:     int   = 128    ,
    num_filters:   int   = 32     ,
    label_count:   int   = 1
) -> Model:

    """
    Creates a basic convolutional network
    """

    input = Input(shape = (tile_size, tile_size, 1))

    # # --------------------------------- #
    # # Feature Map Generation            #
    # # --------------------------------- #

    c1 = conv2d_block(input, num_filters * 1)
    c2 = conv2d_block(c1, 1)

    # # --------------------------------- #
    # # Dense Hidden Layer                #
    # # --------------------------------- #

    f0 = Flatten()(c2)
    d0 = Dense(1024, activation='relu')(f0)

    # --------------------------------- #
    # Output Layer                      #
    # --------------------------------- #

    output = Dense(label_count, activation='sigmoid')(d0)

    # --------------------------------- #
    # Model Creation and Compilation    #
    # --------------------------------- #

    model = Model(
        inputs  = [input ],
        outputs = [output], 
        name    = model_name
    )

    model.compile(
        optimizer = SGD(learning_rate=0.005),
        loss      = 'mean_squared_error',
        metrics   = ['mean_squared_error'],
    )

    return model