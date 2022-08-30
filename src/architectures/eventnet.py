"""
 Created By:   Andrew Player
 File Name:    model.py
 Date Created: 01-25-2021
 Description:  Contains a network for sar classification
"""


from tensorflow                  import Tensor
from tensorflow.keras.layers     import Conv2D, Conv2DTranspose, MaxPooling2D, Input, concatenate
from tensorflow.keras.layers     import Flatten, Dense
from tensorflow.keras.models     import Model
from tensorflow.keras.optimizers import Adam 


def conv2d_block(
    input_tensor: Tensor ,
    num_filters:  int
) -> Tensor:

    """
    UNET style 2D-Convolution Block for encoding / generating feature maps.
    """

    x = Conv2D(
        filters            = num_filters,
        kernel_size        = (3, 3)     ,
        kernel_initializer = 'he_normal',
        padding            = 'same',
        activation         = 'relu'
    )(input_tensor)

    x = Conv2D(
        filters            = num_filters,
        kernel_size        = (3, 3)     ,
        kernel_initializer = 'he_normal',
        padding            = 'same',
        activation         = 'relu'
    )(x)

    return x


def transpose_block(
    input_tensor:  Tensor,
    concat_tensor: Tensor,
    num_filters:   int
) -> Tensor:

    """
    Learned Upscaling for decoding
    """

    x = Conv2DTranspose(
        filters      = num_filters,
        kernel_size  = (2, 2),
        strides      = (2, 2),
        padding      = 'same'
    )(input_tensor)
    
    x = concatenate([x, concat_tensor])
    
    y = conv2d_block(x, num_filters)

    return y


def create_eventnet(
    model_name:    str   = 'model',
    tile_size:     int   = 1024   ,
    num_filters:   int   = 16     ,
    learning_rate: float = 1e-4
) -> Model:

    """
    Creates a U-Net style model
    """

    input = Input(shape = (tile_size, tile_size, 1))


    # --------------------------------- #
    # Feature Map Generation            #
    # --------------------------------- #

    c1 = conv2d_block(input, 25)
    m1 = MaxPooling2D((2, 2), strides=2)  (c1)


    # --------------------------------- #
    # Dense Classification Layer        #
    # --------------------------------- #

    f0 = Flatten()(m1)
    d0 = Dense(100, 'sigmoid')(f0)


    # --------------------------------- #
    # Output Layer                      #
    # --------------------------------- #

    output = Dense(1, 'sigmoid')(d0)


    # --------------------------------- #
    # Model Creation and Compilation    #
    # --------------------------------- #

    model = Model(
        inputs  = [input ],
        outputs = [output], 
        name    = model_name
    )

    model.compile(
        loss      = 'mean_squared_error',
        metrics   = ['mean_squared_error'],
        optimizer = Adam(learning_rate = learning_rate)
    )

    return model
