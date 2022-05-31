"""
 Created By:   Andrew Player
 File Name:    model.py
 Date Created: 01-25-2021
 Description:  Contains a network for sar classification
 Credits: https://arxiv.org/pdf/2010.13268.pdf
"""


from typing                      import Tuple

from src.nn_functions            import loss_fn_tf

from tensorflow                  import Tensor
from tensorflow.keras.layers     import Activation, AveragePooling2D, BatchNormalization, Bidirectional
from tensorflow.keras.layers     import Conv2D, Conv2DTranspose, Input, concatenate, Cropping2D
from tensorflow.keras.models     import Model
from tensorflow.keras.optimizers import Adam 


def conv2d_block(
    input_tensor: Tensor ,
    num_filters:  int
) -> Tensor:

    """
    UNET-ish 2D-Convolution Block for encoding / generating feature maps.
    """

    x = Conv2D(
        filters            = num_filters,
        kernel_size        = (3, 3)     ,
        kernel_initializer = 'he_normal',
        padding            = 'valid'
    )(input_tensor)

    x = Conv2D(
        filters            = num_filters,
        kernel_size        = (3, 3)     ,
        kernel_initializer = 'he_normal',
        padding            = 'valid'
    )(x)

    x = BatchNormalization()(x)
    
    x = Activation('relu')  (x)
    
    x = AveragePooling2D()  (x)

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
        filters = num_filters * 8,
        kernel  = (3, 3),
        strides = (2, 2),
        padding = 'valid'
    )(input_tensor)
    
    #TODO: Implement this
    x = Cropping2D(cropping=((0, 0), (0, 0)))(x)
    
    x = concatenate([x, concat_tensor])
    
    x = conv2d_block(x, num_filters * 8)

    return x


def create_unet(
    model_name:    str   = 'model',
    tile_size:     int   = 1024   ,
    num_filters:   int   = 16     ,
    dropout:       float = 0.2    ,
    learning_rate: float = 1e-4
) -> Model:

    """
    Creates a model combining a UNET style encoder-decoder with an LSTM.

    References:
    -----------
    https://arxiv.org/pdf/2010.13268.pdf
    """

    input = Input(shape = (tile_size, tile_size, 1))
    
    flat_shape = (tile_size / 8 , num_filters * 8)

    unflat_shape = (
        tile_size / 16,
        tile_size / 16,
        num_filters * 4
    )


    # --------------------------------- #
    # Feature Map Generation            #
    # --------------------------------- #

    c1 = conv2d_block(input, num_filters * 1)
    c2 = conv2d_block(c1   , num_filters * 2)
    c3 = conv2d_block(c2   , num_filters * 4)
    c4 = conv2d_block(c3   , num_filters * 8)


    # --------------------------------- #
    # Learned Upscaling                 #
    # --------------------------------- #

    u8  = transpose_block(c3 , c4, num_filters * 8)
    u9  = transpose_block(u8 , c3, num_filters * 4)
    u10 = transpose_block(u9 , c2, num_filters * 2)
    u11 = transpose_block(u10, c1, num_filters * 1)


    # --------------------------------- #
    # Output Layer                      #
    # --------------------------------- #

    output = Conv2D(
        name       = 'last_layer',
        kernel     = (1, 1),
        filters    =  1    ,
        activation = 'linear'
    )(u11)


    # --------------------------------- #
    # Model Creation and Compilation    #
    # --------------------------------- #

    model = Model(
        inputs  = [input ],
        outputs = [output], 
        name    = model_name
    )

    model.compile(
        loss      = loss_fn_tf,
        metrics   = ['mean_squared_error'],
        optimizer = Adam(learning_rate = learning_rate)
    )

    return model
