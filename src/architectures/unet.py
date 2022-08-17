"""
 Created By:   Andrew Player
 File Name:    model.py
 Date Created: 01-25-2021
 Description:  Contains a network for sar classification
"""


from tensorflow                  import Tensor
from tensorflow.keras.layers     import Conv2D, Conv2DTranspose, MaxPooling2D, Input, concatenate
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


def create_unet(
    model_name:    str   = 'model',
    tile_size:     int   = 1024   ,
    num_filters:   int   = 16     ,
    dropout:       float = 0.2    ,
    learning_rate: float = 1e-4
) -> Model:

    """
    Creates a U-Net style model
    """

    input = Input(shape = (tile_size, tile_size, 1))


    # --------------------------------- #
    # Feature Map Generation            #
    # --------------------------------- #

    c1 = conv2d_block(input, num_filters *  1)
    m1 = MaxPooling2D((2, 2), strides=2)  (c1)
    c2 = conv2d_block(m1   , num_filters *  2)
    m2 = MaxPooling2D((2, 2), strides=2)  (c2)
    c3 = conv2d_block(m2   , num_filters *  4)
    m3 = MaxPooling2D((2, 2), strides=2)  (c3)
    c4 = conv2d_block(m3   , num_filters *  8)
    m4 = MaxPooling2D((2, 2), strides=2)  (c4)
    c5 = conv2d_block(m4   , num_filters * 16)


    # --------------------------------- #
    # Learned Upscaling                 #
    # --------------------------------- #

    u8  = transpose_block(c5 , c4, num_filters * 8)
    u9  = transpose_block(u8 , c3, num_filters * 4)
    u10 = transpose_block(u9 , c2, num_filters * 2)
    u11 = transpose_block(u10, c1, num_filters * 1)


    # --------------------------------- #
    # Output Layer                      #
    # --------------------------------- #

    output = Conv2D(
        name        = 'last_layer',
        kernel_size = (1, 1),
        filters     =  1    ,
        activation  = 'linear',
        padding     = 'same'
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
        loss      = 'mean_squared_error',
        metrics   = ['mean_squared_error'],
        optimizer = Adam(learning_rate = learning_rate)
    )

    return model
