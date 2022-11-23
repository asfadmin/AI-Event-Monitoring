"""
 Created By:   Andrew Player
 File Name:    model.py
 Date Created: 01-25-2021
 Description:  Contains a network for sar classification
"""


from tensorflow                   import Tensor
from tensorflow.keras.layers      import Conv2D, Conv2DTranspose, Input, concatenate, MaxPooling2D, Activation, Dropout
from tensorflow.keras.models      import Model
from tensorflow.keras.optimizers  import Adam
from tensorflow.keras             import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


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
    tile_size:     int   = 512    ,
    num_filters:   int   = 64     ,
    learning_rate: float = 1e-4   ,
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
    c5 = Dropout(0.2)(c5)


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
        padding     = 'same'
    )(u11)

    output = Activation('linear', dtype='float32')(output)

    # --------------------------------- #
    # Model Creation and Compilation    #
    # --------------------------------- #

    model = Model(
        inputs  = [input ],
        outputs = [output], 
        name    = model_name
    )


    # TODO: Test huber loss, and maybe some others as well, and binary_crossentropy for the classifier. Maybe test with SGD instead of Adam, as well.
    model.compile(
        loss      = 'huber',
        metrics   = ['mean_squared_error', 'mean_absolute_error'],
        optimizer = Adam(learning_rate = learning_rate)
    )

    return model
