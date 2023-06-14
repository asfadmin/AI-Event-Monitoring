Model Types
===========

UNet - :py:mod:`insar_eventnet.architectures.unet`
    The UNet model is the reccomended model for masking non-timeseries data, it utilizes a UNet architecture modified to use strided convolutions for downsampling instead of pooling.
UNet3D - :py:mod:`insar_eventnet.architectures.unet3d`
    The corresponding model to UNet for masking timeseries data. **This model is currently experimental**.
ResNet - :py:mod:`insar_eventnet.architectures.resnet`
    Not reccomended due to worse masking performance in comparison with UNet.
EventNet - :py:mod:`insar_eventnet.architectures.eventnet`
    This is the binary prediction model which infers the presence of an event given a generated mask.