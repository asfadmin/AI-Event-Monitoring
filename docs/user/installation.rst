Install the Library
===================
There are three ways to install the library: through pip, conda, or by cloning the source.

Install through pip
-------------------

Environment Setup
^^^^^^^^^^^^^^^^^
You need some way of providing the GDAL library

Ubuntu
""""""
Install the ``libgdal-dev`` package with::
    
    sudo apt install libgdal-dev

Some people have noted that apt has troubles resolving dependencies for the ``libgdal-dev`` package. `This stackoverflow answer may provide assistance <https://stackoverflow.com/a/72887401/>`_.

Arch Linux
""""""""""
Install the ``python-gdal`` package with::
    
    sudo pacman -S python-gdal

Library
^^^^^^^
::

    pip install insar-eventnet

Install through conda
---------------------
::

    conda install insar-eventnet

Install from Source
-------------------

Clone the repository::

    git clone https://github.com/asfadmin/AI-Event-Monitoring.git

Activate the conda environment from the environment.yaml file

    conda create env -f environment.yaml
    conda activate insar-eventnet

Then, build and install the package

    pip install .

Setup
=====

In your desired working directory run the setup command::

    insar-eventnet setup

This will create the data directory with the below structure::

    data/
      └──input/
        └──products/
        └──aoi/
      └──working/
        └──real/
        └──synthetic/
      └──output/
        └──models/
        └──mask/
        └──tensorboard/