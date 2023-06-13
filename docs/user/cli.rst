Using the Command Line interface
================================
The ``insar-eventnet`` cli allows you to run inference, train models, and generate simulated datasets from the comfort of your commandline.

List Commands
----------------------
::

    insar-eventnet --help

Detailed Usage Information
------------------------------
::
    
    insar-eventnet [command] --help

Predict Event Using Cloud API
------------------------------
::
    
    insar-eventnet predict-event

View a Random Synthetic Masked/Wrapped Pair
----------------------------------------------
::

    insar-eventnet show-random


View a Random Simulated Masked/Wrapped Pair
----------------------------------------------
::

    insar-eventnet simulate


Create a Synthetic Masked/Wrapped Dataset
-------------------------------------------
::
    
    insar-eventnet make-synthetic-dataset [dataset-name] [dataset-size] --tile_size [nxn-size-of-images]


Create a Simulated Masked/Wrapped Dataset
-------------------------------------------
::

    insar-eventnet make-simulated-dataset [dataset-name] [dataset-size] --tile_size [nxn-size-of-images]


View a Pair from a Dataset
-----------------------------
::
    
    insar-eventnet show [path/to/dataset.npz]


Train a Model
----------------
::
    
    insar-eventnet train-model [model-name] [path/to/training-set] [path/to/testing-set] --epochs [num-of-epochs]


Test a Model
---------------
::
    
    insar-eventnet test-model [path/to/model]


Mask a Real Interferogram
-------------------------
::
    
    insar-eventnet mask [path/to/model] [path/to/product_folder] --tile_size [size-of-tiles-used-to-train]
