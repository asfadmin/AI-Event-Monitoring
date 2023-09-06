Using the Library
=================

Generating Simulated Events
---------------------------

Generation of simulated events can be done with :func:`insar_eventnet.sarsim:gen_simulated_time_series` for positive events with deformation and :func:`insar_eventnet.sarsim:gen_sim_noise` for negative events with noise only.

The following example generates a few events and plots them

.. literalinclude:: /../examples/plot_simulated_event.py
    :language: python

Generating Simulated Datasets
-----------------------------

Simulated Datasets can be generated using the :func:`insar_eventnet.io:_make_simulated_dataset` function.

The following example creates a simulated dataset in ``data/working/synthetic/simulated_dataset``

.. literalinclude:: /../examples/generate_simulated_dataset.py
    :language: python

.. note:: Notice the use of :func:`insar_eventnet.io:_create_directories` to create the data directory which our simulated dataset is stored in.

Generating Masks from Wrapped Interferograms
--------------------------------------------

The :func:`insar_eventnet.inference:mask` can be used to infer masks and presence values.

The following example downloads models and uses them to infer and plot masks and presence values from the prompted path of a wrapped interferogram.

.. literalinclude:: /../examples/infer_mask.py
    :language: python
.. note::
    The initialize function both creates the directory structure and downloads models for the user.

Training UNet and EventNet Models
---------------------------------

The :func:`insar_eventnet.training:train` function can be used to train models.

The following example trains a new model off of a simulated dataset and then uses that model to run inference on an image.

.. literalinclude:: /../examples/train_model.py
    :language: python