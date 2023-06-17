Using the Library
=================

Generating Simulated Events
---------------------------

Generation of simulated events can be done with :func:`insar_eventnet.sarsim:gen_simulated_time_series` for positive events with deformation and :func:`insar_eventnet.sarsim:gen_sim_noise` for negative events with noise only.

The following example generates a few events and plots them

.. code-block:: python

    import matplotlib.pyplot as plt
    
    from insar_eventnet.config    import SYNTHETIC_DIR
    from insar_eventnet.sarsim    import gen_simulated_deformation, gen_sim_noise

    seed       = 232323
    tile_size  = 512
    event_type = 'quake'

    unwrapped_def, masked_def, wrapped_def, presence_def = gen_simulated_deformation(
        seed       = seed,
        tile_size  = tile_size,
        event_type = event_type
    )
    unwrapped_mix, masked_mix, wrapped_mix, presence_mix = gen_sim_noise(
        seed      = seed,
        tile_size = tile_size
    )

    print(f"Deformation Presence: {presence_def}")
    print(f"Mixed Noise Presence: {presence_mix}")

    _, [axs_unwrapped_def, axs_wrapped_def, axs_mask_def] = plt.subplots(1, 3, sharex=True, sharey=True, tight_layout=True)

    _, [axs_unwrapped_mix, axs_wrapped_mix, axs_mask_mix] = plt.subplots(1, 3, sharex=True, sharey=True, tight_layout=True)

    axs_unwrapped_def.set_title("Deformation Event")
    axs_unwrapped_mix.set_title("Atmospheric/Topographic Noise")

    axs_unwrapped_def.imshow(unwrapped_def, origin='lower', cmap='jet')
    axs_unwrapped_mix.imshow(unwrapped_mix, origin='lower', cmap='jet')
    axs_wrapped_def.imshow(wrapped_def, origin='lower', cmap='jet')
    axs_wrapped_mix.imshow(wrapped_mix, origin='lower', cmap='jet')
    axs_mask_def.imshow(masked_def, origin='lower', cmap='jet', vmin=0.0, vmax=1.0)
    axs_mask_mix.imshow(masked_mix, origin='lower', cmap='jet', vmin=0.0, vmax=1.0)

    plt.show()

Generating Simulated Datasets
-----------------------------

Simulated Datasets can be generated using the :func:`insar_eventnet.io:make_simulated_dataset` function.

The following example creates a simulated dataset in ``data/working/synthetic/simulated_dataset``

.. code-block:: python


    import matplotlib.pyplot as plt

    from insar_eventnet.config    import SYNTHETIC_DIR
    from insar_eventnet.io        import make_simulated_dataset, split_dataset, create_directories

    name       = "simulated_dataset"

    amount     = 2000
    seed       = 0
    tile_size  = 512
    crop_size  = 512
    split      = 0.2

    create_directories()

    name, count, dir_name, distribution, dataset_info = make_simulated_dataset(
        name,
        SYNTHETIC_DIR,
        amount,
        seed,
        tile_size,
        crop_size
    )

    dataset_path = SYNTHETIC_DIR.__str__() + '/' + dir_name

    num_train, num_validation = split_dataset(dataset_path, split)

.. note:: Notice the use of :func:`insar_eventnet.io:create_directories` to create the data directory which our simulated dataset is stored in.

Generating Masks from Wrapped Interferograms
--------------------------------------------

The :func:`insar_eventnet.inference:mask` can be used to infer masks and presence values.

The following example downloads models and uses them to infer and plot masks and presence values from the prompted path of a wrapped interferogram.

.. code-block:: python

    from tensorflow.keras.models import load_model
    from insar_eventnet.inference import mask, plot_results
    from insar_eventnet.io import initialize
        
    tile_size = 512
    crop_size = 512

    mask_model_path = 'models/masking_model'
    pres_model_path = 'models/classification_model'
    image_path      = input('Image Path: ')
    image_name      = image_path.split('/')[-1].split('.')[0]
    output_path     = f'masks_inferred/{image_name}_mask.tif'

    initialize()
    mask_model = load_model(mask_model_path)
    pres_model = load_model(pres_model_path)

    mask, presence = mask(
        model_path = mask_model_path,
        pres_model_path = pres_model_path,
        product_path = image_path,
        tile_size = tile_size,
        crop_size = crop_size
    )

    if np.mask(presence) > 0.7:
        print("Positive")
    else:
        print("Negative")
    
    plot_results(wrapped, mask, presence)
.. note::
    The initialize function both creates the directory structure and downloads models for the user.
