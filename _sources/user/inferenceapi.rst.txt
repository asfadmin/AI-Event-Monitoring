Use the Cloud Inference API
===========================

The UNet and EventNet models are hosted in an Amazon SageMaker instance for convienient use without installing any library.

API Specification
-----------------

The API is located at ``https://aevrv4z4vf.execute-api.us-west-2.amazonaws.com/test-3/predict-event``. Make a JSON post request in the format

.. code-block:: json

    {
        "usgs_event_id": "<USGS Event ID>",
        "product_name": "<Product Name>"
    }

Using the API in Python
-----------------------

.. literalinclude:: /../examples/cloud_inference.py
    :language: python