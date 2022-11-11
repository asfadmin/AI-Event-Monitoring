import os
import json

from io import BytesIO

import flask
import numpy as np

from flask import request

from src.inference import mask
from src.io import get_image_array


ping_test_image = 'tests/test_image.tif'
mask_model_path = '/opt/ml/models/mask_model'
pres_model_path = '/opt/ml/models/pres_model'
tile_size       = 512


app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():    

    image = get_image_array(ping_test_image)

    masked, pres_mask = mask(mask_model_path, pres_model_path, image, tile_size)

    if np.mean(pres_mask) > 0.0:
        presense = True
    else:
        presense = False

    response = {
        "presense": presense,
    }

    return flask.Response(response=json.dumps(response), status=200, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def invocations():


    """
    Main function to run inference on the image
    :return:
    """

    status = 200

    try:

        byteImg = BytesIO(request.get_data())
        with open("image.tif", "wb") as f:
            f.write(byteImg.getbuffer())

        image = get_image_array("image.tif")

        _, presence_mask = mask(mask_model_path, pres_model_path, image, tile_size)

        if np.mean(presence_mask) > 0.0:
            presense = True
        else:
            presense = False

        result = {
            "presense": presense,
        }
        
    except Exception as err:

        status = 500

        result = {
            "error": str(err),
            "presense": 2
        }
                
        print("invocations() err: {}".format(err))

    return flask.Response(response=json.dumps(result), status=status, mimetype='application/json')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)