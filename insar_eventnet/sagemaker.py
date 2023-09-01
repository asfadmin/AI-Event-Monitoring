"""
 Summary
 -------
 Amazon SageMaker CLI

 Notes
 ------
 Created by Andrew Player.
"""

import json
import os

import flask
import numpy as np
import requests
from tensorflow.keras import models

from insar_eventnet import inference, io


def sagemaker_server():
    """
    Masks events in the given wrapped interferogram using a tensorflow model and plots
    it, with the option to save.

    ARGS:\n
    model_path       path to model to mask with.\n
    pres_model_path  path to model that predicts whether there is an event.\n
    image_path       path to wrapped interferogram to mask.\n
    """

    mask_model_path = "/opt/ml/model/models/mask_model"
    pres_model_path = "/opt/ml/model/models/pres_model"

    print(os.listdir("/opt/ml"))
    print(os.listdir("/opt/ml/model"))
    print(os.listdir("/opt/ml/model/models"))

    mask_model = models.load_model(mask_model_path)
    pres_model = models.load_model(pres_model_path)

    try:
        event_list_res = requests.get(
            "https://gm3385dq6j.execute-api.us-west-2.amazonaws.com/events"
        )
        event_list_res.status_code

    except requests.exceptions.RequestException as e:
        print("Could not connect to event list API. Using test event list.")
        raise SystemExit(e)

    def get_image_from_sarviews(
        usgs_event_id: str = "us6000jkpr",
        granule_name: str = "S1AA_20230126T212437_20230219T212436_VVR024_INT80_G_weF_3603",
    ):
        product_name = granule_name + ".zip"

        event_list = event_list_res.json()
        event_obj = next(
            (
                item
                for item in event_list
                if ("usgs_event_id" in item)
                and (item["usgs_event_id"] == usgs_event_id)
            ),
            None,
        )
        event_id = event_obj["event_id"]

        event_get_res = requests.get(
            f"https://gm3385dq6j.execute-api.us-west-2.amazonaws.com/events/{event_id}"
        )
        event_get_res.status_code

        event_get_list = event_get_res.json()
        event_obj = next(
            (
                item
                for item in event_get_list["products"]
                if item["files"]["product_name"] == product_name
            ),
            None,
        )

        product_url = event_obj["files"]["product_url"]

        os.system(f"wget --quiet {product_url}")
        os.system(f"unzip -qq {product_name}")
        os.system(f"ls {granule_name}")

        return granule_name + "/" + granule_name + "_unw_phase.tif"

    app = flask.Flask(__name__)

    @app.route("/ping", methods=["GET"])
    def ping():
        """
        SageMaker expects a ping endpoint to test the server.
        This confirms that the container is working.
        """

        try:
            response = {
                "presense": True,
            }

            return flask.Response(
                response=json.dumps(response), status=200, mimetype="application/json"
            )
        except ConnectionError:
            print("Could not connect to server")
        except Exception as e:
            print(f"Caught {type(e)}: {e}")
            return flask.Response(
                response=json.dumps({"error": str(e)}),
                status=500,
                mimetype="application/json",
            )

    @app.route("/invocations", methods=["POST"])
    def invocations():
        """
        Main route for inference through SageMaker.
        """

        status = 200

        try:
            content = flask.request.json

            usgs_event_id = content["usgs_event_id"]
            granule_name = content["product_name"]

            image_path = get_image_from_sarviews(usgs_event_id, granule_name)

            image, dataset = io.get_image_array(image_path)
            wrapped_image = np.angle(np.exp(1j * image))

            masked, presence_mask, presence_vals = inference.mask_with_model(
                mask_model, pres_model, wrapped_image, tile_size=512
            )

            if np.mean(presence_mask) > 0.0:
                presense = True
            else:
                presense = False

            result = {
                "presense": presense,
            }

        except Exception as err:
            status = 500

            result = {"error": str(err), "presense": 2}

            print("invocations() err: {}".format(err))

        return flask.Response(
            response=json.dumps(result), status=status, mimetype="application/json"
        )

    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    sagemaker_server()
