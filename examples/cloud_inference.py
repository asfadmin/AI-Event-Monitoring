# Run inference using the cloud api
import requests


def model_inference(usgs_event_id, product_name):
    url = "https://aevrv4z4vf.execute-api.us-west-2.amazonaws.com/test-3/predict-event"

    r = requests.post(
        url, json={"usgs_event_id": usgs_event_id, "product_name": product_name}
    )
    print(r.json())


print(
    model_inference(
        "us6000jkpr", "S1AA_20230126T212437_20230219T212436_VVR024_INT80_G_weF_3603"
    )
)
