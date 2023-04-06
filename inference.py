import requests
import click
@click.group()
def cli():
    pass


@cli.command ("predict-event")
@click.argument("usgs-event-id", type=str)
@click.argument("product-name", type=str)
def model_inference(usgs_event_id, product_name):
 url = "https://aevrv4z4vf.execute-api.us-west-2.amazonaws.com/test-3/predict-event"

 r = requests.post(
 url,
 json={
  "usgs_event_id": usgs_event_id,
  "product_name": product_name
 })
 print(r.json())


if __name__ == "__main__":
 cli()
