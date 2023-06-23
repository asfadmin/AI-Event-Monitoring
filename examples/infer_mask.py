from tensorflow.keras.models import load_model
from insar_eventnet.inference import mask, plot_results
from insar_eventnet.io import initialize

tile_size = 512
crop_size = 512

mask_model_path = "data/output/models/mask_model"
pres_model_path = "data/output/models/pres_model"
image_path = input("Image Path: ") # Prompt user for input interferogram
image_name = image_path.split("/")[-1].split(".")[0]
output_path = f"masks_inferred/{image_name}_mask.tif"

# The initialize function downloads the pretrained models
initialize()
mask_model = load_model(mask_model_path)
pres_model = load_model(pres_model_path)

mask, presence = mask(
    model_path=mask_model_path,
    pres_model_path=pres_model_path,
    product_path=image_path,
    tile_size=tile_size,
    crop_size=crop_size,
)

if np.mask(presence) > 0.7:
    print("Positive")
else:
    print("Negative")

plot_results(wrapped, mask, presence)
