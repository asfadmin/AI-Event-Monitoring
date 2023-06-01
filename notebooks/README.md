# OSL Notebooks

This folder contains the OpenSARlab demonstration notebooks.

## Setup

Assuming you have created an OSL account and launched your server, first you must create the rtc_analysis conda environment for the notebooks to run inside of. Open the notebook located at `conda_environments/Create_OSL_Conda_Environments.ipynb` and run the "Select a Conda Environment to Create" cell and all cells before it. Select the `rtc_analysis` environment and then run the rest of the cells. This will create the `rtc_analysis` environment which is needed to run the provided notebooks. Before running the provided notebooks you must select this environment.

To get the provided notebooks into OSL, clone the git repository using the terminal action in OSL. Once you have done this, open the notebook located at `AI-Event-Monitoring/notebooks/download_models_and_data.ipynb` and run it to download the models and dataset.

## Notebooks

### download_models_and_data.ipynb
Downloads the pre-trained models and dataset.

### evaluate_model.ipynb
Uses the model to run inferences on the dataset.

### generate_masks.ipynb
Performs inferences on interferograms passed to it and plots the mask of the event.

### model_creation.ipynb
Demos the training of both the masking model and binary classification model.