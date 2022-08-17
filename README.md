# AI-Event-Monitoring

The goal of this project is to create a deep learning model that can recognize and mask significant deformation events in InSAR interferograms.

## Table of Content 

- [Setup](#setup)
- [Commands](#commands)
- [Running Unit Tests](#tests)
- [Synthetic Interferograms](#synth)
- [Simulated Interferograms](#sim)
- [Preliminary Results](#results)
- [References](#references)
    
# Setup <a name="setup"></a>
This project uses poetry for dependency management.</br>
First, install poetry and then run these commands:
```
poetry install
poetry shell
```
Once you are in the poetry virtual environment, you can run the setup command:
```
python aievents.py setup
```
This will add the data directory which is structured like this:
```
data/
      └──input/
        └──products/
        └──aoi/
      └──working/
        └──real/
        └──synthetic/
      └──output/
        └──models/
        └──mask/
        └──tensorboard/
```
You should now be ready to run everything.

# Basic Commands <a name="commands"></a>

### For a List of Commands
```bash
python aievents.py --help
```

### For Detailed Usage Information
```bash
python aievents.py [command] --help
```

### Viewing a Random Synthetic Masked/Wrapped Pair
```bash
python aievents.py show-random
```

### Viewing a Random Simulated Masked/Wrapped Pair
```bash
python aievents.py simulate
```

### Creating a Synthetic Masked/Wrapped Dataset
```bash
python aievents.py make-synthetic-dataset [dataset-name] [dataset-size] --tile_size [nxn-size-of-images]
```

### Creating a Simulated Masked/Wrapped Dataset
```bash
python aievents.py make-simulated-dataset [dataset-name] [dataset-size] --tile_size [nxn-size-of-images]
```

### Viewing a Pair from a Dataset
```bash
python aievents.py show [path/to/dataset.npz]
```

### Training a Model
```bash
python aievents.py train-model [model-name] [path/to/training-set] [path/to/testing-set] --epochs [num-of-epochs]
```

### Testing a Model
```bash
python aievents.py test-model [path/to/model]
```

### Mask a Real Interferogram
```bash
python aievents.py mask [path/to/model] [path/to/product_folder] --tile_size [size-of-tiles-used-to-train]
```

# Running Unit Tests <a name="tests"></a>
Currently, test coverage is limited. However, they can be run with pytest by simply typing:<br>
```bash
pytest
```
in the root of the project directory.

# Synthetic Interferograms <a name="synth"></a>
Synthetic Interferograms are generated using more simple math than the simulated ones. This means that the datasets can be created
more quickly; although, the simulated interferogram generation is still fairly quick and recommended over this.

### Synthetic Masked/Wrapped Pair Example
![synth_example](https://user-images.githubusercontent.com/19739107/185218653-7e7b89e9-8ac6-4307-936f-448f1e446ed5.png)

# Simulated Interferograms <a name="sim"></a>
Simulated Interferograms are comprised of simulated deformation using Okada's model, simulated turbulent atmospheric error using a FFT method,
simulated topographic atmospheric error, and simulated incoherence from turbulent atmospheric error. Most of the functions related to the simulation 
come from this project by [Matthew Gaddes](https://github.com/matthew-gaddes/SyInterferoPy) which was used for this [2019 JGR:SE paper](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019JB017519).

### Simulated Masked/Wrapped Pair Example 
![sim_example](https://user-images.githubusercontent.com/19739107/185218696-424eeffa-6f7d-4735-a6c9-081a6823560d.png)

### Simulated Masked/Wrapped Pair Example (Atmospheric Phase Error only)
![git_topo_example](https://user-images.githubusercontent.com/19739107/185219924-803c6e10-4c14-4f55-bab3-d445d4a5892d.png)

# Basic Model's Mask Examples <a name="results"></a>
These results come from a basic model trained on a simulated dataset with 1000 samples, 900 for training and 100 for validation.

### An Earthquake in Iran
![git_result1](https://user-images.githubusercontent.com/19739107/185219423-d384a519-861e-4c60-a585-d2cd8b21c34b.png)

### A Negative from the Coast of Greenland
![7thSimNegative](https://user-images.githubusercontent.com/19739107/185219618-e7ac3274-9cbd-4679-b5be-aa8f1d3d0336.png)

# References
Gaddes, M. E., Hooper, A., & Bagnardi, M. (2019). Using machine learning to automatically detect volcanic unrest in a time series of interferograms. Journal of Geophysical Research: Solid Earth, 124, 12304– 12322. https://doi.org/10.1029/2019JB017519
