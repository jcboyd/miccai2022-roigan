# miccai2022-cyclegan

Code to train region-guided CycleGANs [[1]](#Reference).

* ```main.py``` contains training code.
* ```src/models.py``` defines GAN generator and discriminators.
* ```src/utils.py``` defines utility functions for training and graphing.
* ```config/``` defines ```.yml``` configuration files to set experiment parameters. 

## Installation

The Anaconda environment is specified in ```environment.yml```. The environment can be recreated using,

```
conda env create -f environment.yml
```

Tested with single NVIDIA P100 GPU, running Cuda 10.0.130, and PyTorch 1.9.0 with torchvision 0.10.0.

## Usage

```main.py``` is the training code, which requires two parameters
* ```job_number``` specifies a unique identifier for writing outputs
* ```config``` specifies configuration file path

See ```slurm_submit.sh``` for example.

## Config files

See [config/README.md](config/README.md) for a description of configuration options.

## Data

Experiments performed on [CAMELYON16](https://camelyon16.grand-challenge.org/) and data from the Gustave Roussy Institute.

## Reference

[1] *Region-guided CycleGANs for Stain Transfer in Whole Slide Images*, Joseph Boyd, Irène Villar, Marie-Christine Mathieu, Eric Deutsch, Nikos Paragios, Maria Vakalopoulou, and Stergios Christodoulidis, MICCAI 2022 (in press)

### Model outputs

![Mosaic](http://jcboyd.github.io/assets/miccai2022-cyclegan/prosaic_mosaic.png)
