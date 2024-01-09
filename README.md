# RegressionFlow on BIRAFFE2

Aim of this project is to predict future emotions of player (data from BIRAFFE2 dataset) in the form of probability blob in arousal x valence model. I changed as little as possible in original files.

RegressionFlow:
https://github.com/maciejzieba/regressionFlow

BIRAFFE2:
https://zenodo.org/record/5786104

## Installation

Project requires Nvidia GPU with CUDA capabilities.

### Python packages

It is essential to install install all requirements:
```
pip install -r requirements.txt
```

Setup on which project was tested:
Ubuntu 22
Python 3.10.12
CUDA 11.8
Nvidia Quadro M2000m

Remember to match PyTorch with your CUDA and Python version. In requirements is specified version for CUDA 11.8.


### WEMD module

In order to calculate [Wasserstein Earth Mover's Distance (WEMD)](https://en.wikipedia.org/wiki/Wasserstein_metric), 
one must compile the WEMD C++ library. We use the same implementation of as the implementation of [Overcoming Limitations of Mixture Density Networks: A Sampling and Fitting Framework for Multimodal Future Prediction
](https://github.com/lmb-freiburg/Multimodal-Future-Prediction) paper.

Installation steps:

```bash
cd wemd
mkdir build
cd build
cmake ..
make
```

As a result, `wemd/lib/libwemd.so` file should be created.


## Project run

You need to create folder structures in root of this project and extract appropriate data:

Structure of directory should look like this:

---data\
------BIRAFFE2\
---------BIRAFFE2-gamepad\
------------ *.csv files\
---------BIRAFFE2-games\
------------ *.csv files\
---------BIRAFFE2-photo\
------------ *.csv files\
---------BIRAFFE2-metadata.csv

/data/BIRAFFE2/BIRAFFE2-gamepad/\
https://zenodo.org/records/5786104/files/BIRAFFE2-gamepad.zip

/data/BIRAFFE2/BIRAFFE2-games/\
https://zenodo.org/records/5786104/files/BIRAFFE2-games.zip

/data/BIRAFFE2/BIRAFFE2-photo/\
https://zenodo.org/records/5786104/files/BIRAFFE2-photo.zip

/data/BIRAFFE2/\
https://zenodo.org/records/5786104/files/BIRAFFE2-metadata.csv


Next step is data preparation. Run file (you need to go to this directory and run from there this script):\
/biraffe2_helpers/biraffe2_prepare_data.py

This script will generate 2 directories with prepared data:\
/data/BIRAFFE2/test_data\
/data/BIRAFFE2/train_data


In order to run training, run this script from root directory:\
train_regression_biraffe2.py

In order to run testing, run this script from root directory:\
test_biraffe2.py

You can also play with some parameters - they are specified in file args.py. Some of the are hardcoded and to change them you have to modify them in code.

## Instructions from original RegressionFlow repo

Needed only for different datasets than BIRAFFE2. Some things doesn't work and I left them as they were. Settings (args) are hardcoded in order to much easier running, so there is no need to give parameters as cmd.


### Stanford Drone Dataset (SDD) 

The train and test data are located at:

https://lmb.informatik.uni-freiburg.de/resources/binaries/Multimodal_Future_Prediction/sdd_train.zip

https://lmb.informatik.uni-freiburg.de/resources/binaries/Multimodal_Future_Prediction/sdd_test.zip

The datasets should be located in `data\SDD\train` and `data\SDD\test` locations. 

You can run the training procedure with the script `train_regression_SDD.py`.

### NGSIM  Dataset

First, the data should be obtained from:

https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj

```bash
Specifically you will need these files:
US-101:
'0750am-0805am/trajectories-0750am-0805am.txt'
'0805am-0820am/trajectories-0805am-0820am.txt'
'0820am-0835am/trajectories-0820am-0835am.txt'

I-80:
'0400pm-0415pm/trajectories-0400-0415.txt'
'0500pm-0515pm/trajectories-0500-0515.txt'
'0515pm-0530pm/trajectories-0515-0530.txt'
```

The files should be further processed using `prepocess_data.m` from:

https://github.com/nachiket92/conv-social-pooling

After processing files `TrainSet.mat`, `ValSet.mat`, and `TestSet.mat` should be created.  

### CPI Dataset

Synthetic Car-Pedestrian Interaction dataset introduced [here](https://github.com/lmb-freiburg/Multimodal-Future-Prediction), which can be generated with utilities in `cpi_generation` directory.

To obtain our setup:

```bash
cd cpi_generation/

# train dataset 
python CPI-generate.py --output_folder cpi/train --n_scenes 20000 --history 3 --n_gts 20 --dist 20

# test dataset
python CPI-generate.py --output_folder cpi/test --n_scenes 54 --history 3 --n_gts 1000 --dist 20
```

The model for this dataset can be trained with train_regression_CPI.py.