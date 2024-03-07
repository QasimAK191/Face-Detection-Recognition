# Face-Detection-CPS803-main
Presentation Video: https://youtu.be/_jbV5Tu_RwE


## Creating conda environment
1. Run `conda env create -f environment.yml` from inside the repository.
  - This creates a Conda environment called `face-detection`
2. Run `source activate face-detection`
  - This activates the `face-detection` environment

## Extract Images
1. Unzip the images.zip file which will create a directory of a sample dataset containing images with faces

## Running all models:
1. Run the Python file main.py which will import all modules and code files, and run each model

## Code Structure
Our code runs of main.py, each model inherits from the Model class (model.py) and every image is processed as an Image class (image.py)
util.py and preprocess.py deal with image and model metrics related functions
We also have a demo application that runs on webcam as demo_webcamapp.py

## Dataset Acknowledgment
The dataset included is a truncated version of the dataset taken from the Wider Faces provided by the multimedia laboratory of the Department of Information Engineering from The Chinese University of Hong Kong
