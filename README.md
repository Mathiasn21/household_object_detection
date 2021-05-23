
# Household Object Detection  
 The following objects are the target objects:
 - Chair
 - Couch
 - Cup(Coffee Cup)
 - Painting
 - Plant

## Tools
The following tools were utilized in this project:
 - [Makesense AI](https://www.makesense.ai/)
 - [PyCharm](https://www.jetbrains.com/pycharm/)
 - [Detectron2](https://github.com/facebookresearch/detectron2)
 - [Ninja](https://ninja-build.org/)

## Installation
Do note that when executing this project, it is highly recommended to setup a python venv, as that will not adjust any system wide packages.

This project utilizes detectron2 for detecting objects in images. Hence, it is required to install this. To do this, it is highly recommended to follow the official installation guide at:[detectron2 link](https://github.com/facebookresearch/detectron2) . It is highly recommended to install Ninja on windows, if detectron2 is installed on this platform, as it allows for a quicker installation. Additionally, this py package is already available on linux. Furthemore the IDE used is pycharm, along with python 3.8, and was launched at top level inside the folder, "project_files".

Required packages are found in requirements.txt and installing them can be done using pip. Eg: pip install -r requirements.txt

## Running the code
 1. Resizing the images are done by running the script called, "resize_images.py".
 2. Preprocessing and generating synthetic data is done by running the script, "preprocess_images.py".
 3. Training the model is done by running the script, "detectron2_training.py".
 4. Inference using the model is done by running the script "detectron2_inference.py".
 5. Plotting the stored metrics are done by running the script "plot_metrics.py"

The project files contain all neccessary information to run each of these scripts individually. However, they were initially ran in order as they stand.