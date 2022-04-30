# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] id="OKmbjgu8e8gU"
# # Introduction
# Hello, this notebook is a step by step guide to training a custom object detection model using transfer learning from a Tensorflow pre-trained model. 
#
# I recommend using Google Collab as you can utilize their free GPU provided to quicken your training process. The drawback however, is that you have to run the entire process from start to finish in one go as Google Collab's runtime will expire after a certain period of inactivity and you would have to run all commands and upload all files used in this runtime again. 
#
# If you've just pulled Custom_Object_Detection.py from the object-detection-protoype repo, you'll first need to convert this back to .ipynb format before using Google Collab. Simply open this .py file with Jupyter and save the file, a .ipynb file will be generated in the folder and you can use upload this python notebook to Google Collab.
#
# Lets get started.

# + [markdown] id="BTYtcb-pvlZb"
# # Installation
# Theres a bunch of stuff you will need to install sequentially before you can train a custom object detection model. Follow the instructions below and there should be no issues. Alternatively, you can follow the official documentation provided by Tensorflow [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html).

# + [markdown] id="vDsKubVlg6J5"
# #### Tensorflow Package Installation & Changing Runtime
# Before you begin, you need to `change the Runtime type` of this notebook to GPU. To do this, find 'Runtime' at the top of this notebook, select 'Change runtime type' and change to 'GPU' under the dropdown box.
#
# Don't be alarmed if you see the following errors in the printout because the package will automatically source and install the correct version of these components after encountering these errors.
# - ERROR: tensorflow 2.5.0 has requirement gast==0.4.0
# - ERROR: tensorflow 2.5.0 has requirement grpcio~=1.34.0
# - ERROR: tensorflow 2.5.0 has requirement h5py~=3.1.0
# - ERROR: tensorflow 2.5.0 has requirement tensorflow-estimator<2.6.0,>=2.5.0rc0
#

# + colab={"base_uri": "https://localhost:8080/"} id="xuUBh3tjA9AM" outputId="0cef3853-e967-42df-d176-4fe341a153b9"
# Install Tensorflow package
# !pip install tensorflow-gpu==2.4.1

# + [markdown] id="fWNcMVqOjEkA"
# Verify your installation - the code cell below is to check if you have downloaded Tensorflow. It also shows you how much faster using a GPU is compared to a CPU!

# + colab={"base_uri": "https://localhost:8080/"} id="XrqcCYc0i1qg" outputId="a7375088-a07c-4df8-f8a2-2d65e9552a7d"
# %tensorflow_version 2.x
import tensorflow as tf
import timeit

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise SystemError('GPU device not found')

def cpu():
  with tf.device('/cpu:0'):
    random_image_cpu = tf.random.normal((100, 100, 100, 3))
    net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
    return tf.math.reduce_sum(net_cpu)

def gpu():
  with tf.device('/device:GPU:0'):
    random_image_gpu = tf.random.normal((100, 100, 100, 3))
    net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
    return tf.math.reduce_sum(net_gpu)
  
# We run each op once to warm up; see: https://stackoverflow.com/a/45067900
cpu()
gpu()

# Run the op several times.
print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))

# + [markdown] id="33EU0D7okLld"
# #### Download the TensorFlow Model Garden.
# Here we download the models folder required using Git. After running the cell below, you should have a single folder named `models` in your runtime files. You can check it by pressing on the `Files` icon on the left of your screen followed by the `Refresh` button.

# + colab={"base_uri": "https://localhost:8080/"} id="Upeoz8JvCCo2" outputId="4b7e66ee-0752-4823-fef8-5a0f28f8d393"
# Download models folder
# !git clone https://github.com/tensorflow/models.git

# + colab={"base_uri": "https://localhost:8080/"} id="9sPgDYYjCCrb" outputId="efe3b618-af02-43f5-9907-9637f01593d5"
# Change directory to research folder
# %cd /content/models/research/

# + [markdown] id="QxBGdq-llbH6"
# #### Protobuf Installation/Compilation
# The Tensorflow Object Detection API uses Protobufs to configure the model and its training parameters. We will download this with the cell below.

# + id="kL_24DP0CCtz"
# From within /content/models/research
# !protoc object_detection/protos/*.proto --python_out=.

# + [markdown] id="SSVO0nHGmaQV"
# #### COCO API Installation
# Tensorflow v2 uses the `pycocotools` package as a dependency of its Object Detection API. Ideally this package gets installed when we install the Object Detection API later on but installation via that method is often prone to failure. Therefore we'll explicitly download it beforehand to save ourself from potential trouble. Run the cell below.

# + colab={"base_uri": "https://localhost:8080/"} id="vHX3eqT9CCwX" outputId="a4106158-be6d-4def-e6ca-0b0285e91cb7"
# Download cocoapi
# !git clone https://github.com/cocodataset/cocoapi.git

# + [markdown] id="WGI9nqYa6OID"
# #### Protobuf Compilation
# Now we need to compile Protobuf. We first change change our directory to enter the PythonAPI folder followed by using the `make` command to compile Protobuf. Run the cells below.

# + colab={"base_uri": "https://localhost:8080/"} id="GrjFa2gMCCyt" outputId="72c0765c-066b-43b1-cf8c-c5b366ef8298"
# Change directory to PythonAPI folder
# %cd cocoapi/PythonAPI

# + colab={"base_uri": "https://localhost:8080/"} id="J3I1cv5SCC1T" outputId="31148eeb-f541-48e0-9c5d-a4e35e3c4109"
# Compile
# !make

# + id="KnxnMz7PCC3q"
# Change path
# %cp -r pycocotools /content/models/research/

# + [markdown] id="tn1pMtJNDaof"
# #### Object Detection API Installation
# Installation of the object detection API is done via installing the `object_detection` package. This is done by running the setup.py script from within `models\research`, which will be done in the following cells.

# + colab={"base_uri": "https://localhost:8080/"} id="WMHamIviCC8q" outputId="8698c8a1-d4f9-4086-925e-d8c125a6eb53"
# Change directory to models/research
# %cd ../../

# + id="gXkQuYoeCC_P"
# Change path
# %cp object_detection/packages/tf2/setup.py .

# + [markdown] id="Glyge2pfkocb"
# Don't be alarmed if you see the following errors in the printout from the cell below because the package will automatically source and install the correct version of these components after encountering these errors.
# - ERROR: tensorflow 2.5.0 has requirement grpcio~=1.34.0
# - ERROR: tensorflow-gpu 2.4.1 has requirement gast==0.3.3
# - ERROR: tensorflow-gpu 2.4.1 has requirement h5py~=2.10.0
# - ERROR: tensorflow-gpu 2.4.1 has requirement tensorflow-estimator<2.5.0,>=2.4.0
# - ERROR: multiprocess 0.70.11.1 has requirement dill>=0.3.3
# - ERROR: google-colab 1.0.0 has requirement requests~=2.23.0
# - ERROR: datascience 0.10.6 has requirement folium==0.2.1
# - ERROR: apache-beam 2.29.0 has requirement avro-python3!=1.9.2,<1.10.0,>=1.8.1

# + colab={"base_uri": "https://localhost:8080/"} id="cmmc3E9sCDBr" outputId="61a25921-3448-4f56-fa30-9577e4126f70"
# Install Object Detection API
# !python -m pip install .

# + [markdown] id="a_Mb5_dcQKfr"
# While installing the Object Detection API, the package `h5py` gets upgraded from version 2.10.0 to version 3.1.0. However, the current Tensorflow version we're using requires version 2.10.0, hence we will need to downgrade the version of `h5py` back to 2.10.0. We can do so with the cell below. You will be prompted to `RESTART RUNTIME`, **DO NOT** restart your runtime!

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="mvq4GNvsO2YQ" outputId="5bcf5f06-3625-4ec8-f064-b8c334b3de3e"
# !pip install tensorflow h5py==2.10

# + colab={"base_uri": "https://localhost:8080/", "height": 35} id="nPMF6wJHP2va" outputId="b377e3d6-0ed1-4ae0-86c7-f9c57296a689"
# Check if your current h5py version is 2.10.0
import h5py
h5py.__version__

# + [markdown] id="JNmk6hF_t48e"
# #### Test your Installation
# Sanity check - if all went well, your Tensorflow Object Detection API should have installed properly. Run the cell below, you should observe a printout with a bunch of `RUN` and `OK`, followed by `RAN <20+> tests in <20+>s OK (skipped = 1)`.

# + colab={"base_uri": "https://localhost:8080/"} id="LEwOaWHfCDEX" outputId="98e60e35-dda0-464a-dcac-976a39ea942b"
# !python object_detection/builders/model_builder_tf2_test.py

# + [markdown] id="LJsEfdeHw5Mk"
# # Training Custom Object Detection Model
# At this point, your workspace would have been set up proper and you can start doing the fun stuff. Again, you can refer to the offical documentation provided by Tensorflow [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html) for an alternative guide on how to train a custom object detection model.

# + [markdown] id="1yDt3HY_zV00"
# #### Setting Up Folders
# You will need to manually create a new folder `training_demo` that exists on the same directory alongside the `models` folder. After which you will need to create 5 other folders within the `training_demo` folder with the titles `annotations`, `exported_models`, `images`, `models` and `pre-trained-models`. Within your `images` folder, create 2 subfolders titled `test` and `train`.
#
# Now you will need to upload some files to specific folders. Upload the file `label_map.pbtxt` to the `annotations` folder. Also upload four files titled `export_tflite_graph_tf2.py`, `exporter_main_v2.py`, `generate_tfrecord.py` and `model_main_tf2.py` to the `training_demo` folder.
#
# At this point, the structure of your files should look like this:
#
# - models
# - training_demo
#     - annotations
#         - label_map.pbtxt
#     - exported_models
#     - images
#         - test
#         - train
#     - models
#     - pre-trained-models
#     - export_tflite_graph_tf2.py
#     - exporter_main_v2.py
#     - generate_tfrecord.py
#     - model_main_tf2.py

# + [markdown] id="bgH5dnFnuC6j"
# #### Downloading a Pre-trained Model from Tensorflow Zoo
# Now that your files have been set up, you will need to download a pre-trained model from the Tensorflow model zoo. Visit [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) and choose a model that you would like to use. 
#
# For this tutorial, I chose `SSD MobileNet V2 FPNLite 320x320` as I'm looking for a model that outputs Boxes and has a relatively fast detection speed. At this point, I would recommend you to do a quick Google search on the model that you decide upon. 
#
# In particular, look at what kind of objects the model was trained to detect. For example, if the model was trained on the COCO dataset, you should find out what images are present in the COCO dataset and see if the objects in the dataset are similar to the objects in the images you plan to use this custom model for.
#
# Another crucial thing to look out for is the size of the model. For reference, `SSD MobileNet V2 FPNLite 320x320` has a size of 10.08MB and I found this to be an optimal size. If the model size is too large, the model will take a very long time to load when it is being called in the mobile application later on. 
#
# After you have chosen a model, click on the its name and select `Copy Link Address`. After you have copied the link address, replace the URL in the cell below with it.

# + colab={"base_uri": "https://localhost:8080/"} id="vZ-p02icCDGz" outputId="d3c44d60-3d95-441c-e4c3-9be3cc1a521d"
# Change directory to training_demo/pre-trained-models
# %cd /content/training_demo/pre-trained-models

# + colab={"base_uri": "https://localhost:8080/"} id="zTdTLBE5CDJR" outputId="38e1ab34-6c4e-41b3-91e8-0d53f8280ca9"
# Replace this URL with your copied link address
# !wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz

# + [markdown] id="2NaY1NiJwaMS"
# Now you should have a `tar.gz` file located in the folder `training_demo/pre-trained-models`. You will need to extract the contents of this file using the `tar -xvf` command in the cell below. If you downloaded a Tensorflow pre-trained model that is different from the one used in this guide, you will need to replace the name of the file with the name of your downloaded `tar.gz` file.
#

# + colab={"base_uri": "https://localhost:8080/"} id="VPn7sT6kCDdb" outputId="aa7f78af-8d2b-4947-9edc-7e8fe7634725"
# Replace the filename with the name of your downloaded tar.gz file
# !tar -xvf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz

# + [markdown] id="eo7FpYtkyCgX"
# If you have successfully extracted the contents of the tar.gz file, your `pre-trained-models` folder should have the following architecture.
# - pre-trained-models
#     - ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8
#         - checkpoint
#         - saved_model
#         - pipeline.config
#     - ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz

# + [markdown] id="NLUnMu1NzKqC"
# #### Preparing Images
# Now you will need to prepare the images to feed into the pre-trained model you downloaded. Thankfully, you should already have pre-labelled images taken from Tictag. If you don't, then you can look into using `labelImg` do so - refer [here](https://www.youtube.com/watch?v=VsZvT69Ssbs) for a guide.
#
# More specifically, what you will need is a `.xml` file <b>for every image</b> that you intend to use. Each `.xml` contains meta information regarding its image and more importantly, the bounding box coordinates for the objects that have been labelled in the image.
#
#
# If you are using labelled images from Tictag, chances are that the bounding box coordinates are in float format. You will need to <b>convert this to integer format</b> before feeding it into the model. 
#
# To do this, create a folder on your local machine and put the python script `xml_float_to_int_converter.py` in it. Download the images and their respective `.xml` files from Tictag and transfer them into the same folder. Run the python script. The bounding box coordinates located within the `<bnbbox>` tag should now be in integer format. Now you can upload the images along with their `.xml` files into the `train` and `test` folders found in `training_demo/images`. I used a 80/20 split of images.

# + [markdown] id="-myOEgIBCl5M"
# #### Preparing Label Map
# Now that your images have been uploaded, you will need to edit the `label_map.pbtxt` of your model accordingly. This file can be found in `training_demo/annotations`.
#
# Open the file by double clicking on it. Change the `name` of the item to correspond to the name found within the `<name>` tag of your `.xml` files. Remove or add more items in `label_map.pbtxt` according to the number of items you have in your `.xml` files.
#
# For example, I only have 1 object with the name <em>1_CAN10</em>, so I will rename the first item to that and remove the second item entirely.

# + [markdown] id="6MbYRHeFQgEA"
# #### Create Tensorflow Records
# Now that we have generated our annotations and split our dataset into the desired train and test subsets, it is time to convert our annotations into the `TFRecord` format. We can do so using the `generate_tfrecord.py` script we copied into the `training_demo` folder earlier on. After running the script, you should see 2 new files generated under the `training_demo\annotations` folder, `test.record` and `train.record`.

# + colab={"base_uri": "https://localhost:8080/"} id="gfD3eKXhPlu-" outputId="801de6e5-d9b8-4e46-f3a1-dc90d1a1ec94"
# Change directory to training_demo folder
# %cd /content/training_demo

# + colab={"base_uri": "https://localhost:8080/"} id="RChaocznPlxw" outputId="bdcf6c5c-3a3b-4acd-8fb6-a500150ef236"
# Create train data TFRecords:
# !python generate_tfrecord.py -x /content/training_demo/images/train -l /content/training_demo/annotations/label_map.pbtxt -o /content/training_demo/annotations/train.record

# Create test data TFRecords:
# !python generate_tfrecord.py -x /content/training_demo/images/test -l /content/training_demo/annotations/label_map.pbtxt -o /content/training_demo/annotations/test.record

# + [markdown] id="gNWccYzLUqnh"
# #### Configuring a Training Job
# For clarity, we will create a new folder that will solely contain components related to our new model. 
#
# To do so, create a new folder under `training_demo/models` with the name `my_generated_model`. 
#
# After you have done that, download the `pipeline.config` file found under `training_demo/pre-trained-models/<your-downloaded-model-folder>`. Upload this `pipeline.config` file to `training_demo/models/my_generated_model`.
#
# Your folder structure within `training_demo` should look like this:
# - ...
# - models
#     - my_generated_model
#         - pipeline.config
# - ...

# + [markdown] id="Hk7dXC9jY-ca"
# Now we'll need to edit some lines in the `pipeline.config` file in the `my_generated_model` folder. Refer to [this](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#configure-the-training-pipeline) to look at the changes that we need to make (highlighted in yellow). Some points to take note of:
# - To get paths for the required files, right click on the desired file in your runtime files and press `Copy path`.
# - For the argument of `fine_tune_checkpoint: `, copy the path of `training_demo/models/pre-trained-models/<your-downloaded-model-folder>/checkpoint/ckpt-0.index` but remove `.index` from the file path.
# - You may want to change the argument for `num_steps` (around line 166) to something smaller at first (eg: 1000) so that your model will take a shorter time to finish training. You can increase this number once you have ensured that the training of the model runs without any errors.
# - Don't worry if your pipeline.config does not have `use_bfloat16:`, you can ignore this line.

# + [markdown] id="n0800UGteoAy"
# #### Training the Model
# Time to begin training our model. After running the cell below, our model will begin training for however many steps we initialized in our `pipeline.config` file. It will print out its current status after every 100 steps. If you see an output similar to this `INFO:tensorflow:Step 100 per-step time 0.444s loss=0.332
# I0527 06:24:10.668986 139885325678464 model_lib_v2.py:683] Step 100 per-step time 0.444s loss=0.332` then congrats! Time to chill and wait for your model to train.
#
# Following what people have said online, it seems that it is advisable to allow your model to reach a TotalLoss of at least 2 (ideally 1 and lower) if you want to achieve “fair” detection results. Obviously, lower TotalLoss is better, however very low TotalLoss should be avoided, as the model may end up overfitting the dataset, meaning that it will perform poorly when applied to images outside the dataset.

# + colab={"base_uri": "https://localhost:8080/"} id="2zohkSV9Pl2n" outputId="cc080ff9-1126-4434-85b8-b83a522dd640"
# Training the Model
# !python model_main_tf2.py --model_dir=/content/training_demo/models/my_generated_model --pipeline_config_path=/content/training_demo/models/my_generated_model/pipeline.config

# + [markdown] id="lMRixBOUi7oq"
# #### Exporting a Trained Model
# Now that we have completed training our model, we need to extract the newly trained inference graph, which will be used later to perform object detection. We can do so using the `exporter_main_v2.py` script that we copied into our `training_demo` folder earlier on. Run the cell below.
#

# + colab={"base_uri": "https://localhost:8080/"} id="np0RP8r7Pl5N" outputId="852e3962-7443-4acc-edf5-6b6c1c90168e"
# Exporting the model
# !python exporter_main_v2.py --input_type image_tensor --pipeline_config_path /content/training_demo/models/my_generated_model/pipeline.config --trained_checkpoint_dir /content/training_demo/models/my_generated_model --output_directory /content/training_demo/exported_models


# + [markdown] id="Ii5k-ZuNlesP"
# After running the cell above, your `training_demo/exported_models` folder should have the following structure:
# - training_demo
#     - ...
#     - exported_models
#         - checkpoint
#         - saved_model
#             - assets
#             - variables
#             - saved_model.pb
#         - pipeline.config
#     - ...

# + [markdown] id="V5c00J7xr44B"
# #### Utilizing a Trained Model
# Finally, to check the fruits of your labour, we can run the cell below to get the results you would expect to see when using your model on images. To do so, replace `IMAGE_PATHS` with the image path of any image from the `training_demo/images/train` folder.

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="AYAgPGj5Pl7v" outputId="fedfa444-1eee-4700-c534-ed617a8f1c49"
"""
Object Detection (On Image) From TF2 Saved Model
=====================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse
from google.colab.patches import cv2_imshow

# Enable GPU dynamic memory allocation
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# Replace this with a path to an image from your train folder
IMAGE_PATHS = '/content/training_demo/images/train/NCAG2003016_201205_091938.jpg'

PATH_TO_MODEL_DIR = '/content/training_demo/exported_models'

PATH_TO_LABELS = '/content/training_demo/annotations/label_map.pbtxt'

MIN_CONF_THRESH = float(0.60)

# LOAD THE MODEL

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))




print('Running inference for {}... '.format(IMAGE_PATHS), end='')

image = cv2.imread(IMAGE_PATHS)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

# input_tensor = np.expand_dims(image_np, 0)
detections = detect_fn(input_tensor)
print(detections['detection_scores'])

# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
               for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

image_with_detections = image.copy()

# SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
viz_utils.visualize_boxes_and_labels_on_image_array(
      image_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=0.7,
      agnostic_mode=False)

print('Done')
# DISPLAYS OUTPUT IMAGE
cv2_imshow(image_with_detections)
# CLOSES WINDOW ONCE KEY IS PRESSED

# + [markdown] id="PEWz26HwHGAb"
# # Exporting Trained Custom Object Detection Model
# Now that we have trained our custom object detection model, we can proceed to download it so that we can use it for subsequent steps. We can do this by zipping the content of `training_demo/exported_models/saved_model` and downloading the zipped folder.

# + colab={"base_uri": "https://localhost:8080/"} id="72m8RUd2HXpK" outputId="5244bf0f-3892-4216-80c0-19d796cdc2ea"
# Zip the folder saved_model
# !zip -r /content/training_demo/exported_models/saved_model.zip /content/training_demo/exported_models/saved_model

# + colab={"base_uri": "https://localhost:8080/", "height": 17} id="3tN3pu12H1O4" outputId="66a3af86-0047-445e-c172-ffb03a84de84"
# Download the zipped folder
from google.colab import files
files.download("/content/training_demo/exported_models/saved_model.zip")

# + [markdown] id="6uzj8dzmInNl"
# **END OF GUIDE**
