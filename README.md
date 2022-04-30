# object-detection-prototype

## Table Of Contents
1. [Context](#context)
    1. [Purpose](#purpose)
    2. [Limitation](#limitation)
    3. [High-level Overview](#high-level-overview)
2. [Implementation](#implementation)
    1. [Step 1 - Training a Custom Object Detection Model](#training-a-custom-object-detection-model)
    2. [Step 2 - Converting Saved Tensorflow Model to Tensorflowjs Format](#converting-saved-tensorflow-model-to-tensorflowjs-format)
    3. [Step 3 - Fitting Tensorflowjs Model into React Native Application](#fitting-tensorflowjs-model-into-react-native-application)

## Context
### Purpose <a name="purpose"></a>
The goal of this use case is to explore the use of object detection models to assist taggers in bounding box tasks by suggesting bounding boxes to be drawn in an image. Think of a *suggest bounding box* button that's available on taggers' mobile screens that draws a rough bounding box around objects when pressed, leaving the only work required by the taggers to be adjusting the bounding box accordingly.

Although there exists pre-trained object detection models that are trained to detect common everyday occuring items like cars and humans, these models are likely to perform poorly on our datasets as these models have never *seen* our images. Thus there is a need to train custom object detection models that can be used for our use case. Fret not, we'll not be training these models from scratch but instead carry out transfer learning from these pre-trained models so we can make use of their pre-trained model weights.

### Limitation <a name="limitation"></a>
A glaring limitation here is that it will be pretty inconvenient to have this function available for every job with different objects. Since a model only performs well on data that it has seen before, we would need to train a new model each time we would like to use this function on a new dataset. 

Perhaps in the future we'll figure out a way to incorporate using tagger-annotated data to train the model and use the model on the remainder of the un-annotated data. Or we'll only use this function when its a large enough dataset and the effort it takes to train the model and have this feature available justifies the effort. Anywho, the main goal of this use case is sort of like a proof of concept to see if venturing into this functionality is a viable option.


### High-level Overview <a name='high-level-overview'></a>
The purpose of this repository is to create a custom object detection model in Tensorflow and use it in a react native application. As training a custom object detection model requires images containing the object that you're interested to have detected, you would need to follow the steps in the guide to train a custom object detection model. Afterwards, you would need to convert the custom object detection model into a format that can be used in our React Native application. 

Creating the custom object detection model will take up the bulk of your time whereas converting the format of the model is relatively simple. Moreover, the React Native application has already been built so all that you have to do is to replace the template model with the model you just created.

Essentially, these are the three main steps required:

1. Training a Custom Object Detection Model
2. Converting Saved Tensorflow Model to Tensorflowjs Format
3. Fitting Tensorflowjs Model into React Native Application

As step 1 is a long process, its instructions are found in `Custom_Object_Detection.py` within the `custom_object_detection_model` folder. Steps 2 and 3 are relatively shorter so their instructions can be found on this markdown file. 

## Implementation <a name='implementation'></a>

### Step 1 - Training a Custom Object Detection Model <a name='training-a-custom-object-detection-model'></a>
There are three main steps when training a custom object detection model: Installation, Training the model and Exporting the trained model. A full guide on how to carry out these steps can be found in `Custom_Object_Detection.py` within the `custom_object_detection_model` folder. All required files are also contained in the folder. **Only proceed with this markdown file after you have completed following all the steps in `Custom_Object_Detection.py`.**

### Step 2 - Converting Saved Tensorflow model to Tensorflowjs Format <a name='converting-saved-tensorflow-model-to-tensorflowjs-format'></a>
At this point you should have the folder `saved_model` generated, the next step is to convert this Tensorflow model to the Tensorflowjs format. We can do so using a Tensorflowjs converter.

Firstly, we need to create a new virtual environment and install some packages. This guide uses Anaconda to do this.

Launch a terminal of your choice, then create a new virtual environment and activate it.
```
conda create -n env_tensorflowjs
conda activate env_tensorflowjs
```

Install the following packages.
```
pip install tensorflow
pip install tensorflowjs
pip install tensorflowjs[wizard]
```

Start the converter by running the following command.
```
tensorflowjs_wizard
```

The converter will first ask for the path of the model folder. Input the filepath to the `saved_model` folder on your local machine and hit enter.
<p align="center">
    <img width="900" alt="saved_model_path" src="https://user-images.githubusercontent.com/56946413/119811419-a64bb880-bf19-11eb-8399-71cde479acc1.png">
</p>

Next, we need to select the input model format. Since ours is `Tensorflow Saved Model`, use arrow keys to toggle to that option and hit enter.
<p align="center">
    <img width="900" alt="input_model_format" src="https://user-images.githubusercontent.com/56946413/119811531-cb402b80-bf19-11eb-8cc8-0d88e42f4984.png">
</p>

The converter will ask for what tags were used for the saved model. Select `serve` and hit enter.

Next, the converter will ask for the signature name of the model. Use arrow keys to toggle to the option that shows `inputs: 1 of 1` and hit enter.
<p align="center">
    <img width="900" alt="model_signature" src="https://user-images.githubusercontent.com/56946413/119808980-3b997d80-bf17-11eb-9529-b1db93c44ab6.png">
</p>

The converter then asks if we would like to compress the model. Use arrow keys to select `No compression (Higher accuracy)` and hit enter.

Following which, the converter will ask a series of questions. Keep hitting enter to select the default option for each question till you reach the point where the converter asks for which directory it should save the converted model in.
<p align="center">
    <img width="900" alt="default_options" src="https://user-images.githubusercontent.com/56946413/119809805-10635e00-bf18-11eb-8774-06d80f436ad9.png">
</p>

At this point, create a new folder on your local machine with the name `converted_saved_model` and copy its file path. Paste the file path into the converter and hit enter.
<p align="center">
    <img width="900" alt="output_model_path" src="https://user-images.githubusercontent.com/56946413/119810407-97b0d180-bf18-11eb-82ff-a0be26810ad9.png">
</p>

After the converter is done, you should see the following files in the `converted_saved_model` folder.
<p align="center">
    <img width="900" alt="converted_saved_model" src="https://user-images.githubusercontent.com/56946413/119811051-405f3100-bf19-11eb-8cc0-f1d0f68de16e.png">
</p>

We're done using the converter at this point, feel free to deactivate your virtual environment.
```
conda deactivate
```

Now we need to combine all the `.bin` files into a single `.bin` file. In a terminal, navigate to the `converted_saved_model` folder and enter the command below. Note: if your model produces more than 3 shard files, edit the command accordingly by adding more `group1-shardXofTOTAL.bin`, `type` and `> group1-shard.bin` remains the same.

```
type group1-shard1of3.bin group1-shard2of3.bin group1-shard3of3.bin > group1-shard.bin
```

All your `.bin` files should be consolidated to a single one. Double check by adding up all the file sizes of the `.bin` files, it should be the same as the newly generated `group1-shard.bin`. You can now delete the individual `.bin` files as we will not be using them anymore.


### Step 3 - Fitting Tensorflowjs Model into React Native Application <a name='fitting-tensorflowjs-model-into-react-native-application'></a>
At this point, our entire trained tensorflow object detection model has been condensed down to two files, `model.json` and `group1-shard.bin`. We will now be fitting these 2 files into a react native application.

Navigate to `react_native_app/app/assets/model` and place the 2 files `model.json` and `group1-shard.bin` into this folder.

In a terminal, navigate to the folder containing `App.js` and run the command `npm start`. With a mobile device or a simulator/emulator, connect to the react native application via Expo Go. After the application has loaded, you will be able to input a photo and have the photo returned with bounding boxes drawn on object identified during your model training.

END OF GUIDE
