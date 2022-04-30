# react_native_app

This folder contains the code required to launch the react native application to test out your model. The code works for both Android and IOS devices.

**You will need to add the files `group1-shard.bin` and `model.json` of the model you created into the folder `app/assets/model`.**

## App.js

#### Overview of how it runs
1. The code first loads the model with `tf.loadGraphModel(bundleResource)`
2. The user will then be prompted to select an image from the phone gallery
3. The image is then converted to a Tensorflow tensor with `imageToTensor(source)`
4. This tensor is then fed into the model with `model.executeAsync(imageTensor)`
5. The model outputs a lot of information, but we're only interested in `detectionBoxes` and `detectionScores`
6. The code iterates through all the model's predictions (100 of them) and only accepts predictions that are above the `defaultDetectionScoreCutOff`, which has been arbitarily set to 0.9
7. The `detectionBoxes` index points of accepted predictions are then appended to the array `boundingBoxes`
8. Boxes are then drawn on the image based on these index points before being returned to the user

#### Image Tensor Shape
Depending on the pre-trained model you chose, you might need to edit the function `imageToTensor()` accordingly. For instance, the model that's used here, *ssd mobilenet v2 fpnlite 320x320*, allows for any input shape, values between 0 and 255 and requires 4 dimensions. Therefore `imageToTensor()` accordingly prepares the tensors by creating the 4th dimension (since the model accepts any input shape and values are already between 0 and 255).
