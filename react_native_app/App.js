import React, { useState, useEffect }  from 'react';
import { StyleSheet, View, Image, TouchableOpacity, Text } from 'react-native';
import Svg, {Rect} from 'react-native-svg';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO, decodeJpeg } from '@tensorflow/tfjs-react-native';
import Constants from 'expo-constants';
import * as Permissions from 'expo-permissions';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';

// Adjust accordingly btw 0 and 1, lower = more lenient in identifying object
const defaultDetectionScoreCutOff = 0.9 

async function getPermissionAsync() {
  if (Constants.platform.ios) {
    const { status } = await Permissions.askAsync(Permissions.CAMERA_ROLL);
    if (status !== "granted") {
      alert("Permission for camera access required.");
    }
  }
}

async function imageToTensor(source) {
    // expo-file-system method to acquire imageTensor
    const fileUri = source.uri;
    const imgB64 = await FileSystem.readAsStringAsync(fileUri, {
      encoding: FileSystem.EncodingType.Base64,
    });
    const imgBuffer = tf.util.encodeString(imgB64, 'base64').buffer;
    const raw = new Uint8Array(imgBuffer);
    const imageTensor = decodeJpeg(raw);
    
    // get height, width and array of values
    const height = imageTensor.shape[0]
    const width = imageTensor.shape[1]
    console.log('height and width: ' + height + ' ' + width);
    imageTensorDataSync = imageTensor.dataSync();

    // expand dimensions
    const img4 = tf.tensor4d(imageTensorDataSync, [1, height, width, 3]);
    const tempOutput = [img4, source.uri, height, width];
    return tempOutput
  }

export default function App() {
  const [isTfReady, setTfReady] = useState(false); // gets and sets the Tensorflow.js module loading status
  const [model, setModel] = useState(null); // gets and sets the locally saved Tensorflow.js model
  const [image, setImage] = useState(null); // gets and sets the image selected from the user
  const [predictions, setPredictions] = useState(null); // gets and sets the predicted value from the model
  const [error, setError] = useState(false); // gets and sets any errors
  const [imgUrlNew, setImgUrlNew] = useState(null) // gets and sets image submited by user
  const [boundingBoxes, setBoundingBoxes] = useState(null) // gets and sets bounding boxes coordinates
  const [imgHeight, setImgHeight] = useState(null) // gets and sets image height
  const [imgWidth, setImgWidth] = useState(null) // gets and sets image width

  const detectionScoreCutOff = defaultDetectionScoreCutOff; // minimum detection score req before bounding box is considered
  var imgUrl = 'https://idsb.tmgrup.com.tr/ly/uploads/images/2020/05/13/35552.jpeg' // default starting pict
  var boundingBoxesReal = [{ // default starting boundingBox
    idx: 0,
    ymin: 1,
    xmin: 1,
    ymax: 2,
    xmax: 2
  }]
  var newImgHeight = 720; // default starting image height

  useEffect(() => {
    (async () => {
      await tf.ready(); // wait for Tensorflow.js to get ready
      setTfReady(true); // set the state
      console.log('tf is ready...');

      // bundle the model files and load the model:
      const modelJson = require("./app/assets/model/model.json");
      const weights = require("./app/assets/model/group1-shard.bin");
      const bundleResource = bundleResourceIO(modelJson, weights);
      const loadedModel = await tf.loadGraphModel(bundleResource);
      setModel(loadedModel);
      getPermissionAsync(); // get the permission for camera roll access for iOS users
    })();
  }, []);
  
  async function handlerSelectImage() {
    try {
      console.log('User selecting image...');
      let response = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        quality: 1, // go for highest quality possible
      },
      );

      if (!response.cancelled) {
        const source = { uri: response.uri };
        setImage(source); // put image path to the state

        // convert image to a tensor
        imageToTensorOutput = await imageToTensor(source);            
        const imageTensor = imageToTensorOutput[0];
        imgTensorArray = imageTensor.arraySync();        
        setImgUrlNew(imageToTensorOutput[1]);
        setImgHeight(imageToTensorOutput[2]);
        setImgWidth(imageToTensorOutput[3]);

        console.log('does device support 32 bit textures?: ' + tf.ENV.getBool('WEBGL_RENDER_FLOAT32_CAPABLE'));
        console.log('is TensorFlow.js using 32 bit textures?: ' + tf.ENV.getBool('WEBGL_RENDER_FLOAT32_ENABLED'));
        console.log('TensorFlow.js backend: ' + tf.getBackend());

        const output = await model.executeAsync(imageTensor); // feed tensor into model
        const detectionBoxes = output[1].dataSync();
        const detectionScores = output[4].dataSync();
        setPredictions(true);

        // loop through results, keep only those with detection scores above detectionScoreCutOff
        var tempArray = []
        for (let i = 0; i < detectionScores.length; i++) {
          if (detectionScores[i] >= detectionScoreCutOff) {
            console.log('detection score for prediction #' + i + ': ' + detectionScores[i]);
            var firstIndex = i * 4
            tempArray.push({
              idx: i,
              ymin: detectionBoxes[firstIndex],
              xmin: detectionBoxes[firstIndex + 1],
              ymax: detectionBoxes[firstIndex + 2],
              xmax: detectionBoxes[firstIndex + 3]
            })
          }
        }
        console.log(tempArray);
        setBoundingBoxes(tempArray);
      }
    } catch (error) {
      setError(error);
    }
  }

  function reset() {
    setPredictions(null);
    setImage(null);
    setError(false);
  }

  let status, statusMessage, showReset;
  const resetLink = (
    <Text onPress={reset} style={styles.reset}>
      Restart
    </Text>
  );

  if (!error) {
    if (isTfReady && model && !image && !predictions) {
      status = "modelReady";
      statusMessage = "Model is ready.";
    } else if (model && image && predictions) {
      status = "finished";
      statusMessage = "Prediction finished.";
      showReset = true;
    } else if (model && image && !predictions) {
      status = "modelPredict";
      statusMessage = "Model is predicting...";
    } else {
      status = "modelLoad";
      statusMessage = "Model is loading...";
    }
  } else {
    statusMessage = "Unexpected error occured.";
    showReset = true;
    console.log(error);
  }

  if (imgUrlNew != null) { // replace default pict
    imgUrl = imgUrlNew;
  }

  if (boundingBoxes != null){ // replace default bbox
    boundingBoxesReal = boundingBoxes;
    var scale = imgWidth/300; // get scale of image displayed
    newImgHeight = imgHeight/scale;
  }

  return (
    <View style={styles.container}>
        <Text style={styles.status}>
          {statusMessage} {showReset ? resetLink : null}
        </Text>
        <TouchableOpacity
          style={styles.imageContainer}
          onPress={model && !predictions ? handlerSelectImage : () => {}} // Activates handler only if the model has been loaded and there are no predictions done yet
        >
        <Image
        style = {{width:300, height:300, borderWidth:2, borderColor:'green', resizeMode:'contain'}}
        source = {{uri:imgUrl}}>
        </Image>
        <Svg height = '300' width = '300' style = {{marginTop:-300}}>
          {
            boundingBoxesReal.map((bbox) => {
              return (
                <Rect
                key = {bbox.idx}
                x = {bbox.xmin*300}
                y = {(300-newImgHeight)/2 + bbox.ymin*newImgHeight}
                width = {(bbox.xmax-bbox.xmin)*300}
                height = {(bbox.ymax-bbox.ymin)*newImgHeight}
                stroke = {'red'}
                fill = 'none'
                />
              )
            })
          }
        </Svg>
        </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: "#FFFFFF",
    alignItems: "center",
    justifyContent: "center",
    flex: 1,
  },
  status: { marginBottom: 20 
  },
  reset: { color: "blue" 
  },
  imageContainer: {
    width: 300,
    height: 300,
    borderRadius: 20,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "grey",
    borderColor: "grey",
    borderWidth: 3,
    borderStyle: "dotted",
  },
});