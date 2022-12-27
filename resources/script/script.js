import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js';
const button = document.getElementsByClassName('myBtn')[0];
const image = document.getElementsByClassName('myImg')[0];
const box = document.getElementsByClassName('box')[0];
const correct = document.getElementsByClassName('correct')[0]

let model = null
let touch
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let coord = { x: 0, y: 0 };
let normalize
canvas.addEventListener("mousedown", start);
canvas.addEventListener("touchstart", start);
canvas.addEventListener("touchend", stop);
button.disabled = true
box.addEventListener("mouseup", stop);
window.addEventListener("resize", resize);


resize();

function resize() {
  ctx.canvas.width = 512;
  ctx.canvas.height = 512;
}
function reposition(event) {
  try{
  coord.x = event.clientX - canvas.offsetLeft;
  coord.y = event.clientY - canvas.offsetTop;
  }
  catch(e){
    
    coord.x = touch.clientX - canvas.offsetLeft;
    coord.y = touch.clientY - canvas.offsetTop;
  }
}
function toucher(e){
  touch = e.touches[0];
  var mouseEvent = new MouseEvent("mousemove", {
    clientX: touch.clientX,
    clientY: touch.clientY
  });
  canvas.dispatchEvent(mouseEvent);
  draw()
}
function start(event) {
  event.preventDefault();
  event.stopPropagation();
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("touchmove", toucher, false);
  reposition(event);
}
function stop() {
    canvas.removeEventListener("mousemove", draw);
    canvas.removeEventListener("touchmove", toucher);
   let canvasUrl = canvas.toDataURL();
    image.src = canvasUrl;
    ctx.clearRect(0,0, canvas.width, canvas.height)

}
function draw(event) {

  ctx.beginPath();
  ctx.lineWidth = 12;
  ctx.lineCap = "round";
  ctx.strokeStyle = "white";
  ctx.moveTo(coord.x, coord.y);
  reposition(event);
  ctx.lineTo(coord.x, coord.y);
  ctx.stroke();
}

button.addEventListener('click', ()=>{

  let imageTensor = tf.browser.fromPixels(image);
  let croppedTensor = tf.slice(imageTensor,[-5,-5,0],[60,60,3])
  let resizedTensor = tf.image.resizeBilinear(croppedTensor,[28,28],true);
  let resizedInt = resizedTensor.toInt()
  let grayScale = resizedInt.mean(2);
  let grayFloat = grayScale.toFloat();
  let grayDims = grayFloat.expandDims(-1);
  let grayTen = grayDims.toInt();

 // tf.browser.toPixels(resizedInt, canvas)
  normalize = grayTen.reshape([-1]);
  imageTensor.dispose();
  croppedTensor.dispose();
  resizedTensor.dispose();
  tf.dispose(grayScale);
  grayFloat.dispose();
  grayDims.dispose();
  grayTen.dispose();
  resizedInt.dispose()

 evaluate()

})

const PREDICTION_ELEMENT = document.getElementById('prediction');

// Grab a reference to the MNIST input values (pixel data).

const INPUTS = TRAINING_DATA.inputs;


// Grab reference to the MNIST output values.

const OUTPUTS = TRAINING_DATA.outputs;


// Shuffle the two arrays in the same way so inputs still match outputs indexes.
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// Input feature Array is 2 dimensional, each element represent an image and each pixel
// is an array 0f 784 values representing 0 - 255 possible values in a 28 x 28 picture.

const INPUTS_TENSOR = tf.tensor2d(INPUTS);


// Output feature Array is 1 dimensional.
//one-hot takes two parameters, the tensor that needs to be encoded and number of classes that represent the data
const outputTensor = tf.tensor1d(OUTPUTS, 'int32')
const OUTPUTS_TENSOR = tf.oneHot(outputTensor, 10);
outputTensor.dispose()


try{
  model = await tf.loadLayersModel('localstorage://numberDetector');
 PREDICTION_ELEMENT.innerText = 'Model Loaded';
 button.disabled = false

}
catch(e){
 train()
}

async function train() { 
// Now actually create and define model architecture.
model = tf.sequential();


model.add(tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}));

model.add(tf.layers.dense({units: 16, activation: 'relu'}));

//softmax ensures all outputs add up to one, essentially providing percentage confidences for each output
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));


model.summary();

    // Compile the model with the defined optimizer and specify our loss function to use.
    model.compile({
  //Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent
  // procedure to update network weights iterative based in training data.
      optimizer: 'adam',
  //categoricalCrossentropy generally calculates the difference between two or more probability distributions
      loss: 'categoricalCrossentropy',
  // log the metric ‘accuracy’ as a measure of how many images are predicted correctly from the training data
      metrics: ['accuracy']
  
    });
    let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
  
      shuffle: true,        // Ensure data is shuffled again before using each epoch.
  
      validationSplit: 0.2,
  
      batchSize: 512,       // Update weights after every 512 examples.      
  
      epochs: 75,           // Go over the data 50 times!
    
    });
  
    OUTPUTS_TENSOR.dispose();
  
    INPUTS_TENSOR.dispose();
    PREDICTION_ELEMENT.innerText = 'Training complete';
    button.disabled = false

  }
let x
  function evaluate() {

  
   
  
    let answer = tf.tidy(function() {
  
      let newInput = normalize.expandDims();
      
      
  
      let output = model.predict(newInput);
  
      const values = output.squeeze().dataSync();
      x = Array.from(values);
      
      return output.squeeze().argMax();    
  
    });
  
    answer.array().then(function(index) {
      if(Math.max(...x) > 0.8){
      correct.innerText = index;
      PREDICTION_ELEMENT.innerText = 'Model Loaded'
      }
      else{
        PREDICTION_ELEMENT.innerText = 'Can\'t make any predictions right now';

      }
      answer.dispose()
      normalize.dispose();
      async function saveModel () {
        await model.save('localstorage://numberDetector');
      }
      saveModel()
      console.log(tf.memory().numTensors)
    });
  
  
  }
