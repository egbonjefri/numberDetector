const button = document.getElementsByClassName('myBtn')[0];
const image = document.getElementsByClassName('myImg')[0];
const box = document.getElementsByClassName('box')[0];
const correct = document.getElementsByClassName('correct')[0]
let model = null
let touch
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let coord = { x: 20, y: 0 };
let normal
canvas.addEventListener("mousedown", start);
canvas.addEventListener("touchstart", start);
canvas.addEventListener("touchend", stop);
button.disabled = true
box.addEventListener("mouseup", stop);
window.addEventListener("resize", resize);
image.width = 28;
image.height = 28

resize();

function resize() {
  ctx.canvas.width = 256;
  ctx.canvas.height = 256;
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
    canvas.addEventListener("mousemove", draw, false);
    canvas.addEventListener("touchmove", toucher, false);
  reposition(event);
}
function stop() {
    canvas.removeEventListener("mousemove", draw, false);
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


  evaluate(image)

})
const PREDICTION_ELEMENT = document.getElementById('prediction');



try{
  const URL = "https://teachablemachine.withgoogle.com/models/Pywquvgpr/";
  const modelURL = URL + "model.json";
  const metadataURL = URL + "metadata.json";


  model = Object.freeze(await tmImage.load(modelURL, metadataURL));
 PREDICTION_ELEMENT.innerText = 'Model Loaded';
 button.disabled = false

}
catch(e){
console.log(e)
}





 async function evaluate(myInput) {

    
      const prediction = await model.predict(myInput);
      let x;
      for (let i = 0; i < prediction.length; i++){
        if(prediction[i].probability > 0.5 && prediction[i].probability <= 1) {
          x = i
        }
      }
      if(x == undefined || x > 9){
        PREDICTION_ELEMENT.innerText = 'Can\'t make any predictions right now';
      }
      else{
      correct.innerText = x;
      PREDICTION_ELEMENT.innerText = '';
      }
   
  }

