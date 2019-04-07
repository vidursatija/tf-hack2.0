const video = document.querySelector('video');
const canvas = window.canvas = document.querySelector('canvas');
var videoSelect = document.querySelector('#videoSource');

canvas.width = 200;
canvas.height = 200;
const button = document.getElementById('take_snapshot');
button.onclick = function () {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  console.log(canvas.toDataURL());
};

navigator.mediaDevices.enumerateDevices().then(gotDevices).then(getStream).catch(handleError);

videoSelect.onchange = getStream;

function gotDevices(deviceInfos) {
  for (var i = 0; i !== deviceInfos.length; ++i) {
    var deviceInfo = deviceInfos[i];
    var option = document.createElement('option');
    option.value = deviceInfo.deviceId;
    if (deviceInfo.kind === 'videoinput') {
      option.text = deviceInfo.label || 'camera ' +
        (videoSelect.length + 1);
      videoSelect.appendChild(option);
      console.log(deviceInfo.label);
    } else {
      console.log('Found one other kind of source/device: ', deviceInfo);
    }
  }
}

function getStream() {
  if (window.stream) {
    window.stream.getTracks().forEach(function(track) {
      track.stop();
    });
  }

  var constraints = {
    audio: false,
    video: {
      deviceId: {exact: videoSelect.value}
    }
  };

  navigator.mediaDevices.getUserMedia(constraints).
    then(gotStream).catch(handleError);
}


function gotStream(stream) {
  window.stream = stream; 
  video.srcObject = stream;
}

function handleError(error) {
  console.log('Error: ', error);
}

 // REQUEST
// $.ajax({
//   type: "POST",
//   url: "<url>",
//   data: { 
//      imgBase64: dataURL
//   }
// }).done(function(o) {
//   console.log('saved'); 
// });