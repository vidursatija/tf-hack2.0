const video = document.querySelector('video');
const canvas = window.canvas = document.querySelector('canvas');
var videoSelect = document.querySelector('#videoSource');

const button = document.getElementById('take_snapshot');

button.onclick = function () {
	canvas.width = video.videoWidth;
	canvas.height = video.videoHeight;
	canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
	
	var b64Image = canvas.toDataURL('image/jpeg')
	var base64ImageContent = b64Image.replace(/^data:image\/(png|jpg);base64,/, "");
	var blob = base64ToBlob(base64ImageContent, 'image/png');
	var formData = new FormData();
	formData.append('picture', blob);

	// POST HERE
	// $.ajax({
	// 	type: "POST",
	// 	url: "<url>",
	// 	data: {
	// 		imgBase64: formData
	// 	}
	// }).done(function (o) {
	// 	console.log('SENT', o);
	// });
}


function base64ToBlob(base64, mime) {
	mime = mime || '';
	var sliceSize = 1024;
	var byteChars = window.atob(base64);
	var byteArrays = [];

	for (var offset = 0, len = byteChars.length; offset < len; offset += sliceSize) {
		var slice = byteChars.slice(offset, offset + sliceSize);

		var byteNumbers = new Array(slice.length);
		for (var i = 0; i < slice.length; i++) {
			byteNumbers[i] = slice.charCodeAt(i);
		}

		var byteArray = new Uint8Array(byteNumbers);

		byteArrays.push(byteArray);
	}

	return new Blob(byteArrays, {
		type: mime
	});
}


navigator.mediaDevices.enumerateDevices().then(gotDevices).then(getStream).catch(handleError);

videoSelect.onchange = getStream;

function gotDevices(deviceInfos) {

	for (var i = 0; i !== deviceInfos.length; ++i) {

		var deviceInfo = deviceInfos[i];
		var option = document.createElement('option');
		option.value = deviceInfo.deviceId;

		if (deviceInfo.kind === 'videoinput') {
			option.text = deviceInfo.label || 'camera ' + (videoSelect.length + 1);
			videoSelect.appendChild(option);
			console.log(deviceInfo.label);

		} else {
			console.log('Found one other kind of source/device: ', deviceInfo);
		}
	}
}

function getStream() {
	if (window.stream) {
		window.stream.getTracks().forEach(function (track) {
			track.stop();
		});
	}

	var constraints = {
		audio: false,
		video: {
			deviceId: {
				exact: videoSelect.value
			}
		}
	};

	navigator.mediaDevices.getUserMedia(constraints).then(gotStream).catch(handleError);
}


function gotStream(stream) {
	window.stream = stream;
	video.srcObject = stream;
}

function handleError(error) {
	console.log('Error: ', error);
}
