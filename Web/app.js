const video = document.querySelector('video');
const canvas = window.canvas = document.querySelector('canvas');
var videoSelect = document.querySelector('#videoSource');

const button = document.getElementById('take_snapshot');

button.onclick = function () {
	canvas.width = video.videoWidth;
	canvas.height = video.videoHeight;
	canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

	var ImageURL = canvas.toDataURL('image/jpeg')
	var block = ImageURL.split(";");
	var contentType = block[0].split(":")[1];
	var realData = block[1].split(",")[1];

	var blob = b64toBlob(realData, contentType);
	var formData = new FormData();
	formData.append('image', blob);

	// POST HERE
	$.ajax({
		type: "POST",
		url: "https://f48c6e9c.ngrok.io/image",
		data: formData,
		dataType: "json",
		contentType: false,
		processData: false,
		cache: false,
	}).done(function (response) {
		console.log(response);
	});
}


function b64toBlob(b64Data, contentType, sliceSize) {
	contentType = contentType || '';
	sliceSize = sliceSize || 512;

	var byteCharacters = atob(b64Data);
	var byteArrays = [];

	for (var offset = 0; offset < byteCharacters.length; offset += sliceSize) {
		var slice = byteCharacters.slice(offset, offset + sliceSize);

		var byteNumbers = new Array(slice.length);
		for (var i = 0; i < slice.length; i++) {
			byteNumbers[i] = slice.charCodeAt(i);
		}

		var byteArray = new Uint8Array(byteNumbers);

		byteArrays.push(byteArray);
	}

	var blob = new Blob(byteArrays, {
		type: contentType
	});
	return blob;
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