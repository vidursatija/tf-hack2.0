// Recycle Labels

var yellowBinItems = {
	899: 'water bottle',
	654: 'can',
	441: 'beer bottle',
	550: 'envelope',
	638: 'mailbox, letter box',
	442: 'beer glass',
	924: 'plate',
	1000: 'toilet tissue, toilet paper, bathroom tissue'
}

var redBinItems = {
	530: 'diaper, nappy, napkin',
	846: 'syringe',
	932: 'bagel, beigel',
	934: 'cheeseburger',
	935: 'hotdog, hot dog, red hot',
	938: 'broccoli',
	944: 'cucumber, cuke',
	948: 'mushroom',
	950: 'strawberry',
	951: 'orange',
	952: 'lemon',
	954: 'pineapple, ananas',
	955: 'banana',
	964: 'pizza, pizza pie',
	880: 'umbrella',
	729: 'plastic bag',
	911: 'wooden spoon',
	624: 'letter opener, paper knife, paperknife',
	957: 'custard apple',
}


const video = document.querySelector('video');
const canvas = window.canvas = document.querySelector('canvas');
var videoSelect = document.querySelector('#videoSource');

const button = document.getElementById('take_snapshot');

button.onclick = function () {
	canvas.width = video.videoWidth;
	canvas.height = video.videoHeight;
	canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
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

		/* TODO: Remove on final
		var device_name = deviceInfo.label.toLowerCase();
		console.log(device_name,"*");
		if (deviceInfo.kind === 'videoinput' && !(device_name.includes("face"))) {
		*/

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


// Live camera view
$('.disc--camera').on('click', function () {
	$("video").addClass("is-gone");
	$("canvas").removeClass('is-gone');
	$('.crop').addClass('crop--snapshot');
	$('.crop-cam').text('(still)');
	$('.disc--camera').fadeOut();
	$('.disc--retake').removeClass('is-faded');

	setTimeout(function () {
		$('.disc--retake').addClass('is-moved');
		$('.disc--share').removeClass('is-gone');
		$('.crop').removeClass('crop--snapshot');
	}, 500);

});

$('.disc--retake').on('click', function () {
	$("canvas").addClass("is-gone");
	$("video").removeClass('is-gone');
	$('.disc--share').addClass('is-gone');
	$('.disc--retake').removeClass('is-moved');
	$('h4').addClass("is-gone");

	setTimeout(function () {
		$('.disc--retake').addClass('is-faded');
		$('.disc--camera').fadeIn(100);
	}, 500);

});

$('.disc--share').on('click', function () {
	console.log("SHARE CLICKED");
	var ImageURL = canvas.toDataURL('image/jpeg')
	var block = ImageURL.split(";");
	var contentType = block[0].split(":")[1];
	var realData = block[1].split(",")[1];

	var blob = b64toBlob(realData, contentType);
	var formData = new FormData();
	formData.append('image', blob);

	var xmlhttp = new XMLHttpRequest();
	xmlhttp.open("POST", "https://hack3r.herokuapp.com/image");
	xmlhttp.onreadystatechange = function () {
		if (xmlhttp.readyState == 4 && xmlhttp.status == 200) {
			
			var response = JSON.parse(xmlhttp.responseText);
			var pred = response["name"];
			var value = response["value"];
			var text = "Sorry, " + pred + " cannot be recycled 😢";

			if(yellowBinItems.hasOwnProperty(value)){
				text = "Yay! " + pred + " can be recycled 🎉" ;
			}

			$('h4').removeClass("is-gone");
			$('h4').text(text);
		}
	}
	xmlhttp.send(formData);
});
