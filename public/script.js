const socket = io("/");
let socketclientid;
const myPeer = new Peer();

let speechRec = new p5.SpeechRec("en-US", gotSpeech);

let mediaRecorder;
let continuous = true;
let interim = true;
// shim for AudioContext when it's not avb.
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext; //audio context to help us record

function gotSpeech() {
  if (speechRec.resultValue) {
    socket.emit("message", {
      message: speechRec.resultString,
      roomId: ROOM_ID,
    });
  }
}

let transcriptDiv = document.getElementById("transcript");

socket.on("clientid", (id) => {
  socketclientid = id;
  console.log(socketclientid);
});

socket.on("message", (payload) => {
  const { message, user } = payload;
  console.log(message);
  let d = document.createElement("div");
  d.textContent = message;
  d.classList.add("bubble");
  if (transcriptDiv.children.length < 1) {
    d.classList.add("mt-auto");
  }
  if (user.toString() == socketclientid.toString()) d.classList.add("me");
  else d.classList.add("other");
  transcriptDiv.appendChild(d);
  transcriptDiv.scrollTo(0, transcriptDiv.scrollHeight);
});

const myVideo = document.createElement("video");
var transcriptContainer = document.getElementById("transcript");

var seconds = 00;
var tens = 00;
var interval;

myVideo.muted = true;
const peers = {};
navigator.mediaDevices
  .getUserMedia({
    audio: true,
    video: false,
  })
  .then((stream) => {
    addVideoStream(myVideo, stream);
    myPeer.on("call", (call) => {
      call.answer(stream);
      const video = document.createElement("video");
      call.on("stream", (userVideoStream) => {
        audioContext = new AudioContext();
        input = audioContext.createMediaStreamSource(stream);
        mediaRecorder = new Recorder(input, { numChannels: 1 });
        initCall();
        addVideoStream(video, userVideoStream);
      });
    });

    socket.on("user-connected", (userId) => {
      audioContext = new AudioContext();
      input = audioContext.createMediaStreamSource(stream);
      mediaRecorder = new Recorder(input, { numChannels: 1 });
      initCall();
      connectToNewUser(userId, stream);
    });
  });

socket.on("user-disconnected", (userId) => {
  if (peers[userId]) {
    peers[userId].close();
    speechRec.stop();
    mediaRecorder.stop();
    window.location.reload();
  }
});

myPeer.on("open", (id) => {
  socket.emit("join-room", ROOM_ID, id);
});

function connectToNewUser(userId, stream) {
  const call = myPeer.call(userId, stream);
  const video = document.createElement("video");
  call.on("stream", (userVideoStream) => {
    addVideoStream(video, userVideoStream);
  });
  call.on("close", () => {
    video.remove();
  });

  peers[userId] = call;
}

function initCall() {
  transcriptContainer.classList.remove("opacity-0");
  calling.innerHTML = `<span id="appendMin">00</span>:<span id="appendSeconds">00</span>`;
  clearInterval(interval);
  interval = setInterval(startTimer, 1000);
  speechRec.start(continuous, interim);
  mediaRecorder.record();
  console.log(mediaRecorder.recording);
  recording = setInterval(startRecording, 5000);
}

function startRecording() {
  var tempMediaRecorder = mediaRecorder;
  mediaRecorder = new Recorder(input, { numChannels: 1 });
  if (tempMediaRecorder.recording) {
    tempMediaRecorder.stop();
    mediaRecorder.record();
  }
  tempMediaRecorder.exportWAV(createDownloadLink);
  var tempMediaRecorder;
}

function createDownloadLink(blob) {
  var url = URL.createObjectURL(blob);
  console.log(url);
  let formData = new FormData();
  formData.append("audioBlob", blob);
  console.log("audioBlob", blob);
  $.ajax({
    type: "POST",
    url: "http://localhost:5000/get-blob-data",
    data: formData,
    contentType: false,
    processData: false,
    success: function (result) {
      console.log("success", result);
    },
    error: function (result) {
      console.log("sorry an error occured");
    },
  });
}

function startTimer() {
  var appendMin = document.getElementById("appendMin");
  var appendSeconds = document.getElementById("appendSeconds");
  tens++;
  if (tens <= 9) {
    appendSeconds.innerHTML = "0" + tens;
  }
  if (tens > 9) {
    appendSeconds.innerHTML = tens;
  }
  if (tens > 60) {
    seconds++;
    appendMin.innerHTML = "0" + seconds;
    tens = 0;
    appendSeconds.innerHTML = "0" + 0;
  }
  if (seconds > 9) {
    appendMin.innerHTML = seconds;
  }
}

function addVideoStream(video, stream) {
  video.srcObject = stream;
  video.addEventListener("loadedmetadata", () => {
    video.play();
  });
}
