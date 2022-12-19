const socket = io("/");
let socketclientid;
const myPeer = new Peer();

let speechRec = new p5.SpeechRec("en-US", gotSpeech);

let continuous = true;
let interim = false;

speechRec.start(continuous, interim);

function gotSpeech() {
  if (speechRec.resultValue) {
    console.log(speechRec.resultString);
    console.log(socketclientid);
    socket.emit("message", {
      message: speechRec.resultString,
      roomId: ROOM_ID,
    });
  }
}

let transcriptDiv = document.getElementById("transcript");

socket.on("clientid", (id) => {
  socketclientid = id;
});

socket.on("message", (payload) => {
  const { message, user } = payload;
  console.log(message);
  let d = document.createElement("div");
  d.textContent = message;
  d.classList.add("bubble");
  if (user.toString() == socketclientid.toString()) d.classList.add("me");
  else d.classList.add("other");
  transcriptDiv.appendChild(d);
  transcriptDiv.scrollTo(0, transcriptDiv.scrollHeight);
});

const myVideo = document.createElement("video");
const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
const calling = document.getElementById("calling");
var transcriptContainer = document.getElementById("transcript");

var seconds = 00;
var tens = 00;
var interval;

myVideo.muted = true;
const peers = {};
navigator.mediaDevices
  .getUserMedia({
    audio: true,
  })
  .then((stream) => {
    addVideoStream(myVideo, stream);

    myPeer.on("call", (call) => {
      call.answer(stream);
      const video = document.createElement("video");
      call.on("stream", (userVideoStream) => {
        initCall();
        addVideoStream(video, userVideoStream);
      });
    });

    socket.on("user-connected", (userId) => {
      initCall();
      connectToNewUser(userId, stream);
    });
  });

socket.on("user-disconnected", (userId) => {
  if (peers[userId]) peers[userId].close();
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

    let chunks = [];
    let mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.start();
    mediaRecorder.pause();
    // console.log(mediaRecorder.state);
    mediaRecorder.requestData();
    mediaRecorder.ondataavailable = (ev) => {
      // console.log(ev.data);
      // chunks.push(ev.data);
      // console.log(chunks);
    };
    setTimeout(() => {
      mediaRecorder.stop();
    }, 1500);
    mediaRecorder.onstop = (ev) => {
      let blob = new Blob(chunks, { type: "audio/wav" });
      // console.log(blob);
      chunks = [];
      let videoURL = window.URL.createObjectURL(blob);
      // console.log(videoURL);
    };
  });
}
