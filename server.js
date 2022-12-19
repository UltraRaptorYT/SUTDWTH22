const express = require("express");
const app = express();
const server = require("http").Server(app);
const io = require("socket.io")(server);
const PORT = 8081;

// Create a socket namespace for chat messages

app.set("view engine", "ejs");
app.use(express.static("public"));


app.get("/", (req, res) => {
  res.render("index");
});

app.get("/class/enter", (req, res) => {
  res.render("class_enter");
})

app.get("/class/:room", (req, res) => {
  res.render("class_call", { roomId: req.params.room });
});

app.get("/:room", (req, res) => {
  res.render("room", { roomId: req.params.room });
});

app.get("/ws/incoming", (req, res) => {
  res.render("incoming_call");
});

app.get("/ws/call/:room",(req,res)=>{
  res.render("call",{ roomId: req.params.room})
  });

io.on("connection", (socket) => {
  console.log("New user connected");
  socket.emit("clientid", socket.id);
  socket.on("join-room", (roomId, userId) => {
    socket.join(roomId);
    socket.to(roomId).broadcast.emit("user-connected", userId);
    socket.on("disconnect", () => {
      socket.to(roomId).broadcast.emit("user-disconnected", userId);
    });
  });
  socket.on("message", (payload) => {
    const { roomId, message } = payload;
    io.to(roomId).emit("message", { user: socket.id, message });
  });
});

server.listen(process.env.PORT || PORT);
console.log(
  `Web server is listening at http://localhost:${process.env.PORT || PORT}/`
);
