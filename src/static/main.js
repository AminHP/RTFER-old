////////////////////////// video stream //////////////////////////

function initVideoStream()
{
    if(navigator.mediaDevices.getUserMedia)
        navigator.mediaDevices.getUserMedia(constraints).then(getUserMediaSuccess).catch(errorHandler);
    else
        alert("Your browser does not support getUserMedia API");

    canvas.width = width;
    canvas.height = height;
    setInterval(drawAll, 1000. / 30);
}

function getUserMediaSuccess(stream)
{
    video.src = window.URL.createObjectURL(stream);
}

function errorHandler(error)
{
    console.log("error: " + error);
}


////////////////////////// socketio //////////////////////////

function connect()
{
    url = location.protocol + '//' + location.hostname + ':' + location.port + '/fer';
    socket = io.connect(url);
    socket.on('predicted', onPredicted);
}

function sendBySocket(event, message)
{
    socket.emit(event, message);
}


////////////////////////// ui //////////////////////////

function drawAll()
{
    context.drawImage(video, 0, 0, width, height);

    // draw face rects
    for (i in lastResult)
    {
        border = lastResult[i].border;
        emotion = lastResult[i].emotion;

        // draw border
        context.beginPath();
        context.lineWidth = 2;
        context.strokeStyle = 'red';
        context.rect(border.x, border.y, border.width, border.height);
        context.stroke();

        // draw emotion
        fontSize = border.height / 5;
        context.font = fontSize + "px Comic Sans MS";
        context.fillStyle = "green";
        context.textAlign = "center";
        context.fillText(emotion, border.x + (border.width / 2), border.y + fontSize);
    }
}


////////////////////////// control //////////////////////////

function onPredicted(message)
{
    //console.log(message);
    lastResult = message;

    emotions = [];
    for (i in message)
        emotions.push(message[i].emotion);
    console.log("emotions: " + emotions);
}

function predict()
{
    startTime = (new Date()).getTime();
    canvas.toBlob(
        function (blob) {
            message = { image: blob };
            sendBySocket('predict', message);
            endTime = (new Date()).getTime();
            //console.log("image size: " + blob.size);
            //console.log("process time: " + (endTime - startTime));
        },
        'image/jpeg',
        quality
    );
}


function run()
{
    setInterval(predict, 1000. / fps);
}

function start()
{
    initVideoStream();
    connect();
    run();
}
