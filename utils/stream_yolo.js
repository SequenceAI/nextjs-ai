export async function startStreaming(videoElement, backendUrl) {
  const socket = new WebSocket(backendUrl);

  let drawingCanvas = null; // Canvas for drawing bounding boxes

  socket.onopen = () => {
    console.log("WebSocket connection established. Starting to send frames.");

    const videoParent = videoElement.parentNode;
    videoParent.style.position = 'relative';

    // Create a canvas for sending frames
    const captureCanvas = document.createElement("canvas");
    captureCanvas.width = videoElement.videoWidth;
    captureCanvas.height = videoElement.videoHeight;
    const captureContext = captureCanvas.getContext("2d");

    // Dynamically create an overlay canvas for drawing
    drawingCanvas = document.createElement("canvas");
    drawingCanvas.width = videoElement.offsetWidth;
    drawingCanvas.height = videoElement.offsetHeight;
    drawingCanvas.style.position = "absolute";
    drawingCanvas.style.left = '0px';
    drawingCanvas.style.top = '0px';
    drawingCanvas.style.zIndex = '10';
    videoParent.appendChild(drawingCanvas);

    const sendFrame = () => {
      captureContext.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);
      captureCanvas.toBlob((blob) => {
        if (blob) {
          blob.arrayBuffer().then((arrayBuffer) => {
            socket.send(arrayBuffer);
          });
        } else {
          console.error("Failed to create blob from canvas. Retrying...");
          // setTimeout(sendFrame, 100); // Retry after a short delay
        }
      }, "image/jpeg");
    };

    // Send frames every 100ms
    const intervalId = setInterval(() => {
      if (socket.readyState === WebSocket.OPEN) {
        sendFrame();
      } else {
        console.log("WebSocket is not open. Waiting to send frames.");
      }
    }, 100);

    socket.onclose = () => {
      console.log("WebSocket connection closed. Stopping frame transmission.");
      clearInterval(intervalId);
      if (drawingCanvas && drawingCanvas.parentNode) {
        drawingCanvas.parentNode.removeChild(drawingCanvas); // Clean up the drawing canvas
      }
    };
  };

  socket.onerror = (error) => {
    console.error("WebSocket error:", error);
  };

  socket.onmessage = (event) => {

     const data = JSON.parse(event.data);
    // if (data && onUpdateDetectionResults) {
    //   onUpdateDetectionResults(data);
    // }


    if (!drawingCanvas) return;
    console.log("Received data:", event.data);
    const message = JSON.parse(event.data);

    const ctx = drawingCanvas.getContext("2d");
    ctx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);

    if (message.detections) {
      const detections = JSON.parse(message.detections);
      detections.forEach((det) => {
        const { box, name, confidence } = det;
        ctx.strokeStyle = "blue";
        ctx.lineWidth = 2;
        ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
        ctx.font = "18px Arial";
        ctx.fillStyle = "yellow";
        ctx.fillText(`${name} (${confidence.toFixed(2)})`, box.x1, box.y1 - 10);
      });
    }
  };
}