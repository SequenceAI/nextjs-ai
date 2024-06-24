from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import time
import asyncio
from ultralytics import YOLOWorld, YOLO
import cv2
import numpy as np
app = FastAPI()
# Initialize the FastAPI app
origins = [
    "http://localhost:3000",  # Add the URL of your Next.js frontend
    # Add more origins as needed
]
# Your OpenAI client setup
model = YOLOWorld('yolov8s-world.pt')  # Use the appropriate model
pose_model = YOLO("yolov8n-pose.pt")  # load an official model
@app.get("/api/python")
def hello_world():
    return {"message": "Hello World"}


@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                print("inside frame")
                results = model.predict(frame)
                detections = results[0].tojson()

                # Sending the detections as JSON
                await websocket.send_json({"detections": detections})

            await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await websocket.close()

# @app.websocket("/ws/video")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         data = await websocket.receive_bytes()
#         nparr = np.frombuffer(data, np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#
#         if frame is not None:
#             print("inside frame")
#             results = model.predict(frame)
#             # print('results', results[0].show())
#             # print('results', results[0].tojson())
#             detections = results[0].tojson()
#
#             # Sending the detections as JSON
#             await websocket.send_json({"detections": detections})
#
#         await asyncio.sleep(0.01)

