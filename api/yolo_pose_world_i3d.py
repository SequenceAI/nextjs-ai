import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from ultralytics import YOLO

# Keypoints and skeleton configuration
KEYPOINTS_NAME = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# Load models
i3d_model_url = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
i3d = hub.load(i3d_model_url).signatures['default']

KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
labels_path = tf.keras.utils.get_file('label_map.txt', KINETICS_URL)
with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

yolo_world_model = YOLO("yolov8s-world.pt")
yolo_pose_model = YOLO("yolov8n-pose.pt")

# Helper functions
def preprocess_frame(frame, resize=(224, 224)):
    frame = cv2.resize(frame, resize)
    frame = frame[:, :, [2, 1, 0]]  # Convert BGR to RGB
    return frame / 255.0

def predict_action(frames):
    model_input = tf.constant(frames, dtype=tf.float32)[tf.newaxis, ...]
    logits = i3d(model_input)['default'][0]
    probabilities = tf.nn.softmax(logits)
    top_5_indices = np.argsort(probabilities)[::-1][:5]
    return [(labels[i], probabilities[i].numpy()) for i in top_5_indices]

def draw_predictions(frame, top_5):
    for i, (label, prob) in enumerate(top_5):
        text = f"{label}: {prob:.2f}"
        cv2.putText(frame, text, (10, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return frame

def draw_yolo_detections(frame, results, label_color=(0, 255, 0)):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = result.names[cls]
            text = f"{label}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2, cv2.LINE_AA)

def draw_pose_estimations(frame, results):
    for result in results:
        for i, box in enumerate(result.boxes):
            if int(box.cls[0]) == 0:  # Person class
                keypoints = result.keypoints[i].xy[0].cpu().numpy()
                confs = result.keypoints[i].conf[0].cpu().numpy()
                for j, (x, y) in enumerate(keypoints):
                    if confs[j] > 0.5:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                        text = f"{KEYPOINTS_NAME[j]}: ({int(x)}, {int(y)})"
                        cv2.putText(frame, text, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                for start, end in SKELETON:
                    if confs[start] > 0.5 and confs[end] > 0.5:
                        cv2.line(frame, (int(keypoints[start][0]), int(keypoints[start][1])),
                                 (int(keypoints[end][0]), int(keypoints[end][1])), (0, 0, 155), 2)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

frames = []
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for I3D model
    preprocessed_frame = preprocess_frame(frame)
    frames.append(preprocessed_frame)

    if len(frames) == 64:  # Use 64 frames as input for the I3D model
        top_5_predictions = predict_action(np.array(frames))
        frame = draw_predictions(frame, top_5_predictions)
        frames.pop(0)  # Remove the oldest frame to maintain a fixed window of 64 frames

    # YOLO-World object detection
    yolo_world_results = yolo_world_model.predict(source=frame, save=False, save_txt=False, show=False)
    draw_yolo_detections(frame, yolo_world_results, label_color=(0, 255, 0))

    # YOLO-Pose estimation
    yolo_pose_results = yolo_pose_model.predict(source=frame, save=False, save_txt=False, show=False)
    draw_pose_estimations(frame, yolo_pose_results)

    # Display the frame
    cv2.imshow('Webcam Detection and Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
