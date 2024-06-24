from ultralytics import YOLOWorld, YOLO


model = YOLOWorld('yolov8s-world.pt')  # Use the appropriate model

model.predict(source=0,show=True)