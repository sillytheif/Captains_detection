from ultralytics import YOLO

model = YOLO('yolov8m.yaml')
model = YOLO("yolov8m.pt") 
results = model.train(data='match.yaml',epochs=400)
