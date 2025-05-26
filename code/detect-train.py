from ultralytics import YOLO


model = YOLO("./ultralytics/cfg/models/v8/yolov8.yaml").load("./weights/yolov8s.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="./ultralytics/cfg/datasets/VOC-2.yaml", 
                      epochs=2, 
                      imgsz=640)