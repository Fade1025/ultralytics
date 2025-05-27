from ultralytics import YOLO
import os

path = os.getcwd()

model_path = os.path.join(path, "ultralytics", "cfg", "models", "11", "yolo11-pose-edit.yaml")
weight_path = os.path.join(path, "weights/yolo11m-pose.pt")
data_path = os.path.join(path, "ultralytics", "cfg", "datasets", "coco-pose-edit.yaml")

model = YOLO(model_path).load(weight_path)  # build from YAML and transfer weights

# Train the model
results = model.train(data=data_path, 
                      epochs=100, 
                      imgsz=640,
                      batch=64)

#原始yolo11
# model = YOLO("./ultralytics/cfg/models/11/yolo11-pose.yaml").load("/home/xyhpc/文档/yolo_learn/ultralytics/weights/yolo11m-pose.pt")  # build from YAML and transfer weights

# # Train the model
# results = model.train(data="./ultralytics/cfg/datasets/coco-pose.yaml", 
#                       epochs=100, 
#                       imgsz=640,
#                       batch=64)
