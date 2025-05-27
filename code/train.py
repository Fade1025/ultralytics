from matplotlib import scale
from ultralytics import YOLO
import os
if __name__ == "__main__":

    path = os.getcwd()

    version = "11"  # Specify the YOLO version
    if version == "v8":
        model_path = os.path.join(path, "ultralytics", "cfg", "models", "v8", "yolov8s.yaml")
        weight_path = os.path.join(path, "weights/yolov8s.pt")
        data_path = os.path.join(path, "ultralytics/cfg/datasets/VOC-edit.yaml")
    elif version == "11":
        model_path = os.path.join(path, "ultralytics", "cfg", "models", "11", "yolo11n-pose-edit.yaml")
        
        data_path = os.path.join(path, "ultralytics", "cfg", "datasets", "coco-pose-edit.yaml")
        
    
    model = YOLO(model_path)
    # Train the model
    results = model.train(data=data_path, 
                          epochs=100, 
                          imgsz=640,
                          batch=16)  # Adjust batch size as needed