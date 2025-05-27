from ultralytics import YOLO
import os
if __name__ == "__main__":

    path = os.getcwd()

    model_path = os.path.join(path, "ultralytics", "cfg", "models", "v8", "yolov8m.yaml")
    weight_path = os.path.join(path, "weights/yolov8s.pt")
    data_path = os.path.join(path, "ultralytics/cfg/datasets/VOC-edit.yaml")
    model = YOLO(model_path)

    # Train the model
    results = model.train(data=data_path, 
                          epochs=100, 
                          imgsz=640,
                          batch=16)  # Adjust batch size as needed