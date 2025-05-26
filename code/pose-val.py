from ultralytics import YOLO

# Load a model
model = YOLO("/home/xyhpc/文档/yolo_learn/ultralytics/weights/yolo11m-pose.pt")

# Validate the model
metrics = model.val(data="/home/xyhpc/文档/yolo_learn/ultralytics/ultralytics/cfg/datasets/coco-pose.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")
print(metrics.box.map)  # map50-95