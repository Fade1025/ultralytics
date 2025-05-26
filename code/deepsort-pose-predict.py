import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def calculate_iou(box1, box2):
    """计算两个矩形框的IOU（交并比）"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算交集区域
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    
    # 计算并集区域
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

# 初始化模型和跟踪器
model = YOLO('weights/yolo11m-pose.pt')
tracker = DeepSort(max_age=10, max_iou_distance=0.8,max_cosine_distance=0.4)

# 视频输入输出设置
cap = cv2.VideoCapture("/home/xyhpc/文档/yolo_learn/ultralytics/ultralytics/assets/Whiplash.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('outputs/output_video.mp4', fourcc, fps, (frame_width, frame_height))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO-Pose检测
    results = model(frame, verbose=False)[0]
    
    # 当前帧的检测数据
    current_boxes = []
    current_keypoints = []
    detections = []
    
    if results.boxes is not None and results.keypoints is not None:
        for i, (box, kps) in enumerate(zip(results.boxes.xyxy.cpu(), results.keypoints.data.cpu())):
            x1, y1, x2, y2 = box.numpy()
            conf = results.boxes.conf.cpu().numpy()[i]
            keypoints = kps.numpy()
            
            # 保存当前帧的检测数据
            current_boxes.append([x1, y1, x2, y2])
            current_keypoints.append(keypoints)
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, keypoints))

    # 更新跟踪器
    tracks = tracker.update_tracks(detections, frame=frame)

    # 绘制跟踪结果
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        # 获取跟踪框信息
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1_trk, y1_trk, x2_trk, y2_trk = map(int, ltrb)
        
        # 绘制跟踪框和ID
        cv2.rectangle(frame, (x1_trk, y1_trk), (x2_trk, y2_trk), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1_trk, y1_trk - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 寻找最佳匹配的检测框（用于获取关键点）
        best_iou = 0.3
        best_kps = None
        trk_box = [x1_trk, y1_trk, x2_trk, y2_trk]
        
        for i, det_box in enumerate(current_boxes):
            iou = calculate_iou(trk_box, det_box)
            if iou > best_iou:
                best_iou = iou
                best_kps = current_keypoints[i]
        
        # 仅绘制关键点（不画连线）
        if best_kps is not None:
            for kp in best_kps:
                x, y, conf = kp
                # 过滤低置信度并检查坐标有效性
                if conf > 0.2 and 0 <= x < frame_width and 0 <= y < frame_height:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    # 写入输出视频
    out.write(frame)
    
    # 打印进度
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"已处理 {frame_count} 帧...")

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
print("处理完成，结果已保存至 output_video.mp4")