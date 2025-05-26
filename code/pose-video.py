import cv2
from ultralytics import YOLO

def process_video(input_path, output_video_path, output_txt_path):
    # 初始化模型
    model = YOLO('runs/pose/train7-yolo-11-hands-200/weights/best.pt')  # 使用姿态估计模型
    
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    # 获取视频参数
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # 创建文本文件
    with open(output_txt_path, 'w') as f:
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 使用YOLO进行推理
            results = model.predict(frame, conf=0.5, verbose=False)
            
            # 提取关键点数据
            keypoints_data = results[0].keypoints.data.cpu().numpy()
            
            # 写入关键点数据
            f.write(f"Frame {frame_count}\n")
            for person_id, person_kpts in enumerate(keypoints_data):
                f.write(f"Person {person_id}:\n")
                for kpt_id, (x, y, conf) in enumerate(person_kpts):
                    f.write(f"{kpt_id:2}: ({x:6.1f}, {y:6.1f}), conf: {conf:.2f}\n")
                f.write("\n")
            
            # 绘制骨骼点并写入视频
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            
            frame_count += 1
            print(f"Processed frame {frame_count}", end='\r')

    # 释放资源
    cap.release()
    out.release()
    print("\nProcessing completed!")

if __name__ == "__main__":
    input_video = "ultralytics/assets/fall-01-cam0.mov"
    output_video = "outputs/output.mp4"
    output_txt = "outputs/keypoints.txt"
    
    process_video(input_video, output_video, output_txt)