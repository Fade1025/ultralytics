import os
from glob import glob

# 设置路径
input_root = '/home/xyhpc/文档/yolo_learn/ultralytics/coco-pose/labels-original'
output_root = '/home/xyhpc/文档/yolo_learn/ultralytics/coco-pose/edited-labels'

# 需要保留的关键点索引
selected_kpt_idx = [0, 5, 6, 9, 10, 11, 12, 15, 16]

def process_label_line(line, line_num, file_path):
    parts = line.strip().split()
    if len(parts) < 5 + 17 * 3:
        print(f"[⚠️跳过] {file_path}, line {line_num}: 关键点数不足，共 {len(parts)} 字段")
        return None

    cls, box = parts[0], parts[1:5]
    kpts = list(map(float, parts[5:]))

    if len(kpts) != 51:
        print(f"[⚠️跳过] {file_path}, line {line_num}: 关键点部分不是51个值")
        return None

    # 裁剪关键点
    new_kpts = []
    for idx in selected_kpt_idx:
        i = idx * 3
        new_kpts.extend(kpts[i:i+3])

    return ' '.join([cls] + box + list(map(str, new_kpts)))

def process_all_labels(input_dir, output_dir):
    label_files = glob(os.path.join(input_dir, '*.txt'))
    os.makedirs(output_dir, exist_ok=True)

    for file in label_files:
        with open(file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for i, line in enumerate(lines):
            new_line = process_label_line(line, i + 1, file)
            if new_line:
                new_lines.append(new_line)

        out_path = os.path.join(output_dir, os.path.basename(file))
        with open(out_path, 'w') as f:
            f.write('\n'.join(new_lines) + '\n')

        print(f'✅ 处理完成: {file} -> {out_path}，共 {len(new_lines)} 行')

def main():
    for subfolder in ['train2017', 'val2017']:
        input_dir = os.path.join(input_root, subfolder)
        output_dir = os.path.join(output_root, subfolder)
        process_all_labels(input_dir, output_dir)

if __name__ == '__main__':
    main()