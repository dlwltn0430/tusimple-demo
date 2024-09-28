# preprocess_labels.py
import json
import os
import cv2

# 경로 설정
label_dir = 'datasets/tusimple/labels'  # TuSimple 라벨 파일 경로
image_dir = 'datasets/tusimple/images'  # 이미지 경로
output_dir = 'datasets/train/labels'    # YOLO 형식 라벨 파일을 저장할 경로

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# TuSimple JSON 데이터를 YOLO 형식으로 변환하는 함수
def convert_label(json_file, output_dir):
    # JSON 파일을 한 줄씩 읽어서 처리
    with open(json_file, 'r') as f:
        for line in f:  # JSON 객체가 한 줄씩 있을 때 각각 처리
            try:
                item = json.loads(line)  # 각 줄을 JSON 객체로 변환
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in file {json_file}: {e}")
                continue
            
            lanes = item['lanes']  # 차선 좌표 리스트
            h_samples = item['h_samples']  # 차선 y 좌표 (고정 높이)
            raw_file = item['raw_file']  # 이미지 파일 경로
            
            # 이미지 경로에서 'clips/' 부분 제거 및 이미지 파일 경로 구성
            image_path = os.path.join(image_dir, raw_file.replace('clips/', ''))

            # 라벨 파일의 이름 구성 (폴더 구조와 이미지 이름을 반영)
            relative_path = os.path.relpath(image_path, image_dir)  # 이미지 경로에서 root 디렉토리로부터의 상대 경로
            folder_structure = os.path.dirname(relative_path).replace(os.sep, '_')  # 폴더 구조를 '_'로 결합
            label_filename = f"{folder_structure}_{os.path.basename(raw_file).replace('.jpg', '.txt')}"  # 새로운 파일명

            # 라벨 파일의 경로 설정
            output_path = os.path.join(output_dir, label_filename)

            # 이미지 크기를 읽어옴
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error loading image {image_path}")
                continue
            img_height, img_width = img.shape[:2]

            # YOLO 세그멘테이션 라벨 형식: 클래스 번호와 폴리곤 좌표
            with open(output_path, 'w') as out_file:
                for lane in lanes:
                    # 유효한 좌표만 필터링 (x 좌표가 -2인 경우 차선이 없음)
                    lane_points = [(x, y) for x, y in zip(lane, h_samples) if x >= 0]

                    # 폴리곤 좌표를 YOLO 형식으로 변환
                    if len(lane_points) > 1:
                        polygon = []
                        for x, y in lane_points:
                            # 이미지 크기 대비 상대 좌표로 변환
                            rel_x = x / img_width
                            rel_y = y / img_height
                            polygon.append(f"{rel_x} {rel_y}")

                        # YOLO 형식: 클래스 번호와 폴리곤 좌표
                        out_file.write(f"0 {' '.join(polygon)}\n")

if __name__ == '__main__':
    # 모든 JSON 라벨 파일을 처리
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]

    for label_file in label_files:
        json_path = os.path.join(label_dir, label_file)
        convert_label(json_path, output_dir)

    print(f"Converted {len(label_files)} label files to YOLO format.")