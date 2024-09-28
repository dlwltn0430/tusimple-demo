import cv2
import numpy as np
import os
import json

# 경로 설정
image_dir = 'datasets/tusimple/images'  # 실제 이미지가 저장된 경로
label_dir = 'datasets/tusimple/labels'  # label.json 파일이 있는 경로
mask_dir = 'datasets/train/masks'       # 마스크가 저장될 경로

os.makedirs(mask_dir, exist_ok=True)

def create_lane_mask(image_path, labels, mask_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    # 차선 좌표에 따라 마스크 생성
    for item in labels:
        lanes = item['lanes']  # 여러 차선 좌표 리스트
        h_samples = item['h_samples']  # 차선 y 좌표 (고정 높이)

        for lane in lanes:
            # 유효한 좌표만 필터링 (x 좌표가 -2인 경우 차선이 없음)
            lane_points = [(x, y) for x, y in zip(lane, h_samples) if x >= 0]

            # 차선 좌표에 따라 마스크에 그리기
            for x, y in lane_points:
                cv2.circle(mask, (x, y), 5, 255, -1)  # 차선 위치에 원을 그림

    # 마스크 저장
    cv2.imwrite(mask_path, mask)

if __name__ == '__main__':
    # 모든 label.json 파일을 처리
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]

    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)

        # JSON 파일을 한 줄씩 읽어서 각 줄을 개별적으로 처리
        with open(label_path, 'r') as f:
            labels = []
            for line in f:
                try:
                    labels.append(json.loads(line))  # 각 줄을 JSON 객체로 변환
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in file {label_path}: {e}")
                    continue

        # 레이블 파일의 각 데이터에서 이미지 파일 경로 추출 및 처리
        for item in labels:
            raw_file = item['raw_file']  # ex: clips/0601/1495058713562969635/20.jpg
            image_path = os.path.join(image_dir, raw_file.replace('clips/', ''))

            # 기존의 폴더 구조를 파일명에 반영하여 마스크 파일명 생성
            relative_path = raw_file.replace('clips/', '')
            folder_structure = relative_path.replace(os.sep, '_').replace('/', '_')  # 폴더 구조와 파일명 결합
            mask_filename = f"{folder_structure}"

            mask_path = os.path.join(mask_dir, mask_filename)

            # 마스크 생성
            create_lane_mask(image_path, [item], mask_path)

            print(f"Mask saved at {mask_path}")