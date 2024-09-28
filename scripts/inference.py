from ultralytics import YOLO
import argparse
import os

# 파라미터 설정
parser = argparse.ArgumentParser(description="YOLOv8 Inference for Lane Detection")
parser.add_argument('--source', type=str, help="Path to input image", required=True)
parser.add_argument('--model', type=str, default="C:/Users/computer1/Desktop/lane/runs/segment/train45/weights/best.pt", help="Path to trained model")
parser.add_argument('--output_dir', type=str, default="results/output", help="Directory to save results")
args = parser.parse_args()

# 출력 디렉토리 생성
os.makedirs(args.output_dir, exist_ok=True)

# 모델 로드
model = YOLO(args.model)

# 이미지 추론
results = model(args.source, task='segment', conf=0.25)

# 결과 출력 (리스트 각 항목에 대해 show와 save 수행)
for result in results:
    result.show()  # 시각적으로 결과 보여줌
    result.save(args.output_dir)  # 결과 저장

print(f"Inference completed. Results saved in {args.output_dir}.")
