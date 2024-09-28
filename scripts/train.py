from ultralytics import YOLO

# YOLOv8n-seg 모델 불러오기
model = YOLO('models/yolov8n-seg.pt')

# 데이터셋 경로와 설정
data_config = 'config/data.yaml'

if __name__ == '__main__':
    # 모델 학습
    model.train(data=data_config, epochs=100, imgsz=640, save=True, task='segment')  

    # 학습 완료 후 best.pt 모델 저장
    print("Model training completed.")
