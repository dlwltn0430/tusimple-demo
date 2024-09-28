import os
import shutil
import random

# 데이터셋 경로 설정
dataset_dir = 'datasets/tusimple/images'
train_dir = 'datasets/train/images'
val_dir = 'datasets/val/images'

# 디렉토리 생성
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 재귀적으로 하위 폴더에서 모든 이미지 파일을 찾음
def find_all_images(directory):
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                image_files.append(os.path.join(root, file))
    return image_files

# 데이터셋을 불러와서 train/val으로 나누기
def split_dataset(image_files, split_ratio=0.8):
    random.shuffle(image_files)
    split_index = int(len(image_files) * split_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]
    return train_files, val_files

# 이미지 파일을 대상 폴더로 복사
def move_files(files, target_dir):
    for file in files:
        # 폴더 구조를 '_'로 연결하여 파일 이름 생성
        relative_path = os.path.relpath(file, dataset_dir)  # 예: 0313-1/60/20.jpg
        folder_structure = os.path.dirname(relative_path)   # 예: 0313-1/60
        file_name = os.path.basename(relative_path)         # 예: 20.jpg
        
        # 새로운 파일명 생성: 0313-1_60_20.jpg
        new_filename = f"{folder_structure.replace(os.sep, '_')}_{file_name}"
        
        # 파일을 대상 폴더로 복사
        shutil.copy(file, os.path.join(target_dir, new_filename))

if __name__ == '__main__':
    # 모든 이미지 파일을 검색
    image_files = find_all_images(dataset_dir)

    # train/val 파일 나누기
    train_files, val_files = split_dataset(image_files)

    # 파일 복사
    move_files(train_files, train_dir)
    move_files(val_files, val_dir)

    print(f"Train set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")