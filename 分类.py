import os
import random
import shutil
from pathlib import Path

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    将 YOLO 格式的数据集划分为训练集、验证集和测试集。
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "比例之和必须等于1"
    
    images = list(Path(images_dir).glob("*.jpg"))
    random.shuffle(images)
    
    train_count = int(len(images) * train_ratio)
    val_count = int(len(images) * val_ratio)
    
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]
    
    for split_name, split_images in zip(["train", "val", "test"], [train_images, val_images, test_images]):
        image_output_path = Path(output_dir) / "images" / split_name
        label_output_path = Path(output_dir) / "labels" / split_name
        image_output_path.mkdir(parents=True, exist_ok=True)
        label_output_path.mkdir(parents=True, exist_ok=True)
        
        for image in split_images:
            label_file = Path(labels_dir) / (image.stem + ".txt")
            
            shutil.copy(image, image_output_path / image.name)
            if label_file.exists():
                shutil.copy(label_file, label_output_path / label_file.name)
    
    print("数据集划分完成！")

# 使用示例
split_dataset(
    images_dir="data/images", 
    labels_dir="data/labels",
    output_dir="data/split"
)
