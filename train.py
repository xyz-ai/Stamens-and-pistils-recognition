from ultralytics import YOLO

# 选择 YOLOv8 的检测模型（n, s, m, l, x 代表不同规模）
model = YOLO("yolov8s.pt")  # 使用 YOLOv8 小模型

# 训练模型
model.train(data="data.yaml", epochs=50, batch=8, imgsz=640, device="cpu")
