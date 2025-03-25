from ultralytics import YOLO

# 加载训练好的模型（确保路径正确）
model = YOLO("runs/detect/train5/weights/best.pt")

# 导出 ONNX 格式
model.export(format="onnx")

# 导出 TorchScript 格式
model.export(format="torchscript")
