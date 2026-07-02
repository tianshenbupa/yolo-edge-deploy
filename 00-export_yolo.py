import torch
from ultralytics import YOLO

# 1. 自动从官方加载一个最轻量、最适合边缘端的目标检测模型 YOLOv8n (Nano)
# 它包含工业级的所有算子，但体积小、速度极快
print("📦 正在加载 YOLOv8n 模型...")
model = YOLO("yolov8n.pt") 

# 2. 一键导出为万能的 ONNX 格式
# format="onnx" 是核心
# dynamic=True 表示允许输入的图片尺寸是动态的（面试加分项）
print("🔄 正在将模型转换为通用的 ONNX 格式...")
success_path = model.export(format="onnx", dynamic=True)

print(f"\n🎉 大功告成！ONNX 模型已成功生成，保存在: {success_path}")