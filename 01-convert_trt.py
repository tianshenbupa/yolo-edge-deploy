from ultralytics import YOLO

# 1. 核心修复：必须从原始的 .pt 权重开始直接编译成 TensorRT
print("📦 正在加载原始 PyTorch 模型...")
model = YOLO("yolov8n.pt")

# 2. 一键编译转换成英伟达独占的 TensorRT 格式 (.engine)
# half=True 表示开启 FP16 半精度加速，这能让你的 5060 Ti 速度再翻倍！
print("🚀 正在激活 TensorRT 编译器，针对你的 RTX 5060 Ti 进行硬件级暴力优化...")
success_path = model.export(format="engine", half=True)

print(f"\n🎉 恭喜！英伟达 TensorRT 加速引擎已生成：{success_path}")