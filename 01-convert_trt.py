import os
from ultralytics import YOLO

# =================================================================
# ⚙️ 边缘部署实验控制台：在这里选择你的编译模式
# =================================================================
# 选项 "FP16" : 编译为半精度浮点数引擎（速度快，精度几乎无损）
# 选项 "INT8" : 编译为8位整型引擎（速度狂飙，体积减半，但需要校准集防止精度崩塌）
MODE = "INT8"  # 👈 你可以在这里随时修改为 "FP16" 或 "INT8" 进行对比实验

# 1. 核心：加载原始 PyTorch 模型
print("📦 正在加载原始 PyTorch 权重文件...")
model = YOLO("yolov8n.pt")

print(f"🚀 当前激活实验模式: 【{MODE}】")
print("⚠️ 正在唤醒英伟达 TensorRT 编译器，进行底层算子融合与计算图重构...")

# 2. 根据用户选择，执行不同的编译策略
if MODE == "FP16":
    # 开启 half=True 激活 FP16 模式
    success_path = model.export(
        format="engine", 
        half=True, 
        workspace=4
    )
    print(f"\n🎉 【FP16】加速引擎已成功生成！存储路径: {success_path}")

elif MODE == "INT8":
    # 开启 int8=True 并指定 data 触发量化校准流水线
    # 框架会自动下载包含 8 张图的 coco8 极小数据集作为 Calibration Dataset
    success_path = model.export(
        format="engine", 
        int8=True, 
        data="coco8.yaml", 
        workspace=4
    )
    print(f"\n🎉 【INT8】极致量化引擎已成功生成！存储路径: {success_path}")
    print("💡 提示: 检查你项目目录下，会多出一个专属的 INT8 .engine 文件。")

else:
    print("❌ 模式选择错误，请输入 'FP16' 或 'INT8'")