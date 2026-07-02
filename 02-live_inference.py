import cv2
import time
import os
from ultralytics import YOLO

# =================================================================
# ⚙️ 边缘部署实验控制台
# =================================================================
ENGINE_MODE = "FP16"  # 👈 随时修改为 "FP16" 或 "INT8" 展开对撞实验
input_video = "test.mp4"    # 👈 你要测试的视频源

# ==========================================
# 📂 自动化目录架构：以“输入视频”为核心建立档案库
# ==========================================
BASE_OUTPUT_DIR = "experiment_results"

# 自动提取视频名称（例如从 "test.mp4" 提取出 "test"）
video_basename = os.path.splitext(os.path.basename(input_video))[0]

# 组装该视频的专属存放文件夹名称，例如: experiment_results/test_results
current_exp_dir = os.path.join(BASE_OUTPUT_DIR, f"{video_basename}_results")

# 自动创建专属文件夹
os.makedirs(current_exp_dir, exist_ok=True)

# 根据模式，将输出文件安全地放入专属文件夹中
if ENGINE_MODE == "FP16":
    model_path = "yolov8n.engine"
    output_video = os.path.join(current_exp_dir, "output_trt_fp16.mp4")
elif ENGINE_MODE == "INT8":
    model_path = "yolov8n.engine" 
    output_video = os.path.join(current_exp_dir, "output_trt_int8.mp4")
else:
    print("❌ 模式选择错误，请输入 'FP16' 或 'INT8'")
    exit()

# ==========================================
# 1. 动态加载对应的 TensorRT 加速引擎
# ==========================================
print(f"🚀 正在加载 【{ENGINE_MODE}】 模式的 TensorRT 加速引擎...")
model = YOLO(model_path, task="detect")

cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print(f"❌ 无法打开视频文件 {input_video}")
    exit()

# ==========================================
# 2. 初始化视频写入器 (VideoWriter)
# ==========================================
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps_input, (width, height))

print(f"🎬 视频加载成功！当前引擎基座: {model_path}")
print(f"📂 档案库激活！当前成果将安全写入专属文件夹: '{output_video}'...")

# ==========================================
# 3. 视频流处理循环 (无弹窗模式)
# ==========================================
prev_time = time.time()
frame_count = 0

while True:
    success, frame = cap.read()
    if not success:
        print(f"\n✅ 视频处理完毕！总共处理了 {frame_count} 帧。")
        print(f"💾 【独立归档成功】: 请前往该子文件夹整理对比: {current_exp_dir}")
        break

    current_time = time.time()

    # TensorRT 高速推理 
    results = model(frame, stream=True, verbose=False)

    for r in results:
        annotated_frame = r.plot()

    # 计算端到端性能指标
    processing_time = current_time - prev_time
    fps = 1 / processing_time if processing_time > 0 else 0
    latency_ms = processing_time * 1000
    prev_time = current_time

    # 绘制性能指标标签
    cv2.putText(annotated_frame, f"Mode: {ENGINE_MODE}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Latency: {latency_ms:.1f} ms", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

    out.write(annotated_frame)
    
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"⚡ [{ENGINE_MODE} 模式] 正在处理中... 当前瞬时 FPS: {fps:.1f} | 延迟: {latency_ms:.1f} ms")

cap.release()
out.release()