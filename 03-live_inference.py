import cv2
import time
from ultralytics import YOLO

# ==========================================
# 1. 加载模型与输入流
# ==========================================
print("🚀 正在加载 TensorRT 加速引擎...")
model = YOLO("yolov8n.engine", task="detect")

input_video = "test.mp4"
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print(f"❌ 无法打开视频文件 {input_video}")
    exit()

# ==========================================
# 2. 初始化视频写入器 (VideoWriter)
# ==========================================
# 获取原视频的分辨率和帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = int(cap.get(cv2.CAP_PROP_FPS))

# 定义编码器并创建输出文件
output_video = "output_trt.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps_input, (width, height))

print(f"🎬 视频加载成功！正在后台全速推理，并将结果写入 '{output_video}'...")

# ==========================================
# 3. 视频流处理循环 (无弹窗模式)
# ==========================================
prev_time = time.time()
frame_count = 0

while True:
    success, frame = cap.read()
    if not success:
        print(f"\n✅ 视频处理完毕！总共处理了 {frame_count} 帧。")
        print(f"💾 请在当前目录下查看结果文件: {output_video}")
        break

    current_time = time.time()

    # TensorRT 高速推理
    results = model(frame, stream=True, verbose=False)

    for r in results:
        annotated_frame = r.plot()

    # 计算 FPS
    processing_time = current_time - prev_time
    fps = 1 / processing_time if processing_time > 0 else 0
    latency_ms = processing_time * 1000
    prev_time = current_time

    # 绘制性能指标
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Latency: {latency_ms:.1f} ms", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

    # 【核心改变】：将渲染好的帧写入输出视频，而不是尝试用 imshow 弹窗
    out.write(annotated_frame)
    
    frame_count += 1
    # 每处理 30 帧在终端打印一次进度，让你知道程序没卡死
    if frame_count % 30 == 0:
        print(f"⚡ 正在处理... 当前瞬时 FPS: {fps:.1f}")

# 释放资源
cap.release()
out.release()