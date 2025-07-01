# predict.py
import os
import glob
from ultralytics import YOLO

def find_latest_weights(runs_dir="runs/ip102", stage_prefix="stage2_unfreeze"):
    """
    在 runs/ip102 下找到最新的 stage2_unfreeze/weights/best.pt
    """
    pattern = os.path.join(runs_dir, f"{stage_prefix}/weights/best*.pt")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"在 {runs_dir} 中未找到任何 best.pt 文件")
    return sorted(paths)[-1]

if __name__ == "__main__":
    # 1. 自动定位权重
    weights = find_latest_weights()
    print(f"使用权重：{weights}")

    # 2. 加载模型
    model = YOLO(weights)

    # 3. 批量推理
    results = model.predict(
        source="images/test",   # 测试集目录
        imgsz=640,              # 可选：推理尺寸
        conf=0.25,              # 置信度阈值
        iou=0.45,               # NMS IoU 阈值
        device=0,               # GPU id 或 "cpu"
        save=True,              # 保存可视化图片
        save_txt=True,          # 保存 txt 标注
        project="runs/detect",  # 输出根目录
        name="predict_ip102",   # 子目录名
        exist_ok=True           # 覆盖已有结果
    )

    # 4. 简单打印每张图的 summary
    for r in results:
        print(f"{r.path.name}: {len(r.boxes)} 个目标，耗时 {r.time.inference*1000:.1f} ms")
