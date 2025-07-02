# predict_ip102.py
import os
from ultralytics import YOLO

if __name__ == "__main__":
    # 自动找到最新的 best.pt
    weights_dir = "runs/ip102/yolo11n_ip102_rtx4060/weights"
    best = sorted([f for f in os.listdir(weights_dir) if f.startswith("best")])[-1]
    weights = os.path.join(weights_dir, best)
    print(f"Using weights: {weights}")

    model = YOLO(weights)
    results = model.predict(
        source="images/test",  # 你的 test 目录
        conf=0.25,
        save=True,     # 保存带框的图片到 runs/predict
        save_txt=True, # 保存每张图片的 txt 结果
        project="runs/ip102/predict",
        name="on_test",
        exist_ok=True
    )
