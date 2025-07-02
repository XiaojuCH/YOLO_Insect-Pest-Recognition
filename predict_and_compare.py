# predict_and_compare.py

import os
import cv2
from ultralytics import YOLO
import numpy as np

# —— 配置 —— #
WEIGHTS_DIR    = "runs/ip102/yolo11n_ip102_rtx4060/weights"
TEST_IMG_DIR   = "images/test"
TEST_LABEL_DIR = "labels/test"
OUT_DIR        = "runs/ip102/compare"
CONF_THRESH    = 0.25
IOU_THRESH     = 0.5

# —— 工具函数 —— #
def xywhn2xyxy(box, shape):
    h, w = shape
    xc, yc, bw, bh = box
    x1 = int((xc - bw/2) * w); y1 = int((yc - bh/2) * h)
    x2 = int((xc + bw/2) * w); y2 = int((yc + bh/2) * h)
    return [x1, y1, x2, y2]

def load_gt(path, shape):
    gts = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                cls, x, y, w, h = map(float, line.split())
                gts.append((int(cls), xywhn2xyxy((x,y,w,h), shape)))
    return gts

def iou(a, b):
    xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/area if area>0 else 0

# —— 主流程 —— #
if __name__ == "__main__":
    # 1. 找最新 best.pt
    bests = sorted([f for f in os.listdir(WEIGHTS_DIR) if f.startswith("best")])
    assert bests, "没找到 best.pt"
    weights = os.path.join(WEIGHTS_DIR, bests[-1])
    print("使用权重:", weights)

    # 2. 加载模型并推理
    model = YOLO(weights)
    results = model.predict(
        source=TEST_IMG_DIR, conf=CONF_THRESH, save=False, save_txt=False, imgsz=512
    )

    # 3. 准备输出文件夹
    correct_dir   = os.path.join(OUT_DIR, "correct")
    incorrect_dir = os.path.join(OUT_DIR, "incorrect")
    for d in (correct_dir, incorrect_dir):
        os.makedirs(d, exist_ok=True)

    # 4. 逐图绘制 & 分拣
    for res in results:
        img_path = res.path
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # 4.1 画预测框
        preds = []
        for *xyxy, conf, cls in res.boxes.data.cpu().numpy():
            x1,y1,x2,y2 = map(int, xyxy)
            preds.append((int(cls), (x1,y1,x2,y2), float(conf)))
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, f"P{int(cls)}:{conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # 4.2 画真值框
        lbl = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        gts = load_gt(os.path.join(TEST_LABEL_DIR, lbl), (h,w))
        for cls, box in gts:
            x1,y1,x2,y2 = box
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(img, f"GT{cls}", (x1, y2+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        # 4.3 判断“完全正确”条件
        used = set()
        ok = True
        # 每个 GT 至少有一个 P 匹配
        for gt_cls, gt_box in gts:
            match = False
            for i,(p_cls,p_box,_) in enumerate(preds):
                if i in used: continue
                if p_cls==gt_cls and iou(gt_box,p_box)>=IOU_THRESH:
                    used.add(i); match=True; break
            if not match:
                ok=False; break
        # 无多余高置信度误检
        if ok and len(used)!=len(preds):
            ok=False

        # 4.4 保存
        dst = correct_dir if ok else incorrect_dir
        fn = os.path.basename(img_path)
        cv2.imwrite(os.path.join(dst, fn), img)

    print("处理完成，结果保存在：")
    print("  正确：  ", correct_dir)
    print("  错误：  ", incorrect_dir)
