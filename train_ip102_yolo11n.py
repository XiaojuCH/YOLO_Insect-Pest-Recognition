# train_ip102_yolo11n_full_gpu.py

import os, torch
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer

# —— 环境 & 冲突 —— #
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
os.environ["PYTORCH_NO_PIN_MEMORY"] = "1"

class DynamicResolutionCallback:
    """在 30%/60% 训练进度自动切 512→640→768"""
    def __init__(self, resolutions=[512, 640, 768]):
        self.resolutions = resolutions
        self.stages = [0.3, 0.6]

    def on_train_epoch_start(self, trainer: BaseTrainer):
        curr, total = trainer.epoch, trainer.args.epochs
        frac = curr/total
        if frac < self.stages[0]:
            sz = self.resolutions[0]
        elif frac < self.stages[1]:
            sz = self.resolutions[1]
        else:
            sz = self.resolutions[2]

        if sz != trainer.args.imgsz:
            print(f"\nEpoch {curr}: 切换 imgsz {trainer.args.imgsz}→{sz}")
            trainer.args.imgsz = sz
            trainer.train_loader = trainer.get_dataloader(
                trainer.trainset,
                batch_size=trainer.args.batch,
                imgsz=sz
            )

if __name__ == "__main__":
    # 1) 加载 YOLO11n 预训练模型
    model = YOLO("yolo11m.pt")

    # 2) 训练配置 —— 去掉 accumulate，放大 batch 和 workers
    args = dict(
        data="data.yaml",
        epochs=10,
        imgsz=512,
        batch=32,          # 直接用较大 batch
        device=0,

        # —— 优化器 & LR —— #
        pretrained=True,
        optimizer="AdamW",
        lr0=0.01,          # 直接使用较大 lr0，RTX4060 足够
        lrf=0.1,
        cos_lr=True,
        warmup_epochs=5,
        warmup_bias_lr=0.1,
        warmup_momentum=0.8,
        patience=50,

        # —— 内存/性能 —— #
        half=True,         # FP16 加速
        cache="disk",      # 磁盘缓存安全
        workers=8,         # 利用更多 CPU 线程
        rect=True,         # 矩形训练
        multi_scale=True,  # 多尺度

        # —— 数据增强 —— #
        augment=True,
        auto_augment="v2",
        mosaic=0.8,
        mixup=0.1,
        copy_paste=0.2,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, flipud=0.1,
        translate=0.2, scale=0.5,
        degrees=10.0, shear=5.0, perspective=0.1,

        # —— 正则化 —— #
        dropout=0.1,
        weight_decay=0.05,

        # —— 输出 —— #
        project="runs/ip102",
        name="yolo11n_ip102_fullgpu",
        exist_ok=True,
        plots=True,
        val=True
    )

    # 3) 启动训练并拿到 Trainer
    trainer = model.train(**args)

    # 4) 注册动态分辨率回调
    trainer.add_callback("on_train_epoch_start", DynamicResolutionCallback())

    # 5) 运行训练
    trainer.run()
