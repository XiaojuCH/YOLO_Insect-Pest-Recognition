# train_ip102_yolo11n.py

import os
from copy import deepcopy
import torch
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer

# 解决 OpenMP & pin_memory 冲突
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
os.environ["PYTORCH_NO_PIN_MEMORY"] = "1"

class DynamicResolutionCallback:
    """在 50%/80% 训练进度自动切 512→640→768"""
    def __init__(self, resolutions=[512, 640, 768], stages=[0.5, 0.8]):
        self.resolutions = resolutions
        self.stages = stages

    def on_train_epoch_start(self, trainer: BaseTrainer):
        curr, total = trainer.epoch, trainer.args.epochs
        frac = curr / total
        if frac < self.stages[0]:
            sz = self.resolutions[0]
        elif frac < self.stages[1]:
            sz = self.resolutions[1]
        else:
            sz = self.resolutions[2]
        if sz != trainer.args.imgsz:
            print(f"\nEpoch {curr}: 切换 imgsz {trainer.args.imgsz} → {sz}")
            trainer.args.imgsz = sz
            trainer.train_loader = trainer.get_dataloader(
                trainer.trainset,
                batch_size=trainer.args.batch,
                imgsz=sz
            )

class EMACallback:
    """简单 EMA，提高模型稳定性"""
    def __init__(self, decay=0.999):
        self.decay = decay
        self.ema = None

    def on_train_start(self, trainer: BaseTrainer):
        self.ema = deepcopy(trainer.model.model).eval().to(trainer.device)

    def on_train_batch_end(self, trainer: BaseTrainer):
        with torch.no_grad():
            for e, p in zip(self.ema.parameters(), trainer.model.model.parameters()):
                e.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def on_train_end(self, trainer: BaseTrainer):
        path = os.path.join(trainer.save_dir, "weights", "ema_best.pt")
        torch.save(self.ema.state_dict(), path)
        print(f"EMA 模型已保存: {path}")

if __name__ == "__main__":
    # 加载官方 nano 预训练
    model = YOLO("yolo11n.pt")

    # ---- 阶段1：冻结 backbone ---- #
    print(">>> Stage 1: Freeze backbone (20 epochs)")
    trainer1 = model.train(
        data="data.yaml",
        epochs=20,
        imgsz=512,
        batch=32,
        device=0,

        pretrained=True,
        optimizer="AdamW",
        lr0=0.005 * (32/16),  # batch=32 学习率线性放大
        lrf=0.1,
        cos_lr=True,
        warmup_epochs=5,
        warmup_bias_lr=0.1,
        warmup_momentum=0.8,
        patience=50,

        half=True,
        cache="disk",
        workers=4,

        augment=True,
        auto_augment="v2",
        mosaic=1.0,
        close_mosaic=40,      # 40 轮后关闭 mosaic
        mixup=0.1,
        copy_paste=0.2,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, flipud=0.2,
        translate=0.2, scale=0.5,
        degrees=10.0, shear=5.0,

        project="runs/ip102",
        name="stage1_freeze",
        exist_ok=True,

        plots=True,
        val=True,

        freeze=[0, 10]       # 冻结前 10 层 backbone
    )
    trainer1.add_callback("on_train_epoch_start", DynamicResolutionCallback())
    trainer1.add_callback("on_train_start", EMACallback())
    trainer1.run()

    # ---- 阶段2：全量训练 ---- #
    print("\n>>> Stage 2: Unfreeze all (30 epochs)")
    best1 = os.path.join(trainer1.save_dir, "weights", "best.pt")
    model2 = YOLO(best1)
    trainer2 = model2.train(
        data="data.yaml",
        epochs=30,
        imgsz=512,
        batch=32,
        device=0,

        pretrained=True,
        optimizer="AdamW",
        lr0=0.005 * (32/16),
        lrf=0.1,
        cos_lr=True,
        warmup_epochs=5,
        warmup_bias_lr=0.1,
        warmup_momentum=0.8,
        patience=50,

        half=True,
        cache="disk",
        workers=4,

        augment=True,
        auto_augment="v2",
        mosaic=1.0,
        close_mosaic=40,
        mixup=0.1,
        copy_paste=0.2,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, flipud=0.2,
        translate=0.2, scale=0.5,
        degrees=10.0, shear=5.0,

        project="runs/ip102",
        name="stage2_unfreeze",
        exist_ok=True,

        plots=True,
        val=True
        # freeze=None    # 默认就是解冻所有层
    )
    trainer2.add_callback("on_train_epoch_start", DynamicResolutionCallback())
    trainer2.add_callback("on_train_start", EMACallback())
    trainer2.run()
