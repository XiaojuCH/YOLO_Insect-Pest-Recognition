# train_ip102_yolo11n_rtx4060.py
import os
import torch
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
import gc

# 解决 OpenMP & pin_memory 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 优化显存碎片

# 自定义回调函数实现动态分辨率训练
class DynamicResolutionCallback:
    def __init__(self, resolutions=[640, 768, 896]):
        self.resolutions = resolutions
        self.stages = [0.25, 0.6]  # 在25%和60%epoch切换分辨率
    
    def on_train_epoch_start(self, trainer: BaseTrainer):
        current_epoch = trainer.epoch
        total_epochs = trainer.args.epochs
        
        # 动态调整分辨率
        if current_epoch / total_epochs < self.stages[0]:
            new_res = self.resolutions[0]
        elif current_epoch / total_epochs < self.stages[1]:
            new_res = self.resolutions[1]
        else:
            new_res = self.resolutions[2]
            
        if new_res != trainer.args.imgsz:
            print(f"Epoch {current_epoch}: 切换分辨率 {trainer.args.imgsz} -> {new_res}")
            trainer.args.imgsz = new_res
            # 清理缓存
            torch.cuda.empty_cache()
            gc.collect()
            # 重新创建数据加载器
            trainer.train_loader = trainer.get_dataloader(trainer.trainset, batch_size=trainer.args.batch)

# 自定义回调实现EMA (指数移动平均)
class EMACallback:
    def __init__(self, decay=0.9999):
        self.decay = decay
        self.ema_model = None
    
    def on_train_start(self, trainer: BaseTrainer):
        # 初始化EMA模型
        self.ema_model = YOLO("yolo11n.pt")  # 加载相同架构
        self.ema_model.model.load_state_dict(trainer.model.model.state_dict())
        self.ema_model.model.eval()
        self.ema_model.model.to(trainer.device)
    
    def on_train_batch_end(self, trainer: BaseTrainer):
        # 更新EMA权重
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.model.parameters(), 
                                              trainer.model.model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
    
    def on_train_end(self, trainer: BaseTrainer):
        # 保存EMA模型
        ema_save_path = os.path.join(trainer.save_dir, "weights", "ema_best.pt")
        torch.save(self.ema_model.model.state_dict(), ema_save_path)
        print(f"EMA模型已保存至: {ema_save_path}")

if __name__ == "__main__":
    # 加载 YOLO11n 官方预训练
    model = YOLO("yolo11n.pt")

    # 训练配置 - 充分利用RTX 4060性能
    args = dict(
        data="data.yaml",
        epochs=50,
        imgsz=512,  # 初始分辨率提高
        batch=32,   # RTX 4060可支持更大batch
        device=0,    # 使用GPU
        
        # ===== 核心优化 =====
        close_mosaic=20,  # 稍晚关闭mosaic增强
        cos_lr=True,      # 余弦学习率衰减
        
        # ===== 学习率优化 =====
        pretrained=True,
        optimizer="AdamW",
        lr0=0.01,         # 更高初始学习率
        lrf=0.01,         # 最终学习率为0.01*lr0
        warmup_epochs=3,   # 更短的预热
        warmup_bias_lr=0.1,
        warmup_momentum=0.9,
        patience=30,       # 更早停止耐心
        
        # ===== 内存/速度优化 =====
        half=True,         # 半精度训练
        workers=8,         # 增加workers充分利用CPU
        single_cls=False,  # 多类别训练
        
        # ===== 数据增强优化 =====
        augment=True,
        auto_augment="v2",
        mosaic=1.0,        # 保持高强度mosaic
        mixup=0.2,         # 保持mixup强度
        copy_paste=0.3,    # 增加copy-paste强度
        erasing=0.4,       # 随机擦除增强
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, flipud=0.3,
        translate=0.2, scale=0.5,
        degrees=15.0, shear=8.0,
        
        # ===== 正则化增强 =====
        dropout=0.2,       # 更强的Dropout正则化
        weight_decay=0.05, # 权重衰减
        
        # ===== 日志与输出 =====
        project="runs/ip102",
        name="yolo11n_ip102_rtx4060",
        exist_ok=True,
        plots=True,
        val=True,
        save_period=10,    # 每10个epoch保存一次
        resume=False
    )
    
    # 创建训练器并添加回调
    trainer = model.train(**args)
    trainer.add_callback("on_train_epoch_start", DynamicResolutionCallback(resolutions=[640, 768, 896]))
    trainer.add_callback("on_train_start", EMACallback(decay=0.9999))
    
    # 开始训练
    results = trainer.run()
    