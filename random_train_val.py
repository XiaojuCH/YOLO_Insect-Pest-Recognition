# split_trainval.py
import os, random

ROOT = "ImageSets/Main"
random.seed(42)

# 1. 读取 trainval 列表
with open(os.path.join(ROOT, "trainval.txt")) as f:
    samples = [x.strip() for x in f if x.strip()]

# 2. 随机打乱
random.shuffle(samples)

# 3. 划分比例（90% 训练，10% 验证）
n = len(samples)
n_val = int(n * 0.1)
val_samples   = samples[:n_val]
train_samples = samples[n_val:]

# 4. 写入文件
with open(os.path.join(ROOT, "train.txt"), "w") as f:
    f.write("\n".join(train_samples))
with open(os.path.join(ROOT, "val.txt"),   "w") as f:
    f.write("\n".join(val_samples))

print(f"Total: {n}, Train: {len(train_samples)}, Val: {len(val_samples)}")
