# val_to_test.py
import os
import random
import shutil

# TODO: 修改为你的绝对路径
root = r"D:\Projects_\IP102_YOLO\dataset\Detection\IP102"

# 原 val 目录下的图片和标签
img_val_dir = os.path.join(root, "images", "val")
lbl_val_dir = os.path.join(root, "labels", "val")
# 新 test 目录
img_test_dir = os.path.join(root, "images", "test")
lbl_test_dir = os.path.join(root, "labels", "test")

os.makedirs(img_test_dir, exist_ok=True)
os.makedirs(lbl_test_dir, exist_ok=True)

# 读取所有 val 下的图片
val_imgs = [f for f in os.listdir(img_val_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# 随机抽 20% 做 test
test_num = int(len(val_imgs) * 0.2)
test_imgs = random.sample(val_imgs, test_num)

for img in test_imgs:
    src_img = os.path.join(img_val_dir, img)
    dst_img = os.path.join(img_test_dir, img)
    shutil.move(src_img, dst_img)

    lbl_name = os.path.splitext(img)[0] + ".txt"
    src_lbl = os.path.join(lbl_val_dir, lbl_name)
    dst_lbl = os.path.join(lbl_test_dir, lbl_name)

    if os.path.exists(src_lbl):
        # 如果原来就有标签，直接移动
        shutil.move(src_lbl, dst_lbl)
    else:
        # 否则创建一个空文件
        open(dst_lbl, 'w').close()

print(f"已将 {len(test_imgs)} 张图像从 val 移到 test，并同步处理对应的标签。")
