import os

# TODO: 把下面路径改成你本地的实际路径
images_dir = 'images/train'
labels_dir = 'labels/train'

# 读取文件名（无扩展名）
image_files = [os.path.splitext(f)[0] for f in os.listdir(images_dir)
               if f.lower().endswith(('.jpg', '.png'))]
label_files = [os.path.splitext(f)[0] for f in os.listdir(labels_dir)
               if f.lower().endswith('.txt')]

# 找出缺失标签的图像
missing = sorted(set(image_files) - set(label_files))
print(f"共 {len(missing)} 张图像缺少标签：")
for name in missing:
    print(" ", name)

# 自动为缺失标签的图像创建空 .txt 文件
for name in missing:
    open(os.path.join(labels_dir, f"{name}.txt"), 'w').close()
print("已为缺失的图像生成空标签文件。")
