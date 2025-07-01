import os, glob

labels_dir = "labels/train"  # 根据实际路径修改
txts = glob.glob(os.path.join(labels_dir, "*.txt"))

non_empty = [p for p in txts if os.path.getsize(p) > 0]
empty     = [p for p in txts if os.path.getsize(p) == 0]

print(f"Total .txt: {len(txts)}")
print(f"Non-empty: {len(non_empty)}")
print(f"Empty:     {len(empty)}")

print("\n🎯 一些非空文件示例：")
for p in non_empty[:5]:
    print(" ", os.path.basename(p))

print("\n⚠️ 一些空文件示例：")
for p in empty[:5]:
    print(" ", os.path.basename(p))