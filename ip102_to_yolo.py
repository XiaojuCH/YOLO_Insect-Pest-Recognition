# ip102_to_yolo.py
import os
import xml.etree.ElementTree as ET
from shutil import copyfile

CLASSES = [str(i) for i in range(102)]  # ["0","1",...,"101"]


ROOT = os.getcwd()  # 假设在 IP102/ 目录下运行

def convert_xml(xml_path, txt_path):
    # 1) 简单检查：文件是否以 <annotation> 开头
    with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
        first = f.readline().strip()
    if not first.startswith("<annotation"):
        print(f"⚠️ Skip malformed XML (no <annotation>): {xml_path}")
        return

    # 2) 尝试解析，并在失败时捕获
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"⚠️ XML parse error {xml_path}: {e}. Skipping.")
        return

    size = root.find("size")
    if size is None:
        print(f"⚠️ No <size> in {xml_path}. Skipping.")
        return

    w = float(size.find("width").text)
    h = float(size.find("height").text)
    lines = []
    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls not in CLASSES:
            continue
        cid = CLASSES.index(cls)
        b = obj.find("bndbox")
        xmin = float(b.find("xmin").text)
        ymin = float(b.find("ymin").text)
        xmax = float(b.find("xmax").text)
        ymax = float(b.find("ymax").text)
        x_c = ((xmin + xmax) / 2) / w
        y_c = ((ymin + ymax) / 2) / h
        bw  = (xmax - xmin) / w
        bh  = (ymax - ymin) / h
        lines.append(f"{cid} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

    # 3) 写入 YOLO 格式 txt（即使空，也生成一个空文件以示无目标）
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def prepare(split):
    xml_dir = os.path.join(ROOT, "Annotations")
    img_dir = os.path.join(ROOT, "JPEGImages")
    with open(os.path.join(ROOT, "ImageSets/Main", f"{split}.txt")) as f:
        names = [x.strip() for x in f if x.strip()]
    os.makedirs(f"{ROOT}/images/{split}", exist_ok=True)
    os.makedirs(f"{ROOT}/labels/{split}", exist_ok=True)

    for nm in names:
        jpg_src = os.path.join(img_dir, nm + ".jpg")
        xml_src = os.path.join(xml_dir, nm + ".xml")
        jpg_dst = os.path.join(ROOT, "images", split, nm + ".jpg")
        txt_dst = os.path.join(ROOT, "labels", split, nm + ".txt")

        if not os.path.exists(jpg_src):
            print(f"⚠️ Missing image: {jpg_src}, skipping.")
            continue
        copyfile(jpg_src, jpg_dst)

        if not os.path.exists(xml_src):
            # 若 XML 缺失，则生成空的 txt
            open(txt_dst, "w").close()
            print(f"⚠️ Missing XML for {nm}, created empty label.")
            continue

        convert_xml(xml_src, txt_dst)

    print(f"[{split}] done: processed {len(names)} entries")

if __name__ == "__main__":
    # 首先确保有 train.txt, val.txt
    for s in ("train", "val"):
        prepare(s)