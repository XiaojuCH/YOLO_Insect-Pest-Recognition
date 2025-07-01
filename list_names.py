# list_names.py
import os, glob, xml.etree.ElementTree as ET

xml_dir = "Annotations"  # 请按实际路径修改
names = set()

for xml_path in glob.glob(os.path.join(xml_dir, "*.xml")):
    try:
        tree = ET.parse(xml_path)
    except:
        continue
    root = tree.getroot()
    for obj in root.findall("object"):
        cls = obj.find("name").text.strip()
        names.add(cls)

print(f"共发现 {len(names)} 种类别名称：")
for n in sorted(names):
    print(" ", repr(n))