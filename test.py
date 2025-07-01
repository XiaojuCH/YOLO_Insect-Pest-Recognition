from ultralytics import YOLO

model = YOLO("runs/ip102/yolo11n_ip102/weights/best.pt")
# 推理
model.predict(source="images/test", conf=0.25, save=True)
# 如果 test 有标签，可以直接评估
metrics = model.val(data="data.yaml", split="test")
print("mAP50 =", metrics.box.map50)
