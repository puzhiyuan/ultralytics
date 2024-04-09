from ultralytics import YOLO

model = YOLO(model="yolov8s.yaml", verbose=True)  # 原始配置
# model = YOLO(model="LSKmodel.yaml", verbose=True)  # 加入LSK模块
# model = YOLO(model="LSKmodelV2.yaml", verbose=True)  # 加入LSK模块

# server
# model.train(data="kitti.yaml", epochs=150, batch=64, workers=16)
# pc
model.train(data="kitti.yaml", epochs=4, batch=4)