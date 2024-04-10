from ultralytics import YOLO

# model = YOLO(model="yolov8s.yaml", verbose=True)
# model = YOLO(model="LSK.yaml", verbose=True)  # LSK
# model = YOLO(model="LSKV2.yaml", verbose=True)  # LSKV2
# model = YOLO(model="Swin.yaml", verbose=True)  # Swin
# model = YOLO(model="DW_C2_C3STR.yaml", verbose=True)  # DW_C2_Swin
model = YOLO(model="BiFPN.yaml", verbose=True)  # DW_C2_Swin

# server
model.train(data="kitti.yaml", epochs=150, batch=64, workers=16)
# pc
# model.train(data="kitti.yaml", epochs=4, batch=4)