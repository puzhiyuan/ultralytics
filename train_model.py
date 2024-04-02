from ultralytics import YOLO


model = YOLO(model="LSKmodel.yaml", verbose=True, cache=True)

# model.train(data="kitti.yaml", epoch=150, batch=64, workers=16)
model.train(data="kitti.yaml", epochs=150, batch=4)
