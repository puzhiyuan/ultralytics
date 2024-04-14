from ultralytics import YOLO


# model = YOLO(model="LSK.yaml", verbose=True)  # LSK
# model = YOLO(model="LSKV2.yaml", verbose=True)  # LSKV2
# model = YOLO(model="Swin.yaml", verbose=True)  # Swin
# model = YOLO(model="DW_C2_C3STR.yaml", verbose=True)  # DW_C2_Swin
# model = YOLO(model="BiFPN.yaml", verbose=True)  # DW_C2_Swin


def train(model):
    model = YOLO(model=model, verbose=True)
    # server
    # model.train(data="kitti.yaml", epochs=150, batch=32, workers=16)
    # pc
    model.train(data="kitti.yaml", epochs=4, batch=4)


def main():
    mods_used = ["YOLOv8_CBAM.yaml", "YOLOv8_GAM.yaml", "YOLOv8_SA.yaml", "YOLOv8_SimAM.yaml", "YOLOv8_SK.yaml",
                 "YOLOv8_ASPP.yaml", "YOLOv8_BasicRFB.yaml", "YOLOv8_SimSPPF.yaml", "YOLOv8_SPPELAN.yaml",
                 "YOLOv8_SPPFCSPC.yaml", "YOLOv8_bigFMap.yaml", "YOLOv8_SEAM.yaml", "YOLOv8_MultiSEAM.yaml", ]
    mods = ["yolov8.yaml", "YOLOv8_bigFMap.yaml", "YOLOv8_BF_ECA.yaml"]

    for mod in mods:
        train(mod)


if __name__ == '__main__':
    main()
