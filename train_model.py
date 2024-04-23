from ultralytics import YOLO


def train(model):
    model = YOLO(model=model, verbose=True)
    # server
    model.train(data="kitti.yaml", epochs=150, batch=32, workers=16, )  # device=(0,1))


def main():
    mods_used = ["YOLOv8_CBAM.yaml", "YOLOv8_GAM.yaml", "YOLOv8_SA.yaml", "YOLOv8_SimAM.yaml", "YOLOv8_SK.yaml",
                 "YOLOv8_ASPP.yaml", "YOLOv8_BasicRFB.yaml", "YOLOv8_SimSPPF.yaml", "YOLOv8_SPPELAN.yaml",
                 "YOLOv8_SPPFCSPC.yaml", "YOLOv8_bigFMap.yaml", "YOLOv8_SEAM.yaml", "YOLOv8_MultiSEAM.yaml", 
                 "YOLOv8_BF_ECA.yaml", "yolov8.yaml","neck_v2.yaml","neck_v2_ECA.yaml","neck_v2_CBAM.yaml", 
                 "neck_v2_Res_CBAM.yaml", "neck_v2_CoT.yaml", "neck_v2_SA.yaml", "neck_v2_GAM.yaml", 
                 "neck_Bb_CoT.yaml", "neck_Bb_ECA.yaml", "neck_v2_MS.yaml", ]
    
    mods = ["neck_v2_BasicRFB.yaml", "neck_v2_ASPP.yaml", "neck_v2_SimSPPF.yaml",]
    for mod in mods:
        train(mod)


if __name__ == '__main__':
    main()
