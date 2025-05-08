from ultralytics import YOLO

# 1. Load the pretrained YOLOv8m model
model = YOLO('yolov8m.pt')

# 2. Start training with custom settings
model.train(
    data='./data.yaml',  # path to your 640×640 grayscale dataset :contentReference[oaicite:0]{index=0}
    device='cuda:0',                # use the first GPU (3080 Ti, 12 GB VRAM) :contentReference[oaicite:1]{index=1}
    epochs=100,                     # number of training epochs
    imgsz=640,                      # training image size 640×640 :contentReference[oaicite:2]{index=2}
    batch=16,                       # start with 16; adjust up to ~20–24 for 12 GB VRAM :contentReference[oaicite:3]{index=3}turn0search6
    workers=8,                      # DataLoader workers for I/O parallelism :contentReference[oaicite:4]{index=4}

    # Color & light augmentations
    hsv_h=0.015,      # hue shift ±1.5% :contentReference[oaicite:5]{index=5}
    hsv_s=0.7,        # saturation shift ±70% :contentReference[oaicite:6]{index=6}
    hsv_v=0.4,        # brightness shift ±40% :contentReference[oaicite:7]{index=7}

    # Geometric augmentations
    degrees=10.0,     # random rotation ±10° :contentReference[oaicite:8]{index=8}
    translate=0.1,    # random translation ±10% :contentReference[oaicite:9]{index=9}
    scale=0.5,        # random scale [50–150]% :contentReference[oaicite:10]{index=10}
    shear=2.0,        # random shear ±2° :contentReference[oaicite:11]{index=11}
    perspective=0.001,# random perspective transformation :contentReference[oaicite:12]{index=12}
    flipud=0.5,       # vertical flip with 50% probability :contentReference[oaicite:13]{index=13}
    fliplr=0.5,       # horizontal flip with 50% probability :contentReference[oaicite:14]{index=14}

    # Advanced mix augmentations
    mosaic=1.0,       # always apply mosaic :contentReference[oaicite:15]{index=15}
    mixup=0.2,        # mixup with 20% probability :contentReference[oaicite:16]{index=16}
    copy_paste=0.1,    # copy-paste with 10% probability :contentReference[oaicite:17]{index=17}

    ##saving properties
    project='runs/detect',   # root directory for all runs
    name='klilprofiles',    # this run’s specific sub-folder
)


