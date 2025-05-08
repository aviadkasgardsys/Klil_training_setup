# test.py
import argparse
import os
import cv2
from ultralytics import YOLO
 ## command : python3 test.py path/to/your_image.jpg --weights runs/detect/train12/weights/best.pt

def run_test(
    source: str,
    weights: str,
    imgsz: int,
    conf: float,
    device: str,
    outdir: str
):
    # 1. Prepare output folder
    os.makedirs(outdir, exist_ok=True)

    # 2. Load & resize image
    img = cv2.imread(source)
    if img is None:
        raise FileNotFoundError(f"Cannot load '{source}'")
    img_resized = cv2.resize(img, (imgsz, imgsz))

    # 3. Load model
    model = YOLO(weights)

    # 4. Run inference (passing the numpy array directly)
    results = model.predict(
        source=img_resized,
        imgsz=imgsz,
        conf=conf,
        device=device,
        save=False     # we'll save manually below
    )

    # 5. Annotate (red boxes + labels by r.plot) & save each result
    for i, r in enumerate(results):
        annotated = r.plot()  # draws red boxes and class names
        # Convert RGB -> BGR for OpenCV
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(outdir, f"result_{i}.jpg")
        cv2.imwrite(out_path, annotated_bgr)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Test YOLOv8 model on any image (resized to square) with red annotations only"
    )
    p.add_argument("source",    help="Path to input image")
    p.add_argument("--weights", default="runs/detect/train12/weights/best.pt",
                   help="Path to your trained weights (.pt)")
    p.add_argument("--imgsz",   type=int, default=640,
                   help="Size (both width & height) to resize input to")
    p.add_argument("--conf",    type=float, default=0.25,
                   help="Confidence threshold")
    p.add_argument("--device",  default="cuda:0",
                   help="Device, e.g. 'cpu' or 'cuda:0'")
    p.add_argument("--outdir",  default="results",
                   help="Folder to save annotated outputs")
    args = p.parse_args()

    run_test(
        source=args.source,
        weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        outdir=args.outdir
    )