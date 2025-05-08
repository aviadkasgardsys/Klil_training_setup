# tests.py
import argparse
import os
import cv2
from ultralytics import YOLO

def run_single(
    img_path: str,
    model: YOLO,
    imgsz: int,
    conf: float,
    device: str,
    outdir: str,
    index: int
):
    """Load one image, run inference, and save the annotated result."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️  Skipping '{img_path}': not a valid image")
        return

    # 1. Resize & infer
    img_resized = cv2.resize(img, (imgsz, imgsz))
    results = model.predict(
        source=img_resized,
        imgsz=imgsz,
        conf=conf,
        device=device,
        save=False
    )

    # 2. Plot & save
    for r in results:
        annotated = r.plot()  # draws red boxes + class names
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        fname = f"result_{index}.jpg"
        out_path = os.path.join(outdir, fname)
        cv2.imwrite(out_path, annotated_bgr)
        print(f"✅ Saved: {out_path}")

def main():
    p = argparse.ArgumentParser(
        description="Run YOLOv8 on one image or all in a folder"
    )
    p.add_argument("source", help="Path to an image or a folder of images")
    p.add_argument("--weights", default="runs/detect/train12/weights/best.pt",
                   help="Path to your trained weights (.pt)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Resize both width & height to this size")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold")
    p.add_argument("--device", default="cuda:0",
                   help="Device (e.g. 'cpu' or 'cuda:0')")
    p.add_argument("--outdir", default="results",
                   help="Folder to save annotated outputs")
    args = p.parse_args()

    # create output folder
    os.makedirs(args.outdir, exist_ok=True)

    # load model once
    model = YOLO(args.weights)

    # gather files
    paths = []
    if os.path.isdir(args.source):
        for fn in sorted(os.listdir(args.source)):
            full = os.path.join(args.source, fn)
            if os.path.isfile(full) and fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                paths.append(full)
    else:
        paths = [args.source]

    if not paths:
        print(f"❌ No valid images found in '{args.source}'")
        return

    # process each
    for idx, img_path in enumerate(paths):
        run_single(
            img_path=img_path,
            model=model,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            outdir=args.outdir,
            index=idx
        )

if __name__ == "__main__":
    main()
