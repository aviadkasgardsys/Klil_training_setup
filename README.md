# YOLOv8 Object Detection Project

This repository provides scripts and configuration to train, test, and evaluate a YOLOv8 object detection model using Ultralytics.

## 🗂 Project Structure

```text
├── data.yaml           # Dataset configuration (paths + class names)
├── train.py            # Optional wrapper for `model.train`
├── test.py             # Test a single image with red annotations
├── tests.py            # Test a multiple image with red annotations
├── tests.py            # Batch test script (iterates over ./tests folder)
├── tests/              # Folder containing images for batch testing
├── datasets/           # Raw images + labels
│   ├── train/          # Training images + labels
│   └── val/            # Validation images + labels
├── runs/               # Ultralytics outputs (weights, validation outputs)
├── results/            # Saved metrics (JSON/CSV) and annotated outputs
└── README.md           # Project overview and usage
```

## 📦 Installation

1. Clone the repo:

   ```bash
   git clone <your-repo-url>
   cd <your-repo>
   ```
2. Setup Python environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. (Optional) Install Ultralytics YOLOv8:

   ```bash
   pip install ultralytics
   ```

## 🏋️ Training

Use the `train.py` wrapper

```bash
python3 train.py
```
## 🔍 Testing a Single Image

Use `test.py` to infer on any image (resized to square 640×640):

```bash
python3 test.py path/to/image.jpg \
  --weights runs/detect/train/weights/best.pt \
  --conf 0.25 \
  --device cuda:0 \
  --outdir results/single
```

## 🔄 Batch Testing

Run `tests.py` at project root to process all images in `./tests`:

```bash
python tests.py ./tests \
  --weights runs/detect/train/weights/best.pt \
  --imgsz 640 \
  --conf 0.25 \
  --device cuda:0 \
  --outdir results/batch
```

## 📊 Viewing Statistics

After validation, metrics are saved in `results/val_stats.json` and `results/val_stats.csv`.
Open the CSV or JSON to see:

* **Precision**: Fraction of predicted boxes that are correct
* **Recall**: Fraction of ground-truth boxes detected
* **mAP50**: Mean Average Precision at IoU=0.50
* **mAP50-95**: Mean AP averaged over IoUs from 0.50 to 0.95
* **Fitness**: Composite score used during hyperparameter search

Example:

```json
{
  "metrics/precision(B)": 0.835,
  "metrics/recall(B)": 0.606,
  "metrics/mAP50(B)": 0.740,
  "metrics/mAP50-95(B)": 0.506,
  "fitness": 0.530
}
```

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Commit your changes (`git commit -m "Add feature"`)
4. Push to branch (`git push origin feat/my-feature`)
5. Open a pull request

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
