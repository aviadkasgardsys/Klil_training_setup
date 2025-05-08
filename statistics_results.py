import os
import json
import pandas as pd
import ultralytics
from ultralytics import YOLO

# ─── CONFIG ────────────────────────────────────────────────────────────────────
WEIGHTS = 'runs/detect/train12/weights/best.pt'
DATA    = './data.yaml'
IMGSZ   = 640
BATCH   = 16
DEVICE  = 'cuda:0'

# Paths to your training stats (you should have saved these during train)
TRAIN_STATS_JSON = 'runs/detect/train12/train_stats.json'

# ─── UTILITY ───────────────────────────────────────────────────────────────────
def describe_metric(key: str) -> str:
    """Return a human‐readable description for a given metric key."""
    return {
        'metrics/precision(B)':    "Precision @ IoU=0.50: Fraction of detected boxes that are correct.",
        'metrics/recall(B)':       "Recall @ IoU=0.50: Fraction of ground‐truth boxes detected.",
        'metrics/mAP50(B)':        "mAP @ IoU=0.50: Mean Average Precision at IoU=0.50.",
        'metrics/mAP50-95(B)':     "mAP @ IoU=0.50:0.95: Average mAP across IoUs from 0.50 to 0.95.",
    }.get(key, "No description available.")

# ─── LOAD TRAIN STATS ─────────────────────────────────────────────────────────
if os.path.exists(TRAIN_STATS_JSON):
    with open(TRAIN_STATS_JSON) as f:
        train_stats = json.load(f)
else:
    train_stats = {}
    print(f"⚠️  Warning: Train stats not found at {TRAIN_STATS_JSON}. Overfitting gaps will be empty.")

# ─── RUN VALIDATION ────────────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)

model = YOLO(WEIGHTS)
val_metrics: ultralytics.utils.metrics.DetMetrics = model.val(
    data=DATA, imgsz=IMGSZ, batch=BATCH, device=DEVICE
)
val_stats = val_metrics.results_dict

# ─── BUILD COMBINED DICT ───────────────────────────────────────────────────────
combined = {}
for k, v in val_stats.items():
    desc = describe_metric(k)
    train_v = train_stats.get(k)
    gap = None if train_v is None else train_v - v
    combined[k] = {
        'value':             v,
        'description':       desc,
        'train_value':       train_v,
        'overfitting_gap':   gap
    }

# ─── DUMP JSON & CSV ──────────────────────────────────────────────────────────
json_out = 'results/val_stats_detailed.json'
with open(json_out, 'w') as f:
    json.dump(combined, f, indent=4)

# Flatten for CSV: one row per metric
rows = []
for k, info in combined.items():
    rows.append({
        'metric':            k,
        'value':             info['value'],
        'train_value':       info['train_value'],
        'overfitting_gap':   info['overfitting_gap'],
        'description':       info['description']
    })
df = pd.DataFrame(rows)
csv_out = 'results/val_stats_detailed.csv'
df.to_csv(csv_out, index=False)

print(f"✅ Saved detailed stats:\n  JSON → {json_out}\n  CSV  → {csv_out}")
