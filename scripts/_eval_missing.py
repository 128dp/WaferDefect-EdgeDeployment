import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from config import Config
from models import build_model
from dataset import load_wm811k
from utils import evaluate_model

cfg = Config()
_, _, test_loader, _ = load_wm811k(cfg)

for name in ["resnet34", "mobilenet_v2"]:
    m = build_model(name, cfg.num_classes, pretrained=False)
    ckpt = torch.load(f"checkpoints/{name}_best.pt", map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    r = evaluate_model(m, test_loader, "cpu", cfg.class_names)
    print(f"{name}  test macro F1={r['macro_f1']:.4f}  acc={r['accuracy']*100:.2f}%")
