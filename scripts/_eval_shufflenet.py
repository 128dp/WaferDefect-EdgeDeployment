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

for label, arch, ckpt_file in [
    ("EfficientNet-B0 standalone",  "efficientnet_b0",  "efficientnet_b0_best.pt"),
    ("ShuffleNetV2 standalone",     "shufflenet_v2",    "shufflenet_v2_best.pt"),
    ("ShuffleNetV2 distilled (R18)","shufflenet_v2",    "shufflenet_v2_distilled_best.pt"),
]:
    m = build_model(arch, cfg.num_classes, pretrained=False).to(cfg.device)
    ckpt = torch.load(f"checkpoints/{ckpt_file}", map_location=cfg.device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    r = evaluate_model(m, test_loader, cfg.device, cfg.class_names)
    print(f"{label}: test macro F1 = {r['macro_f1']:.4f}  |  acc = {r['accuracy']*100:.2f}%")
