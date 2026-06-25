import torch
from config import Config
from models import build_model
from dataset import load_wm811k
from utils import evaluate_model

cfg = Config()
_, _, test_loader, _ = load_wm811k(cfg)
m = build_model("resnet18", cfg.num_classes, pretrained=False)
ckpt = torch.load("checkpoints/resnet18_best.pt", map_location="cpu")
m.load_state_dict(ckpt["model_state"])
m.eval()
r = evaluate_model(m, test_loader, "cpu", cfg.class_names)
print(f"Dense resnet18 test macro F1: {r['macro_f1']:.4f}")
