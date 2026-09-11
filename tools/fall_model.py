"""Pinned AGPL-3.0 model, restricted checkpoint loading; weights not bundled.

Model: melihuzunoglu/human-fall-detection, HF revision
97261b03632a9715a27481b2cfdac6a78e26d2d9. Snapshot labels are fall candidates,
not medical conclusions. The loader implementation is original project code.
"""
import hashlib
import torch
from torch.nn import BatchNorm2d, Upsample, ModuleList, SiLU, MaxPool2d, Sequential, Conv2d, Identity
from ultralytics.nn.modules.block import SPPF, Bottleneck, C2PSA, DFL, C3k2, Attention, C3k, PSABlock
from ultralytics.nn.modules.conv import Conv, DWConv, Concat
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from pathlib import Path

EXPECTED_SHA256 = "3f56ad30358d5c63bf8dbc0c1299cf68818c3d291dfb10c94107b94110aadd4c"

def load_model(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.stat().st_size>10_000_000:
        raise ValueError('Invalid bounded model file')
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Checkpoint SHA256 mismatch: " + str(path))
    allowed = [BatchNorm2d, Upsample, ModuleList, SiLU, MaxPool2d, Sequential, Conv2d, Identity, SPPF, Bottleneck, C2PSA, DFL, C3k2, Attention, C3k, PSABlock, Conv, DWConv, Concat, Detect, DetectionModel]
    found = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    if found != {c.__module__ + "." + c.__name__ for c in allowed}:
        raise ValueError("Checkpoint global set changed")
    with torch.serialization.safe_globals(allowed):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = ckpt["model"]
    if model.names != {0: "fallen", 1: "sitting", 2: "standing"}:
        raise ValueError("Checkpoint class map changed")
    return model.float().eval()


def make_predictor(path):
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    predictor=DetectionPredictor(overrides={'imgsz':640,'conf':.25,'device':'cpu','verbose':False,'save':False})
    predictor.setup_model(model=load_model(path),verbose=False)
    return predictor
