"""The chapter's CNN: data, preprocessing, training, and the frozen reference.

    conv(3->32) conv(32->32) pool conv(32->64) conv(64->64) pool
    conv(64->128) conv(128->128) global-average-pool linear(128->10)

Every convolution is 3 x 3, stride 1, zero padding 1, followed by ReLU. Pooling is
2 x 2 max with stride 2. Training uses batch normalization after each convolution;
`fold()` folds it into the convolution's weights and bias, so the network the
chapter serves is convolutions, ReLUs, two pools and a linear head.

The book's numbers use the weights in model/weights.npz, trained by `train()` below
on an RTX 5070 (86.9% top-1 on the 3,925 validation images). Retraining gives
different weights, because GPU training is not bitwise reproducible, and so slightly
different numbers.

Data: Imagenette, 160 px version (fast.ai), about 94 MB, downloaded into data/ on
first use. Every image is resized so its short side is 160 and center-cropped to
160 x 160 once; the network sees the center 128 x 128 crop.
"""
from __future__ import annotations

import json
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
CACHE = DATA / "imagenette2-160-160px.npz"
WEIGHTS = HERE / "model" / "weights.npz"

CHANNELS = (3, 32, 32, 64, 64, 128, 128)
POOL_AFTER = (1, 3)
CLASSES = 10
CLASS_NAMES = ["tench", "English springer", "cassette player", "chain saw", "church",
               "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
CROP = 128
SIDE = 160
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def download():
    """Fetch and unpack Imagenette into data/, once."""
    root = DATA / "imagenette2-160"
    if root.exists():
        return root
    DATA.mkdir(exist_ok=True)
    archive = DATA / "imagenette2-160.tgz"
    if not archive.exists():
        print("downloading", URL)
        urllib.request.urlretrieve(URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(DATA, filter="data")
    return root


def load_split(split: str):
    """All images of one split as uint8 (N, 160, 160, 3), labels as int64 (N,)."""
    from PIL import Image

    root = download() / split
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    xs, ys = [], []
    for label, cls in enumerate(classes):
        for f in sorted((root / cls).glob("*.JPEG")):
            im = Image.open(f).convert("RGB")
            w, h = im.size
            s = SIDE / min(w, h)
            im = im.resize((max(SIDE, round(w * s)), max(SIDE, round(h * s))),
                           Image.BILINEAR)
            w, h = im.size
            l, t = (w - SIDE) // 2, (h - SIDE) // 2
            xs.append(np.asarray(im.crop((l, t, l + SIDE, t + SIDE)), dtype=np.uint8))
            ys.append(label)
    return np.stack(xs), np.array(ys, dtype=np.int64)


def dataset():
    """Train and validation arrays, decoded once and cached in data/."""
    if CACHE.exists():
        z = np.load(CACHE)
        return z["xtr"], z["ytr"], z["xva"], z["yva"]
    xtr, ytr = load_split("train")
    xva, yva = load_split("val")
    np.savez(CACHE, xtr=xtr, ytr=ytr, xva=xva, yva=yva)
    return xtr, ytr, xva, yva


def validation_images():
    """The 3,925 validation images, uint8 (N, 160, 160, 3), and their labels."""
    if not CACHE.exists():
        dataset()
    z = np.load(CACHE)
    return z["xva"], z["yva"]


def normalize(images_uint8: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float32, normalized per channel."""
    x = images_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def center_crop(images: np.ndarray, offset_x: int | None = None) -> np.ndarray:
    """The 128 x 128 crop at vertical center; the horizontal offset defaults to
    center, and the panning stream of section 6.5 moves it one pixel per frame."""
    t = (SIDE - CROP) // 2
    l = t if offset_x is None else offset_x
    return images[:, t:t + CROP, l:l + CROP]


def torch_net(batchnorm: bool = True):
    import torch.nn as nn

    layers = []
    for i in range(len(CHANNELS) - 1):
        layers.append(nn.Conv2d(CHANNELS[i], CHANNELS[i + 1], 3, padding=1,
                                bias=not batchnorm))
        if batchnorm:
            layers.append(nn.BatchNorm2d(CHANNELS[i + 1]))
        layers.append(nn.ReLU(inplace=True))
        if i in POOL_AFTER:
            layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Linear(CHANNELS[-1], CLASSES))


def fold(net) -> dict:
    """Fold each BatchNorm into the convolution before it."""
    import torch.nn as nn

    out, k = {}, 0
    mods = list(net)
    for i, m in enumerate(mods):
        if isinstance(m, nn.Conv2d):
            bn = mods[i + 1]
            scale = (bn.weight / (bn.running_var + bn.eps).sqrt()).detach()
            w = m.weight.detach() * scale[:, None, None, None]
            b = (bn.bias - bn.running_mean * scale).detach()
            out[f"w{k}"] = w.cpu().numpy().astype(np.float32)
            out[f"b{k}"] = b.cpu().numpy().astype(np.float32)
            k += 1
        if isinstance(m, nn.Linear):
            out["head_w"] = m.weight.detach().cpu().numpy().astype(np.float32)
            out["head_b"] = m.bias.detach().cpu().numpy().astype(np.float32)
    return out


def frozen_torch_net(params: dict):
    """The folded network as a PyTorch module: the cuDNN reference."""
    import torch

    net = torch_net(batchnorm=False)
    convs = [m for m in net if isinstance(m, torch.nn.Conv2d)]
    with torch.no_grad():
        for k, c in enumerate(convs):
            c.weight.copy_(torch.from_numpy(params[f"w{k}"]))
            c.bias.copy_(torch.from_numpy(params[f"b{k}"]))
        net[-1].weight.copy_(torch.from_numpy(params["head_w"]))
        net[-1].bias.copy_(torch.from_numpy(params["head_b"]))
    return net.eval()


def evaluate(net, x, y, batch: int = 500) -> float:
    import torch

    net.eval()
    hits = 0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            hits += int((net(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum())
    return hits / len(x)


def train(epochs: int = 40, batch: int = 128, seed: int = 0, out=WEIGHTS) -> dict:
    """Train with batch normalization, fold it, and save the weights to `out`."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda"
    xtr, ytr, xva, yva = dataset()
    xtr_t = torch.from_numpy(normalize(xtr)).to(dev)
    ytr_t = torch.from_numpy(ytr).to(dev)
    xva_t = torch.from_numpy(normalize(center_crop(xva))).to(dev)
    yva_t = torch.from_numpy(yva).to(dev)

    net = torch_net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=5e-4)
    steps = epochs * (len(xtr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    t0, log = time.time(), []
    for ep in range(epochs):
        net.train()
        perm = torch.from_numpy(rng.permutation(len(xtr))).to(dev)
        for s in range(len(xtr) // batch):
            idx = perm[s * batch:(s + 1) * batch]
            oy, ox = rng.integers(0, SIDE - CROP + 1, size=2)
            xb = xtr_t[idx, :, oy:oy + CROP, ox:ox + CROP]
            if rng.random() < 0.5:
                xb = xb.flip(3)
            loss = F.cross_entropy(net(xb), ytr_t[idx], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        acc = evaluate(net, xva_t, yva_t)
        log.append({"epoch": ep + 1, "loss": float(loss), "val_top1": acc})
        print(f"epoch {ep + 1:3d}  loss {float(loss):.3f}  val top-1 {acc:.4f}  "
              f"{time.time() - t0:.0f} s", flush=True)
    params = fold(net.eval())
    acc_folded = evaluate(frozen_torch_net(params).to(dev), xva_t, yva_t)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **params)
    print("folded network val top-1:", acc_folded)
    return {"val_top1_folded": acc_folded, "log": log}


if __name__ == "__main__":
    # Retrains into model/retrained.npz, leaving the book's weights untouched.
    train(out=HERE / "model" / "retrained.npz")
