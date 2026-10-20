"""Chapter 6, section 6.1.3: the network on one GPU.

    conv_layer(device_id, x_padded, w, b, _SRC)   listing 6.7. One convolution layer
                                                  (listing 6.6) over a padded input.
    max_pool(y)                                   listing 6.7. 2 x 2 max pooling.
    head(features, params)                        listing 6.8. Global average pool
                                                  and the linear classifier.
    cnn_forward(device_id, image, params, _SRC)   listing 6.8. The whole network:
                                                  six convolutions, two pools, the
                                                  head. Returns the final feature
                                                  map and the ten logits.

`load_weights()` reads the frozen weights the book's numbers were measured with
(model/weights.npz): batch normalization already folded into each convolution.
"""

from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np

_SRC = Path(__file__).with_name("kernels.cu").read_text()
WEIGHTS = Path(__file__).with_name("model") / "weights.npz"


def load_weights(path=WEIGHTS):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def conv_layer(device_id, x_padded, w, b, _SRC):
    with cp.cuda.Device(device_id):
        _MODULE = cp.RawModule(code=_SRC, options=("--std=c++14",))
        _layer = _MODULE.get_function("ConvLayerKernel")
        x = cp.ascontiguousarray(cp.asarray(x_padded, dtype=cp.float32))
        C, H, W = x.shape[0], x.shape[1] - 2, x.shape[2] - 2
        M = w.shape[0]
        y = cp.empty((M, H, W), dtype=cp.float32)
        grid = ((W + 15) // 16, (H + 15) // 16, M)
        _layer(grid, (16, 16, 1), (x, cp.asarray(w), cp.asarray(b), y,
                                   np.int32(C), np.int32(H), np.int32(W)))
        return y


def max_pool(y):
    M, H, W = y.shape
    return y.reshape(M, H // 2, 2, W // 2, 2).max(axis=(2, 4))


CHANNELS = (3, 32, 32, 64, 64, 128, 128)
POOL_AFTER = (1, 3)


def head(features, params):
    pooled = features.mean(axis=(1, 2))
    w, b = cp.asarray(params["head_w"]), cp.asarray(params["head_b"])
    return (w * pooled).sum(axis=1) + b


def cnn_forward(device_id, image, params, _SRC):
    with cp.cuda.Device(device_id):
        x = cp.asarray(image, dtype=cp.float32)
        for k in range(6):
            x = cp.pad(x, ((0, 0), (1, 1), (1, 1)))
            x = conv_layer(device_id, x, params[f"w{k}"], params[f"b{k}"], _SRC)
            if k in POOL_AFTER:
                x = max_pool(x)
        return x, head(x, params)


if __name__ == "__main__":
    from model import CLASS_NAMES, validation_images, normalize, center_crop

    images, labels = validation_images()
    params = load_weights()
    for i in range(0, len(images), len(images) // 8)[:8]:
        _, logits = cnn_forward(0, normalize(center_crop(images[i:i + 1]))[0], params, _SRC)
        print(f"image {i:4d}: predicted {CLASS_NAMES[int(cp.argmax(logits))]:<16s}"
              f" label {CLASS_NAMES[labels[i]]}")
