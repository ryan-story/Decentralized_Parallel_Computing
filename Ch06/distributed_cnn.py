"""Chapter 6, section 6.3.1: the network split across 16 workers.

    pad_with_halos(tiles, i, grid)                listing 6.9. Tile i's padded
                                                  input: its own values, a one-cell
                                                  halo from each of its eight
                                                  neighbours, and zero padding at the
                                                  image's edge.
    distributed_cnn(device_ids, image, params, _SRC, grid=4)
                                                  listing 6.10. Every layer: exchange
                                                  halos, convolve every tile
                                                  concurrently, wait for all of them.

The image is cut into a 4 x 4 grid of tiles, one worker each. Device IDs may repeat:
`[0] * 16` gives 16 workers that share one GPU.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cupy as cp

from cnn import POOL_AFTER, conv_layer, head, max_pool


def pad_with_halos(tiles, i, grid):
    C, T, _ = tiles[i].shape
    row, col = divmod(i, grid)
    span = {-1: (slice(T - 1, T), slice(0, 1)), 0: (slice(0, T), slice(1, T + 1)),
            1: (slice(0, 1), slice(T + 1, T + 2))}
    p = cp.zeros((C, T + 2, T + 2), dtype=cp.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            r, c = row + dy, col + dx
            if 0 <= r < grid and 0 <= c < grid:
                (sy, ty), (sx, tx) = span[dy], span[dx]
                p[:, ty, tx] = tiles[r * grid + c][:, sy, sx]
    return p


def distributed_cnn(device_ids, image, params, _SRC, grid=4):
    x = cp.asarray(image, dtype=cp.float32)
    T = x.shape[1] // grid
    tiles = [x[:, r * T:(r + 1) * T, c * T:(c + 1) * T]
             for r in range(grid) for c in range(grid)]
    with ThreadPoolExecutor(max_workers=len(tiles)) as pool:
        for k in range(6):
            padded = [pad_with_halos(tiles, i, grid) for i in range(len(tiles))]
            futures = [pool.submit(conv_layer, device_ids[i], padded[i],
                                   params[f"w{k}"], params[f"b{k}"], _SRC)
                       for i in range(len(tiles))]
            tiles = [f.result() for f in futures]
            if k in POOL_AFTER:
                tiles = [max_pool(t) for t in tiles]
    rows = [cp.concatenate(tiles[r * grid:(r + 1) * grid], axis=2)
            for r in range(grid)]
    features = cp.concatenate(rows, axis=1)
    return features, head(features, params)


if __name__ == "__main__":
    from cnn import _SRC, cnn_forward, load_weights
    from model import validation_images, normalize, center_crop

    images, _ = validation_images()
    params = load_weights()
    image = normalize(center_crop(images[:1]))[0]
    f1, l1 = cnn_forward(0, image, params, _SRC)
    f2, l2 = distributed_cnn([0] * 16, image, params, _SRC)
    print("16 workers, bitwise equal to one GPU:",
          bool(cp.array_equal(f1, f2) and cp.array_equal(l1, l2)))
