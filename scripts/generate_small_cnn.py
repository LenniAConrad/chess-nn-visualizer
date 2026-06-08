#!/usr/bin/env python3
"""Generate a compact LC0J CNN model for the visualizer.

The C++ loader expects the CRTK `LC0J` flat binary layout: dimensions, a 112-plane
input convolution, residual blocks, policy/value heads, and a compressed policy
map. This writes a deterministic untrained model that is tiny enough for local
visualization while still exercising the real CNN forward path.
"""

import math
import os
import random
import struct
import sys


def env_int(name, default):
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


INPUT_CHANNELS = 112
TRUNK = env_int("CNN_TRUNK", 32)
BLOCKS = env_int("CNN_BLOCKS", 4)
POLICY_CHANNELS = 73
VALUE_CHANNELS = env_int("CNN_VALUE_CHANNELS", 16)
VALUE_HIDDEN = env_int("CNN_VALUE_HIDDEN", 64)
POLICY_MAP = 73 * 64
WDL_OUTPUTS = 3

rng = random.Random(env_int("CNN_SEED", 20260608))


def w_u32(buf, v):
    buf.append(struct.pack("<I", int(v) & 0xFFFFFFFF))


def w_i32(buf, v):
    buf.append(struct.pack("<i", int(v)))


def w_u8(buf, v):
    buf.append(struct.pack("<B", int(v) & 0xFF))


def w_floatarray(buf, values):
    w_u32(buf, len(values))
    if values:
        buf.append(struct.pack("<%df" % len(values), *values))


def rand_weights(out_dim, in_dim, kernel):
    fan_in = max(1, in_dim * kernel * kernel)
    scale = 1.0 / math.sqrt(fan_in)
    return [rng.gauss(0.0, scale) for _ in range(out_dim * fan_in)]


def w_conv(buf, out_channels, in_channels, kernel):
    w_u32(buf, out_channels)
    w_u32(buf, in_channels)
    w_u32(buf, kernel)
    w_floatarray(buf, rand_weights(out_channels, in_channels, kernel))
    w_floatarray(buf, [0.0] * out_channels)


def w_dense(buf, out_dim, in_dim):
    w_u32(buf, out_dim)
    w_u32(buf, in_dim)
    scale = 1.0 / math.sqrt(max(1, in_dim))
    w_floatarray(buf, [rng.gauss(0.0, scale) for _ in range(out_dim * in_dim)])
    w_floatarray(buf, [0.0] * out_dim)


def main():
    out_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "models/lc0-cnn-small-112p-4x32-policy4672-wdl3.bin"
    )
    b = [b"LC0J"]
    w_u32(b, 1)  # version
    w_u32(b, INPUT_CHANNELS)
    w_u32(b, TRUNK)
    w_u32(b, BLOCKS)
    w_u32(b, POLICY_CHANNELS)
    w_u32(b, VALUE_CHANNELS)
    w_u32(b, VALUE_HIDDEN)
    w_u32(b, POLICY_MAP)
    w_u32(b, WDL_OUTPUTS)

    w_conv(b, TRUNK, INPUT_CHANNELS, 3)
    for _ in range(BLOCKS):
        w_conv(b, TRUNK, TRUNK, 3)
        w_conv(b, TRUNK, TRUNK, 3)
        w_u8(b, 0)  # no squeeze-excitation unit

    w_conv(b, TRUNK, TRUNK, 1)
    w_conv(b, POLICY_CHANNELS, TRUNK, 1)
    w_conv(b, VALUE_CHANNELS, TRUNK, 1)
    w_dense(b, VALUE_HIDDEN, VALUE_CHANNELS * 64)
    w_dense(b, WDL_OUTPUTS, VALUE_HIDDEN)

    w_u32(b, POLICY_MAP)
    for idx in range(POLICY_MAP):
        w_i32(b, idx)

    data = b"".join(b)
    with open(out_path, "wb") as f:
        f.write(data)
    print("wrote %s (%.2f MB)" % (out_path, len(data) / 1e6))


if __name__ == "__main__":
    main()
