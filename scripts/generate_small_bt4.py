#!/usr/bin/env python3
"""Generate a small, real-architecture BT4J (v2) network for the visualizer.

chess-rtk only ships the full 706 MB BT4 (1024x15x32h). That is far too large to
keep around for a teaching/visualizer app, so this writes a *small* network in
the exact `BT4J` v2 on-disk format that the C++ loader
(`src/nn/lc0_bt4/Bt4BinLoader.cpp`) reads. It is a genuine BT4 transformer — the
native forward pass runs canonical-112 encoding, multi-head attention, an
attention policy head and a WDL value head over it — just with small dimensions
and freshly initialised (untrained) weights. Attention still varies by position
because the queries/keys are projections of the board-derived token embeddings.

Default dimensions (≈0.5M params, ≈2 MB):
  embedding 96 · 4 encoder blocks · 4 heads · FFN 192 · policy 48 · value 24.

Smolgen, input preproc/gates/embedding-FFN are intentionally disabled to keep
the file small and the format minimal; the loader/forward handle their absence.
Set `BT4_D`, `BT4_BLOCKS`, `BT4_HEADS`, and related environment variables to
generate a different compact variant.
"""

import math
import os
import random
import struct
import sys


def env_int(name, default):
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


# --- network dimensions ------------------------------------------------------
NAME = os.environ.get("BT4_NAME", "bt4-tiny-96x4x4h")
INPUT_FORMAT = "BT4_CANONICAL_112"   # must equal this for canonical encoding
INPUT_EMBEDDING = "PE_LEGACY"        # anything != PE_MAP/PE_DENSE -> plain dense
INPUT_CHANNELS = 112
TOKENS = 64
D = env_int("BT4_D", 96)             # embedding size (divisible by HEADS)
N_BLOCKS = env_int("BT4_BLOCKS", 4)
HEADS = env_int("BT4_HEADS", 4)
FFN = env_int("BT4_FFN", 192)
POLICY_SIZE = 1858
LN_EPS = 1.0e-3
POLICY_EMB = env_int("BT4_POLICY_EMB", 48)
POLICY_DMODEL = env_int("BT4_POLICY_DMODEL", 48)
VALUE_EMB = env_int("BT4_VALUE_EMB", 24)
VALUE_HIDDEN = env_int("BT4_VALUE_HIDDEN", 96)
ACT = "MISH"                         # NONE/RELU/MISH/SWISH/TANH

if D % HEADS != 0:
    raise SystemExit("BT4_D must be divisible by BT4_HEADS")

rng = random.Random(env_int("BT4_SEED", 20240608))


def alpha(n):
    return (2.0 * max(1, n)) ** -0.25


# --- little-endian writers ---------------------------------------------------
def w_i32(buf, v):
    buf.append(struct.pack("<i", int(v)))


def w_u32(buf, v):
    buf.append(struct.pack("<I", int(v) & 0xFFFFFFFF))


def w_f32(buf, v):
    buf.append(struct.pack("<f", float(v)))


def w_bool(buf, v):
    buf.append(struct.pack("<B", 1 if v else 0))


def w_string(buf, s):
    raw = s.encode("ascii")
    w_i32(buf, len(raw))
    buf.append(raw)


def w_floatarray(buf, values):
    w_i32(buf, len(values))
    buf.append(struct.pack("<%df" % len(values), *values))


def rand_weights(out_dim, in_dim):
    # Xavier-ish init keeps activations (and the LN inputs) well-scaled.
    scale = 1.0 / math.sqrt(in_dim)
    return [rng.gauss(0.0, scale) for _ in range(out_dim * in_dim)]


def w_dense(buf, in_dim, out_dim):
    w_i32(buf, in_dim)
    w_i32(buf, out_dim)
    w_floatarray(buf, rand_weights(out_dim, in_dim))
    w_floatarray(buf, [0.0] * out_dim)            # zero bias


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "models/lc0-bt4-tiny-96x4x4h.bin"
    b = []
    w_u32(b, 0x4A345442)   # "BT4J"
    w_u32(b, 2)            # version

    # --- Architecture ---
    w_string(b, NAME)
    w_string(b, INPUT_FORMAT)
    w_string(b, INPUT_EMBEDDING)
    w_i32(b, INPUT_CHANNELS)
    w_i32(b, TOKENS)
    w_i32(b, D)
    w_i32(b, N_BLOCKS)
    w_i32(b, HEADS)
    w_i32(b, POLICY_SIZE)
    w_f32(b, LN_EPS)
    w_i32(b, FFN)
    w_i32(b, 0)   # smolgenHiddenChannels
    w_i32(b, 0)   # smolgenHiddenSize
    w_i32(b, 0)   # smolgenPerHeadDim
    w_i32(b, 0)   # smolgenGlobalSize
    w_string(b, ACT)       # defaultActivation
    w_string(b, "SWISH")   # smolgenActivation (unused)
    w_string(b, ACT)       # ffnActivation
    w_bool(b, False)       # hasInputPreproc
    w_bool(b, False)       # hasInputEmbFfn
    w_bool(b, False)       # hasInputGates
    w_bool(b, False)       # hasSmolgen

    # --- Input stack (just the embedding dense for PE_LEGACY) ---
    w_dense(b, INPUT_CHANNELS, D)

    # --- Encoder blocks ---
    w_i32(b, N_BLOCKS)
    a = alpha(N_BLOCKS)
    for _ in range(N_BLOCKS):
        w_i32(b, HEADS)            # attention.heads
        w_dense(b, D, D)           # query
        w_dense(b, D, D)           # key
        w_dense(b, D, D)           # value
        w_dense(b, D, D)           # out
        w_dense(b, D, FFN)         # ffnIn
        w_dense(b, FFN, D)         # ffnOut
        w_floatarray(b, [1.0] * D)  # ln1 gamma
        w_floatarray(b, [0.0] * D)  # ln1 beta
        w_floatarray(b, [1.0] * D)  # ln2 gamma
        w_floatarray(b, [0.0] * D)  # ln2 beta
        w_string(b, ACT)
        w_f32(b, a)

    # hasSmolgen == False -> no shared smolgenW

    # --- Policy head ---
    w_dense(b, D, POLICY_EMB)      # embedding
    w_i32(b, 0)                    # policy encoder block count (none)
    w_dense(b, POLICY_EMB, POLICY_DMODEL)   # query
    w_dense(b, POLICY_EMB, POLICY_DMODEL)   # key
    w_floatarray(b, rand_weights(4, POLICY_DMODEL))  # promotionWeights [4, dmodel]
    w_string(b, ACT)

    # --- Value head ---
    w_dense(b, D, VALUE_EMB)               # embedding
    w_dense(b, TOKENS * VALUE_EMB, VALUE_HIDDEN)  # fc1 (flattened tokens*emb)
    w_dense(b, VALUE_HIDDEN, 3)            # fc2 -> WDL
    w_string(b, ACT)

    data = b"".join(b)
    with open(out_path, "wb") as f:
        f.write(data)
    print("wrote %s (%.2f MB)" % (out_path, len(data) / 1e6))


if __name__ == "__main__":
    main()
