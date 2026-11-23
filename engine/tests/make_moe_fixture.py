#!/usr/bin/env python3
"""Write a tiny synthetic mixture-of-experts GGUF for the SSD streaming tests.

The model is a qwen3moe with random weights (fixed seed, so the file is the same
on every run and platform), a dummy "test" tokenizer and nothing else. It only
needs to be a real model to llama.cpp's loader and graph: the tests compare a
resident run against a streamed run of the same file, so the weights carry no
meaning.

Pure Python (struct and array), so CI needs no numpy.

    make_moe_fixture.py --out DIR [--expert-type f16|q8_0|f32] [--mtp] [--seed N]

--mtp writes a qwen35moe instead: four trunk layers and one NextN/MTP layer that
carries routed experts of its own, the layout GLM-5.3-Flash has. llama.cpp leaves
the MTP tensors in the file unless it is told to load them.

--glm writes the shape of GLM-5.3-Flash at a small size: a dense leading layer,
three layers of routed experts, KDA and MLA attention with the k-pool indexer, and
an MTP layer. Its experts are placeholders. Python cannot make GLM's IQ types, so
`ssd_stream_test glm-fixture` fills and quantizes them. Widths are 256 because IQ
weights need whole rows of 256.
"""

from __future__ import annotations

import argparse
import array
import random
import struct
import sys
from pathlib import Path

ARCH = "qwen3moe"
ALIGN = 32

# Shape: three layers so the next-layer router prefetch runs on two transitions,
# 16 experts with 4 used so routing picks a different subset per token.
N_VOCAB = 256
N_EMBD = 128
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 32
N_FF_EXP = 64
N_EXPERT = 16
N_EXPERT_USED = 4
N_LAYER = 3
N_LAYER_MTP_TRUNK = 4  # four trunk layers of experts take exactly 3 MiB, a fifth would show in the receipt
SSM_KERNEL, SSM_RANK, SSM_GROUPS = 4, 2, 2  # linear attention: the state size is HEAD_DIM
N_CTX_TRAIN = 65536

# gguf value types
U32, F32V, BOOL, STRING, ARRAY = 4, 6, 7, 8, 9
# ggml tensor types
GGML_F32, GGML_F16, GGML_Q8_0 = 0, 1, 8


def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, vtype: int, value) -> bytes:
    out = _str(key) + struct.pack("<I", vtype)
    if vtype == U32:
        out += struct.pack("<I", value)
    elif vtype == F32V:
        out += struct.pack("<f", value)
    elif vtype == BOOL:
        out += struct.pack("<B", 1 if value else 0)
    elif vtype == STRING:
        out += _str(value)
    else:
        raise ValueError(vtype)
    return out


def _kv_array(key: str, etype: int, values) -> bytes:
    out = _str(key) + struct.pack("<IIQ", ARRAY, etype, len(values))
    for v in values:
        if etype == STRING:
            out += _str(v)
        elif etype == F32V:
            out += struct.pack("<f", v)
        elif etype == U32:
            out += struct.pack("<I", v)
        else:
            raise ValueError(etype)
    return out


def _f16_bits(x: float) -> bytes:
    return struct.pack("<e", x)


def quantize_q8_0(values: array.array) -> bytes:
    """ggml's reference Q8_0: blocks of 32, an f16 scale amax/127 and 32 int8."""
    assert len(values) % 32 == 0
    out = bytearray()
    for b in range(0, len(values), 32):
        block = values[b:b + 32]
        amax = max(abs(v) for v in block)
        d = amax / 127.0
        inv = 1.0 / d if d else 0.0
        out += _f16_bits(d)
        out += struct.pack("<32b", *(max(-128, min(127, int(round(v * inv)))) for v in block))
    return bytes(out)


class Tensor:
    def __init__(self, name: str, dims: list[int], ggml_type: int, data: bytes):
        self.name, self.dims, self.type, self.data = name, dims, ggml_type, data


# GLM-5.3-Flash at a small size; the real values are in the comments.
GLM_ARCH = "glm5-next"
GLM_N_EMBD = 256  # 4096
GLM_N_FF = 512  # 12288
GLM_N_FF_EXP = 256  # 2048
GLM_N_EXPERT = 32  # 288
GLM_N_EXPERT_USED = 4  # 8
GLM_N_TRUNK = 4  # 45: one dense layer (3 in the real file), then routed experts
GLM_N_HEAD = 4  # 64
GLM_KDA_HEAD_DIM = 32  # 128
GLM_HEAD_COUNT_KV = [0, 0, 1, 0, 1]  # zero is a KDA layer; the last entry is the MTP layer
GLM_Q_LORA, GLM_KV_LORA = 64, 64  # 1536, 512
GLM_HEAD_MLA = 32  # 256
GLM_IDX_HEADS, GLM_IDX_DIM, GLM_IDX_TOP_K, GLM_KPOOL = 4, 16, 32, 4  # 32, 128, 2048, 4
GLM_HC = 4  # the arch accepts no other value
GLM_SSM_KERNEL = 4


def build_glm(seed: int) -> list[Tensor]:
    rng = random.Random(seed)

    def weights(n: int, fan_in: int) -> array.array:
        bound = (3.0 / fan_in) ** 0.5
        return array.array("f", (rng.uniform(-bound, bound) for _ in range(n)))

    def constant(n: int, value: float) -> array.array:
        return array.array("f", [value] * n)

    def f32(name: str, dims: list[int], values: array.array) -> Tensor:
        if sys.byteorder != "little":
            values = array.array("f", values)
            values.byteswap()
        return Tensor(name, dims, GGML_F32, values.tobytes())

    def mat(name: str, cols: int, rows: int) -> Tensor:
        return f32(name, [cols, rows], weights(cols * rows, cols))

    def placeholder(name: str, dims: list[int]) -> Tensor:
        return Tensor(name, dims, GGML_F32, bytes(dims[0] * dims[1] * dims[2] * 4))

    embd, ff_exp, n_expert = GLM_N_EMBD, GLM_N_FF_EXP, GLM_N_EXPERT
    hc_mix = (2 + GLM_HC) * GLM_HC
    d_inner = GLM_KDA_HEAD_DIM * GLM_N_HEAD
    tensors = [
        f32("token_embd.weight", [embd, N_VOCAB], weights(embd * N_VOCAB, 3)),
        f32("output_norm.weight", [embd], constant(embd, 1.0)),
        mat("output.weight", embd, N_VOCAB),
    ]
    for il in range(GLM_N_TRUNK + 1):
        p = f"blk.{il}."
        tensors += [
            f32(p + "attn_norm.weight", [embd], constant(embd, 1.0)),
            f32(p + "ffn_norm.weight", [embd], constant(embd, 1.0)),
        ]
        if il < GLM_N_TRUNK:
            for side in ("attn", "ffn"):
                tensors += [
                    mat(f"{p}hc_{side}_fn.weight", GLM_HC * embd, hc_mix),
                    f32(f"{p}hc_{side}_base.weight", [hc_mix], constant(hc_mix, 0.0)),
                    f32(f"{p}hc_{side}_scale.weight", [3], constant(3, 1.0)),
                ]
        if GLM_HEAD_COUNT_KV[il] == 0:
            for name in ("q", "k", "v"):
                tensors.append(f32(f"{p}ssm_conv1d_{name}.weight", [GLM_SSM_KERNEL, 1, d_inner],
                                   weights(GLM_SSM_KERNEL * d_inner, GLM_SSM_KERNEL)))
            tensors += [
                mat(p + "attn_q.weight", embd, d_inner),
                mat(p + "attn_k.weight", embd, d_inner),
                mat(p + "attn_v.weight", embd, d_inner),
                mat(p + "ssm_f_a.weight", embd, GLM_KDA_HEAD_DIM),
                mat(p + "ssm_f_b.weight", GLM_KDA_HEAD_DIM, d_inner),
                mat(p + "ssm_beta.weight", embd, GLM_N_HEAD),
                f32(p + "ssm_a", [GLM_N_HEAD], constant(GLM_N_HEAD, -1.0)),
                f32(p + "ssm_dt.bias", [d_inner], constant(d_inner, 1.0)),
                mat(p + "ssm_g_a.weight", embd, GLM_KDA_HEAD_DIM),
                mat(p + "ssm_g_b.weight", GLM_KDA_HEAD_DIM, d_inner),
                f32(p + "ssm_norm.weight", [GLM_KDA_HEAD_DIM], constant(GLM_KDA_HEAD_DIM, 1.0)),
                mat(p + "attn_output.weight", d_inner, embd),
            ]
        else:
            q_b = GLM_N_HEAD * GLM_HEAD_MLA
            k_b = GLM_HEAD_MLA * GLM_KV_LORA * GLM_N_HEAD
            tensors += [
                f32(p + "attn_q_a_norm.weight", [GLM_Q_LORA], constant(GLM_Q_LORA, 1.0)),
                f32(p + "attn_kv_a_norm.weight", [GLM_KV_LORA], constant(GLM_KV_LORA, 1.0)),
                mat(p + "attn_q_a.weight", embd, GLM_Q_LORA),
                mat(p + "attn_q_b.weight", GLM_Q_LORA, q_b),
                mat(p + "attn_kv_a_mqa.weight", embd, GLM_KV_LORA),
                f32(p + "attn_k_b.weight", [GLM_HEAD_MLA, GLM_KV_LORA, GLM_N_HEAD], weights(k_b, GLM_HEAD_MLA)),
                f32(p + "attn_v_b.weight", [GLM_KV_LORA, GLM_HEAD_MLA, GLM_N_HEAD], weights(k_b, GLM_KV_LORA)),
                mat(p + "attn_output.weight", q_b, embd),
                f32(p + "indexer.k_norm.weight", [GLM_IDX_DIM], constant(GLM_IDX_DIM, 1.0)),
                f32(p + "indexer.k_norm.bias", [GLM_IDX_DIM], constant(GLM_IDX_DIM, 0.0)),
                mat(p + "indexer.proj.weight", embd, GLM_IDX_HEADS),
                mat(p + "indexer.attn_k.weight", embd, GLM_IDX_DIM),
                mat(p + "indexer.attn_q_b.weight", GLM_Q_LORA, GLM_IDX_HEADS * GLM_IDX_DIM),
                mat(p + "indexer_compressor_gate.weight", embd, GLM_IDX_DIM),
                mat(p + "indexer_compressor_ape.weight", GLM_IDX_DIM, GLM_KPOOL),
            ]
        if il == 0:
            tensors += [
                mat(p + "ffn_gate.weight", embd, GLM_N_FF),
                mat(p + "ffn_down.weight", GLM_N_FF, embd),
                mat(p + "ffn_up.weight", embd, GLM_N_FF),
            ]
        else:
            tensors += [
                # a sharper router, as in the qwen3moe fixture
                f32(p + "ffn_gate_inp.weight", [embd, n_expert], weights(embd * n_expert, embd // 16)),
                f32(p + "exp_probs_b.bias", [n_expert], constant(n_expert, 0.0)),
                placeholder(p + "ffn_gate_exps.weight", [embd, ff_exp, n_expert]),
                placeholder(p + "ffn_up_exps.weight", [embd, ff_exp, n_expert]),
                placeholder(p + "ffn_down_exps.weight", [ff_exp, embd, n_expert]),
                mat(p + "ffn_gate_shexp.weight", embd, ff_exp),
                mat(p + "ffn_up_shexp.weight", embd, ff_exp),
                mat(p + "ffn_down_shexp.weight", ff_exp, embd),
            ]
    p = f"blk.{GLM_N_TRUNK}.nextn."
    tensors += [
        mat(p + "eh_proj.weight", 2 * embd, embd),
        f32(p + "enorm.weight", [embd], constant(embd, 1.0)),
        f32(p + "hnorm.weight", [embd], constant(embd, 1.0)),
    ]
    return tensors


def glm_kvs() -> list[bytes]:
    a = GLM_ARCH
    return [
        _kv("general.architecture", STRING, a),
        _kv("general.name", STRING, "openharness-ssd-glm-fixture"),
        _kv("general.alignment", U32, ALIGN),
        _kv(f"{a}.context_length", U32, N_CTX_TRAIN),
        _kv(f"{a}.embedding_length", U32, GLM_N_EMBD),
        _kv(f"{a}.block_count", U32, GLM_N_TRUNK + 1),
        _kv(f"{a}.nextn_predict_layers", U32, 1),
        _kv(f"{a}.feed_forward_length", U32, GLM_N_FF),
        _kv(f"{a}.leading_dense_block_count", U32, 1),
        _kv(f"{a}.expert_feed_forward_length", U32, GLM_N_FF_EXP),
        _kv(f"{a}.expert_count", U32, GLM_N_EXPERT),
        _kv(f"{a}.expert_used_count", U32, GLM_N_EXPERT_USED),
        _kv(f"{a}.expert_shared_count", U32, 1),
        _kv(f"{a}.expert_gating_func", U32, 2),
        _kv(f"{a}.expert_weights_norm", BOOL, True),
        _kv(f"{a}.expert_weights_scale", F32V, 2.5),
        _kv(f"{a}.swiglu_clamp_exp", F32V, 10.0),
        _kv(f"{a}.attention.head_count", U32, GLM_N_HEAD),
        _kv_array(f"{a}.attention.head_count_kv", U32, GLM_HEAD_COUNT_KV),
        _kv(f"{a}.attention.key_length", U32, GLM_KV_LORA),
        _kv(f"{a}.attention.value_length", U32, GLM_KV_LORA),
        _kv(f"{a}.attention.key_length_mla", U32, GLM_HEAD_MLA),
        _kv(f"{a}.attention.value_length_mla", U32, GLM_HEAD_MLA),
        _kv(f"{a}.attention.q_lora_rank", U32, GLM_Q_LORA),
        _kv(f"{a}.attention.kv_lora_rank", U32, GLM_KV_LORA),
        _kv(f"{a}.attention.layer_norm_rms_epsilon", F32V, 1e-5),
        _kv(f"{a}.attention.indexer.head_count", U32, GLM_IDX_HEADS),
        _kv(f"{a}.attention.indexer.key_length", U32, GLM_IDX_DIM),
        _kv(f"{a}.attention.indexer.top_k", U32, GLM_IDX_TOP_K),
        _kv(f"{a}.attention.indexer.kpool", U32, GLM_KPOOL),
        _kv(f"{a}.rope.dimension_count", U32, 0),
        _kv(f"{a}.rope.freq_base", F32V, 10000.0),
        _kv(f"{a}.ssm.conv_kernel", U32, GLM_SSM_KERNEL),
        _kv(f"{a}.kda.head_dim", U32, GLM_KDA_HEAD_DIM),
        _kv(f"{a}.kda.gate_lower_bound", F32V, -5.0),
        _kv(f"{a}.hyper_connection.count", U32, GLM_HC),
        _kv(f"{a}.hyper_connection.sinkhorn_iterations", U32, 2),  # 20
        _kv(f"{a}.hyper_connection.epsilon", F32V, 1e-6),
        _kv(f"{a}.vocab_size", U32, N_VOCAB),
        _kv("tokenizer.ggml.model", STRING, "test"),
        _kv_array("tokenizer.ggml.tokens", STRING, [f"tok_{i}" for i in range(N_VOCAB)]),
        _kv_array("tokenizer.ggml.scores", F32V, [0.0] * N_VOCAB),
    ]


def build(seed: int, expert_type: str, mtp: bool = False) -> list[Tensor]:
    rng = random.Random(seed)

    def weights(n: int, fan_in: int) -> array.array:
        bound = (3.0 / fan_in) ** 0.5  # unit output variance for unit-variance input
        return array.array("f", (rng.uniform(-bound, bound) for _ in range(n)))

    def ones(n: int) -> array.array:
        return array.array("f", [1.0] * n)

    def f32(name: str, dims: list[int], values: array.array) -> Tensor:
        if sys.byteorder != "little":
            values = array.array("f", values)
            values.byteswap()
        return Tensor(name, dims, GGML_F32, values.tobytes())

    def expert(name: str, dims: list[int], fan_in: int) -> Tensor:
        n = dims[0] * dims[1] * dims[2]
        values = weights(n, fan_in)
        if expert_type == "q8_0":
            return Tensor(name, dims, GGML_Q8_0, quantize_q8_0(values))
        if expert_type == "f16":
            return Tensor(name, dims, GGML_F16, struct.pack(f"<{len(values)}e", *values))
        return f32(name, dims, values)

    tensors = [
        f32("token_embd.weight", [N_EMBD, N_VOCAB], weights(N_EMBD * N_VOCAB, 3)),  # unit-ish variance
        f32("output_norm.weight", [N_EMBD], ones(N_EMBD)),
        f32("output.weight", [N_EMBD, N_VOCAB], weights(N_EMBD * N_VOCAB, N_EMBD)),
    ]
    q_dim, kv_dim = N_HEAD * HEAD_DIM, N_HEAD_KV * HEAD_DIM
    if mtp:
        for il in range(N_LAYER_MTP_TRUNK + 1):
            p = f"blk.{il}."
            tensors += [
                f32(p + "attn_norm.weight", [N_EMBD], ones(N_EMBD)),
                f32(p + "post_attention_norm.weight", [N_EMBD], ones(N_EMBD)),
            ]
            if il < N_LAYER_MTP_TRUNK and il % 2 == 0:  # the arch's memory needs a recurrent layer in the trunk
                key_dim, value_dim = HEAD_DIM * SSM_GROUPS, HEAD_DIM * SSM_RANK
                conv_dim = key_dim * 2 + value_dim
                tensors += [
                    f32(p + "attn_qkv.weight", [N_EMBD, conv_dim], weights(N_EMBD * conv_dim, N_EMBD)),
                    f32(p + "attn_gate.weight", [N_EMBD, value_dim], weights(N_EMBD * value_dim, N_EMBD)),
                    f32(p + "ssm_conv1d.weight", [SSM_KERNEL, conv_dim], weights(SSM_KERNEL * conv_dim, SSM_KERNEL)),
                    f32(p + "ssm_dt.bias", [SSM_RANK], ones(SSM_RANK)),
                    # -exp(A_log): a decay, so it has to be negative or the state grows without bound
                    f32(p + "ssm_a", [SSM_RANK], array.array("f", [-1.0] * SSM_RANK)),
                    f32(p + "ssm_beta.weight", [N_EMBD, SSM_RANK], weights(N_EMBD * SSM_RANK, N_EMBD)),
                    f32(p + "ssm_alpha.weight", [N_EMBD, SSM_RANK], weights(N_EMBD * SSM_RANK, N_EMBD)),
                    f32(p + "ssm_norm.weight", [HEAD_DIM], ones(HEAD_DIM)),
                    f32(p + "ssm_out.weight", [value_dim, N_EMBD], weights(value_dim * N_EMBD, value_dim)),
                ]
            else:
                tensors += [
                    f32(p + "attn_q.weight", [N_EMBD, q_dim * 2], weights(N_EMBD * q_dim * 2, N_EMBD)),
                    f32(p + "attn_k.weight", [N_EMBD, kv_dim], weights(N_EMBD * kv_dim, N_EMBD)),
                    f32(p + "attn_v.weight", [N_EMBD, kv_dim], weights(N_EMBD * kv_dim, N_EMBD)),
                    f32(p + "attn_output.weight", [q_dim, N_EMBD], weights(q_dim * N_EMBD, q_dim)),
                    f32(p + "attn_q_norm.weight", [HEAD_DIM], ones(HEAD_DIM)),
                    f32(p + "attn_k_norm.weight", [HEAD_DIM], ones(HEAD_DIM)),
                ]
            tensors += [
                f32(p + "ffn_gate_inp.weight", [N_EMBD, N_EXPERT], weights(N_EMBD * N_EXPERT, N_EMBD // 16)),
                expert(p + "ffn_gate_exps.weight", [N_EMBD, N_FF_EXP, N_EXPERT], N_EMBD),
                expert(p + "ffn_up_exps.weight", [N_EMBD, N_FF_EXP, N_EXPERT], N_EMBD),
                expert(p + "ffn_down_exps.weight", [N_FF_EXP, N_EMBD, N_EXPERT], N_FF_EXP),
                f32(p + "ffn_gate_inp_shexp.weight", [N_EMBD], ones(N_EMBD)),
                f32(p + "ffn_gate_shexp.weight", [N_EMBD, N_FF_EXP], weights(N_EMBD * N_FF_EXP, N_EMBD)),
                f32(p + "ffn_up_shexp.weight", [N_EMBD, N_FF_EXP], weights(N_EMBD * N_FF_EXP, N_EMBD)),
                f32(p + "ffn_down_shexp.weight", [N_FF_EXP, N_EMBD], weights(N_FF_EXP * N_EMBD, N_FF_EXP)),
            ]
        p = f"blk.{N_LAYER_MTP_TRUNK}.nextn."
        tensors += [
            f32(p + "eh_proj.weight", [2 * N_EMBD, N_EMBD], weights(2 * N_EMBD * N_EMBD, 2 * N_EMBD)),
            f32(p + "enorm.weight", [N_EMBD], ones(N_EMBD)),
            f32(p + "hnorm.weight", [N_EMBD], ones(N_EMBD)),
        ]
        return tensors
    for il in range(N_LAYER):
        p = f"blk.{il}."
        tensors += [
            f32(p + "attn_norm.weight", [N_EMBD], ones(N_EMBD)),
            f32(p + "attn_q.weight", [N_EMBD, q_dim], weights(N_EMBD * q_dim, N_EMBD)),
            f32(p + "attn_k.weight", [N_EMBD, kv_dim], weights(N_EMBD * kv_dim, N_EMBD)),
            f32(p + "attn_v.weight", [N_EMBD, kv_dim], weights(N_EMBD * kv_dim, N_EMBD)),
            f32(p + "attn_output.weight", [q_dim, N_EMBD], weights(q_dim * N_EMBD, q_dim)),
            f32(p + "attn_q_norm.weight", [HEAD_DIM], ones(HEAD_DIM)),
            f32(p + "attn_k_norm.weight", [HEAD_DIM], ones(HEAD_DIM)),
            f32(p + "ffn_norm.weight", [N_EMBD], ones(N_EMBD)),
            # a sharper router, so each token picks a clearly different subset
            f32(p + "ffn_gate_inp.weight", [N_EMBD, N_EXPERT], weights(N_EMBD * N_EXPERT, N_EMBD // 16)),
            expert(p + "ffn_gate_exps.weight", [N_EMBD, N_FF_EXP, N_EXPERT], N_EMBD),
            expert(p + "ffn_up_exps.weight", [N_EMBD, N_FF_EXP, N_EXPERT], N_EMBD),
            expert(p + "ffn_down_exps.weight", [N_FF_EXP, N_EMBD, N_EXPERT], N_FF_EXP),
        ]
    return tensors


def write(path: Path, tensors: list[Tensor], mtp: bool = False) -> None:
    arch = "qwen35moe" if mtp else ARCH
    extra = [
        _kv(f"{arch}.nextn_predict_layers", U32, 1),
        _kv(f"{arch}.full_attention_interval", U32, 2),  # layers 0 and 2 are recurrent, 1 and 3 attention
        _kv(f"{arch}.expert_shared_feed_forward_length", U32, N_FF_EXP),
        _kv_array(f"{arch}.rope.dimension_sections", U32, [HEAD_DIM // 4, HEAD_DIM // 8, HEAD_DIM // 8, 0]),
        _kv(f"{arch}.ssm.conv_kernel", U32, SSM_KERNEL),
        _kv(f"{arch}.ssm.inner_size", U32, 64),
        _kv(f"{arch}.ssm.state_size", U32, HEAD_DIM),
        _kv(f"{arch}.ssm.time_step_rank", U32, SSM_RANK),
        _kv(f"{arch}.ssm.group_count", U32, SSM_GROUPS),
    ] if mtp else []
    kvs = extra + [
        _kv("general.architecture", STRING, arch),
        _kv("general.name", STRING, "openharness-ssd-fixture"),
        _kv("general.alignment", U32, ALIGN),
        _kv(f"{arch}.context_length", U32, N_CTX_TRAIN),
        _kv(f"{arch}.embedding_length", U32, N_EMBD),
        _kv(f"{arch}.block_count", U32, N_LAYER_MTP_TRUNK + 1 if mtp else N_LAYER),
        _kv(f"{arch}.feed_forward_length", U32, N_FF_EXP * N_EXPERT_USED),
        _kv(f"{arch}.expert_feed_forward_length", U32, N_FF_EXP),
        _kv(f"{arch}.attention.head_count", U32, N_HEAD),
        _kv(f"{arch}.attention.head_count_kv", U32, N_HEAD_KV),
        _kv(f"{arch}.attention.key_length", U32, HEAD_DIM),
        _kv(f"{arch}.attention.value_length", U32, HEAD_DIM),
        _kv(f"{arch}.attention.layer_norm_rms_epsilon", F32V, 1e-6),
        _kv(f"{arch}.rope.freq_base", F32V, 10000.0),
        _kv(f"{arch}.expert_count", U32, N_EXPERT),
        _kv(f"{arch}.expert_used_count", U32, N_EXPERT_USED),
        _kv(f"{arch}.vocab_size", U32, N_VOCAB),
        _kv("tokenizer.ggml.model", STRING, "test"),
        _kv_array("tokenizer.ggml.tokens", STRING, [f"tok_{i}" for i in range(N_VOCAB)]),
        _kv_array("tokenizer.ggml.scores", F32V, [0.0] * N_VOCAB),
    ]
    write_file(path, kvs, tensors)


def write_file(path: Path, kvs: list[bytes], tensors: list[Tensor]) -> None:
    def pad(n: int) -> int:
        return (ALIGN - n % ALIGN) % ALIGN

    infos = bytearray()
    offset = 0
    for t in tensors:
        infos += _str(t.name) + struct.pack("<I", len(t.dims))
        infos += b"".join(struct.pack("<Q", d) for d in t.dims)
        infos += struct.pack("<IQ", t.type, offset)
        offset += len(t.data) + pad(len(t.data))

    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs) + bytes(infos)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(header)
        f.write(b"\0" * pad(len(header)))
        for t in tensors:
            f.write(t.data)
            f.write(b"\0" * pad(len(t.data)))
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write a tiny synthetic mixture-of-experts GGUF.")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--expert-type", choices=["f16", "q8_0", "f32"], default="f16",
                    help="routed expert type (f32 is refused at load: the CPU expert sink cannot compute it)")
    ap.add_argument("--mtp", action="store_true", help="a qwen35moe with a NextN/MTP layer")
    ap.add_argument("--glm", action="store_true", help="GLM-5.3-Flash's shape, F32 with placeholder experts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.glm:
        path = args.out / "ssd-glm-base.gguf"
        write_file(path, glm_kvs(), build_glm(args.seed))
        print(f"wrote {path} ({path.stat().st_size} bytes)")
        return 0
    path = args.out / f"ssd-moe-{'mtp-' if args.mtp else ''}{args.expert_type}.gguf"
    write(path, build(args.seed, args.expert_type, args.mtp), args.mtp)
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
