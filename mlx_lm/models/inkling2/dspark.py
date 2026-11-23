# Copyright © 2026 Apple Inc.
"""Inkling 2 DSpark draft model (`mtp.safetensors`).

Port of vLLM `vllm/models/inkling2/nvidia/dspark.py`. One block is the anchor
(the last target token) and `block_size - 1` mask embeddings. Three dense
layers attend, without a causal mask, to the block and to context K/V made from
mixed target features. A recurrent Markov head then corrects the logits of
each mask position from left to right.
"""

from dataclasses import dataclass

import mlx.core as mx
from mlx import nn

from mlx_lm.models.cache import KVCache


@dataclass
class DraftArgs:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    local_layer_ids: list
    sliding_window_size: int
    block_size: int
    markov_rank: int
    target_layer_ids: list

    @classmethod
    def from_dict(cls, d: dict) -> "DraftArgs":
        assert d.get("markov_head_type") == "rnn_hidden"
        assert not d.get("q_bias") and not d.get("o_bias")
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def _rope(x: mx.array, offset: int, theta: float) -> mx.array:
    return mx.fast.rope(
        x, x.shape[-1], traditional=False, base=theta, scale=1.0, offset=offset
    )


class DraftAttention(nn.Module):
    def __init__(self, args: DraftArgs, is_local: bool):
        super().__init__()
        h, d = args.hidden_size, args.head_dim
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = d
        self.theta = args.rope_theta
        self.window = args.sliding_window_size if is_local else 0
        self.q_proj = nn.Linear(h, self.num_heads * d, bias=False)
        self.k_proj = nn.Linear(h, self.num_kv_heads * d, bias=False)
        self.v_proj = nn.Linear(h, self.num_kv_heads * d, bias=False)
        self.o_proj = nn.Linear(self.num_heads * d, h, bias=False)
        self.k_proj_ctx = nn.Linear(h, self.num_kv_heads * d, bias=False)
        self.v_proj_ctx = nn.Linear(h, self.num_kv_heads * d, bias=False)
        eps = args.rms_norm_eps
        self.q_norm = nn.RMSNorm(d, eps=eps)
        self.k_norm = nn.RMSNorm(d, eps=eps)
        self.k_norm_ctx = nn.RMSNorm(d, eps=eps)

    def _heads(self, x: mx.array, n: int) -> mx.array:
        B, L, _ = x.shape
        return x.reshape(B, L, n, self.head_dim)

    def context_kv(self, context: mx.array, offset: int):
        k = self.k_norm_ctx(self._heads(self.k_proj_ctx(context), self.num_kv_heads))
        v = self._heads(self.v_proj_ctx(context), self.num_kv_heads)
        k = _rope(k.transpose(0, 2, 1, 3), offset, self.theta)
        return k, v.transpose(0, 2, 1, 3)

    def __call__(self, x: mx.array, offset: int, ctx_k, ctx_v) -> mx.array:
        B, L, _ = x.shape
        q = self.q_norm(self._heads(self.q_proj(x), self.num_heads))
        k = self.k_norm(self._heads(self.k_proj(x), self.num_kv_heads))
        v = self._heads(self.v_proj(x), self.num_kv_heads)
        q = _rope(q.transpose(0, 2, 1, 3), offset, self.theta)
        k = _rope(k.transpose(0, 2, 1, 3), offset, self.theta)
        v = v.transpose(0, 2, 1, 3)
        if ctx_k is not None:
            # A local layer sees the window before the block and the block.
            if self.window:
                ctx_k, ctx_v = ctx_k[..., -self.window :, :], ctx_v[..., -self.window :, :]
            k = mx.concatenate([ctx_k, k], axis=2)
            v = mx.concatenate([ctx_v, v], axis=2)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=None
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class DraftMLP(nn.Module):
    def __init__(self, args: DraftArgs):
        super().__init__()
        h, i = args.hidden_size, args.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x).astype(mx.float32)
        up = self.up_proj(x).astype(mx.float32)
        return self.down_proj((nn.silu(gate) * up).astype(x.dtype))


class DraftLayer(nn.Module):
    def __init__(self, args: DraftArgs, is_local: bool):
        super().__init__()
        eps = args.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.dflare_norm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.self_attn = DraftAttention(args, is_local)
        self.mlp = DraftMLP(args)


class MarkovHead(nn.Module):
    def __init__(self, args: DraftArgs, vocab_size: int):
        super().__init__()
        h, r = args.hidden_size, args.markov_rank
        self.markov_w1 = nn.Embedding(vocab_size, r)
        self.rnn_proj = nn.Linear(2 * r + 2 * h, 3 * r, bias=True)
        self.hidden_out = nn.Linear(r, h, bias=True)
        self.dflare_norm = nn.RMSNorm(h, eps=args.rms_norm_eps)


class DSparkDraft(nn.Module):
    def __init__(self, args: DraftArgs, vocab_size: int):
        super().__init__()
        self.args = args
        self.layers = [
            DraftLayer(args, i in args.local_layer_ids)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.markov_head = MarkovHead(args, vocab_size)
        self.fuse_weight = mx.zeros((args.num_hidden_layers + 1, len(args.target_layer_ids)))
        self.mask_embed = mx.zeros((args.hidden_size,))

    def attach_target(self, target) -> None:
        """Share the target's embeddings and LM head; derive the folded
        Markov projection (`lm_head @ hidden_out`) and the fp32 weights."""
        lm = target.model.language_model
        self._embed = lm.embed
        self._lm_head = target.lm_head
        text = target.args.text
        self.vocab = text.unpadded_vocab_size
        self.softcap = text.final_logit_softcapping
        head = self.markov_head
        r = self.args.markov_rank
        w = target.lm_head.weight
        proj = head.hidden_out.weight.astype(mx.float32)
        rows = []
        for s in range(0, w.shape[0], 16384):
            rows.append((w[s : s + 16384].astype(mx.float32) @ proj).astype(w.dtype))
        self.fold_weight = mx.concatenate(rows)
        self.fold_bias = (
            w.astype(mx.float32) @ head.hidden_out.bias.astype(mx.float32)
        ).astype(w.dtype)
        rnn_w = head.rnn_proj.weight.astype(mx.float32)
        self.rnn_w_step = rnn_w[:, : 2 * r].T
        self.rnn_w_ctx = rnn_w[:, 2 * r :].T
        self.rnn_b = head.rnn_proj.bias.astype(mx.float32)
        self.fuse_alpha = mx.softmax(self.fuse_weight.astype(mx.float32), axis=-1)
        mx.eval(self.fold_weight, self.fold_bias, self.rnn_w_step, self.rnn_w_ctx)
        self.reset()

    def reset(self) -> None:
        self.ctx = [KVCache() for _ in self.layers]

    def mix(self, features: list, row: int, norm: nn.RMSNorm) -> mx.array:
        """RMSNorm of the fuse-weighted sum of the target features (fp32)."""
        mixed = sum(
            f.astype(mx.float32) * self.fuse_alpha[row, t] for t, f in enumerate(features)
        )
        return mx.fast.rms_norm(mixed, norm.weight, norm.eps).astype(features[0].dtype)

    def store_context(self, features: list, offset: int) -> None:
        """Add context K/V for target positions `offset, offset + 1, ...`."""
        for i, layer in enumerate(self.layers):
            context = self.mix(features, i, layer.dflare_norm)
            k, v = layer.self_attn.context_kv(context, offset)
            cache = self.ctx[i]
            assert cache.offset == offset
            cache.update_and_fetch(k, v)

    def trim_context(self, n: int) -> None:
        for c in self.ctx:
            c.trim(n)

    def backbone(self, anchor: mx.array, offset: int) -> mx.array:
        """Hidden states of the block at positions `offset ...`; `anchor` is
        the token at `offset`."""
        B = anchor.shape[0]
        n_mask = self.args.block_size - 1
        x = self._embed(anchor.reshape(B, 1))
        masks = mx.broadcast_to(self.mask_embed.astype(x.dtype), (B, n_mask, x.shape[-1]))
        x = mx.concatenate([x, masks], axis=1)
        residual = None
        for layer, cache in zip(self.layers, self.ctx):
            residual = x if residual is None else residual + x
            x = layer.input_layernorm(residual)
            ctx_k = ctx_v = None
            if cache.keys is not None:
                ctx_k, ctx_v = cache.keys[..., : cache.offset, :], cache.values[..., : cache.offset, :]
            x = layer.self_attn(x, offset, ctx_k, ctx_v)
            residual = residual + x
            x = layer.mlp(layer.post_attention_layernorm(residual))
        return self.norm(residual + x)

    def draft(self, anchor: mx.array, offset: int, last_features: list) -> mx.array:
        """Greedy draft of the `block_size - 1` tokens after `anchor`.

        `last_features` are the target features of the token before the anchor.
        Returns (B, block_size - 1) token ids, as a lazy array.
        """
        r = self.args.markov_rank
        hidden = self.backbone(anchor, offset)[:, 1:]  # (B, n, H)
        B, n, _ = hidden.shape
        base = self._lm_head(hidden)  # (B, n, V) bf16
        f_rnn = self.mix(last_features, -1, self.markov_head.dflare_norm)  # (B, 1, H)
        inputs = mx.concatenate(
            [hidden, mx.broadcast_to(f_rnn, hidden.shape)], axis=-1
        ).astype(mx.float32)
        context = inputs @ self.rnn_w_ctx + self.rnn_b  # (B, n, 3r) fp32
        state = mx.zeros((B, r), mx.bfloat16)
        prev = anchor.reshape(B)
        tokens = []
        bf = mx.bfloat16
        for i in range(n):
            step_in = mx.concatenate(
                [state, self.markov_head.markov_w1(prev).astype(bf)], axis=-1
            ).astype(mx.float32)
            raw = (step_in @ self.rnn_w_step + context[:, i]).astype(bf)
            gate, cand, out = mx.split(raw, 3, axis=-1)
            gate = mx.sigmoid(gate)
            state = gate * state + (1 - gate) * mx.tanh(cand)
            latent = mx.tanh(out)
            bias = latent @ self.fold_weight.T + self.fold_bias
            logits = base[:, i] + (0 if getattr(self, "no_markov", False) else bias)
            logits = mx.tanh(logits / self.softcap) * self.softcap
            prev = mx.argmax(logits[:, : self.vocab], axis=-1).astype(mx.int32)
            tokens.append(prev)
        return mx.stack(tokens, axis=1)


def load_draft(target, weights_dir: str) -> DSparkDraft:
    import json
    from pathlib import Path

    d = Path(weights_dir)
    config = json.loads((d / "config.json").read_text())["mtp_config"]
    args = DraftArgs.from_dict(config)
    text = target.args.text
    num_visits = len(target.model.language_model.layer_visits)
    assert args.target_layer_ids[-1] == num_visits, "last id is the final hidden"
    draft = DSparkDraft(args, text.vocab_size)
    draft.load_weights(list(mx.load(str(d / "mtp.safetensors")).items()), strict=True)
    mx.eval(draft.parameters())
    draft.attach_target(target)
    return draft
