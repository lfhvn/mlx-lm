"""Speculative decode with a fake draft on the tiny random looped model of
../common/test_inkling2.py: output must equal plain greedy decoding, for any
acceptance pattern (checks cache rollback, incl. rotating and conv caches).

    .venv/bin/python -m unittest spec_bench/test_spec.py
"""

import random, sys, unittest
from types import SimpleNamespace

import mlx.core as mx

sys.path.insert(0, "/Users/lhm/inkling-orchestra/engine/common")
from test_inkling2 import LOOP_CONFIG, UNPADDED_VOCAB, create_model  # noqa: E402

from mlx_lm.models.inkling2.speculative import (  # noqa: E402
    _forward,
    greedy_generate,
    make_spec_cache,
    speculative_generate,
)


class FakeDraft:
    """Drafts the true continuation, corrupted at random positions."""

    def __init__(self, truth, num_visits, p_wrong, seed):
        self.args = SimpleNamespace(target_layer_ids=[1, 5, num_visits], block_size=8)
        self.truth, self.p_wrong = truth, p_wrong
        self.rng = random.Random(seed)
        self.offsets = []

    def reset(self):
        self.ctx_end = 0

    def store_context(self, feats, offset):
        assert offset == self.ctx_end, (offset, self.ctx_end)
        self.ctx_end += feats[0].shape[1]

    def draft(self, anchor, offset, last):
        assert offset == self.ctx_end
        out = []
        for j in range(1, 8):
            t = self.truth[offset + j] if offset + j < len(self.truth) else 0
            if self.rng.random() < self.p_wrong:
                t = (t + 1) % UNPADDED_VOCAB
            out.append(t)
        return mx.array([out], mx.int32)


class TestSpec(unittest.TestCase):
    def test_matches_greedy(self):
        model = create_model(config=LOOP_CONFIG)
        nv = len(model.model.language_model.layer_visits)
        for prompt_len in (5, 30):
            prompt = mx.random.randint(0, UNPADDED_VOCAB, (prompt_len,)).tolist()
            base, _ = greedy_generate(model, prompt, 60)
            for p_wrong in (0.0, 0.15, 0.5, 1.0):
                draft = FakeDraft(prompt + base, nv, p_wrong, seed=prompt_len)
                stats = {}
                spec, _ = speculative_generate(model, draft, prompt, 60, stats=stats)
                self.assertEqual(spec, base, f"prompt {prompt_len} p_wrong {p_wrong}")

    def test_aux_does_not_change_logits(self):
        model = create_model(config=LOOP_CONFIG)
        nv = len(model.model.language_model.layer_visits)
        tokens = mx.random.randint(0, UNPADDED_VOCAB, (1, 12))
        plain = model(tokens)
        logits, feats = _forward(model, tokens, None, [0, 4, nv])
        self.assertTrue(mx.array_equal(plain, logits))
        self.assertEqual(len(feats), 3)
        self.assertTrue(all(f.shape == (1, 12, 64) for f in feats))


if __name__ == "__main__":
    unittest.main()
