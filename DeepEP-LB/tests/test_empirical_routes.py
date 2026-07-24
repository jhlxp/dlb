"""Correctness checks for empirical token-level TopK route generation."""

from __future__ import annotations

import unittest

import torch

from empirical_routes import (
    DEFAULT_DISTRIBUTION,
    distribution_sha256,
    empirical_expert_hotness,
    empirical_topk,
)


class EmpiricalRoutesTest(unittest.TestCase):
    """Validate data provenance and token-level TopK invariants."""

    def test_distribution_digest(self) -> None:
        """Keep the copied empirical input byte-identical to its source."""
        self.assertEqual(
            distribution_sha256(),
            "55cdb477e3952fb9a14ac44735936b4f6675dd69310e5cdc9737c10ab8b34e4d",
        )

    def test_topk_shape_uniqueness_and_distribution(self) -> None:
        """Generate EP8-sized routes that follow the expert hotness."""
        topk_idx, topk_weights = empirical_topk(
            rank=0,
            num_tokens=4096,
            num_topk=8,
            num_experts=256,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(topk_idx.shape), (4096, 8))
        self.assertEqual(tuple(topk_weights.shape), (4096, 8))
        self.assertEqual(int(topk_idx.min()), 0)
        self.assertLess(int(topk_idx.max()), 256)

        ordered = topk_idx.sort(dim=1).values
        self.assertTrue(bool(torch.all(ordered[:, 1:] != ordered[:, :-1])))
        torch.testing.assert_close(
            topk_weights.sum(dim=1),
            torch.ones(4096),
            atol=2e-7,
            rtol=2e-7,
        )

        hotness = torch.tensor(
            empirical_expert_hotness(
                256,
                path=DEFAULT_DISTRIBUTION,
            ),
            dtype=torch.float32,
        )
        realized = torch.bincount(
            topk_idx.flatten(), minlength=256
        ).to(torch.float32)
        correlation = torch.corrcoef(
            torch.stack((hotness, realized))
        )[0, 1]
        self.assertGreater(float(correlation), 0.98)
        self.assertEqual(int(realized.sum()), 4096 * 8)


if __name__ == "__main__":
    unittest.main()

