"""Generate token-level TopK routes from the empirical hotness distribution."""

from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

import torch


DEFAULT_DISTRIBUTION = (
    Path(__file__).resolve().parent
    / "data"
    / "empirical_pooled_distribution.csv"
)
DEFAULT_SEED = 20260624


def distribution_sha256(path: Path = DEFAULT_DISTRIBUTION) -> str:
    """Return the content digest used to identify an empirical input."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pooled_heat_values(
    path: Path = DEFAULT_DISTRIBUTION,
) -> list[float]:
    """Expand the pooled heat/count CSV into its empirical samples."""
    values: list[float] = []
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"heat", "count"} <= set(
            reader.fieldnames
        ):
            raise ValueError(f"{path} must contain heat and count columns")
        for row in reader:
            heat = float(row["heat"])
            count = int(row["count"])
            if heat < 0 or count < 0:
                raise ValueError(f"{path} contains a negative heat/count")
            values.extend([heat] * count)
    if not values:
        raise ValueError(f"{path} contains no empirical samples")
    return values


def empirical_expert_hotness(
    num_experts: int,
    *,
    path: Path = DEFAULT_DISTRIBUTION,
    seed: int = DEFAULT_SEED,
) -> list[float]:
    """Reproduce the HTSIM midpoint-quantile expert hotness sampling."""
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    values = sorted(read_pooled_heat_values(path))
    count = len(values)
    hotness = [
        values[min(count - 1, int((expert + 0.5) / num_experts * count))]
        for expert in range(num_experts)
    ]
    random.Random(seed).shuffle(hotness)
    if sum(value > 0 for value in hotness) < num_experts:
        raise ValueError("all sampled expert hotness values must be positive")
    return hotness


def empirical_topk(
    *,
    rank: int,
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    device: torch.device,
    path: Path = DEFAULT_DISTRIBUTION,
    seed: int = DEFAULT_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate reproducible, unique weighted TopK experts for each token."""
    if not 0 < num_topk <= num_experts:
        raise ValueError("num_topk must be in [1, num_experts]")
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")

    hotness = torch.tensor(
        empirical_expert_hotness(
            num_experts,
            path=path,
            seed=seed,
        ),
        dtype=torch.float32,
        device=device,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1009 * rank)
    topk_idx = torch.multinomial(
        hotness.expand(num_tokens, -1),
        num_samples=num_topk,
        replacement=False,
        generator=generator,
    ).contiguous()
    selected_hotness = hotness[topk_idx]
    topk_weights = (
        selected_hotness
        / selected_hotness.sum(dim=1, keepdim=True)
    ).contiguous()

    sorted_indices = topk_idx.sort(dim=1).values
    if torch.any(sorted_indices[:, 1:] == sorted_indices[:, :-1]):
        raise AssertionError("empirical TopK contains duplicate experts")
    return topk_idx, topk_weights
