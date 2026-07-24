"""Validate compact Rail quotas across production-relevant distributions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from dlb_test_extension import load_dlb_test_extension
from empirical_routes import distribution_sha256, empirical_topk


RAILS = 4
LOG_ROOT = Path(__file__).resolve().parent / "logs"


def model_derived_loads() -> list[int]:
    """Count token-deduplicated traffic from four source Rails."""
    device = torch.device("cuda")
    loads: list[int] = []
    for source_rail in range(RAILS):
        topk_idx, _ = empirical_topk(
            rank=source_rail,
            num_tokens=4096,
            num_topk=8,
            num_experts=256,
            device=device,
        )
        destination_ranks = torch.div(
            topk_idx, 32, rounding_mode="floor"
        )
        sends_to_remote_server = torch.any(
            destination_ranks >= RAILS, dim=1
        )
        loads.append(int(sends_to_remote_server.sum().item()))
    return loads


def main() -> None:
    """Check conservation and exact floor/ceil Rail balance."""
    extension = load_dlb_test_extension()
    cases = {
        "balanced": [6500, 6500, 6500, 6500],
        "skewed": [3000, 5000, 7000, 11000],
        "extreme_hotspot": [0, 0, 0, 26000],
        "all_cross_server_directed": [5000, 6000, 7000, 8000],
        "model_derived": model_derived_loads(),
    }
    results: dict[str, dict[str, object]] = {}
    for name, loads in cases.items():
        source_loads = torch.tensor(
            [[0, 0, 0, 0], loads],
            dtype=torch.int32,
            device="cuda",
        )
        quotas, balanced = extension.run_dlb_compact_quota(
            source_loads, 0, 0
        )
        torch.cuda.synchronize()
        selected = balanced[1]
        torch.testing.assert_close(
            quotas[1].sum(dim=-1, dtype=torch.int32),
            source_loads[1],
        )
        assert int(selected.sum().item()) == sum(loads)
        assert int(selected.max().item() - selected.min().item()) <= 1
        quota_rows = quotas[1].cpu().tolist()
        moved = sum(
            quota_rows[source][selected_rail]
            for source in range(RAILS)
            for selected_rail in range(RAILS)
            if source != selected_rail
        )
        results[name] = {
            "source_rail_loads": loads,
            "selected_rail_loads": selected.cpu().tolist(),
            "quota_rows": quota_rows,
            "moved_tokens": moved,
            "moved_fraction": moved / max(sum(loads), 1),
        }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_dir = LOG_ROOT / f"{timestamp}-dlb-distributions"
    log_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "passed",
        "distribution_sha256": distribution_sha256(),
        "cases": results,
    }
    (log_dir / "result.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"log_dir={log_dir}")


if __name__ == "__main__":
    main()
