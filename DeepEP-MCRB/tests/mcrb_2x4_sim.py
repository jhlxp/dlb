"""2x4 Rail simulation for Mask-Constrained Rail Balancing.

This test validates the scheduling idea without adding a planner stage to the
DeepEP data path. The production integration should inline the same selection
rule into existing notify/dispatch scans.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


NUM_RAILS = 4
NUM_SERVERS = 2
NUM_EXPERTS = 256
NUM_TOPK = 8
DEFAULT_LOG_ROOT = Path(__file__).resolve().parent / "logs"
DEFAULT_EMPIRICAL_DISTRIBUTION = (
    Path(__file__).resolve().parent
    / "data/empirical_pooled_distribution.csv"
)


@dataclass(frozen=True)
class Message:
    """A deduplicated token message from one source Rail to one remote server."""

    source_rail: int
    dst_mask: int


def mask_contains(mask: int, rail: int) -> bool:
    """Return whether the compact destination mask contains a Rail."""

    return ((mask >> rail) & 1) != 0


def forwarding_cost(source_rail: int, selected_rail: int, dst_mask: int) -> int:
    """Count intra-server forwarding hops for one token message."""

    dst_count = dst_mask.bit_count()
    source_stage = int(selected_rail != source_rail)
    dst_forward = dst_count - int(mask_contains(dst_mask, selected_rail))
    return source_stage + dst_forward


def compute_quota_row(loads: list[int], source_rail: int) -> list[int]:
    """Compute the selected-Rail quota row for one source Rail."""

    total = sum(loads)
    targets = [
        total // NUM_RAILS + int(rail < total % NUM_RAILS)
        for rail in range(NUM_RAILS)
    ]
    deficits = [max(targets[rail] - loads[rail], 0) for rail in range(NUM_RAILS)]
    quotas = [0] * NUM_RAILS

    for producer in range(NUM_RAILS):
        direct = min(loads[producer], targets[producer])
        if producer == source_rail:
            quotas[producer] = direct

        surplus = loads[producer] - direct
        for selected in range(NUM_RAILS):
            moved = min(surplus, deficits[selected])
            if producer == source_rail:
                quotas[selected] += moved
            surplus -= moved
            deficits[selected] -= moved
    return quotas


def choose_rail_from_quota(
    source_rail: int,
    dst_mask: int,
    quotas: list[int],
    used: list[int],
    *,
    allow_extra_forward: bool,
) -> int:
    """Choose source Rail or a destination Rail using a quota row."""

    if dst_mask == 0:
        raise ValueError("dst_mask must be non-zero")

    source_is_destination = mask_contains(dst_mask, source_rail)
    for rail in range(NUM_RAILS):
        if rail == source_rail or not mask_contains(dst_mask, rail):
            continue
        if source_is_destination and not allow_extra_forward:
            continue
        if used[rail] < quotas[rail]:
            used[rail] += 1
            return rail

    used[source_rail] += 1
    return source_rail


def balance_messages(
    messages: list[Message],
    *,
    allow_extra_forward: bool = False,
) -> dict[str, object]:
    """Run quota-row MCRB over a source-server to destination-server tile."""

    original_loads = [0] * NUM_RAILS
    original_tile = [[0] * NUM_RAILS for _ in range(NUM_RAILS)]
    for message in messages:
        original_loads[message.source_rail] += 1
        for dst_rail in range(NUM_RAILS):
            if mask_contains(message.dst_mask, dst_rail):
                original_tile[message.source_rail][dst_rail] += 1

    total = len(messages)
    target = (total + NUM_RAILS - 1) // NUM_RAILS
    selected_loads = [0] * NUM_RAILS
    selected_tile = [[0] * NUM_RAILS for _ in range(NUM_RAILS)]
    selected = []
    quotas = [
        compute_quota_row(original_loads, source_rail)
        for source_rail in range(NUM_RAILS)
    ]
    used = [[0] * NUM_RAILS for _ in range(NUM_RAILS)]

    for message in messages:
        rail = choose_rail_from_quota(
            message.source_rail,
            message.dst_mask,
            quotas[message.source_rail],
            used[message.source_rail],
            allow_extra_forward=allow_extra_forward,
        )
        selected.append(rail)
        selected_loads[rail] += 1
        for dst_rail in range(NUM_RAILS):
            if mask_contains(message.dst_mask, dst_rail):
                selected_tile[rail][dst_rail] += 1

    moved = sum(
        int(rail != message.source_rail)
        for rail, message in zip(selected, messages)
    )
    original_forwarding = sum(
        forwarding_cost(message.source_rail, message.source_rail, message.dst_mask)
        for message in messages
    )
    selected_forwarding = sum(
        forwarding_cost(message.source_rail, rail, message.dst_mask)
        for rail, message in zip(selected, messages)
    )
    violations = sum(
        int(rail != message.source_rail and not mask_contains(message.dst_mask, rail))
        for rail, message in zip(selected, messages)
    )
    return {
        "total_messages": total,
        "target_per_rail": target,
        "original_tile_src_by_dst": original_tile,
        "original_loads": original_loads,
        "selected_tile_selected_by_dst": selected_tile,
        "selected_loads": selected_loads,
        "original_imbalance": max(original_loads) - min(original_loads),
        "selected_imbalance": max(selected_loads) - min(selected_loads),
        "moved_messages": moved,
        "moved_fraction": moved / max(total, 1),
        "original_forwarding_hops": original_forwarding,
        "selected_forwarding_hops": selected_forwarding,
        "extra_forwarding_hops": selected_forwarding - original_forwarding,
        "third_party_rail_violations": violations,
    }


def synthetic_messages(loads: list[int]) -> list[Message]:
    """Build a controllable source-load case with broad destination coverage."""

    messages: list[Message] = []
    for source_rail, count in enumerate(loads):
        for idx in range(count):
            dst_rail = idx % NUM_RAILS
            messages.append(Message(source_rail=source_rail, dst_mask=1 << dst_rail))
    return messages


def tile_messages(tile: list[list[int]]) -> list[Message]:
    """Build a source-by-destination Rail traffic tile."""

    if len(tile) != NUM_RAILS or any(len(row) != NUM_RAILS for row in tile):
        raise ValueError(f"tile must be {NUM_RAILS}x{NUM_RAILS}")

    messages: list[Message] = []
    for source_rail, row in enumerate(tile):
        for dst_rail, count in enumerate(row):
            messages.extend(
                Message(source_rail=source_rail, dst_mask=1 << dst_rail)
                for _ in range(count)
            )
    return messages


def topk_messages(
    *,
    tokens_per_source_rail: int,
    seed: int,
) -> list[Message]:
    """Generate token-deduplicated messages for a 2-server, 4-Rail topology."""

    rng = random.Random(seed)
    experts_per_rank = NUM_EXPERTS // (NUM_SERVERS * NUM_RAILS)
    messages: list[Message] = []
    for source_rail in range(NUM_RAILS):
        for _ in range(tokens_per_source_rail):
            experts = rng.sample(range(NUM_EXPERTS), NUM_TOPK)
            dst_mask = 0
            for expert in experts:
                rank = expert // experts_per_rank
                server = rank // NUM_RAILS
                rail = rank % NUM_RAILS
                if server == 1:
                    dst_mask |= 1 << rail
            if dst_mask:
                messages.append(Message(source_rail=source_rail, dst_mask=dst_mask))
    return messages


def load_empirical_heats(path: Path) -> list[tuple[int, float]]:
    """Load the empirical heat distribution as (heat, pmf) rows."""

    rows: list[tuple[int, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            heat = int(row["heat"])
            pmf = float(row["pmf"])
            rows.append((heat, pmf))
    if not rows:
        raise ValueError(f"empty empirical distribution: {path}")
    return rows


def sample_heat_distribution(
    rows: list[tuple[int, float]],
    *,
    rng: random.Random,
    num_experts: int,
) -> list[int]:
    """Sample one empirical heat for each expert."""

    heats = [heat for heat, _ in rows]
    weights = [pmf for _, pmf in rows]
    sampled = rng.choices(heats, weights=weights, k=num_experts)
    return [max(1, heat) for heat in sampled]


def weighted_sample_without_replacement(
    weights: list[float],
    *,
    rng: random.Random,
    k: int,
) -> list[int]:
    """Sample k expert indices without replacement using mutable weights."""

    mutable_weights = weights[:]
    selected: list[int] = []
    for _ in range(k):
        total = sum(mutable_weights)
        if total <= 0:
            break
        draw = rng.random() * total
        prefix = 0.0
        for expert, weight in enumerate(mutable_weights):
            prefix += weight
            if prefix >= draw:
                selected.append(expert)
                mutable_weights[expert] = 0.0
                break
    return selected


def empirical_receive_messages(
    *,
    source_loads: list[int],
    seed: int,
    distribution_path: Path,
) -> tuple[list[Message], dict[str, object]]:
    """Generate fixed source sends with heavy-tailed destination experts."""

    rng = random.Random(seed)
    num_destination_experts = NUM_EXPERTS // NUM_SERVERS
    experts_per_destination_rail = num_destination_experts // NUM_RAILS
    heats = sample_heat_distribution(
        load_empirical_heats(distribution_path),
        rng=rng,
        num_experts=num_destination_experts,
    )
    messages: list[Message] = []
    destination_expert_hits = [0] * num_destination_experts
    destination_rail_hits = [0] * NUM_RAILS

    destination_experts = list(range(num_destination_experts))
    for source_rail, num_messages in enumerate(source_loads):
        sampled_experts = rng.choices(
            destination_experts,
            weights=heats,
            k=num_messages,
        )
        for local_expert in sampled_experts:
            dst_rail = local_expert // experts_per_destination_rail
            destination_expert_hits[local_expert] += 1
            destination_rail_hits[dst_rail] += 1
            messages.append(Message(source_rail=source_rail, dst_mask=1 << dst_rail))

    top_experts = sorted(
        range(num_destination_experts),
        key=lambda expert: destination_expert_hits[expert],
        reverse=True,
    )[:16]
    metadata = {
        "distribution": str(distribution_path),
        "source_loads": source_loads,
        "destination_server": 1,
        "destination_experts": num_destination_experts,
        "expert_heat_min": min(heats),
        "expert_heat_max": max(heats),
        "expert_heat_sum": sum(heats),
        "destination_rail_message_hits": destination_rail_hits,
        "top_destination_experts_by_hits": [
            {
                "local_expert": expert,
                "global_expert": NUM_EXPERTS // NUM_SERVERS + expert,
                "rail": expert // experts_per_destination_rail,
                "sampled_heat": heats[expert],
                "hits": destination_expert_hits[expert],
            }
            for expert in top_experts
        ],
    }
    return messages, metadata


def main() -> None:
    """Run the default 2x4 MCRB simulation and write a timestamped log."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-source-rail", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--allow-extra-forward", action="store_true")
    parser.add_argument(
        "--empirical-distribution",
        type=Path,
        default=DEFAULT_EMPIRICAL_DISTRIBUTION,
    )
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    args = parser.parse_args()

    empirical_messages, empirical_metadata = empirical_receive_messages(
        source_loads=[5000, 6000, 7000, 8000],
        seed=args.seed,
        distribution_path=args.empirical_distribution,
    )

    cases = {
        "synthetic_5000_6000_7000_8000": synthetic_messages(
            [5000, 6000, 7000, 8000]
        ),
        "skewed_4x4_expert_tile": tile_messages(
            [
                [4200, 500, 200, 100],
                [600, 4700, 500, 200],
                [400, 800, 5200, 600],
                [200, 500, 1000, 6300],
            ]
        ),
        "random_topk_ep8_ep256": topk_messages(
            tokens_per_source_rail=args.tokens_per_source_rail,
            seed=args.seed,
        ),
        "fixed_send_empirical_receive_ep8_ep256": empirical_messages,
    }
    report = {
        "topology": {
            "servers": NUM_SERVERS,
            "rails_per_server": NUM_RAILS,
            "ranks": NUM_SERVERS * NUM_RAILS,
            "experts": NUM_EXPERTS,
            "topk": NUM_TOPK,
        },
        "algorithm": "MCRB",
        "allow_extra_forward": args.allow_extra_forward,
        "case_metadata": {
            "fixed_send_empirical_receive_ep8_ep256": empirical_metadata,
        },
        "cases": {
            name: balance_messages(
                messages,
                allow_extra_forward=args.allow_extra_forward,
            )
            for name, messages in cases.items()
        },
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_dir = args.log_root / f"{timestamp}-mcrb-2x4-sim"
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "result.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"log_dir={log_dir}")


if __name__ == "__main__":
    main()
