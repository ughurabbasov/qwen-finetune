#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mix Azerbaijani data with replay data for multilingual continual finetuning."
    )
    parser.add_argument("--az_jsonl", type=str, required=True, help="Path to Azerbaijani JSONL.")
    parser.add_argument("--replay_jsonl", type=str, required=True, help="Path to replay JSONL.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to output mixed JSONL.")
    parser.add_argument("--az_ratio", type=float, default=0.3, help="Target Azerbaijani ratio in mixed output.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Total output rows. 0 means use maximum balanced size without replacement.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    if not (0.0 < args.az_ratio < 1.0):
        raise ValueError("--az_ratio must be between 0 and 1.")

    az_path = Path(args.az_jsonl).resolve()
    replay_path = Path(args.replay_jsonl).resolve()
    out_path = Path(args.output_jsonl).resolve()

    az_rows = load_jsonl(az_path)
    replay_rows = load_jsonl(replay_path)

    if len(az_rows) == 0:
        raise ValueError(f"No rows found in az_jsonl: {az_path}")
    if len(replay_rows) == 0:
        raise ValueError(f"No rows found in replay_jsonl: {replay_path}")

    random.seed(args.seed)

    if args.max_samples > 0:
        total_samples = int(args.max_samples)
        az_count = int(round(total_samples * args.az_ratio))
        replay_count = total_samples - az_count
    else:
        total_by_az = int(len(az_rows) / args.az_ratio)
        total_by_replay = int(len(replay_rows) / (1.0 - args.az_ratio))
        total_samples = min(total_by_az, total_by_replay)
        if total_samples <= 0:
            raise ValueError("Could not compute a valid balanced output size from inputs and az_ratio.")
        az_count = int(round(total_samples * args.az_ratio))
        replay_count = total_samples - az_count

    if az_count > len(az_rows):
        raise ValueError(
            f"Requested {az_count} AZ rows but only {len(az_rows)} available. "
            "Lower --max_samples or --az_ratio."
        )
    if replay_count > len(replay_rows):
        raise ValueError(
            f"Requested {replay_count} replay rows but only {len(replay_rows)} available. "
            "Lower --max_samples or raise --az_ratio."
        )

    sampled_az = random.sample(az_rows, az_count)
    sampled_replay = random.sample(replay_rows, replay_count)
    mixed = sampled_az + sampled_replay
    random.shuffle(mixed)
    write_jsonl(out_path, mixed)

    print(f"Saved mixed dataset: {out_path}")
    print(f"Total rows: {len(mixed)}")
    print(f"AZ rows: {len(sampled_az)}")
    print(f"Replay rows: {len(sampled_replay)}")


if __name__ == "__main__":
    main()
