#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Any, Optional

BEGIN_SOL = r"<\|begin_of_solution\|>"
END_SOL   = r"<\|end_of_solution\|>"
SOL_RE = re.compile(BEGIN_SOL + r"(.*?)" + END_SOL, flags=re.DOTALL)


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at {path}:{line_no}: {e}")


def write_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_short_form_with_tags(solution_text: str) -> str:
    """
    Extract <|begin_of_solution|> ... <|end_of_solution|> including tags.
    Return "" if not found.
    """
    if not solution_text:
        return ""
    m = SOL_RE.search(solution_text)
    if not m:
        return ""
    inner = m.group(1).strip()
    return f"<|begin_of_solution|>\n{inner}\n<|end_of_solution|>"


def normalize_domain(domain: Optional[str]) -> str:
    if not domain:
        return "unknown"
    d = str(domain).strip().lower()
    mapping = {
        "maths": "math",
        "mathematics": "math",
        "bio": "biology",
        "chem": "chemistry",
        "phys": "physics",
    }
    return mapping.get(d, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to input jsonl (your overall dataset)")
    ap.add_argument("--out_dir", default="/code/GLoRE-main/dataset/data/demon",
                    help="output dir for domain-split files (default: /code/GLoRE-main/dataset/data/demon)")
    ap.add_argument("--mixed_out", default="/code/GLoRE-main/dataset/data/demon/long_short_form_thought.jsonl",
                    help="output path for mixed (all domains) jsonl")
    ap.add_argument("--keep_fields", default="problem,domain",
                    help="comma-separated fields to keep besides thoughts (default: problem,domain)")
    ap.add_argument("--no_split_by_domain", action="store_true",
                    help="if set, do not write domain-split files; only write mixed_out")
    args = ap.parse_args()

    keep_fields = [x.strip() for x in args.keep_fields.split(",") if x.strip()]

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    mixed_rows: List[Dict[str, Any]] = []
    missing_counter = defaultdict(int)
    total_counter = defaultdict(int)

    for ex in read_jsonl(args.input):
        domain = normalize_domain(ex.get("domain"))
        total_counter[domain] += 1

        long_form = ex.get("solution", "")
        short_form = extract_short_form_with_tags(long_form)
        if short_form == "":
            missing_counter[domain] += 1

        out_ex: Dict[str, Any] = {}

        # keep selected fields
        for k in keep_fields:
            if k in ex:
                out_ex[k] = ex[k]

        # required fields for GLoRE
        out_ex["short_form_thought"] = short_form
        out_ex["long_form_thought"] = long_form

        # add to mixed
        mixed_rows.append(out_ex)

        # add to domain buckets
        buckets[domain].append(out_ex)

    # 1) write mixed file (ALL samples)
    write_jsonl(args.mixed_out, mixed_rows)

    # 2) optionally write per-domain files
    if not args.no_split_by_domain:
        for domain, rows in buckets.items():
            out_path = os.path.join(args.out_dir, f"{domain}_long_short_form_thought.jsonl")
            write_jsonl(out_path, rows)

    # stats
    print("Done.")
    print(f"[MIXED] saved={len(mixed_rows)} -> {args.mixed_out}")
    for d in sorted(total_counter.keys()):
        print(f"  domain={d:<10} total={total_counter[d]:<6} missing_solution_tags={missing_counter[d]}")

    if args.no_split_by_domain:
        print("[NOTE] domain-split files not written (--no_split_by_domain set).")
    else:
        print(f"[SPLIT] written to dir: {args.out_dir}")


if __name__ == "__main__":
    main()
