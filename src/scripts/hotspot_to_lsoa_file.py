"""
Convert hotspot_dict_3y.json into an LSOA list file accepted by the train pipeline.

Usage
-----
    # All forces → one file
    python scripts/hotspot_to_lsoa_file.py

    # Specific forces only
    python scripts/hotspot_to_lsoa_file.py --forces metropolitan merseyside

    # Custom input / output paths
    python scripts/hotspot_to_lsoa_file.py --input data/hotspot_dict_3y.json --output lsoas.json

    # One output file per force
    python scripts/hotspot_to_lsoa_file.py --split --output-dir data/lsoa_lists/

Then pass the result to the pipeline:
    python -m src.train --data data/monthly_counts.csv --lsoa-file lsoas.json --output-dir results/
"""

import argparse
import json
from pathlib import Path


DEFAULT_INPUT = Path("data/hotspot_dict_3y.json")
DEFAULT_OUTPUT = Path("data/lsoas.json")


def load_hotspots(path: Path) -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_lsoa_list(hotspots: dict, forces: list[str] | None) -> list[str]:
    if forces:
        missing = [f for f in forces if f not in hotspots]
        if missing:
            available = ", ".join(sorted(hotspots))
            raise SystemExit(f"Unknown force(s): {missing}\nAvailable: {available}")
        selected = {f: hotspots[f] for f in forces}
    else:
        selected = hotspots

    seen = set()
    codes = []
    for lsoa_list in selected.values():
        for code in lsoa_list:
            if code not in seen:
                seen.add(code)
                codes.append(code)
    return codes


def write_lsoa_file(codes: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"lsoas": codes}, indent=2), encoding="utf-8")
    print(f"Written {len(codes)} LSOAs → {path}")


def main():
    parser = argparse.ArgumentParser(description="Convert hotspot dict to train pipeline input.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--forces", nargs="+", metavar="FORCE",
                        help="Only include these force names (default: all forces).")
    parser.add_argument("--list-forces", action="store_true",
                        help="Print all available force names and exit.")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                      help="Output file path (default: data/lsoas.json).")
    mode.add_argument("--split", action="store_true",
                      help="Write one file per force instead of merging.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/lsoa_lists"),
                        help="Output directory when using --split.")

    args = parser.parse_args()

    hotspots = load_hotspots(args.input)

    if args.list_forces:
        for force, codes in sorted(hotspots.items()):
            print(f"  {force:<30} {len(codes)} LSOAs")
        return

    if args.split:
        forces = args.forces or list(hotspots.keys())
        for force in forces:
            if force not in hotspots:
                print(f"  [skip] unknown force: {force}")
                continue
            write_lsoa_file(hotspots[force], args.output_dir / f"{force}.json")
    else:
        codes = build_lsoa_list(hotspots, args.forces)
        write_lsoa_file(codes, args.output)


if __name__ == "__main__":
    main()
