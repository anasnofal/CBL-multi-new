"""
Reduces lsoa_areas.geojson file size by rounding coordinates to 5 decimal places.
5 d.p. = ~1 metre precision — more than enough for LSOA boundary display.

Usage:
    python scripts/simplify_geojson.py
"""

import json
from pathlib import Path

INPUT  = Path("src/dashboard/app/data/lsoa_areas.geojson")
OUTPUT = Path("src/dashboard/app/data/lsoa_areas.geojson")
PRECISION = 5


def round_coords(geometry):
    def round_ring(ring):
        return [[round(x, PRECISION), round(y, PRECISION)] for x, y in ring]

    t = geometry["type"]
    if t == "Polygon":
        geometry["coordinates"] = [round_ring(r) for r in geometry["coordinates"]]
    elif t == "MultiPolygon":
        geometry["coordinates"] = [
            [round_ring(r) for r in poly]
            for poly in geometry["coordinates"]
        ]
    return geometry


def main():
    print(f"Reading {INPUT} ({INPUT.stat().st_size / 1e6:.1f} MB)...")
    data = json.loads(INPUT.read_text(encoding="utf-8"))

    for feature in data["features"]:
        feature["geometry"] = round_coords(feature["geometry"])

    out = json.dumps(data, separators=(",", ":"))  # no whitespace = smaller file
    OUTPUT.write_text(out, encoding="utf-8")
    print(f"Written {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
