import re
from pathlib import Path

import pandas as pd

from police_forces import FORCE_ALIASES, TARGET_FORCES

INPUT_PATH = Path("data/uk_police_stations_osm.csv")
OUTPUT_PATH = Path("data/force_representative_locations.csv")

MATCH_COLUMNS = ["Operator", "Station", "Police"]
OUTPUT_COLUMNS = [
    "Force",
    "Station",
    "Latitude",
    "Longitude",
    "Operator",
    "Police",
    "Postcode",
    "Match_Source",
    "Match_Keyword",
]


def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value).lower()
    text = text.replace("&amp;", "&")
    text = text.replace("&", " and ")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


FORCE_MATCHES = sorted(
    [(clean_text(keyword), keyword, force) for keyword, force in FORCE_ALIASES.items()],
    # Longer names are checked first so a specific phrase wins over a shorter one.
    key=lambda item: len(item[0]),
    reverse=True,
)


def match_force(row):
    fields = [(column, clean_text(row[column])) for column in MATCH_COLUMNS]

    for clean_keyword, keyword, force in FORCE_MATCHES:
        for column, text in fields:
            if clean_keyword in text:
                return pd.Series([force, column, keyword])

    return pd.Series(["Unknown", None, None])


def choose_representative_station(stations, force):
    headquarters = stations[
        stations["Station"].astype(str).str.contains("headquarters|hq", case=False)
    ]
    operator_matches = stations[stations["Match_Source"] == "Operator"]

    # Assumption: one coordinate is needed to represent each force area.
    # The force headquarters is the best estimate because it is the official central
    # location, not just one local station inside the force area.
    if not headquarters.empty:
        selected_station = headquarters.iloc[0]

    # If no headquarters is found, prefer a row where OSM's operator tag names the
    # force. That is usually more reliable than matching only the station name.
    elif not operator_matches.empty:
        selected_station = operator_matches.iloc[0]

    # Last choice: use the first matched station. This keeps every force in the
    # final file when OSM does not provide a stronger signal.
    else:
        selected_station = stations.iloc[0]

    selected_station = selected_station.copy()
    selected_station["Matched_Force"] = force

    return selected_station


def main():
    stations = pd.read_csv(INPUT_PATH)

    stations[["Matched_Force", "Match_Source", "Match_Keyword"]] = stations.apply(
        match_force, axis=1
    )

    matched_stations = stations[stations["Matched_Force"].isin(TARGET_FORCES)].copy()
    matched_stations = matched_stations.dropna(subset=["Latitude", "Longitude"])
    matched_stations = matched_stations.drop_duplicates(
        subset=["Station", "Latitude", "Longitude", "Matched_Force"]
    )

    representatives = [
        choose_representative_station(group, force)
        for force, group in matched_stations.groupby("Matched_Force")
    ]

    force_locations = pd.DataFrame(representatives)
    force_locations = (
        force_locations.set_index("Matched_Force")
        .reindex(TARGET_FORCES)
        .reset_index()
        .rename(columns={"Matched_Force": "Force"})
    )
    force_locations = force_locations[OUTPUT_COLUMNS]
    force_locations.to_csv(OUTPUT_PATH, index=False)

    missing_coordinates = force_locations[
        force_locations["Latitude"].isna() | force_locations["Longitude"].isna()
    ]

    print("Done.")
    print(f"Input rows: {len(stations)}")
    print(f"Matched station rows: {len(matched_stations)}")
    print(f"Final force locations: {len(force_locations)}")
    print(f"Forces missing coordinates: {len(missing_coordinates)}")
    print(f"Saved force locations to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
