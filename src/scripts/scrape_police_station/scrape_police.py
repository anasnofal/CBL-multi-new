import requests
import pandas as pd
from pathlib import Path
import tqdm

overpass_url = "https://overpass-api.de/api/interpreter"

query = """
[out:json][timeout:300];
area(3600062149)->.uk;
(
  node["amenity"="police"](area.uk);
  way["amenity"="police"](area.uk);
  relation["amenity"="police"](area.uk);
);
out center tags;
"""

headers = {"User-Agent": "CBL-police-stations-project/1.0 (student research)"}

response = requests.post(
    overpass_url, data={"data": query}, headers=headers, timeout=300
)

if response.status_code != 200:
    print("Status code:", response.status_code)
    print(response.text[:2000])
    response.raise_for_status()

data = response.json()

stations_list = []

for element in data.get("elements", []):
    tags = element.get("tags", {})

    lat = element.get("lat")
    lon = element.get("lon")

    if lat is None or lon is None:
        center = element.get("center", {})
        lat = center.get("lat")
        lon = center.get("lon")

    if lat is not None and lon is not None:
        stations_list.append(
            {
                "Station": tags.get("name", "Unknown police station"),
                "Latitude": lat,
                "Longitude": lon,
                "Operator": tags.get("operator"),
                "Police": tags.get("police"),
                "Postcode": tags.get("addr:postcode"),
                "OSM_ID": element.get("id"),
                "OSM_Type": element.get("type"),
            }
        )

police_stations_df = pd.DataFrame(stations_list)

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "uk_police_stations_osm.csv"
police_stations_df.to_csv(output_path, index=False)

print("Saved to:", output_path)
print("Number of police station records:", len(police_stations_df))
print(police_stations_df.head())
