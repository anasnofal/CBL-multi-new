"""
Identify crime hotspots using Local Moran's I for a specific police force and month.
It can:
- Compute hotspots using Local Moran's I for a selected police force (kml name) and month
- Compute (stable) hotspots on the entire street dataframe.
- Create a dictionary with stable hotspots and non hotspots for each police force over the entire dataframe 
    (important note: there is one force missing in the json file: Greater Manchester Police (1,702 LSOA's missing)
    because it contains no information about LSOA's in the all_street.csv  )

To run it needs the following datasets:
- the all_street.csv from combine_police_data.py
- force boundaries: https://data.police.uk/data/boundaries/
named:Police workforce, England and Wales, 31 March 2025: workforce open data tables
- LSOA shapefiles: https://geoportal.statistics.gov.uk/datasets/04c65a08ecff4858bffc16e9ca9356f4_0/explore?location=52.846052%2C-2.465415%2C6
Choose to download the shapefiles

Usage example:
    uv run python -m src.scripts.identify_hotspots -i "data" -o "results" -m "2025-01" -f "wiltshire" -c "all_street.csv" -s "42"
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from libpysal.weights import Queen
import glob
import os
from esda.moran import Moran_Local
from rapidfuzz import process
import argparse
from pathlib import Path
import warnings
import json
import random
from statsmodels.stats.multitest import multipletests
from collections import defaultdict

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent.parent.parent

def set_seed(seed: int = 42):
    """Sets the global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")

def load_lsoa_geometries(input_dir: Path) -> gpd.GeoDataFrame:
    print("Loading LSOA geometries")
    base_path = input_dir / "LSOA_location_info" 

    if not base_path.exists():
        base_path = input_dir

    shapefiles = glob.glob(os.path.join(base_path, "**/*.shp"), recursive=True)
    if not shapefiles:
        raise FileNotFoundError(f"No .shp files found in {base_path}")

    lsoa_gdfs = [gpd.read_file(shp) for shp in shapefiles]
    lsoa_locations = gpd.GeoDataFrame(
        pd.concat(lsoa_gdfs, ignore_index=True),
        crs=lsoa_gdfs[0].crs
    )
    lsoa_locations = lsoa_locations.rename(columns={"LSOA21CD": "LSOA code"})

    return lsoa_locations

def load_force_boundaries(input_dir: Path, target_crs) -> gpd.GeoDataFrame:
    print("Loading police force boundaries")
    kml_folder = os.path.join(input_dir, "force kmls")
    kml_files = glob.glob(os.path.join(kml_folder, "**/*.kml"), recursive=True)

    force_gdfs = []
    for file in kml_files:
        gdf = gpd.read_file(file, driver="KML")
        gdf["force"] = os.path.basename(file).replace(".kml", "")
        force_gdfs.append(gdf)

    forces = gpd.GeoDataFrame(pd.concat(force_gdfs, ignore_index=True))
    forces = forces.to_crs(target_crs)
    forces["geometry"] = forces["geometry"].buffer(0)
    return forces

def load_or_prepare_crime_data(crime_csv_path: Path, forces: gpd.GeoDataFrame, output_file: Path) -> pd.DataFrame:
    if output_file.exists():
        print("Loading prepared crime data from file")
        return pd.read_csv(output_file)

    print("Prepared crime data file not found. Running preparation step...")
    df = prepare_crime_data(crime_csv_path, forces)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Prepared crime data saved to: {output_file}")
    return df

def prepare_crime_data(crime_csv_path: Path, forces: gpd.GeoDataFrame) -> pd.DataFrame:
    print("Preparing crime data")
    df_street_crimes = pd.read_csv(crime_csv_path)

    if "Context" in df_street_crimes.columns:
        df_street_crimes = df_street_crimes.drop(columns=["Context"])
        
    df_street_crimes = df_street_crimes.dropna(subset=["LSOA name", "LSOA code"])
    df_street_crimes = df_street_crimes[df_street_crimes["Reported by"] != "British Transport Police"]

    crime_monthly = (
        df_street_crimes
        .groupby(["Reported by", "LSOA code", "Month"])
        .size()
        .reset_index(name="crimes")
    )

    forces_reported_by = crime_monthly["Reported by"].unique()
    kml_forces = list(forces["force"].unique())

    # remove northern ireland and greater-manchester from the kml since not in the df_Street_crimes database
    if "northern-ireland" in kml_forces:
        kml_forces.remove("northern-ireland")
    if "greater-manchester" in kml_forces:
        kml_forces.remove("greater-manchester")

    # Mapping the reported by force names to kml names, since they are not the same
    map_force_names = {}
    for cf in forces_reported_by:
        match = process.extractOne(cf, kml_forces)
        map_force_names[cf] = match[0]

    # Manually checked in notebook if mapping was correct, below is a correction
    map_force_names["North Yorkshire Police"] = "north-yorkshire"
    map_force_names["Nottinghamshire Police"] = "nottinghamshire"
    map_force_names["South Yorkshire Police"] = "south-yorkshire"

    crime_monthly["force_kml"] = crime_monthly["Reported by"].map(map_force_names)
    return crime_monthly

def get_monthly_hotspots_force(
    crime_monthly: pd.DataFrame,
    lsoa_gdf: gpd.GeoDataFrame,
    forces: gpd.GeoDataFrame,
    month: str,
    police_force: str,
    lsoa_col="LSOA code",
    month_col="Month",
    value_col="crimes",
    significance=0.05,
    seed=42           
):
    """
    Computes hotspots using Local Moran's I for a selected police force (kml name) and month.
    Returns: a geopanda geodataframe with the LSOA geometries, reported by, crime count, month, name of kml force file, 
    Local Moran's I values, p-values, moran quadrants, classification type
    
    """

    print(f"Calculating hotspots for {police_force}")

    # filter data to keep chosen month and police force
    df_filtered = crime_monthly[
        (crime_monthly[month_col] == month) &
        (crime_monthly["force_kml"] == police_force)
    ].copy()

    # get force geometry and make one shape/boundary
    force_geom = forces[forces["force"] == police_force].copy()
    force_geom = force_geom[force_geom.geometry.notnull()].copy()

    if len(force_geom) == 0:
        raise ValueError(f"No valid geometry for {police_force}")

    force_union = force_geom.geometry.unary_union

    # get only lsoa's that are in in the force boundary 
    lsoa_force = lsoa_gdf[
        lsoa_gdf.geometry.centroid.within(force_union.buffer(0))
    ].copy()

    gdf = lsoa_force.merge(df_filtered, on=lsoa_col, how="left")
    gdf[value_col] = gdf[value_col].fillna(0)

    # spatial weights
    w = Queen.from_dataframe(gdf)
    w.transform = "r"

    # Local Morans I
    lisa = Moran_Local(gdf[value_col], w, permutations=999, seed=seed)

    gdf["Is"] = lisa.Is
    gdf["p_sim"] = lisa.p_sim
    gdf["quadrant"] = lisa.q

    # Not using FDR correction, since it is too strict on police forces with not enough LSOA's
    sig = lisa.p_sim < significance

    cluster_map = {
        0: "Non-significant",
        1: "High-High (Hotspot)",
        2: "Low-High",
        3: "Low-Low",
        4: "High-Low"
    }

    clusters = lisa.q.copy()
    clusters[~sig] = 0

    # Create column with cluster label
    gdf["cluster_classification"] = pd.Series(
        clusters,
        index=gdf.index
    ).map(cluster_map)

    return gdf

def plot_moran_clusters_save(hotspots, output_dir: Path, force: str, month: str):
    """Plot function to save image."""
    fig, ax = plt.subplots(figsize=(8, 8))

    hotspots.plot(
        column="cluster_classification",
        categorical=True,
        cmap="tab10",
        edgecolor="white",
        linewidth=0.1,
        legend=True,
        ax=ax
    )

    ax.set_title(f"Moran Cluster Map - {force} ({month})")
    ax.set_axis_off()

    plt.tight_layout()
    
    out_path = output_dir / f"hotspots_{force}_{month}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Image saved: {out_path}")

def save_hotspot_lsoas(hotspots: gpd.GeoDataFrame, output_dir: Path, force: str, month: str):
    """Saves a list of the LSOA codes that are hotspots."""
    hotspot_df = hotspots[hotspots["cluster_classification"] == "High-High (Hotspot)"]
    
    out_path = output_dir / f"hotspot_lsoas_{force}_{month}.csv"
    hotspot_df[["LSOA code", "crimes"]].to_csv(out_path, index=False)
    print(f"  Hotspot list saved: {out_path}")

def get_stable_hotspots(
    crime: pd.DataFrame,
    lsoa_gdf: gpd.GeoDataFrame,
    forces: gpd.GeoDataFrame,
    police_force: str,
    lsoa_col="LSOA code",
    value_col="crimes",
    significance=0.05,
    seed = 42
):
    """
    Calculates (stable) hotspots on the entire street dataframe.
    """
    df_filtered = crime[crime["force_kml"] == police_force].copy()

    force_geom = forces[forces["force"] == police_force].copy()
    force_geom = force_geom[force_geom.geometry.notnull()].copy()

    if len(force_geom) == 0:
        raise ValueError(f"No valid geometry for {police_force}")

    force_union = force_geom.geometry.unary_union

    lsoa_force = lsoa_gdf[
        lsoa_gdf.geometry.centroid.within(force_union.buffer(0))
    ].copy()

    gdf = lsoa_force.merge(df_filtered, on=lsoa_col, how="left")
    gdf[value_col] = gdf[value_col].fillna(0)

    w = Queen.from_dataframe(gdf)
    w.transform = "r"

    lisa = Moran_Local(gdf[value_col], w, permutations=999, seed = seed)

    gdf["Is"] = lisa.Is
    gdf["p_sim"] = lisa.p_sim
    gdf["quadrant"] = lisa.q

    # FDR correction, Benjamini–Hochberg
    _, p_fdr, _, _ = multipletests(lisa.p_sim, alpha=significance, method="fdr_bh")

    gdf["p_fdr"] = p_fdr
    sig = p_fdr < significance

    cluster_map = {
        0: "Non-significant",
        1: "High-High (Hotspot)",
        2: "Low-High",
        3: "Low-Low",
        4: "High-Low"
    }

    clusters = lisa.q.copy()
    clusters[~sig] = 0

    gdf["cluster_classification"] = pd.Series(
        clusters,
        index=gdf.index
    ).map(cluster_map)

    return gdf


def create_overall_hotspot_dict(
    crime: pd.DataFrame,
    lsoa_gdf: gpd.GeoDataFrame,
    forces: gpd.GeoDataFrame,
    force_to_lsoas: dict,
    base_output_dir: Path,
    output_json_path: Path,
    seed=42
) -> dict:
    """
    Loops through all forces, calculates 3-year hotspots, 
    and returns a dictionary mapping force names to nonhotspots and hotspot LSOA code lists.
    """
    hotspot_dict = {}
    unique_forces = crime["force_kml"].dropna().unique()
    
    print(f"\nCalculating (stable hotspots for {len(unique_forces)} forces")
    
    for i, force in enumerate(unique_forces, 1):
        print(f"  [{i}/{len(unique_forces)}] Processing {force}...")
        
        try:
            output_file = base_output_dir / f"stable_hotspots_{force}.gpkg"

            hotspots = load_or_compute_stable_hotspots(
                crime=crime,
                lsoa_gdf=lsoa_gdf,
                forces=forces,
                police_force=force,
                output_file=output_file,
                seed=seed
            )

            hotspot_list = hotspots.loc[
                hotspots["cluster_classification"] == "High-High (Hotspot)",
                "LSOA code"
            ].dropna().unique().tolist()

            all_lsoas = force_to_lsoas.get(force, set())

            hotspot_set = set(hotspot_list)

            non_hotspot_list = list(all_lsoas - hotspot_set)

            hotspot_dict[force] = {
                "hotspots": hotspot_list,
                "non_hotspots": non_hotspot_list
            }

            
        except Exception as e:
            print(f"  [!] Skipping {force} due to error: {e}")

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    check_internal_duplicates(hotspot_dict)
    check_cross_force_duplicates(hotspot_dict)

    with open(output_json_path, "w") as f:
        json.dump(hotspot_dict, f, indent=2)

    print(f"\nHotspot dictionary saved to: {output_json_path}")

    print("Performing duplicates check..")
    internal_duplicates = check_internal_duplicates(hotspot_dict)
    cross_force_duplicates = check_cross_force_duplicates(hotspot_dict)

    if not internal_duplicates:
        print("No duplicates found in a police foce")
    else:
        print(f"Duplicates found inside forces: {internal_duplicates}")

    if not cross_force_duplicates:
        print("No duplicates found across police forces")
    else:
        print(f"Duplicates found across police forces: {cross_force_duplicates}")

    return hotspot_dict

def check_internal_duplicates(hotspot_dict):
    issues = {}

    for force, data in hotspot_dict.items():

        hotspot_list = data if isinstance(data, list) else data["hotspots"]

        duplicates = [
            lsoa for lsoa in hotspot_list
            if hotspot_list.count(lsoa) > 1
        ]

        if duplicates:
            issues[force] = list(set(duplicates))

    return issues

from collections import defaultdict

def check_cross_force_duplicates(hotspot_dict):
    lsoa_to_forces = defaultdict(set)

    for force, data in hotspot_dict.items():

        hotspot_list = data if isinstance(data, list) else data["hotspots"]

        for lsoa in hotspot_list:
            lsoa_to_forces[lsoa].add(force)

    duplicates = {
        lsoa: list(forces)
        for lsoa, forces in lsoa_to_forces.items()
        if len(forces) > 1
    }

    return duplicates

def load_or_compute_stable_hotspots(
    crime: pd.DataFrame,
    lsoa_gdf: gpd.GeoDataFrame,
    forces: gpd.GeoDataFrame,
    police_force: str,
    output_file: Path,
    lsoa_col="LSOA code",
    value_col="crimes",
    significance=0.05,
    seed=42
) -> gpd.GeoDataFrame:

    if output_file.exists():
        print(f"Loading stable hotspots from file: {output_file}")
        return gpd.read_file(output_file)

    print("Stable hotspots file not found. Computing...")

    gdf = get_stable_hotspots(
        crime=crime,
        lsoa_gdf=lsoa_gdf,
        forces=forces,
        police_force=police_force,
        lsoa_col=lsoa_col,
        value_col=value_col,
        significance=significance,
        seed=seed
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_file(output_file, driver="GPKG")

    print(f"Saved stable hotspots to: {output_file}")

    return gdf

def build_uk_hotspots(
    crime: pd.DataFrame,
    lsoa_gdf: gpd.GeoDataFrame,
    forces: gpd.GeoDataFrame,
    base_output_dir: Path,
    seed=42
) -> gpd.GeoDataFrame:

    all_hotspots = []

    for force in crime["force_kml"].dropna().unique():
        print(f"Processing {force}...")

        output_file = base_output_dir / f"stable_hotspots_{force}.gpkg"

        gdf = load_or_compute_stable_hotspots(
            crime=crime,
            lsoa_gdf=lsoa_gdf,
            forces=forces,
            police_force=force,
            output_file=output_file,
            seed=seed
        )

        gdf["force_kml"] = force
        all_hotspots.append(gdf)

    return gpd.GeoDataFrame(pd.concat(all_hotspots, ignore_index=True))

def plot_uk_hotspots(
    hotspots: gpd.GeoDataFrame,
    output_path: Path
):
    fig, ax = plt.subplots(figsize=(10, 14))

    hotspots.plot(
        column="cluster_classification",
        categorical=True,
        cmap="tab10",
        edgecolor="white",
        linewidth=0.05,
        legend=True,
        ax=ax
    )

    ax.set_title("UK Moran Cluster Map (All Forces)")
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def build_lsoa_force_mapping(
    lsoa_gdf: gpd.GeoDataFrame,
    forces_gdf: gpd.GeoDataFrame,
    lsoa_code_col: str = "LSOA code",
    force_col: str = "force",
) -> pd.DataFrame:
    """
    Builds a unique mapping from LSOA → police force using centroid containment.
    Returns a DataFrame with LSOA code and force
    """

    if lsoa_gdf.crs != forces_gdf.crs:
        forces_gdf = forces_gdf.to_crs(lsoa_gdf.crs)

    lsoa_centroids = lsoa_gdf[[lsoa_code_col, "geometry"]].copy()
    lsoa_centroids["geometry"] = lsoa_centroids.geometry.centroid

    mapping = gpd.sjoin(
        lsoa_centroids,
        forces_gdf[[force_col, "geometry"]],
        how="left",
        predicate="within"
    )
    mapping = mapping[[lsoa_code_col, force_col]].drop_duplicates()

    duplicates = mapping[mapping.duplicated(lsoa_code_col, keep=False)]
    if len(duplicates) > 0:
        raise ValueError(
            f"Found {len(duplicates)} LSOAs assigned to multiple forces."
        )

    return mapping

def main():
    parser = argparse.ArgumentParser(description="Identify crime hotspots.")
    parser.add_argument("-i", "--inputdir", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument("-o", "--outdir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("-c", "--crime_file", type=str, default="combined_data.csv")
    parser.add_argument("-m", "--month", type=str, required=True)
    parser.add_argument("-f", "--force", type=str, required=True)
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for Moran's I permutations")
    
    args = parser.parse_args()
    set_seed(args.seed)

    OUTPUT_DIR = args.outdir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    HOTSPOT_DIR = OUTPUT_DIR / "hotspot_results"
    HOTSPOT_DIR.mkdir(parents=True, exist_ok=True)

    STABLE_DIR = HOTSPOT_DIR / "stable_hotspots"
    STABLE_DIR.mkdir(parents=True, exist_ok=True)

    crime_csv_path = args.inputdir / args.crime_file

    lsoa_locations = load_lsoa_geometries(args.inputdir)
    forces = load_force_boundaries(args.inputdir, lsoa_locations.crs)
    lsoa_locations = lsoa_locations.to_crs(forces.crs)
    
    prepared_file = args.inputdir / "prepared_crime_hotspot.csv"

    crime_monthly = load_or_prepare_crime_data(
        crime_csv_path,
        forces,
        prepared_file
    )

    print("Building LSOA → force mapping...")

    lsoa_force_lookup = build_lsoa_force_mapping(
        lsoa_locations,
        forces
    )

    lsoa_to_force = dict(
        zip(
            lsoa_force_lookup["LSOA code"],
            lsoa_force_lookup["force"]
        )
    )

    force_to_lsoas = defaultdict(set)

    for lsoa, force in lsoa_to_force.items():
        force_to_lsoas[force].add(lsoa)
        
    hotspots_gdf = get_monthly_hotspots_force(
        crime_monthly=crime_monthly,
        lsoa_gdf=lsoa_locations,
        forces=forces,
        month=args.month,
        police_force=args.force,
        seed=args.seed
    )

    print("Running stable hotspot analysis...")
    stable_file = STABLE_DIR / f"stable_hotspots_{args.force}.gpkg"

    print("Building UK hotspot map...")

    uk_hotspots = build_uk_hotspots(
    crime=crime_monthly,
    lsoa_gdf=lsoa_locations,
    forces=forces,
    base_output_dir=STABLE_DIR,
    seed=args.seed
    )

    plot_uk_hotspots(
        uk_hotspots,
        STABLE_DIR / "uk_hotspots.png"
    )

    print("Building hotspot dictionary for all forces...")

    hotspot_json = STABLE_DIR / "hotspot_dictionary_test2.json"

    hotspot_dict = create_overall_hotspot_dict(
        crime=crime_monthly,
        lsoa_gdf=lsoa_locations,
        forces=forces,
        force_to_lsoas=force_to_lsoas,
        base_output_dir=STABLE_DIR,
        output_json_path=hotspot_json,
        seed=args.seed
    )

    plot_moran_clusters_save(hotspots_gdf, HOTSPOT_DIR, args.force, args.month)
    save_hotspot_lsoas(hotspots_gdf, HOTSPOT_DIR, args.force, args.month)


    # Check if dictionary is complete or missing LSOA's
    all_dict_lsoas = set()

    for force, data in hotspot_dict.items():
        all_dict_lsoas.update(data.get("hotspots", []))
        all_dict_lsoas.update(data.get("non_hotspots", []))

    expected_lsoas = set(lsoa_locations["LSOA code"].unique())

    # filter crime data for Greater Manchester Police
    # get Greater Manchester geometry from force boundaries
    gmp_geom = forces[forces["force"] == "greater-manchester"].geometry.unary_union

    # LSOAs that fall inside GMP boundary (geometry-based, independent of crime data)
    gmp_lsoas = set(
        lsoa_locations[
            lsoa_locations.geometry.intersects(gmp_geom)
        ]["LSOA code"].unique()
    )

    
    missing_in_output = expected_lsoas - all_dict_lsoas
    extra_in_output = all_dict_lsoas - expected_lsoas
    missing_in_gmp = set(missing_in_output) & gmp_lsoas
    print("Expected LSOAs:", len(expected_lsoas))
    print("LSOAs in hotspot dictionary:", len(all_dict_lsoas))
    print("Missing from dictionary:", len(missing_in_output))
    print("Extra in dictionary:", len(extra_in_output))
    print("Coverage %:", len(all_dict_lsoas) / len(expected_lsoas) * 100)
    print("GMP LSOAs total in crime data:", len(gmp_lsoas))
    print("Missing LSOAs that belong to GMP:", len(missing_in_gmp))
    print("GMP contribution to missing (%):",
      len(missing_in_gmp) / len(missing_in_output) * 100 if missing_in_output else 0)

    print("Example missing:", list(missing_in_output)[:20])
    print("Example extra:", list(extra_in_output)[:20])

    print("Done!")

if __name__ == "__main__":
    main()
