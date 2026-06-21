import pandas as pd
import json

# load the data
df = pd.read_csv("combined_data.csv")
print("data has been read")

# general information
number_of_rows = len(df)
id_counts = df["Crime ID"].value_counts()
duplicate_id_count = int((id_counts > 1).sum())

# duplicates check
dup_rows = df[df.duplicated(keep="first")]
df_dup_missing = dup_rows[dup_rows.isna().sum(axis=1) >= 2]
num_of_dup_percentage = float(round(len(dup_rows)/number_of_rows*100, 2))
percentage_mis_val = len(df_dup_missing)/len(dup_rows)*100

# missing data
# at least two missing values, since context is missing everywhere
rows_with_missing_value = int((df.isna().sum(axis=1) >= 2).sum())
percentage_of_rows_with_missing_value = float(round(rows_with_missing_value/number_of_rows * 100, 2))

df_rows_more_2_missing_val = df[df.isna().sum(axis=1) >= 2]

missing_data_per_month = df_rows_more_2_missing_val.groupby("Month").size().sort_index()
missing_data_per_crime = df_rows_more_2_missing_val.groupby("Crime type").size().sort_values(ascending=False)

missing_counts_column = df.isna().sum()
missing_percent_column = (df.isna().mean() * 100).round(2)

crimes_per_month = df.groupby("Month").size()

crime_types_count = df.groupby("Crime type").size().sort_values(ascending=False)

# calculating percentage of missing per crime type:
missing_per_crime = df_rows_more_2_missing_val.groupby("Crime type").size()
total_per_crime = df.groupby("Crime type").size()
missing_per_crime = missing_per_crime.reindex(total_per_crime.index, fill_value=0)
missing_percentage_per_crime = ((missing_per_crime / total_per_crime) * 100).round(2)

# calculating percentage of missing data per month:
missing_per_month = df_rows_more_2_missing_val.groupby("Month").size()
missing_per_month = missing_per_month.reindex(crimes_per_month.index, fill_value=0)

# percentage of missing rows per month
missing_percentage_per_month = ((missing_per_month / crimes_per_month) * 100).round(2)

top_10_crimes_lsoa = df.groupby("LSOA name").size().sort_values(ascending=False).head(10)

lsoa_counts = df.groupby("LSOA name").size()
lsoa_one_crime = lsoa_counts[lsoa_counts == 1]
num_lsoa_one_crime = len(lsoa_one_crime)

# find out what values are missing in anti-social behaviour category
df_asb = df[df["Crime type"] == "Anti-social behaviour"]

missing_asb_columns = df_asb.isna().sum().sort_values(ascending=False)
missing_asb_columns = missing_asb_columns[missing_asb_columns > 0]

missing_asb_percent = (df_asb.isna().mean() * 100).round(2)
missing_asb_percent = missing_asb_percent[missing_asb_percent > 0].sort_values(ascending=False)


# summary of information in dictionary
eda = {
    "general_info":{
        "total_rows": number_of_rows,
        "num_of_duplicate_ids": duplicate_id_count,
        "num_of_rows_duplicate_rows": len(dup_rows),
        "percentage_of_rows_that_are_duplicate": num_of_dup_percentage

    },
    "missing_data": {
        "rows_with_missing_value": rows_with_missing_value,
        "percentage_of_rows_with_missing_values": percentage_of_rows_with_missing_value,
        "num_missing_values_each_column": missing_counts_column.to_dict(),
        "percent_per_column_missing": missing_percent_column.to_dict(),
        "missing_data_each_month": missing_data_per_month.to_dict(),
        "missing_data_percentage_each_month":missing_percentage_per_month.to_dict(),
        "missing_data_per_crime_type": missing_data_per_crime.to_dict(),
        "missing_data_percentage_crime_type": missing_percentage_per_crime.to_dict(),
        "missing_column_values_in_Anti-social_behaviour": missing_asb_columns.to_dict(),
        "missing_column_in_social_behaviour_percent": missing_asb_percent.to_dict(),
        "Num_of_identical_rows_with_missing_values": len(df_dup_missing),
        "percentage_dupl_with_missing_values": percentage_mis_val
    },
    "Crimes":{
        "crimes_per_month": crimes_per_month.to_dict(),
        "crimes_per_type": crime_types_count.to_dict(),
        "top_10_lsoas_with_most_crimes": top_10_crimes_lsoa.to_dict(),
        "num_lsoa's_with_exactly_one_crime": num_lsoa_one_crime
    }
}

# save information to seperate json file
with open("eda.json", "w") as f:
    json.dump(eda, f, indent=4)

print("JSON file saved")

def add_crimes_per_month_for_station_to_json(df, station_name, json_path="eda.json"):
    """
    Adds a dict of number of crimes per month for a given police station to the json file.
    """

    with open(json_path, "r") as f:
        eda = json.load(f)

    filtered = df[df["Reported by"] == station_name]

    data = (filtered.groupby("Month").size().sort_index())

    result_dict = {str(k): int(v) for k, v in data.items()}

    if "crimes_per_station" not in eda["Crimes"]:
        eda["Crimes"]["crimes_per_station"] = {}

    eda["Crimes"]["crimes_per_station"][station_name] = result_dict

    with open(json_path, "w") as f:
        json.dump(eda, f, indent=4)

add_crimes_per_month_for_station_to_json(df, "Metropolitan Police Service")