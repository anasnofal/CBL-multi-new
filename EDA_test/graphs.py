import os
import json
import pandas as pd
import matplotlib.pyplot as plt

with open("eda.json", "r") as f:
    eda = json.load(f)

os.makedirs("figures", exist_ok=True)

missing_counts = eda["missing_data"]["num_missing_values_each_column"]
missing_percent = eda["missing_data"]["percent_per_column_missing"]

columns = list(missing_counts.keys())
counts_values = list(missing_counts.values())
percent_values = list(missing_percent.values())

#bar chart of missing data count per column
plt.figure(figsize=(12, 6))
plt.bar(columns, counts_values)
plt.xticks(rotation=45, ha="right")
plt.title("Missing Data (Counts per Column)")
plt.tight_layout()
plt.savefig("figures/missing_data_counts.png")
plt.close()

# bar chart of missing data in percentage
plt.figure(figsize=(12, 6))
plt.bar(columns, percent_values)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Percentage (%)")
plt.title("Missing Data (% per Column)")
plt.tight_layout()
plt.savefig("figures/missing_data_percentage.png")
plt.close()

# bar chart of number of crimes of each type
crime_types = eda["Crimes"]["crimes_per_type"]

crime_names = list(crime_types.keys())
crime_values_count = list(crime_types.values())

plt.figure(figsize=(12, 6))
plt.bar(crime_names, crime_values_count)
plt.title("Total Crimes per Crime Type")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Number of Crimes")
plt.tight_layout()
plt.savefig("figures/crimes_per_type_total.png")
plt.close()

#bar chart of missing data per crime type
missing_by_crime_dict = eda["missing_data"]["missing_data_per_crime_type"]

crime_types = list(missing_by_crime_dict.keys())
values_missing = list(missing_by_crime_dict.values())

plt.figure(figsize=(12, 6))
plt.bar(crime_types, values_missing)
plt.xticks(rotation=45, ha="right")
plt.title("Missing Data (≥2 missing values) per Crime Type")
plt.tight_layout()
plt.savefig("figures/missing_data_per_crime.png")
plt.close()

# bar chart of percentage of missing data per crime type
missing_by_crime_dict_perc = eda["missing_data"]["missing_data_percentage_crime_type"]

crime_types = list(missing_by_crime_dict_perc.keys())
values_missing = list(missing_by_crime_dict_perc.values())

plt.figure(figsize=(12, 6))
plt.bar(crime_types, values_missing)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Percentage (%)")
plt.title("Missing Data (≥2 missing values) per Crime Type in percentage")
plt.tight_layout()
plt.savefig("figures/missing_data_per_crime_percentage.png")
plt.close()

# bar chart of percentage of missing data per attribute in anti social behavious
missing_by_column_dict_perc_asb = eda["missing_data"]["missing_column_in_social_behaviour_percent"]

attributes = list(missing_by_column_dict_perc_asb.keys())
values_missing_asb = list(missing_by_column_dict_perc_asb.values())

plt.figure(figsize=(12, 6))
plt.bar(attributes, values_missing_asb)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Percentage (%)")
plt.title("percentage of Missing Data per attribute for anti-social behaviour")
plt.tight_layout()
plt.savefig("figures/missing_data_per_crime_percentage_asb.png")
plt.close()

# line chart of crimes per month
crimes_month = eda["Crimes"]["crimes_per_month"]

months = list(crimes_month.keys())
crime_values = list(crimes_month.values())

plt.figure(figsize=(14, 6))
plt.plot(months, crime_values, marker="o")
plt.xticks(rotation=45)
plt.title("Crimes per Month")
plt.xlabel("Month")
plt.ylabel("Number of Crimes")
plt.tight_layout()
plt.savefig("figures/crimes_per_month.png")
plt.close()

#line chart of missing values per month
missing_data_month = eda["missing_data"]["missing_data_each_month"]

months = list(missing_data_month.keys())
missing_values = list(missing_data_month.values())

plt.figure(figsize=(14, 6))
plt.plot(months, missing_values, marker="o")
plt.xticks(rotation=45)
plt.title("Missing data per Month")
plt.xlabel("Month")
plt.ylabel("Missing data")
plt.tight_layout()
plt.savefig("figures/missing_data_month.png")
plt.close()

#line chart of percentage of missing values per month
missing_data_month_percentage = eda["missing_data"]["missing_data_percentage_each_month"]

months_percentage = list(missing_data_month_percentage.keys())
missing_values_percentage = list(missing_data_month_percentage.values())

plt.figure(figsize=(14, 6))
plt.plot(months_percentage, missing_values_percentage, marker="o")
plt.xticks(rotation=45)
plt.title("Missing data per Month (%)")
plt.xlabel("Month")
plt.ylabel("Missing data (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("figures/missing_data_month_percentage.png")
plt.close()

#plot the two line charts below each other
fig, ax = plt.subplots(2, 1, figsize=(16, 6))

ax[0].plot(months, crime_values)
ax[0].set_title("Crimes per Month")
ax[0].tick_params(axis="x", rotation=45)

ax[1].plot(months, missing_values)
ax[1].set_title("Missing Rows (≥2 missing values) per Month")
ax[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("figures/crimes_vs_missing.png")
plt.close()

#combining the two line charts in one chart
fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(months, crime_values, label="Total Crimes")
ax.plot(months, missing_values, label="Missing Rows (≥2 missing values)")

ax.set_title("Crimes vs Missing Data Over Time")
ax.set_xlabel("Month")
ax.set_ylabel("Count")

ax.tick_params(axis="x", rotation=45)
ax.legend()

plt.tight_layout()
plt.savefig("figures/crimes_vs_missing_combined.png")
plt.close()


# bar chart of top 10 LSOA's with most crimes committed
top_lsoa = eda["Crimes"]["top_10_lsoas_with_most_crimes"]

lsoa_names = list(top_lsoa.keys())
lsoa_values = list(top_lsoa.values())

plt.figure(figsize=(12, 6))
plt.bar(lsoa_names, lsoa_values)
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 LSOAs with Most Crimes")
plt.tight_layout()
plt.savefig("figures/top_10_lsoa.png")
plt.close()



def plot_station(station_name):
    with open("eda.json", "r") as f:
        eda = json.load(f)

    data = eda["Crimes"]["crimes_per_station"][station_name]

    months = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(12, 5))
    plt.plot(months, values, marker="o")
    plt.xticks(rotation=45)
    plt.title(f"Crimes per Month {station_name}")
    plt.tight_layout()
    plt.savefig(f"figures/crimes_per_station_{station_name}.png")
    plt.close()

plot_station("Metropolitan Police Service")