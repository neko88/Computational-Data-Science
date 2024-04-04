import numpy as np
import pandas as pd

totals = pd.read_csv('totals.csv').set_index(keys=['name'])
counts = pd.read_csv('counts.csv').set_index(keys=['name'])

pd_totals_df = pd.DataFrame(totals)
pd_counts_df = pd.DataFrame(counts)

# Find the city with the lowest total percipitation over the year
# Find sum of each row (axis 1), use 'argmin'
    # np.argmin returns the index with the min value

pd_city_totals = pd_totals_df.sum(axis=1) # sum of the 12 columns of each city
pd_city_totals_min = np.argmin(pd_city_totals)

print("City with lowest total precipitation:")
print(pd_city_totals.index[pd_city_totals_min])

# Find the average percipitation for each month
# Total percip of the month (axis 0) / total counts in the month
# Print the array

pd_month_totals = pd_totals_df.sum(axis=0)
pd_month_counts = pd_counts_df.sum(axis=0)
pd_month_avgs = pd_month_totals/pd_month_counts

print("Average precipitation in each month:")
print(pd_month_avgs)

# Find the average percipitation for each city
# sum of total percipitation of a city / sum of counts of a city
pd_city_counts = pd_counts_df.sum(axis=1)
pd_city_avgs = pd_city_totals/pd_city_counts

print("Average precipitation in each city:")
print(pd_city_avgs)