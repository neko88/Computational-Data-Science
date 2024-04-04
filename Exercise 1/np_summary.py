import numpy as np
import pandas as pd

# Load the monthly data percipitation
data = np.load('monthdata.npz')
totals = data['totals']
counts = data['counts']

# Find the city with the lowest total percipitation over the year
# Find sum of each row (axis 1), use 'argmin'
    # np.argmin returns the index with the min value

np_city_totals = np.sum(totals, axis=1)
np_city_totals_min = np.argmin(np_city_totals)

print("Row with lowest total precipitation:")
print(np_city_totals_min)

# Find the average percipitation for each month
np_month_counts = np.sum(counts, axis=0)
np_month_totals = np.sum(totals, axis=0)
np_month_avgs = np_month_totals/np_month_counts

print("Average precipitation in each month:")
print(np_month_avgs)

# Find the average percipitation for each city
# sum of total percipitation of a city / sum of counts of a city
np_city_counts = np.sum(counts,axis=1)
np_city_avgs = np_city_totals/np_city_counts

print("Average precipitation in each city:")
print(np_city_avgs)

# Calculate the total percipitation for each quarter in each city (3 month groups)
# 4 groups of 3 months
# Reshape to 4n by 3 array
# Reshape back to n by 4

q1,q2,q3, q4 = np.hsplit(totals,4)

q1_totals = np.sum(q1, axis=1)
q2_totals = np.sum(q2, axis=1)
q3_totals = np.sum(q3, axis=1)
q4_totals = np.sum(q4, axis=1)

qs = np.concatenate([[q1_totals], [q2_totals], [q3_totals], [q4_totals]])

print("Quarterly precipitation totals:")
print(np.transpose(qs))