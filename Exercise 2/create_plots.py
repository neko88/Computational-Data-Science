"""
python3 create_plots.py pagecounts-20190509-120000.txt pagecounts-20190509-130000.txt
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt


#files: space-separated vals: language, page num, num views, bytes
filename1 = sys.argv[1]
filename2 = sys.argv[2]

# Read the data out of the file
data1 = pd.read_csv(filename1, sep=' ', header=None, index_col=1,
                    names=['lang','page','views','bytes'])
data2 = pd.read_csv(filename2, sep=' ', header=None, index_col=1,
                    names=['lang', 'page', 'views', 'bytes'])

# Sort the data by views
data1_sorted = data1.sort_values(by=['views'], ascending=False)
data2_sorted = data2.sort_values(by=['views'], ascending=False)

# PLOT 1: DISTRIBUTION OF VIEWS
# sort the data by num. views (dcrsing) - sort_values
# plt.plot will plot against 0 to n-1 range
# plot with pandas will use its index as the x-coord
    # to do so otherwise: pass NumPy array (data['views'].values
    # create an explicit range to use as x-coords with "np.arrange"

# produce single plot w. two subplots left n right
plt.figure(figsize=(10,5))      # change fig size

plt.subplot(1,2,1)      # subplots in 1 row, 2 columns, on first
plt.title(" Popularity Distribution")
plt.xlabel("Rank")
plt.ylabel("Views")
plt.plot(data1_sorted['views'].values)       # build plot 1

# PLOT 2: HOURLY VIEWS
# Scatterplot of views of data1 (x-coord) vs. views of data2 (y-coord)

# Combine the two series into the same DataFrame
# Obtain the elements by indices to the new DF so that view counts are beside each other as per page
data_1_2_sorted = pd.DataFrame()
data_1_2_sorted.insert(0, 'views_1', data1_sorted['views'].copy())
data_1_2_sorted.insert(1, 'views_2', data2_sorted['views'].copy())

# Plot a scatterplot of the data
plt.subplot(1,2,2)      # subplots in 1 row, 2 columns, on second
plt.scatter(data_1_2_sorted['views_1'].values,
            data_1_2_sorted['views_2'].values)
plt.xscale('log')
plt.yscale('log')
plt.title("Hourly Correlation")
plt.xlabel("Hour 1 Views")
plt.ylabel("Hour 2 Views")
plt.savefig('wikipedia.png')

