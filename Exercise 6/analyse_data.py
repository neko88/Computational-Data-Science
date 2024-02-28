import numpy as np
import pandas as pd
import time
from implementations import all_implementations
from scipy import stats
from numpy import random
import matplotlib.pyplot as plt


def add_hist(axs, index, data, title, xlab, ylab, color=None):
    r, c = index
    axs[r][c].set_title(title, size=6)
    axs[r][c].set_xlabel(xlab, size=6)
    axs[r][c].set_ylabel(ylab, size=6)
    axs[r][c].hist(data, color=color)

def get_iqr_data(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr_data = data.loc[(data >= q1) & (data <= q3)]
    return iqr_data

def plot_histograms(data, columns, super_title):
    fig, axs = plt.subplots(2, 4)
    fig.suptitle("Distributions of Sorting Algorithms: " + super_title)
    fig.tight_layout()
    row = 0
    col = 0
    title = 0
    for sort in columns:
        colour = (np.random.random(), np.random.random(), np.random.random())
        add_hist(axs, (row, col), data[sort], title=columns[title], xlab='Sort Time', ylab='Frequency', color=colour)
        col += 1
        title += 1
        if col == 4:
            row += 1
            col = 0
    plt.plot()
    plt.show()

def main():
    sort_columns = ['qs1', 'qs2', 'qs3', 'qs4', 'qs5', 'merge1', 'partition_sort']
    data = pd.read_csv('data.csv', header=0)
    print(data.head())

    log_transform = np.log(data.astype(float))
    reciprocal_transform = 1 / (data.astype(float))
    square_transform = data.astype(float) ** 2
    cube_transform = data.astype(float) ** 3

    data_means = pd.DataFrame(columns=sort_columns)
    data_means.loc[0, 'qs1'] = data['qs1'].mean()
    data_means.loc[0, 'qs2'] = data['qs2'].mean()
    data_means.loc[0, 'qs3'] = data['qs3'].mean()
    data_means.loc[0, 'qs4'] = data['qs4'].mean()
    data_means.loc[0, 'qs5'] = data['qs5'].mean()
    data_means.loc[0, 'merge1'] = data['merge1'].mean()
    data_means.loc[0, 'partition_sort'] = data['partition_sort'].mean()

    print("\nThe resulting analysis was by data transformed using Reciprocal Transform\n")
    print("\nSorting Algorithms - Fastest to Slowest Means:\n", data_means.loc[0,:].sort_values())
    plot_histograms(reciprocal_transform, sort_columns, 'Reciprocal Transformed')

    tt_results = pd.DataFrame(columns=sort_columns)
    tt_results.loc['qs1', 'qs1'] = stats.ttest_ind(reciprocal_transform['qs1'].astype(float), reciprocal_transform['qs1'].astype(float)).pvalue
    tt_results.loc['qs1', 'qs2'] = stats.ttest_ind(reciprocal_transform['qs1'].astype(float), reciprocal_transform['qs2'].astype(float)).pvalue
    tt_results.loc['qs1', 'qs3'] = stats.ttest_ind(reciprocal_transform['qs1'].astype(float), reciprocal_transform['qs3'].astype(float)).pvalue
    tt_results.loc['qs1', 'qs4'] = stats.ttest_ind(reciprocal_transform['qs1'].astype(float), reciprocal_transform['qs4'].astype(float)).pvalue
    tt_results.loc['qs1', 'qs5'] = stats.ttest_ind(reciprocal_transform['qs1'].astype(float), reciprocal_transform['qs5'].astype(float)).pvalue
    tt_results.loc['qs1', 'merge1'] = stats.ttest_ind(reciprocal_transform['qs1'].astype(float), reciprocal_transform['merge1'].astype(float)).pvalue
    tt_results.loc['qs1', 'partition_sort'] = stats.ttest_ind(reciprocal_transform['qs1'].astype(float), reciprocal_transform['partition_sort'].astype(float)).pvalue

    tt_results.loc['qs2', 'qs2'] = stats.ttest_ind(reciprocal_transform['qs2'].astype(float), reciprocal_transform['qs2'].astype(float)).pvalue
    tt_results.loc['qs2', 'qs3'] = stats.ttest_ind(reciprocal_transform['qs2'].astype(float), reciprocal_transform['qs3'].astype(float)).pvalue
    tt_results.loc['qs2', 'qs4'] = stats.ttest_ind(reciprocal_transform['qs2'].astype(float), reciprocal_transform['qs4'].astype(float)).pvalue
    tt_results.loc['qs2', 'qs5'] = stats.ttest_ind(reciprocal_transform['qs2'].astype(float), reciprocal_transform['qs5'].astype(float)).pvalue
    tt_results.loc['qs2', 'merge1'] = stats.ttest_ind(reciprocal_transform['qs2'].astype(float), reciprocal_transform['merge1'].astype(float)).pvalue
    tt_results.loc['qs2', 'partition_sort'] = stats.ttest_ind(reciprocal_transform['qs2'].astype(float), reciprocal_transform['partition_sort'].astype(float)).pvalue

    tt_results.loc['qs3', 'qs3'] = stats.ttest_ind(reciprocal_transform['qs3'].astype(float), reciprocal_transform['qs3'].astype(float)).pvalue
    tt_results.loc['qs3', 'qs4'] = stats.ttest_ind(reciprocal_transform['qs3'].astype(float), reciprocal_transform['qs4'].astype(float)).pvalue
    tt_results.loc['qs3', 'qs5'] = stats.ttest_ind(reciprocal_transform['qs3'].astype(float), reciprocal_transform['qs5'].astype(float)).pvalue
    tt_results.loc['qs3', 'merge1'] = stats.ttest_ind(reciprocal_transform['qs3'].astype(float), reciprocal_transform['merge1'].astype(float)).pvalue
    tt_results.loc['qs3', 'partition_sort'] = stats.ttest_ind(reciprocal_transform['qs3'].astype(float), reciprocal_transform['partition_sort'].astype(float)).pvalue

    tt_results.loc['qs4', 'qs4'] = stats.ttest_ind(reciprocal_transform['qs4'].astype(float), reciprocal_transform['qs4'].astype(float)).pvalue
    tt_results.loc['qs4', 'qs5'] = stats.ttest_ind(reciprocal_transform['qs4'].astype(float), reciprocal_transform['qs5'].astype(float)).pvalue
    tt_results.loc['qs4', 'merge1'] = stats.ttest_ind(reciprocal_transform['qs4'].astype(float), reciprocal_transform['merge1'].astype(float)).pvalue
    tt_results.loc['qs4', 'partition_sort'] = stats.ttest_ind(reciprocal_transform['qs4'].astype(float), reciprocal_transform['partition_sort'].astype(float)).pvalue

    tt_results.loc['qs5', 'qs5'] = stats.ttest_ind(reciprocal_transform['qs5'].astype(float), reciprocal_transform['qs5'].astype(float)).pvalue
    tt_results.loc['qs5', 'merge1'] = stats.ttest_ind(reciprocal_transform['qs5'].astype(float), reciprocal_transform['merge1'].astype(float)).pvalue
    tt_results.loc['qs5', 'partition_sort'] = stats.ttest_ind(reciprocal_transform['qs5'].astype(float), reciprocal_transform['partition_sort'].astype(float)).pvalue

    tt_results.loc['merge1', 'merge1'] = stats.ttest_ind(reciprocal_transform['merge1'].astype(float), reciprocal_transform['merge1'].astype(float)).pvalue
    tt_results.loc['merge1', 'partition_sort'] = stats.ttest_ind(reciprocal_transform['merge1'].astype(float), reciprocal_transform['partition_sort'].astype(float)).pvalue

    tt_results.loc['partition_sort', 'partition_sort'] = stats.ttest_ind(reciprocal_transform['partition_sort'].astype(float), reciprocal_transform['partition_sort'].astype(float)).pvalue

    tt_results.set_index(sort_columns)
    tt_results.to_csv('tt_results.csv', index=True)
    tt_results_h0_accept = tt_results.where(tt_results > 0.05).stack().sort_values().index.tolist()
    tt_results_h0_reject = tt_results.where(tt_results < 0.05).stack().sort_values().index.tolist()

    print("\n", )
    print("\nT-Test Analysis: Comparing Algorithms - Means not Different\n",tt_results_h0_accept)
    print("\nT-Test Analysis: Comparing Algorithms - Means Different\n",tt_results_h0_reject)



    '''
    plot_histograms(data, sort_columns, 'Untransformed Data')
    plot_histograms(log_transform, sort_columns, 'Log Transformed')
    plot_histograms(square_transform, sort_columns, 'Square Transformed')
    plot_histograms(cube_transform, sort_columns, 'Cube Transformed')
    iqr_qs1 = get_iqr_data(data['qs1'])
    iqr_qs2 = get_iqr_data(data['qs2'])
    iqr_qs3 = get_iqr_data(data['qs3'])
    iqr_qs4 = get_iqr_data(data['qs4'])
    iqr_qs5 = get_iqr_data(data['qs5'])
    iqr_m = get_iqr_data(data['merge1'])
    iqr_ps = get_iqr_data(data['partition_sort'])

    q1, q3 = np.percentile(data['qs1'], [25, 75])
    iqr = q3 - q1
    lf = q1 - (1.5*iqr)
    hf = q3 + (1.5*iqr)
    print(lf, iqr, hf)
    norm_qs1 = data.loc[(data['qs1'] >= q1) & (data['qs1'] <= q3), 'qs1']
    print(norm_qs1)

    '''



if __name__ == '__main__':
    main()
