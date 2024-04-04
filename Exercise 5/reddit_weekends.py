"""
python3 reddit_weekends.py reddit-counts.json.gz
"""

import sys
import pandas as pd
from scipy import stats
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.axes as axs
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer


OUTPUT_TEMPLATE = (
    "Initial T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann-Whitney U-test p-value: {utest_p:.3g}"
)

""" 
Read the Data:
    The counts of # comments posted daily in each Canadian province sub reddit,
    and in /r/canada itself.

Question:
    are there a different num. of Reddit comments posted on weekdays than on weekends?

Only look at Values of:
    1) 2012 and 2013 data
    2) /r/canada subreddit
"""

def transform_data(data, method=None):
    transformed = data
    if method is None:
        return ("Invalid method. ['log', 'exp', 'sqrt', 'square']")
    if method == 'log':
        transformed = data.apply(lambda x: np.log(x))
    # Ref: https://www.geeksforgeeks.org/implement-sigmoid-function-using-numpy/
    elif method == 'exp':
        transformed = data.apply(lambda x: 1/(1 + np.exp(-x)))
    elif method == 'sqrt':
        transformed = data.apply(lambda x: np.sqrt(x))
    elif method == 'square':
        transformed = data.apply(lambda x: x**2)

    return transformed


def add_hist(axs, index, data, title, xlab, ylab, color=None):
    r, c = index
    axs[r][c].set_title(title, size=6)
    axs[r][c].set_xlabel(xlab, size=6)
    axs[r][c].set_ylabel(ylab, size=6)
    axs[r][c].hist(data, color=color)

def plot_histogram(data, title):
    fig, axs = plt.subplots(2, 3)
    fig.suptitle(title)
    fig.tight_layout()

    add_hist(axs, (0, 0), data['comment_count'],
             "Not Transformed",
             "Comment Count", "Frequency", "Blue")
    add_hist(axs, (0, 1), data['log'],
             "Log Transformed",
             "Comment Count", "Frequency", "Green")
    add_hist(axs, (0, 2), data['exp'],
             "Exp Transformed",
             "Comment Count", "Frequency", "Pink")
    add_hist(axs, (1, 0), data['sqrt'],
             "Square Root Transformed",
             "Comment Count", "Frequency", "Violet")
    add_hist(axs, (1, 1), data['square'],
             "Squared Transformed",
             "Comment Count", "Frequency", "Yellow")
    plt.plot()
    plt.show()

def main():
    # Load the program and files
    file1 = sys.argv[1]
    counts = pd.read_json(file1, lines=True)
    # Keep only 2012 and 2013 data
    counts = counts[counts['date'].dt.year.isin([2012, 2013])]

    """
    Separate the Weekdays from Weekends
    datetime.date.weekday(): return the day of week as an int.
    """
    wkdays = [0, 1, 2, 3, 4]
    wkends = [5, 6]
    weekdays = counts[counts['date'].dt.weekday.isin(wkdays)]
    weekends = counts[counts['date'].dt.weekday.isin(wkends)]
    weekdays = weekdays[weekdays['subreddit'] == 'canada']
    weekends = weekends[weekends['subreddit'] == 'canada']

    # Transform the data to get close to a Normal Distribution
    weekdays.loc[:, 'log'] = transform_data(weekdays['comment_count'], 'log')
    weekdays.loc[:, 'exp'] = transform_data(weekdays['comment_count'], 'exp')
    weekdays.loc[:, 'sqrt'] = transform_data(weekdays['comment_count'], 'sqrt')
    weekdays.loc[:, 'square'] = transform_data(weekdays['comment_count'], 'square')

    weekends.loc[:, 'log'] = transform_data(weekends['comment_count'], 'log')
    weekends.loc[:, 'exp'] = transform_data(weekends['comment_count'], 'exp')
    weekends.loc[:, 'sqrt'] = transform_data(weekends['comment_count'], 'sqrt')
    weekends.loc[:, 'square'] = transform_data(weekends['comment_count'], 'square')

    log_ttest = stats.ttest_ind(weekdays['log'], weekends['log'])
    exp_ttest = stats.ttest_ind(weekdays['exp'], weekends['exp'])
    sqrt_ttest = stats.ttest_ind(weekdays['sqrt'], weekends['sqrt'])
    square_ttest = stats.ttest_ind(weekdays['square'], weekends['square'])

    log_normaltest_weekday = stats.normaltest(weekdays['log'])
    exp_normaltest_weekday = stats.normaltest(weekdays['exp'])
    sqrt_normaltest_weekday = stats.normaltest(weekdays['sqrt'])
    square_normaltest_weekday = stats.normaltest(weekdays['square'])

    log_normaltest_weekend = stats.normaltest(weekends['log'])
    exp_normaltest_weekend = stats.normaltest(weekends['exp'])
    sqrt_normaltest_weekend = stats.normaltest(weekends['sqrt'])
    square_normaltest_weekend = stats.normaltest(weekends['square'])

    log_levene = stats.levene(weekdays['log'], weekends['log'])
    exp_levene = stats.levene(weekdays['exp'], weekends['exp'])
    sqrt_levene = stats.levene(weekdays['sqrt'], weekends['sqrt'])
    square_levene = stats.levene(weekdays['square'], weekends['square'])

    # Check nan
    #print(weekdays.isna().sum())
    #print(weekends.isna().sum())
    """
    STUDENTS T-TEST: scipy.stats.ttest_ind(x1,x2)
    Do T-test on the data; get a p-value.
    Q: Are there different num. comments on weekdays compared to weekends?
    DISTRIBUTION: stats.normaltest, stats.levene
    See if data is normally distributed and if they have equal variances
    """
    counts_ttest = stats.ttest_ind(weekdays['comment_count'], weekends['comment_count'])
    weekday_normaltest = stats.normaltest(weekdays['comment_count'])
    weekend_normaltest = stats.normaltest(weekends['comment_count'])
    counts_levene = stats.levene(weekdays['comment_count'], weekends['comment_count'])


    """
    CENTRAL LIMIT THEOREM: date.isocalendar()
    Group by week for the weekend and weekdays, take their mean.
    Check for normality and equal variance. Apply t-test.
    Do the number of comments on weekends and weekdays for each week differ?
    """
    counts.loc[:, 'week'] = counts['date'].apply(lambda x: x.isocalendar().week)
    counts.loc[:, 'weekday'] = counts['date'].apply(lambda x: x.isocalendar().weekday)

    clt_weekday = weekdays
    clt_weekend = weekends
    clt_weekday.loc[:, 'week'] = clt_weekday['date'].apply(lambda x: x.isocalendar().week)
    clt_weekend.loc[:, 'week'] = clt_weekend['date'].apply(lambda x: x.isocalendar().week)
    clt_weekday = clt_weekday.groupby('week')['comment_count'].mean()
    clt_weekend = clt_weekend.groupby('week')['comment_count'].mean()
    # Apply T-Test
    clt_ttest = stats.ttest_ind(clt_weekday, clt_weekend)
    # Test Normal Distribution
    clt_weekday_normaltest = stats.normaltest(clt_weekday)
    clt_weekend_normaltest = stats.normaltest(clt_weekend)
    # Test Levene Variance
    clt_levene = stats.levene(clt_weekday, clt_weekend)

    """
    NON-PARAMETRIC TEST: Mann-Whitney U-Test
    "Its not equally likely that the larger number of comments occur on weekends vs. weekdays"
    """
    mannwhitney = stats.mannwhitneyu(weekdays['comment_count'], weekends['comment_count'],
                                           alternative='two-sided')

    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=counts_ttest.pvalue,
        initial_weekday_normality_p=weekday_normaltest.pvalue,
        initial_weekend_normality_p=weekend_normaltest.pvalue,
        initial_levene_p=counts_levene.pvalue,
        transformed_weekday_normality_p=sqrt_normaltest_weekday.pvalue,
        transformed_weekend_normality_p=sqrt_normaltest_weekend.pvalue,
        transformed_levene_p=sqrt_levene.pvalue,
        weekly_weekday_normality_p=clt_weekday_normaltest.pvalue,
        weekly_weekend_normality_p=clt_weekend_normaltest.pvalue,
        weekly_levene_p=clt_levene.pvalue,
        weekly_ttest_p=clt_ttest.pvalue,
        utest_p=mannwhitney.pvalue,
    ))

    """
    --- ANALYSIS & HISTOGRAM PRINTS ---
    print("log_ttest", log_ttest)
    print("exp_ttest", exp_ttest)
    print("sqrt_ttest", sqrt_ttest)
    print("square_ttest", square_ttest)
    print("log_normaltest_weekday", log_normaltest_weekday)
    print("exp_normaltest_weekday", exp_normaltest_weekday)
    print("sqrt_normaltest_weekday", sqrt_normaltest_weekday)
    print("square_normaltest_weekday", square_normaltest_weekday)
    print("log_normaltest_weekend", log_normaltest_weekend)
    print("exp_normaltest_weekend", exp_normaltest_weekend)
    print("sqrt_normaltest_weekend", sqrt_normaltest_weekend)
    print("square_normaltest_weekend", square_normaltest_weekend)
    print("log_levene", log_levene)
    print("exp_levene", exp_levene)
    print("sqrt_levene", sqrt_levene)
    print("square_levene", square_levene)
    print("transforms t-test: ", np.sort([log_ttest.pvalue, exp_ttest.pvalue, sqrt_ttest.pvalue, square_ttest.pvalue]))
    print("weekdays, transforms normal test: ", np.sort([log_normaltest_weekday.pvalue, exp_normaltest_weekday.pvalue, sqrt_normaltest_weekday.pvalue, square_normaltest_weekday.pvalue]))
    print("weekends, transforms normal test: ", np.sort([log_normaltest_weekend.pvalue, exp_normaltest_weekend.pvalue, sqrt_normaltest_weekend.pvalue, square_normaltest_weekend.pvalue]))
    print("transforms levene test:", np.sort([log_levene.pvalue, exp_levene.pvalue, sqrt_levene.pvalue, square_levene.pvalue]))

    print("\n - - - - T-TEST - - - -")
    print("\nQ: There are different number of comments on weekdays compared to weekends.")
    print("\nT-Test Results:\n", counts_ttest)
    print("Reject H0 (pvalue < 0.05)?:", counts_ttest.pvalue < 0.05)
    print("\nNormal Test Results:")
    print("Is the weekday distribution Normal (pvalue > 0.05)?:", weekday_normaltest.pvalue > 0.05)
    print("Is the weekend distribution Normal (pvalue > 0.05)?:", weekend_normaltest.pvalue > 0.05)
    print("\nLevene Results:\n", counts_levene)
    print("Do weekday and weekend have different Variances?:", counts_levene.pvalue < 0.05)

    plot_histogram(weekdays, 'Reddit Comment Count for Reddit: /r/canada - Weekdays')
    plot_histogram(weekends, 'Reddit Comment Count for Reddit: /r/canada - Weekends')

    print("\n - - - - CENTRAL LIMIT THEOREM - - - -")
    print("Q: Do the number of comments on weekends and weekdays for each week differ? ")
    print("\nT-Test Results:\n", clt_ttest)
    print("Reject (p < 0.05)?:", clt_ttest.pvalue < 0.05)
    print("\nNormal Test Results:")
    print("Is the weekday distribution Normal (p > 0.05)?:", clt_weekday_normaltest.pvalue > 0.05)
    print("Is the weekend distribution Normal (p > 0.05)?:", clt_weekend_normaltest.pvalue > 0.05)
    print("\nLevene Results:\n", clt_levene)
    print("Do weekday and weekend have different Variances? (p < 0.05)?:", clt_levene.pvalue < 0.05)

    fig, ax = plt.subplots(1,2)
    fig.suptitle("Reddit Comment Count for Reddit: /r/canada - Weekly Differences")
    ax[0].hist(clt_weekday_normaltest, color="Pink")
    ax[0].set_title("Weekday Grouped by Week Num.")
    ax[0].set_xlabel('Comment Count', size=6)
    ax[0].set_ylabel('Frequency', size=6)
    ax[1].hist(clt_weekend_normaltest, color="Yellow")
    ax[1].set_title("Weekend Grouped by Week Num.")
    ax[1].set_xlabel('Comment Count', size=6)
    ax[1].set_ylabel('Frequency', size=6)
    plt.plot()
    plt.show()
    
    print("\n - - - - MANN-WHITNEY-U TEST - - - -")
    print("\nMann-Whitney-U Test Results:\n", mannwhitney)
    print("Reject (p < 0.05)?:", mannwhitney.pvalue < 0.05)
    """

if __name__ == '__main__':
    main()
