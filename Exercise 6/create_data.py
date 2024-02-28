'''
Write a program create_data.py that generates random arrays, and uses time.time (https://docs.python.org/3/library/time.html)
to measure the wall-clock time (https://en.wikipedia.org/wiki/Wall-clock_time)
each function takes to sort them. You may write loops in the create_data.py program.

benchmarking some code to see how it performs, and analysing the results for statistical signifi cance.
'''
import numpy as np
import pandas as pd
import time
from implementations import all_implementations
from scipy import stats
from numpy import random
import matplotlib.pyplot as plt

'''
time python3 create_data.py
# just check the time
timeout -s SIGKILL 60 python3 create_data.py
# kill after 60 seconds
'''

def benchmark_arrays(df, sort_columns, n_data, array_size, max):
    row = 0
    for i in range(n_data):
        random_array = random.randint(max, size=(array_size))
        col = 0
        for sort in all_implementations:
            st = time.time()
            res = sort(random_array)
            en = time.time()
            duration = en - st
            df.loc[row, sort_columns[col]] = duration
            col += 1
        row += 1
    return df


def main():
    sort_columns = ['qs1', 'qs2', 'qs3', 'qs4', 'qs5', 'merge1', 'partition_sort']
    data = pd.DataFrame(columns=sort_columns)
    data = benchmark_arrays(data, sort_columns,
                            n_data=1000,
                            array_size=800,
                            max=10000)
    data.to_csv('data.csv', index=False)
if __name__ == '__main__':
    main()
