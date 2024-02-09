"""
python3 average_ratings.py movie_list.txt movie_ratings.csv output.csv
"""

import numpy as np
import pandas as pd
import sys
import difflib as dl

"""
Find matching titles for each movie from the ratings list.
Function: get close matches (at most 1), return the list's value (ie. title name)
If none, return NaN
"""
def find_matches_in (word, list):
    matches = dl.get_close_matches(word, list, n=1, cutoff=0.3)
    if len(matches) == 0:
        return np.nan
    return matches[0]


def main():
    """
    open the file and read line-at-a-time from the .txt
    open(movie_list).readlines()
    then create a DataFrame
    ** Not comma-separated data since movies can contain commas.
    """

    # Read data from command line
    file_movies = sys.argv[1]
    file_ratings = sys.argv[2]
    file_output = sys.argv[3]

    # Read lines in data
    data = open(file_movies).readlines()

    # Create dataframe
    movie_list = pd.DataFrame(columns=['title', 'rating'])
    movie_list['title'] = data
    movie_list.set_index('title')

    data_ratings = pd.read_csv(file_ratings)
    data_ratings = pd.DataFrame(data_ratings)

    # Replace newline strings
    movie_list = movie_list.replace(r'\n', value='', regex=True)

    # Apply matching function
    data_ratings['matched_titles'] = data_ratings['title'].apply(lambda x: find_matches_in(x, movie_list['title']))
    movie_avg_ratings = pd.DataFrame(data_ratings.groupby('matched_titles')['rating'].mean().round(2),
                                     columns=['rating'])

    # Export results
    movie_avg_ratings.to_csv(file_output, index=True)

    return

if __name__ == '__main__':
    main()