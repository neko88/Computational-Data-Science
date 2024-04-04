import sys
import pandas as pd
import numpy as np
from scipy import stats
OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value:  {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value:  {more_searches_p:.3g} \n'
    '"Did more/less instructors use the search feature?" p-value:  {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value:  {more_instr_searches_p:.3g}'
)

"""
The question we were interested in: do people search more with the new design? We see a few ways to approach that problem:
Did more users use the search feature? (More precisely: did a different fraction of users have search count > 0?)
Did users search more often? (More precisely: is the number of searches per user different?)
H0: People don't search more with the new design.

- Analyze the data; get their p-values

"""

def main():
    searchdata_file = sys.argv[1]
    searches = pd.read_json(searchdata_file, lines=True)
    searches = pd.DataFrame(searches)
    print(searches)

    search_table = pd.DataFrame(columns=['searched','zero search'],
                                     index=['even_uid', 'odd_uid'])

    even_uid = searches[searches['uid'].values % 2 == 0]
    odd_uid = searches[searches['uid'].values % 2 != 0]

    search_table.loc['even_uid', 'searched'] = len(even_uid[even_uid['search_count'] > 0])
    search_table.loc['odd_uid', 'searched'] = len(odd_uid[odd_uid['search_count'] > 0])
    search_table.loc['even_uid', 'zero search'] = len(even_uid) - search_table.loc['even_uid', 'searched']
    search_table.loc['odd_uid', 'zero search'] = len(odd_uid) - search_table.loc['odd_uid', 'searched']

    print("\nAll Users Search Table\n", search_table)

    chi2_contingency = stats.chi2_contingency(search_table)
    mw_u = stats.mannwhitneyu(x=even_uid['search_count'], y=odd_uid['search_count'])

    """
    Chi2ContingencyResult(
    statistic=1.9040140234412852, 
    pvalue=0.1676297094499566, 
    dof=1, 
    expected_freq=array([[102.19823789, 230.80176211],
                         [106.80176211, 241.19823789]]))
    p = 0.17 > 0.05
    Using a significance level of 5%, it is not significant and we would not reject the null hypothesis: 
    “the odd/even uid does not have effect on searching or not searching”. 
    Because scipy.stats.contingency.chi2_contingency performs a two-sided test, 
    the alternative hypothesis does not indicate the direction of the effect. 
    We can use stats.contingency.odds_ratio to support the conclusion that aspirin reduces the risk of ischemic stroke.
    MannwhitneyuResult(statistic=61019.5, pvalue=0.14118207247086972)
    
    p = 0.14 > 0.05
    There is no statistically significant difference between the means of distribution 
    of even uid and distribution of odd uid (new design) in the count of searching or not searching.

    """
    
    even_uid_instructors = searches[ (searches['uid'] % 2 == 0) & (searches['is_instructor'] == True) ]
    odd_uid_instructors = searches[ (searches['uid'] % 2 != 0) & (searches['is_instructor'] == True) ]

    search_table_instructors = pd.DataFrame(columns=['searched', 'zero search'],
                                            index=['even_uid_instructors', 'odd_uid_instructors'])
    search_table_instructors.loc['even_uid_instructors', 'searched'] = len(even_uid_instructors[even_uid_instructors['search_count'] > 0])
    search_table_instructors.loc['odd_uid_instructors', 'searched'] = len(odd_uid_instructors[odd_uid_instructors['search_count'] > 0])
    search_table_instructors.loc['even_uid_instructors', 'zero search'] = len(even_uid_instructors) - search_table_instructors.loc['even_uid_instructors', 'searched']
    search_table_instructors.loc['odd_uid_instructors', 'zero search'] = len(odd_uid_instructors) - search_table_instructors.loc['odd_uid_instructors', 'searched']

    print("\nInstructors Only Search Table\n", search_table_instructors)

    chi2_contingency_instructors = stats.chi2_contingency(search_table_instructors)
    mw_u_instructors = stats.mannwhitneyu(even_uid_instructors['search_count'], odd_uid_instructors['search_count'])

    """
    Chi2ContingencyResult(
    statistic=3.775716244554879, 
    pvalue=0.052001632770999166, 
    dof=1, 
    expected_freq=array([[42.38297872, 77.61702128],
                         [40.61702128, 74.38297872]]))
    
    p = 0.052 > 0.050
    Using a significance level of 5%, it is not significant and we would not reject the null hypothesis: 
    “the odd/even uid instructors does not have effect on searching or not searching”. 
    
    MannwhitneyuResult(statistic=7790.0, pvalue=0.044959434016105145)
    p = 0.045 < 0.05
    There is statistically significant difference between the means of distribution 
    of even uid and distribution of odd uid (new design) in the count of searching or not searching.
    """

    # ...

    # Output
    '''
    print(search_table)
    print(search_table_instructors)
    '''

    print(OUTPUT_TEMPLATE.format(
        more_users_p=chi2_contingency.pvalue,
        more_searches_p=mw_u.pvalue,
        more_instr_p=chi2_contingency_instructors.pvalue,
        more_instr_searches_p=mw_u_instructors.pvalue,
    ))



if __name__ == '__main__':
    main()
