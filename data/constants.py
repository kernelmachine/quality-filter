from pathlib import Path
import pandas as pd

DATA_DIR = Path("/homes/gws/sg01/logistic_regression/")




dependent_variable = [{'feature': 'prob_high_quality', 
                       'zscore': False, 
                       'log': False, 
                       'type': "output", 
                       'new_name': None ,
                       'outlier_removal': None}]
race_features = [
            {'feature': 'asian_ratio', 'zscore': True, 'log': True, 'type': 'race', 'new_name': None , 'outlier_removal': None,'outlier_removal': None},
            {'feature': 'white_ratio', 'zscore': True, 'log': True, 'type': 'race', 'new_name': None ,'outlier_removal': None},
            {'feature': 'black_ratio', 'zscore':True, 'log': True, 'type': 'race', 'new_name': None ,'outlier_removal': None},
            {'feature': 'hispanic_ratio', 'zscore': True, 'log': True, 'type': 'race', 'new_name': None ,'outlier_removal': None},
            {'feature': 'native_ratio', 'zscore': True, 'log': True, 'type': 'race', 'new_name': None ,'outlier_removal': None},
            {'feature': 'mixed_ratio', 'zscore': True, 'log': True, 'type': 'race', 'new_name': None ,'outlier_removal': None},
            {'feature': 'pacific_islander_ratio', 'zscore': True, 'log': True, 'type': 'race', 'new_name': None ,'outlier_removal': None}
]


    

education_features = [
            {'feature': 'num_non_degree_holders', 'zscore': True, 'log': True, 'type': 'wealth', 'new_name': None,'outlier_removal': None},
            {'feature': 'num_degree_holders', 'zscore': True, 'log': True, 'type': 'wealth', 'new_name': None ,'outlier_removal': None},
            {'feature': 'pupil_ratio', 'zscore': True, 'log': True, 'type': 'other', 'new_name': None,'outlier_removal': None},
            {'feature': 'fte_teachers', 'zscore': True, 'log': False, 'type': 'other', 'new_name': None,'outlier_removal': None},

]


population_features = [                   
            {'feature': 'population_density', 'zscore': True, 'log': True, 'type': 'other', 'new_name': None ,'outlier_removal': None},
            {'feature': 'school_size', 'zscore': True, 'log': False, 'type': 'other', 'new_name': None ,'outlier_removal': None},
            {'feature': 'city', 'zscore': True, 'log': True, 'type': 'wealth', 'new_name': None,'outlier_removal': None},
            {'feature': 'suburb', 'zscore': True, 'log': True, 'type': 'wealth', 'new_name': None,'outlier_removal': None},
            {'feature': 'town', 'zscore': True, 'log': True, 'type': 'wealth', 'new_name': None,'outlier_removal': None},
            {'feature': 'rural', 'zscore': True, 'log': True, 'type': 'wealth', 'new_name': None,'outlier_removal': None},
            {'feature': 'POPPCT_RURAL', 'zscore': True, 'log': True, 'type': 'other', 'new_name': 'pct_rural','outlier_removal': None},
]

wealth_features =  [
    {'feature': 'median_household_income', 'zscore': True, 'log':True, 'type': 'wealth', 'new_name': None,'outlier_removal': None},
    {'feature': 'median_home_value', 'zscore': True, 'log': True, 'type': 'wealth', 'new_name': None,'outlier_removal': None},

]

text_features = [{'feature': 'num_tokens', 'zscore': True, 'log': False, 'type': 'text', 'new_name': None, 'outlier_removal': None}]

other_features = [
                  {'feature': 'rep_share_2020', 'zscore': True, 'log': True, 'type': 'other', 'new_name': None ,'outlier_removal': None},
                  {'feature': 'rep_share_2016', 'zscore': True, 'log': True, 'type': 'other', 'new_name': None ,'outlier_removal': None},
                  {'feature': 'rep_share_2012', 'zscore': True, 'log': True, 'type': 'other', 'new_name': None ,'outlier_removal': None},
                  {'feature': 'rep_share_2008', 'zscore': True, 'log': True, 'type': 'other', 'new_name': None ,'outlier_removal': None},
                  {'feature': 'rep_share_2004', 'zscore': True, 'log': True, 'type': 'other', 'new_name': None ,'outlier_removal': None},
                  {'feature': 'rep_share_2000', 'zscore': True, 'log': True, 'type': 'other', 'new_name': None ,'outlier_removal': None},

]


REGRESSION_FEATURES = pd.DataFrame((dependent_variable +
                             race_features + 
                             wealth_features + 
                             population_features + 
                             education_features + 
                             other_features))


