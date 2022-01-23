import pandas as pd
from thefuzz import process
from joblib import Parallel, delayed
from data.constants import DATA_DIR, REGRESSION_FEATURES
import numpy as np
from tqdm.auto import tqdm
from scipy import stats


def initialize_features(df, level='zipcode'):
    """
    initialize feature dataframe at a particular unit of analysis
    """
    features = {}
    zipcode_counts = df.zipcode.value_counts()
    popular_zipcodes= zipcode_counts.loc[zipcode_counts > 1].index
    high_schools = df.loc[df.zipcode.isin(popular_zipcodes)]
    high_schools['category_first'] = df.category.apply(lambda x: x[0])

    features['prob_high_quality'] = df.groupby(level).prob_high_quality.mean()

    features['zipcode'] = df.groupby(level).zipcode.min()
    features = pd.DataFrame(features)
    features.shape
    features = features.reset_index(drop=True)
    county_state = df.groupby(level)[['school_name','zipcode', 'county', 'state']].sample().reset_index(drop=True)
    county_state['base_county'] = county_state.county.str.lower()
    county_state['base_state'] = county_state.state.str.lower()
    county_state['county:state'] = county_state['base_county']  + ":" + county_state['base_state'] 
    county_state['school:county:state'] = (county_state['school_name'].str.lower() + ':' 
                                            + county_state['base_county'] 
                                             + ":" + county_state['base_state'])
    county_state['school:zipcode:county:state'] = (county_state['school_name'].str.lower() + ':' 
                                                    + county_state['zipcode'].astype('str').str.lower() + ":" 
                                                    + county_state['base_county']  + ":" 
                                                    + county_state['base_state'])

    features = features.merge(county_state)

    return features


def merge_zipcode_features(features, path='zipcode_data.jsonl'):
    """
    add in zipcode features from Census
    """
    zipcode_data = pd.read_json(DATA_DIR / path, lines=True).dropna(subset=['zipcode'])

    def num_degree_holders(z):
        if not z:
            return
        totals = sum([x['y'] for x in z[0]['values']])
        return sum([x['y'] for x in z[0]['values'] if "Degree" in x['x']]) / totals

    def num_non_degree_holders(z):
        if not z:
            return
        totals = sum([x['y'] for x in z[0]['values']])
        return sum([x['y'] for x in z[0]['values'] if "High School" in x['x']]) / totals

    zipcode_data['num_degree_holders'] = zipcode_data.educational_attainment_for_population_25_and_over.apply(num_degree_holders)
    zipcode_data['num_non_degree_holders'] = zipcode_data.educational_attainment_for_population_25_and_over.apply(num_non_degree_holders)

    features = features.merge(zipcode_data, on='zipcode',  how='left')
    features.shape

  
    return features
    
def merge_income_features(features, path='18zpallnoagi.csv'):
    irs_data = pd.read_csv(DATA_DIR / path)
    irs_data['income_per_capita'] = stats.zscore(np.log(irs_data['A04800'] / irs_data['N04800'] + 1e-5))
    irs_data['tax_per_capita'] =  stats.zscore(np.log((irs_data['A11901'] /(irs_data['N11901'] + 1e-5)).sort_values() + 1))
    features = features.merge(irs_data,how='left', left_on='zipcode', right_on='ZIPCODE')
    features.shape
    return features

def load_private_school_data(path='ELSI_csv_export_6377702192917155939481.csv'):
    """
    add in private school info from NCES
    """
    private_school_dem = pd.read_csv(DATA_DIR / path)
    private_school_dem = private_school_dem.replace('–', np.nan)
    private_school_dem = private_school_dem.replace('†', np.nan)

    private_school_dem = private_school_dem.replace('2-No', 0)
    private_school_dem = private_school_dem.replace('1-Yes', 1)


    private_school_dem['zipcode'] = private_school_dem['ZIP [Private School] 2017-18'].str.extract("(\d+)").astype('int')
    private_school_dem['school_name'] = private_school_dem['School Name'].str.lower()

    private_school_dem['zipcode'] = private_school_dem['ZIP [Private School] 2017-18'].str.extract("(\d+)").astype('int')
    private_school_dem['school:zipcode:county:state'] = private_school_dem['school_name'].str.lower() + ":" + private_school_dem['zipcode'].astype('str').str.lower() + ":" + private_school_dem['County Name [Private School] 2017-18'].astype(str).str.lower() + ":" + private_school_dem['State Name [Private School] 2017-18'].str.lower()
    private_school_dem['zipcode:county:state'] =  private_school_dem['zipcode'].astype('str').str.lower() + ":" + private_school_dem['County Name [Private School] 2017-18'].astype(str).str.lower() + ":" + private_school_dem['State Name [Private School] 2017-18'].str.lower()

    private_school_dem['black_ratio'] = private_school_dem['Percentage of Black Students [Private School] 2017-18'].str.extract("(\d+\.\d)+").astype('float')/100
    private_school_dem['hispanic_ratio'] = private_school_dem['Percentage of Hispanic Students [Private School] 2017-18'].str.extract("(\d+\.\d)+").astype('float')/100
    private_school_dem['white_ratio'] = private_school_dem['Percentage of White Students [Private School] 2017-18'].str.extract("(\d+\.\d)+").astype('float')/100
    private_school_dem['asian_ratio'] = private_school_dem['Percentage of Asian or Asian/Pacific Islander Students [Private School] 2017-18'].str.extract("(\d+\.\d)+").astype('float')/100
    private_school_dem['mixed_ratio'] = private_school_dem['Percentage of Two or More Races Students [Private School] 2017-18'].str.extract("(\d+\.\d)+").astype('float')/100
    private_school_dem['native_ratio'] = private_school_dem['Percentage of American Indian/Alaska Native Students [Private School] 2017-18'].str.extract("(\d+\.\d)+").astype('float') /100
    private_school_dem['pacific_islander_ratio'] = private_school_dem['Percentage of Nat. Hawaiian or Other Pacific Isl. Students [Private School] 2017-18'].str.extract("(\d+\.\d)+").astype('float')/100
    private_school_dem['locale'] = "Unknown"
    private_school_dem['is_public'] = 0
    private_school_dem['is_private'] = 1
    private_school_dem['total_race'] = 0
    private_school_dem['is_charter'] = 0

    private_school_dem['is_magnet'] = 0
    private_school_dem['school_size'] = private_school_dem['Total Students (Ungraded & K-12) [Private School] 2017-18'].str.extract("(\d+)").astype('float')
    private_school_dem['pupil_ratio'] = private_school_dem['Pupil/Teacher Ratio [Private School] 2017-18'].str.extract("(\d+\.?\d+)").astype('float')
    private_school_dem['fte_teachers'] = private_school_dem['Full-Time Equivalent (FTE) Teachers [Private School] 2017-18'].str.extract("(\d+\.?\d+)").astype('float')
    private_school_dem['county:state'] = private_school_dem['County Name [Private School] 2017-18'].astype(str).str.lower() + ":" + private_school_dem['State Name [Private School] 2017-18'].str.lower()
    private_school_dem['state'] =  private_school_dem['State Name [Private School] 2017-18']

    return private_school_dem
    
    
def load_public_school_data(path='ELSI_csv_export_6377702205999255091950.csv'):
    """
    add in public school info from NCES
    """
    public_school_dem = pd.read_csv(DATA_DIR / path)
    public_school_dem = public_school_dem.loc[public_school_dem['School Level (SY 2017-18 onward) [Public School] 2017-18'] == 'High']
    public_school_dem = public_school_dem.replace('–', np.nan)
    public_school_dem = public_school_dem.replace('†', np.nan)

    public_school_dem = public_school_dem.replace('2-No', 0)
    public_school_dem = public_school_dem.replace('1-Yes', 1)

    public_school_dem['zipcode'] = public_school_dem['Location ZIP [Public School] 2017-18'].str.extract("(\d+)").astype('int')
    public_school_dem['school_name'] = public_school_dem['School Name'].str.lower()
    public_school_dem['school:zipcode:county:state'] = public_school_dem['school_name'].str.lower() +":" + public_school_dem['zipcode'].astype('str').str.lower() + ":" + public_school_dem['County Name [Public School] 2017-18'].astype(str).str.lower() + ":" + public_school_dem['State Name [Public School] 2017-18'].str.lower()
    public_school_dem['zipcode:county:state'] = public_school_dem['zipcode'].astype('str').str.lower() + ":" + public_school_dem['County Name [Public School] 2017-18'].astype(str).str.lower() + ":" + public_school_dem['State Name [Public School] 2017-18'].str.lower()

    public_school_dem['zipcode'] = public_school_dem['Location ZIP [Public School] 2017-18'].str.extract("(\d+)").astype('int')
    public_school_dem['black'] = public_school_dem['Black or African American Students [Public School] 2017-18'].str.extract("(\d+)").astype('float')
    public_school_dem['hispanic'] = public_school_dem['Hispanic Students [Public School] 2017-18'].str.extract("(\d+)").astype('float')
    public_school_dem['white'] = public_school_dem['White Students [Public School] 2017-18'].str.extract("(\d+)").astype('float')
    public_school_dem['asian'] = public_school_dem['Asian or Asian/Pacific Islander Students [Public School] 2017-18'].str.extract("(\d+)").astype('float')
    public_school_dem['mixed'] = public_school_dem['Two or More Races Students [Public School] 2017-18'].str.extract("(\d+)").astype('float')
    public_school_dem['native'] = public_school_dem['American Indian/Alaska Native Students [Public School] 2017-18'].str.extract("(\d+)").astype('float')
    public_school_dem['pacific_islander'] = public_school_dem['Nat. Hawaiian or Other Pacific Isl. Students [Public School] 2017-18'].str.extract("(\d+)").astype('float')
    public_school_dem['total_race'] = public_school_dem['Total Race/Ethnicity [Public School] 2017-18'].str.extract("(\d+)").astype('float')
    for race in ['black', 'hispanic', 'white', 'asian', 'mixed', 'native', 'pacific_islander']:
        public_school_dem[f'{race}_ratio'] = public_school_dem[race] /  public_school_dem['total_race'] 
    public_school_dem['is_public'] = 1
    public_school_dem['is_private'] = 0
    public_school_dem['locale'] = public_school_dem['Urban-centric Locale [Public School] 2017-18']
    public_school_dem['is_charter'] = public_school_dem['Charter School [Public School] 2017-18'].fillna(0)
    public_school_dem['is_magnet'] = public_school_dem['Magnet School [Public School] 2017-18'].fillna(0)
    public_school_dem['school_size'] = public_school_dem['Total Students All Grades (Excludes AE) [Public School] 2017-18'].str.extract("(\d+)").astype('float')
    public_school_dem['pupil_ratio'] = public_school_dem['Pupil/Teacher Ratio [Public School] 2017-18'].str.extract("(\d+\.?\d+)").astype('float')
    public_school_dem['fte_teachers'] = public_school_dem['Full-Time Equivalent (FTE) Teachers [Public School] 2017-18'].str.extract("(\d+\.?\d+)").astype('float')

    public_school_dem['county:state'] =  public_school_dem['County Name [Public School] 2017-18'].astype(str).str.lower() + ":" + public_school_dem['State Name [Public School] 2017-18'].str.lower()
    
    public_school_dem['state'] =  public_school_dem['State Name [Public School] 2017-18']
    return public_school_dem

def get_mappings(school_demographics, items, num_jobs=40, min_threshold=86):
    """
    fuzzy match school names
    """
    mappings = {}
    def find_closest(item, school_dem):
        zipcode = item.split(":")[1]
        candidates = school_dem.loc[school_dem.zipcode == int(zipcode)]['school:zipcode:county:state'].unique().tolist()
        if len(candidates) > 0:
            value = process.extractOne(item, candidates)
            return {"key": item, "value": value[0], "sim": value[1] }
        else:
            return {"key": item, "value": None, "sim": None }
    mappings = Parallel(n_jobs=num_jobs)(delayed(find_closest)(item, school_demographics) for item in tqdm(items))
    mappings = pd.DataFrame(mappings)
    mappings = mappings.loc[mappings.sim > min_threshold]
    mappings['school:zipcode'] = mappings['key']
    return mappings


def impute(df, features = None):
    """
    general data imputation, backing off to zipcode, county, and state features
    """
    if not features:
        features = df.columns
    for feature in features:
        if df[feature].dtype in ('float', 'int'):
            median = df.groupby(['state'])[feature].median().median()
            state_features = df.groupby(['state'])[feature].transform('median').fillna(median).copy()
            county_features = df.groupby(['county:state'])[feature].transform('median').fillna(state_features).copy()
            zipcode_features = df.groupby(['zipcode:county:state'])[feature].transform('median').fillna(county_features).copy()
            df[feature] = df[feature].fillna(zipcode_features)
    return df

def merge_school_features(features, public_school_dem, private_school_dem, on_zipcode=True, items=None):
    """
    merge school features from NCES
    """
    school_demographics = pd.concat([private_school_dem, public_school_dem], 0).drop_duplicates(subset=['school:zipcode:county:state'])
    fs = ['county:state', 'school:zipcode:county:state','zipcode:county:state', 'state', 'zipcode', 'black_ratio', 'hispanic_ratio', 'white_ratio', 'asian_ratio', 'mixed_ratio', 'native_ratio', 'pacific_islander_ratio', 'locale', 'school_size', 'pupil_ratio', 'fte_teachers', 'is_private', 'is_public', 'is_charter','is_magnet']
    school_demographics = school_demographics[fs].copy()
    school_demographics['is_charter'] = school_demographics.is_charter.astype(int).astype('category')
    school_demographics['is_magnet'] = school_demographics.is_magnet.astype(int).astype('category')
    school_demographics['is_private'] = school_demographics.is_private.astype(int).astype('category')
    school_demographics['is_public'] = school_demographics.is_public.astype(int).astype('category')

    school_demographics = impute(school_demographics)
    school_demographics = pd.concat([school_demographics, pd.get_dummies(school_demographics['locale'])], 1)

    school_demographics['city'] = school_demographics['11-City: Large'] + school_demographics['12-City: Mid-size'] + school_demographics['13-City: Small'] 
    school_demographics['suburb'] = school_demographics['21-Suburb: Large'] + school_demographics['22-Suburb: Mid-size'] + school_demographics['23-Suburb: Small'] 
    school_demographics['town'] = school_demographics['31-Town: Fringe'] + school_demographics['32-Town: Distant'] + school_demographics['33-Town: Remote'] 
    school_demographics['rural'] = school_demographics['41-Rural: Fringe'] + school_demographics['42-Rural: Distant'] + school_demographics['43-Rural: Remote'] 
    school_demographics = school_demographics.drop('locale', 1)
    if on_zipcode:
        school_demographics = school_demographics.groupby('zipcode').mean().reset_index()
        features = features.merge(school_demographics, how='left')
    else:
        mappings = get_mappings(school_demographics, items)
        school_demographics = school_demographics.merge(mappings, left_on='school:zipcode:county:state', right_on='value')
        features = features.merge(school_demographics.reset_index(), left_on='school:zipcode:county:state', right_on='key', how='left')
    return features

def merge_urban_features(features, path='county_stats - Pct urban by county.csv'):
    """
    merge urban/rural county features from census
    """
    urban_data = pd.read_csv(DATA_DIR / path).drop_duplicates(subset=['COUNTYNAME', 'STATENAME'])
    urban_data['base_county'] = urban_data['COUNTYNAME'].str.lower()
    urban_data['base_state'] = urban_data['STATENAME'].str.lower()
    features = features.merge(urban_data, on=['base_county', 'base_state'], how='left')
    return features

def merge_political_features(features, path='countypres_2000-2020.tsv'):
    """
    merge 20th century voting patterns (county-level)
    """
    politics = pd.read_csv(DATA_DIR / path, sep='\t')

    politics['county'] = politics.county_name.str.lower()
    politics['state'] = politics.state.str.lower()
    trump_votes = politics.loc[(politics.party.isin(["REPUBLICAN"])) & (politics['mode'] == 'TOTAL') & (politics['county'] != 'san joaquin')]
    trump_votes = trump_votes.groupby(['year','county', 'state']).apply(lambda x: x['candidatevotes'].sum() / (x['totalvotes'].sum() + 1))
    trump_votes = trump_votes.reset_index()

    years = {}
    for year in trump_votes.year.unique():
        years[year] =  trump_votes.loc[trump_votes.year == year].copy()
    for year, data in years.items():
        data[f'rep_share_{year}'] = data[0]
        data['base_county'] = data['county']
        data['base_state'] = data['state']
        data = data.drop(['county', 'state'], axis=1)
        features = features.merge(data,on=['base_county', 'base_state'], how='left')
    features = features.drop(['year_x', 'year_y', '0_x', '0_y'], axis=1)
    return features


def demographic_featurize(df):
    """
    main featurization function
    """
    print('initializing features...')
    features = initialize_features(df, level='school:county:state')
    print('merging zipcode features...')

    features = merge_zipcode_features(features)
    features = merge_income_features(features)

    items = features['school:zipcode:county:state'].unique().tolist()
    print('loading school data...')

    public_school_data = load_public_school_data() 
    private_school_data = load_private_school_data() 
    print('merging school data...')

    features = merge_school_features(features, public_school_data, private_school_data, on_zipcode=False, items=items)
    print('loading and merging county data...')
    
    features = merge_urban_features(features)
    features = merge_political_features(features)

    features['school:zipcode:county:state'] = features['school:zipcode:county:state_x']
    features['county:state'] = features['base_county'] +  ":" +  features['base_state']

    features = features.drop(['school:zipcode:county:state_x'], axis=1)
    features = features.drop_duplicates(subset=['school:zipcode:county:state'])
    return features

def preprocess(features, regression_features , impute_only=False):
    """
    basic preprocessing of features (e.g. outlier removal, logging, zscoring)
    """
    
    features = impute(features)
    features = features.loc[features.median_home_value < 1_000_000]
    if impute_only:
        return features
    

    for ix, feature in regression_features.iterrows():
        if feature['log']:
            features[feature['feature']] = np.log2(features[feature['feature']] + 1e-5)
        if feature['zscore']:
            features[feature['feature']] = stats.zscore(features[feature['feature']])
            if feature.get('remove_outliers'):
                features = features.loc[np.abs(features[feature['feature']]) < feature['remove_outliers']]
        else:
            if feature.get('remove_outliers'):
                features = features.loc[np.abs(stats.zscore(features[feature['feature']])) < feature['remove_outliers']]
        if feature.get('new_name'):
            features[feature['new_name']] = features[feature['feature']]
            features = features.drop([feature['feature']], axis=1)

    features['urban_top_30'] = (features['pct_rural'] < features.pct_rural.quantile(0.3)).astype('category')
    features['rural_top_30'] = (features['pct_rural'] > features.pct_rural.quantile(0.7)).astype('category')
    features['is_charter'] = features['is_charter'].fillna(0)
    features['is_magnet'] = features['is_magnet'].fillna(0)
    # features['is_private'] = features['is_private'].fillna(0)
    # features['is_public'] = features['is_public'].fillna(0)
    features = features.loc[features.school_size > -15]
    features = features.dropna(subset=['is_private', 'is_public'])
    features['pct_rural'] = features['pct_rural']/100
    return features



def text_featurize(df):
    """
    featurize text for the document-level regression
    """
    text_features = {}
    text_features['num_tokens'] = df.num_tokens
    text_features['prob_high_quality'] = df.prob_high_quality

    first_person_words = set(["i", "me", "my", "mine", "our", "ours", "us", "we", 'your', 'yours'])
    third_person_words = set(["he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"])
    text_features['first_person'] = df.text.progress_apply(lambda x: first_person_words & set(x.lower().split()))
    text_features['third_person'] = df.text.progress_apply(lambda x: third_person_words & set(x.lower().split()))

    text_features['topic'] = df.cluster

    text_features = pd.DataFrame(text_features)
    text_features['first_person'] = text_features['first_person'].progress_apply(lambda x: len(x) > 0)
    text_features['third_person'] = text_features['third_person'].progress_apply(lambda x: len(x) > 0)
    
    text_features['num_tokens'] = np.log2(text_features['num_tokens'] + 1e-5)
        
    return text_features




