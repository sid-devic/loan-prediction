import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 0)
pd.set_option('display.max_rows', 500)
# read in loan data
csv = pd.read_csv('loan.csv', low_memory=False)

# plot for visualization
'''
plt.figure(figsize= (12,6))
plt.ylabel('Loan Status')
plt.xlabel('Count')
csv['loan_status'].value_counts().plot(kind = 'barh', grid = True) 
plt.show()
'''

# list of columns and there filled distribution
check_null = csv.isnull().sum(axis=0).sort_values(ascending=False)/float(len(csv))

# for pandas dataframes, 'inplace=True' means that we overwrite our current dataframe with the new one we generated
# by adding/removing values
# drop every column with <60% values
csv.drop(check_null[check_null>0.6].index,axis=1,inplace=True)

# drop all columns with >30 NaN values (empty data)
csv.dropna(axis=0,thresh=30,inplace=True)

# cols for deletion.
# zip_code: Irrelevant
# title: can't use
# emp_title: can't use
# acc_now_delinq: useless, doesn't tell us anything
# application_type: Most are INDEPENDENT, not filed as JOINT loans, so we might as well drop this anyways.
# url: irrel
# id: don't need, already have built in dataframe indexing
# member_id: don't need, this is for the original dataset creators
# pymnt_plan: will not matter for us
# policy_code and index: don't matter
deleted_cols = ['zip_code', 'title', 'emp_title', 'acc_now_delinq', 'application_type', 'url', 'id', 'member_id', 'pymnt_plan', 'policy_code']
csv.drop(deleted_cols, axis=1, inplace=True)

#print(csv.head())
#print(list(csv.columns.values))

# turn interest into a float
csv['int_rate'] = csv['int_rate'].str.split('%').str[0]
csv['int_rate'] = csv.int_rate.astype(float)/100.

# turn term (installments) into strict int
csv['term'] = csv['term'].str.split(' ').str[1]

# fill in missing employer length times with the median, 
# turn into strict year value
csv['emp_length'] = csv['emp_length'].str.extract('(\d+)').astype(float)
csv['emp_length'] = csv['emp_length'].fillna(csv.emp_length.median())

col_dates = csv.dtypes[csv.dtypes == 'datetime64[ns]'].index
for d in col_dates:
    csv[d] = csv[d].dt.to_period('M')

# dict of median values
median_dict = {}
non_ints =[]
for column in csv:
    try:
        median_dict[column] = csv[column].median()
    except Exception:
        non_ints.append(column)
        continue

print(median_dict)
print(non_ints)

csv['amt_difference'] = 'eq'
csv.loc[(csv['funded_amnt'] - csv['funded_amnt_inv']) > 0,'amt_difference'] = 'less'

csv['delinq_2yrs_cat'] = 'no'
csv.loc[csv['delinq_2yrs']> 0,'delinq_2yrs_cat'] = 'yes'

csv['inq_last_6mths_cat'] = 'no'
csv.loc[csv['inq_last_6mths']> 0,'inq_last_6mths_cat'] = 'yes'

csv['pub_rec_cat'] = 'no'
csv.loc[csv['pub_rec']> 0,'pub_rec_cat'] = 'yes'

# Create new metric
csv['acc_ratio'] = csv.open_acc / csv.total_acc

features = ['loan_amnt', 'amt_difference', 'term', 
            'installment', 'grade','emp_length',
            'home_ownership', 'annual_inc','verification_status',
            'purpose', 'dti', 'delinq_2yrs_cat', 'inq_last_6mths_cat', 
            'open_acc', 'pub_rec', 'pub_rec_cat', 'acc_ratio', 'initial_list_status',  
            'loan_status'
           ]

X_clean = csv.loc[csv.loan_status != 'Current', features]
X_clean.head()

mask = (X_clean.loan_status == 'Charged Off')
X_clean['target'] = 0
X_clean.loc[mask,'target'] = 1

cat_features = ['term','amt_difference', 'grade', 'home_ownership', 'verification_status', 'purpose', 'delinq_2yrs_cat', 'inq_last_6mths_cat', 'pub_rec_cat', 'initial_list_status']

# Drop any residual missing value (only 24)
X_clean.dropna(axis=0, how = 'any', inplace = True)

X = pd.get_dummies(X_clean[X_clean.columns[:-2]], columns=cat_features).astype(float)
y = X_clean['target']

