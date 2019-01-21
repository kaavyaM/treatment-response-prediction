## Python version

"""
Import necessary packages
"""
import pandas as pd
from sklearn import preprocessing

"""
Some intro Data reading and description
"""
data = pd.read_csv('data-before-processing.csv')

"""
Print data descriptors
"""
print(data.shape)

# print how many null values are in each column
print(data.isnull().sum())

"""
Data imputing: Fill in all null values with the median of the column
"""
data.fillna(data.median(), inplace=True)

"""
Implements one-hot-encoding for all genes
"""
genes = data.loc[:, 'ABCB1rs10245483':'TRIM55rs7350113']
for column in genes:
    data = pd.get_dummies(data, prefix = column + "_", columns = [column], drop_first = True)

genes2 = data.loc[:, 'ABCB1rs2032583':'TPH1rs211101']
for column in genes2:
    data = pd.get_dummies(data, prefix=column + "_", columns=[column], drop_first=True)

"""
Normalize
"""
all_values = data.values
min_max_scaler = preprocessing.MinMaxScaler()
all_values = min_max_scaler.fit_transform(all_values)
data.iloc[:, :] = all_values

data.to_csv('preprocessed.csv')