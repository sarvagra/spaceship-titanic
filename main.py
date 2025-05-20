# %% Importing packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer  

# %% Read the train and test data, print the first 5 rows
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Add placeholder for 'Transported' in test set
df_test['Transported'] = False  

# Combine both datasets for preprocessing
df = pd.concat([df_train, df_test], sort=False)
df.head()

# %% Find the number of missing values in each column
print(df.isna().sum())

# %% Split the 'Cabin' column into 'Deck', 'Number', and 'Side'
df[['Deck', 'Number', 'Side']] = df['Cabin'].str.split('/', expand=True)
df.drop(columns=['Cabin'], inplace=True)
df.head()

# %% Deal with missing values in 'Deck', 'Number', and 'Side'
df['Deck'] = df['Deck'].fillna('U')
df['Number'] = df['Number'].fillna('-1')
df['Side'] = df['Side'].fillna('U')

# %% Replace categorical values with numerical values
df['Side'] = df['Side'].map({'P': 0, 'S': 1, 'U': 2})
df['Deck'] = df['Deck'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7, 'U': 8})

# %% Drop columns that are not needed
df.drop(columns=['Name','PassengerId'], inplace=True)

# %% Prepare for KNN imputation
# Convert 'Number' to numeric
df['Number'] = pd.to_numeric(df['Number'], errors='coerce')

# Select columns to impute
impute_list = ['Age', 'VIP', 'Number', 'CryoSleep', 'Side', 'Deck','RoomService','FoodCourt','Spa','ShoppingMall','VRDeck','PassengerId']
rest = [col for col in df.columns if col not in impute_list]

# Separate non-impute and impute parts
df_rest = df[rest]
df_to_impute = df[impute_list]

# KNN imputation
imp = KNNImputer()
df_imputed = imp.fit_transform(df_to_impute)
df_imputed = pd.DataFrame(df_imputed, columns=impute_list)

# Concatenate imputed and other data
df = pd.concat([df_rest.reset_index(drop=True), df_imputed.reset_index(drop=True)], axis=1)

# %% Final check
df.head()

# %%
df.isna().sum()
# %%
df['HomePlanet']=df['HomePlanet'].fillna('U')
df['Destination']=df['Destination'].fillna('U')

category_cols=['HomePlanet','Destination']
for col in category_cols:
    df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
df=df.drop(columns=category_cols)

# %% Feature engineering
bill_cols = ['RoomService', 'FoodCourt', 'Spa', 'ShoppingMall', 'VRDeck']

# Fill missing values in bill columns with 0 before calculations
df[bill_cols] = df[bill_cols].fillna(0)

# Create new features
df['amt_spent'] = df[bill_cols].sum(axis=1)        # Total amount spent
df['std_amt_spent'] = df[bill_cols].std(axis=1)    # Std deviation of spending
df['mean_amt_spent'] = df[bill_cols].mean(axis=1)  # Mean spending

# %%
df['3_high_cols']=df['CryoSleep']+df['HomePlanet_Europa']+df['Destination_55 Cancri e']
df['3_low_cols']=df['mean_amt_spent']+df['amt_spent']+df['HomePlanet_Earth']


# %% finding correlation
df.corr()['Transported'].sort_values(ascending=False)
