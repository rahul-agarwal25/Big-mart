# -*- coding: utf-8 -*-
"""
Created on Sun May  4 08:22:08 2025

@author: Rahul
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  2 12:47:00 2025

@author: Rahul
"""

import pandas as pd
import sweetviz
import shap
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt  

df1 = pd.read_csv(r'C:\Users\Rahul\Downloads\train_v9rqX0R.csv')





from sklearn.preprocessing import OrdinalEncoder




df1['Item_Fat_Content'] = df1['Item_Fat_Content'].replace(to_replace=['reg'], value='Regular')

df1['Item_Fat_Content'] = df1['Item_Fat_Content'].replace(to_replace=['LF'], value='Low Fat')

df1['Item_Fat_Content'] = df1['Item_Fat_Content'].replace(to_replace=['low fat'], value='Low Fat')



df1['Item_Fat_Content'].value_counts()


df2 = pd.read_csv(r'C:\Users\Rahul\Downloads\test_AbJTz2l.csv')

df2.isna().sum()

df2['Item_Fat_Content'] = df2['Item_Fat_Content'].replace(to_replace=['reg'], value='Regular')

df2['Item_Fat_Content'] = df2['Item_Fat_Content'].replace(to_replace=['LF'], value='Low Fat')

df2['Item_Fat_Content'] = df2['Item_Fat_Content'].replace(to_replace=['low fat'], value='Low Fat')


df1['Item_Number'] = df1['Item_Identifier'].str[-2:].astype(int)
df2['Item_Number'] = df2['Item_Identifier'].str[-2:].astype(int)

encoder = OrdinalEncoder(categories=[["Low Fat","Regular"]])  
df1["Item_Fat_Content_encoded"] = encoder.fit_transform(df1[["Item_Fat_Content"]])

df1=df1.drop(columns = "Item_Fat_Content")

encoder = OrdinalEncoder(categories=[["Low Fat","Regular"]])  
df2["Item_Fat_Content_encoded"] = encoder.fit_transform(df2[["Item_Fat_Content"]])

df2=df2.drop(columns = "Item_Fat_Content")


#####missing value for item weight

df1['Item_Weight'] = df1['Item_Weight'].fillna(
    df1.groupby('Item_Identifier')['Item_Weight'].transform('mean'))
df1['Item_Weight'] = df1['Item_Weight'].fillna(df1['Item_Weight'].mean())


item_weight_map = df1.groupby('Item_Identifier')['Item_Weight'].mean()


df2['Item_Weight'] = df2.apply(
    lambda row: item_weight_map[row['Item_Identifier']] if pd.isnull(row['Item_Weight']) else row['Item_Weight'],
    axis=1
)

df1.isna().sum()
df2.isna().sum()

df1['Outlet_Size'] = df1['Outlet_Size'].fillna("Medium")


encoder = OrdinalEncoder(categories=[["Small", "Medium", "High"]])  
df1["Outlet_Size_encoded"] = encoder.fit_transform(df1[["Outlet_Size"]])
df1=df1.drop(columns = "Outlet_Size")



df2['Outlet_Size'] = df2['Outlet_Size'].fillna("Medium")


encoder = OrdinalEncoder(categories=[["Small", "Medium", "High"]])  
df2["Outlet_Size_encoded"] = encoder.fit_transform(df2[["Outlet_Size"]])
df2=df2.drop(columns = "Outlet_Size")



df1['Item_Visibility'].replace(0, np.nan, inplace=True)
visibility_means = df1.groupby('Item_Identifier')['Item_Visibility'].mean()
df1['Item_Visibility'] = df1.apply(
    lambda row: visibility_means[row['Item_Identifier']] if pd.isna(row['Item_Visibility']) else row['Item_Visibility'],
    axis=1
)


df2['Item_Visibility'].replace(0, np.nan, inplace=True)
#visibility_means = df2.groupby('Item_Identifier')['Item_Visibility'].mean()
df2['Item_Visibility'] = df2.apply(
    lambda row: visibility_means[row['Item_Identifier']] if pd.isna(row['Item_Visibility']) else row['Item_Visibility'],
    axis=1
)

print(df1[df1["Item_Visibility" ]== 0])


df1["Age"] = 2013 - df1["Outlet_Establishment_Year"]
df1=df1.drop(columns="Outlet_Establishment_Year")

df2["Age"] = 2013 - df2["Outlet_Establishment_Year"]
df2=df2.drop(columns="Outlet_Establishment_Year")

df1['outlet_number'] = df1['Outlet_Identifier'].str[-2:].astype(int)
df2['outlet_number'] = df2['Outlet_Identifier'].str[-2:].astype(int)





df1['Item_broad_category'] = df1['Item_Identifier'].apply(lambda x: x[:2])
df1['Item_broad_category'] = df1['Item_broad_category'].map({'FD': 'Food', 'NC': 'NC', 'DR': 'Bev'})
df1=df1.drop(columns="Item_Identifier")

from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()
df1['Item_Type'] = le.fit_transform(df1['Item_Type'])

le = LabelEncoder()
df1['Item_broad_category'] = le.fit_transform(df1['Item_broad_category'])

#df1 = pd.get_dummies(df1, columns=['Item_Type'], drop_first=True)
#df1 = pd.get_dummies(df1, columns=['Item_broad_category'], drop_first=True)
#df1 = pd.get_dummies(df1, columns=['Outlet_Identifier'], drop_first=True)

encoder = OrdinalEncoder(categories=[["Grocery Store","Supermarket Type1", "Supermarket Type2","Supermarket Type3"]])  
df1["Outlet_Type_encoded"] = encoder.fit_transform(df1[["Outlet_Type"]])
df1=df1.drop(columns = "Outlet_Type")


#df1 = pd.get_dummies(df1, columns=['Outlet_Type'], drop_first=True)

encoder = OrdinalEncoder(categories=[["Tier 1","Tier 2", "Tier 3"]])  
df1["Outlet_Location_Type_encoded"] = encoder.fit_transform(df1[["Outlet_Location_Type"]])
df1=df1.drop(columns = "Outlet_Location_Type")



df2['Item_broad_category'] = df2['Item_Identifier'].apply(lambda x: x[:2])
df2['Item_broad_category'] = df2['Item_broad_category'].map({'FD': 'Food', 'NC': 'NC', 'DR': 'Bev'})
df2=df2.drop(columns="Item_Identifier")

le = LabelEncoder()
df2['Item_Type'] = le.fit_transform(df2['Item_Type'])

le = LabelEncoder()
df2['Item_broad_category'] = le.fit_transform(df2['Item_broad_category'])

#df2 = pd.get_dummies(df2, columns=['Item_Type'], drop_first=True)
#df2 = pd.get_dummies(df2, columns=['Item_broad_category'], drop_first=True)
#df2 = pd.get_dummies(df2, columns=['Outlet_Identifier'], drop_first=True)

encoder = OrdinalEncoder(categories=[["Grocery Store","Supermarket Type1", "Supermarket Type2","Supermarket Type3"]])  
df2["Outlet_Type_encoded"] = encoder.fit_transform(df2[["Outlet_Type"]])
df2=df2.drop(columns = "Outlet_Type")



#df2 = pd.get_dummies(df2, columns=['Outlet_Type'], drop_first=True)

encoder = OrdinalEncoder(categories=[["Tier 1","Tier 2", "Tier 3"]])  
df2["Outlet_Location_Type_encoded"] = encoder.fit_transform(df2[["Outlet_Location_Type"]])
df2=df2.drop(columns = "Outlet_Location_Type")




missing_rows = df1[df1['Item_Weight'].isnull()]
print(missing_rows)

df1= df1.drop(columns="Outlet_Identifier")
df2= df2.drop(columns="Outlet_Identifier")

#y = np.log1p(df1['Item_Outlet_Sales'])
y = df1['Item_Outlet_Sales']
x = df1.drop(columns="Item_Outlet_Sales")

df2.isna().sum()


##################################################
#########################################################
######################################################
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso 
from sklearn.linear_model import ElasticNet 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


base_models = [
   # ('xgb', make_pipeline(StandardScaler(), XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbosity=0))),
    ('lgbm', make_pipeline(StandardScaler(), LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42))),
   # ('rf', make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))),
  # ('gbr', make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))),
    ('dt', make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=6, random_state=42))),
  # ('ada', make_pipeline(StandardScaler(), AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42))),
 ('et', make_pipeline(StandardScaler(), ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42)))
]


meta_model = make_pipeline(StandardScaler(),Ridge())


stacked_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

stacked_model.fit(X_train, y_train)


preds_stacked = stacked_model.predict(df2)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f"Stacked Regressor RMSE (with scaling): {rmse:.2f}")
preds = stacked_model.predict(df2)


preds = voting_model.predict(df2)

from sklearn.ensemble import VotingRegressor

voting_model = VotingRegressor(estimators=base_models,weights = [0,1,0,2,0,0,1], n_jobs=-1)


voting_model.fit(X_train, y_train)
preds = voting_model.predict(X_test)
preds=abs(preds)

rmse = mean_squared_error(y_test, preds, squared=False) 


##################################################
#########################################################
######################################################

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3, 0.5],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [1, 1.5, 2, 3]
}


xgb = XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0)


random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=76,  # More iterations give better results (50 is a good start)
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=2,
    n_jobs=-1
)


random_search.fit(X_train, y_train)


best_model = random_search.best_estimator_
preds = best_model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
preds = best_model.predict(df2)
#preds = np.expm1(preds)
#preds = np.where(preds < 0, 0, preds)
preds=abs(preds)


##################################################
#########################################################
######################################################

from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import randint as sp_randint, uniform


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


lgb = LGBMRegressor(random_state=42)


param_dist = {
    'n_estimators': sp_randint(100, 1000),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': sp_randint(3, 15),
    'num_leaves': sp_randint(20, 150),
    'min_child_samples': sp_randint(10, 100),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}


random_search = RandomizedSearchCV(
    estimator=lgb,
    param_distributions=param_dist,
    n_iter=50,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)


random_search.fit(X_train, y_train)


best_model = random_search.best_estimator_
preds = best_model.predict(df2)

preds_test = best_model.predict(X_test)
preds_0 = np.where(preds < 0, 0, preds)
preds_abs=abs(preds)
rmse = mean_squared_error(y_test, preds_test, squared=False)

importances = best_model.feature_importances_
feature_names = x.columns  


feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Best Parameters:", random_search.best_params_)
print(f"LightGBM RMSE after tuning: {rmse:.2f}")
