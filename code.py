import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
%matplotlib inline
file_name='house-price-data.csv'
df=pd.read_csv(file_name)
df.dtypes
df.drop(df[["id","Unnamed: 0"]], axis=1, inplace=True)
df.describe()
missing_data = df.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")
mean_bathrooms = df["bathrooms"].astype("float").mean(axis=0)
df["bathrooms"].replace(np.nan, mean_bathrooms, inplace=True)
mean_bedrooms = df["bedrooms"].astype("float64").mean(axis=0)
df["bedrooms"].replace(np.nan,mean_bedrooms,inplace=True)
df["bedrooms"].isnull().sum()
df["bathrooms"].isnull().sum()
df_houses_floors = df["floors"].value_counts().to_frame()
df_houses_floors.rename(columns = {"floors":"No. of Houses"}, inplace=True)
df_houses_floors.index.name = "Unique Floors"
df_houses_floors
sns.boxplot(x='waterfront',y='price',data=df)
plt.title("No Waterfront VS With Waterfront")
plt.show()
sns.regplot(x='sqft_above',y='price',data=df)
plt.title("SQFT_ABOVE VS PRICE")
plt.show()
df.corr()["price"].sort_values()
x = df[["long"]]
y = df["price"]
lm = LinearRegression()
lm.fit(x,y)
lm.score(x,y)
x = df[["sqft_living"]]
y = df["price"]
lm = LinearRegression()
lm.fit(x,y)
lm.score(x,y)
features =df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"] ]
lm_multi = LinearRegression()
lm_multi.fit(features,y)
lm_multi.score(features,y)
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(features,y)
pipe.score(features,y)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']
x_train , x_test , y_train , y_test = train_test_split (X , Y , test_size = 0.15 , random_state = 1)
x_test.shape[0]
x_train.shape[0]
from sklearn.linear_model import Ridge
RidgeModel = Ridge (alpha = 0.1)
RidgeModel.fit (x_train , y_train)
RidgeModel.score(x_test,y_test)
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train)
x_test_pr=pr.fit_transform(x_test)
pollyfit = Ridge(alpha=0.1)
pollyfit.fit(x_train_pr , y_train)
pollyfit.score(x_test_pr,y_test)
