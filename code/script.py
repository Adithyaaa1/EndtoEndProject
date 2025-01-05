import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
df= pd.read_csv("train.csv")
df.columns= [col.upper().replace(".","_").strip() for col in df.columns]
ncols2 = df.select_dtypes(include=np.number).columns
ccols2 = df.select_dtypes(exclude=np.number).columns
nmiss2= [col for col in ncols2 if df[col].isnull().sum()>0]
def dropcols2(mcdf): # Initialize the function
    if "DISORDER" in mcdf.columns:
        mcdf.drop(["DISORDER"], axis= 1) # Drop the unnecessary rows
    elif "DISORDER" not in mcdf.columns:
        mcdf
    return mcdf # Return the modified DataFrame
dcfunc= FunctionTransformer(dropcols2) # Wrap the function in a function transformer
OHE2= OneHotEncoder(handle_unknown= "ignore", sparse_output= False) # Turn the categorical data into numeric data.
MFI2= SimpleImputer(strategy= "most_frequent") # Fill missing data with the most frequent value in that column
MI2= SimpleImputer(strategy= "median") # Fill missing data with the median of the column
def multiclassxgbpipe():
    prep= ColumnTransformer(transformers= [("nimp", MI2, nmiss2), ("catpipe", Pipeline(steps= [("imputer", MFI2), ("ohe", OHE2)]), ccols2)]
                    , remainder= "passthrough", verbose_feature_names_out= False).set_output(transform= "pandas")
    pipeline = Pipeline(steps=[
        ("data eng", dcfunc),
        ("preprocessor", prep),
        ("classifier", XGBClassifier(gamma= 3.380397544384568, 
                            colsample_bytree= 0.7654820824760737,
                            learning_rate= 0.08314149149079444,
                            max_depth=8,
                            min_child_weight=9,
                            n_estimators= 883,
                            eval_metric="logloss", random_state=42))]) 
    return pipeline