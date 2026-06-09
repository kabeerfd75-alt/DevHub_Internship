import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes":1,"No":0})

num_cols = X.select_dtypes(include=["int64","float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer([
("num", Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]), num_cols),
("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])

pipeline = Pipeline([
("preprocessor", preprocessor),
("classifier", LogisticRegression(max_iter=1000))
])

grid = GridSearchCV(pipeline, {"classifier__C":[0.1,1,10]}, cv=3)
grid.fit(X,y)

joblib.dump(grid.best_estimator_, "churn_pipeline.pkl")
print("Pipeline exported successfully")
