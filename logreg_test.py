import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import joblib
import sys

#CHECK ARGS
args = sys.argv
if len(args) != 3:
    print("Bad number of arguments")
    exit(1)
filename = args[1]
model_name = args[2]

#IMPORT MODEL
my_model = None
try:
    my_model = joblib.load(model_name)
except FileNotFoundError:
    print("File not found!")
    exit(2)

#LOAD FILE TO DATAFRAME
test_dataset = None
try:
    test_dataset = pd.read_csv(filename)
except FileNotFoundError:
    print("File not found!")
    exit(2)
    
#CLEAN DATASET 
test_dataset = test_dataset.dropna()
X = test_dataset.drop(columns=["Hogwarts House", "Index", "First Name", "Last Name", "Birthday"])
oe_X = OrdinalEncoder(categories=[["Left", "Right"]])
X["Best Hand"] = oe_X.fit_transform(X["Best Hand"].to_numpy().reshape(-1, 1))
X = X.to_numpy()
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y = test_dataset[["Hogwarts House"]].to_numpy().flatten()

#PREDICTIONS
my_predictions = my_model.predict_ovr(X)
print("Y_TEST\n", y)
print("PREDICTIONS\n", my_predictions)