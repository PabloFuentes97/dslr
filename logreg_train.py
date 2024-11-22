import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression, OneVsRestClassifier

#CHECK ARGS
args = sys.argv
if len(args) != 2:
    print("Bad number of arguments")
    exit(1)

filename = args[1]

#LOAD FILE TO DATAFRAME
try:
    train_dataset = pd.read_csv(filename)
except FileNotFoundError:
    print("File not found!")
    exit(2)
    
#CLEAN DATASET 
train_dataset = train_dataset.dropna()
X = train_dataset.drop(columns=["Hogwarts House", "Index", "First Name", "Last Name", "Birthday", 
                            "Best Hand", "Arithmancy", "Care of Magical Creatures", "Potions"])
X = X.to_numpy()
scaler_X = StandardScaler()

#SCALE DATA
X_scaled = scaler_X.fit_transform(X)
y = train_dataset[["Hogwarts House"]].to_numpy().flatten()

#TRAIN DATASET
model = OneVsRestClassifier(LogisticRegression())
model.fit(X_scaled, y)

#SERIALIZE MODEL
joblib.dump(model, "my_model")

