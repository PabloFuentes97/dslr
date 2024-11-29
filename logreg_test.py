import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score
#from logistic_regression import OneVsRestClassifier, LogisticRegression
from itertools import combinations
import joblib
import sys
import matplotlib.pyplot as plt

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
    exit(3)
    
#CLEAN DATASET
test_dataset = test_dataset.fillna(0)
X = test_dataset[["Defense Against the Dark Arts", "Astronomy", "Charms", "Herbology"]]
#X.fillna(0)
X = X.to_numpy()

#SCALE DATA
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
#PREDICTIONS
my_predictions = my_model.predict(X_scaled)
#print(f"MY MODEL ACCURACY: {accuracy_score(y, my_predictions) * 100}%") 
    
#OPEN FILE
try:
    file = open("houses.csv", "w")
except Exception:
    print("Couldn't open file!")
    exit(4)

#WRITE PREDICTIONS TO FILE
file.write("Index,Hogwarts House\n")
for idx, pred in enumerate(my_predictions):
    file.write(f"{idx},{pred}\n")
    
file.close()