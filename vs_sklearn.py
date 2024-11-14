import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression as my_logreg, OneVsRestClassifier as my_OvR, SoftmaxRegression as SR
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt

#TRAIN DATASET
dataset = pd.read_csv("datasets/dataset_train.csv")
dataset = dataset.dropna()

X = dataset.drop(columns=["Hogwarts House", "Index", "First Name", "Last Name", "Birthday"])
oe_X = OrdinalEncoder(categories=[["Left", "Right"]])
X["Best Hand"] = oe_X.fit_transform(X["Best Hand"].to_numpy().reshape(-1, 1))
print("---------X----------\n", X)
X = X.to_numpy()
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y = dataset[["Hogwarts House"]].to_numpy().flatten()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

#MY MODEL
my_model = my_OvR(my_logreg())
my_model.fit(X_train, y_train)

#SKLEARN MODEL
model = OneVsRestClassifier(LogisticRegression())
model.classes_ = ["Gryffindor", "Hufflepuff", "Slytherin", "Ravenclaw"]
model.fit(X_train, y_train)

#PREDICTIONS
#my_predictions = my_model.predict_ovr(X_test)
my_predictions = my_model.predict(X_test)
sk_predictions = model.predict(X_test)

print("Y_TEST\n", y_test)
print("MY PREDICTIONS\n", my_predictions)
print("SK_PREDICTIONS\n", sk_predictions)
print(f"MY_ACCURACY: {(y_test == my_predictions).mean() * 100}%")
print(f"SK_ACCURACY: {model.score(X_test, y_test) * 100}%")

#OPEN FILE
try:
    file = open("houses.csv", "w")
except Exception:
    print("Couldn't open file!")
    exit(4)

#WRITE PREDICTIONS TO FILE
file.write("Index,Hogwarts House")
for idx, pred in enumerate(my_predictions):
    file.write(f"{idx},{pred}\n")
    
file.close()

#SOFTMAX
oe = OrdinalEncoder(categories=[["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]])
my_softmax = SR()
y_train = oe.fit_transform(y_train.reshape(-1, 1)).flatten().astype(int) 
my_softmax.fit(X_train, y_train)
y_test = oe.fit_transform(y_test.reshape(-1, 1)).flatten().astype(int)
softmax_pred = my_softmax.predict(X_test)

acc = accuracy_score(y_test, softmax_pred)
print(f"ACCURACY: {acc * 100}%")