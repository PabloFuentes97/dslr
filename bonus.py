import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logistic_regression as logreg
from sklearn.metrics import accuracy_score

#TRAIN DATASET
dataset = pd.read_csv("dataset_train.csv")
dataset = dataset.dropna()

X = dataset.drop(columns=["Hogwarts House", "Index", "First Name", "Last Name", "Birthday", 
                            "Best Hand", "Arithmancy", "Care of Magical Creatures"])

X = X.to_numpy()
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y = dataset[["Hogwarts House"]].to_numpy().flatten()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

#MY MODEL
my_model = logreg.OneVsRestClassifier(logreg.LogisticRegression())
my_model.fit(X_train, y_train)

#SKLEARN MODEL
sk_model = OneVsRestClassifier(LogisticRegression())
sk_model.classes_ = ["Gryffindor", "Hufflepuff", "Slytherin", "Ravenclaw"]
sk_model.fit(X_train, y_train)

#PREDICTIONS
my_predictions = my_model.predict(X_test)
sk_predictions = sk_model.predict(X_test)

print(f"MY_ACCURACY: {(y_test == my_predictions).mean() * 100}%")
print(f"SK_ACCURACY: {sk_model.score(X_test, y_test) * 100}%")
print(f"MY MODEL ACCURACY: {accuracy_score(y_test, my_predictions) * 100}%")

#SOFTMAX
oe = OrdinalEncoder(categories=[["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]])
my_softmax = logreg.SoftmaxRegression()
y_train = oe.fit_transform(y_train.reshape(-1, 1)).flatten().astype(int) 
my_softmax.fit(X_train, y_train)
y_test = oe.fit_transform(y_test.reshape(-1, 1)).flatten().astype(int)
softmax_pred = my_softmax.predict(X_test)

acc = accuracy_score(y_test, softmax_pred)
print(f"SOFTMAX ACCURACY: {acc * 100}%")


