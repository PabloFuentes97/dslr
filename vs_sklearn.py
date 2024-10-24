import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression as my_logreg, OneVsRestClassifier as my_OvR

#TRAIN DATASET
dataset = pd.read_csv("datasets/dataset_train.csv")
dataset = dataset.dropna()
print("--------DATASET-------\n", dataset)

X = dataset.drop(columns=["Hogwarts House", "Index", "First Name", "Last Name", "Birthday"])
oe_X = OrdinalEncoder(categories=[["Left", "Right"]])
X["Best Hand"] = oe_X.fit_transform(X["Best Hand"].to_numpy().reshape(-1, 1))
print("---------X----------\n", X)
X = X.to_numpy()
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

y = dataset[["Hogwarts House"]].to_numpy().flatten()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

#y_encoder = OrdinalEncoder(categories=[["Gryffindor", "Hufflepuff", "Slytherin", "Ravenclaw"]])
#y = y_encoder.fit_transform(y)
#y = y.flatten()
print("--------Y---------\n", y)

#MY MODEL
my_model = my_OvR(my_logreg())
#my_model = my_logreg(classes=["Gryffindor", "Hufflepuff", "Slytherin", "Ravenclaw"])
#my_model.fit_ovr(X_train, y_train)
my_model.fit(X_train, y_train)

#SKLEARN MODEL
model = OneVsRestClassifier(LogisticRegression())
#model = LogisticRegression(max_iter=1000, multi_class="ovr")
model.classes_ = ["Gryffindor", "Hufflepuff", "Slytherin", "Ravenclaw"]
model.fit(X_train, y_train)

#PREDICTIONS
#my_predictions = my_model.predict_ovr(X_test)
my_predictions = my_model.predict(X_test)
sk_predictions = model.predict(X_test)

'''
sk_proba = model.predict_proba(X_test)
print("SK_PROBA:")
for proba in sk_proba:
    print("[", end="")
    proba_n = len(proba)
    for idx, p in enumerate(proba):
        print(f"{p:,.6f}", end="")
        if idx < proba_n - 1:
            print(" ", end="")
    print("]", "=", np.sum(proba))
'''

print("Y_TEST\n", y_test)
print("MY PREDICTIONS\n", my_predictions)
print("SK_PREDICTIONS\n", sk_predictions)
print(f"MY_ACCURACY: {(y_test == my_predictions).mean() * 100}%")
print(f"SK_ACCURACY: {model.score(X_test, y_test) * 100}%")

#TEST DATASET
'''
test_dataset = pd.read_csv("datasets/dataset_test.csv")

X_test = test_dataset.drop(columns=["Hogwarts House", "Index", "First Name", "Last Name", "Birthday"]).fillna(0)
print(X_test)
X_test["Best Hand"] = oe_X.fit_transform(X_test["Best Hand"].to_numpy().reshape(-1, 1))
scaled_test_X = scaler_X.fit_transform(X_test)

y_test = test_dataset[["Hogwarts House"]].to_numpy()
#y_test = y_encoder.fit_transform(y_test)
y_test = y_test.flatten()

predictions = model.predict(X_test)
print("---------PREDICTIONS------------\n", predictions)
'''
