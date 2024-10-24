import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import math
from copy import copy

def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

class OneVsRestClassifier:
    def __init__(self, classifier):
        self.classifier = classifier
        
    def fit(self, X, y):       
        self.classes = np.unique(y)
        y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
        n_labels = y.shape[1]
        self.models = []
        for m in range(n_labels):
            y_sub = y[:, m]
            model_i = copy(self.classifier)
            model_i.fit(X, y_sub)
            self.models.append(model_i)
            
    def predict(self, X):
        y_prob = np.zeros(shape=(X.shape[0], len(self.classes)))
        
        for m in range(len(self.classes)):
            proba = self.models[m].predict(X)
            y_prob[:, m] = proba

        num_predictions = np.argmax(y_prob, axis=1)
        return [self.classes[i] for i in num_predictions]

class LogisticRegression:
    def __init__(self, learningRate=0.01, iter_n=1000, tolerance=0.0001):
        self.iter_n = iter_n
        self.weights = 0
        self.bias = 0
        self.learningRate = learningRate
        self.tolerance = tolerance    
    
    def fit(self, X, y, batch_size=None):
        samples_n, features_n = X.shape
        self.weights = np.zeros(features_n)
        self.bias = 0
        
        if not batch_size:
            batch_size = len(X)
            
        for _ in range(self.iter_n):
            
            #CREATE BATCH
            batch_sample = np.random.choice(samples_n, batch_size, replace=False)
            batch_X = X[batch_sample] 
            batch_y = y[batch_sample]
            
            y_pred = sigmoid(np.dot(batch_X, self.weights) + self.bias)
            dw = (1 / batch_size) * np.dot(batch_X.T, (y_pred - batch_y)) * self.learningRate 
            db = (1 / batch_size) * np.sum(y_pred - batch_y) * self.learningRate
            
            self.weights = self.weights - dw
            self.bias = self.bias - db
            
            if np.linalg.norm(dw) < self.tolerance:
                break
        
    def predict(self, X):
        return sigmoid(np.dot(X, self.weights) + self.bias)