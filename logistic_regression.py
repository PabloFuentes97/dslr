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
        
    def fit(self, X, y, batch_size=None):       
        self.classes = np.unique(y)
        y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
        classes_n = y.shape[1]
        self.models = []
        for m in range(classes_n):
            y_sub = y[:, m]
            model_i = copy(self.classifier)
            model_i.fit(X, y_sub, batch_size=batch_size)
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
            batch_X = X
            batch_y = y
            n = len(X)
        else:
            n = batch_size
            
        for _ in range(self.iter_n):
            
            if batch_size:
                batch_sample = np.random.choice(samples_n, batch_size, replace=False)
                batch_X = X[batch_sample] 
                batch_y = y[batch_sample]
            
            y_pred = sigmoid(np.dot(batch_X, self.weights) + self.bias)
            
            dw = (1 / n) * np.dot(batch_X.T, (y_pred - batch_y)) 
            db = (1 / n) * np.sum(y_pred - batch_y) 
            
            self.weights = self.weights - dw * self.learningRate 
            self.bias = self.bias - db * self.learningRate
            
            if np.linalg.norm(dw) < self.tolerance:
                break
        
    def predict(self, X):
        return sigmoid(np.dot(X, self.weights) + self.bias)
    
class SoftmaxRegression:
    def __init__(self, lr=0.01, epochs=1000, tolerance=0.0001):
        self.weights = 0
        self.bias = 0
        self.lr = lr
        self.epochs = epochs
        self.tolerance = tolerance
    
    def fit(self, X, y, batch_size=None):
        num_classes = len(np.unique(y))
        num_features = X.shape[1]
        self.weights = np.random.randn(num_features, num_classes)
        self.bias = 1
        for _ in range(self.epochs):
            logits = np.dot(X, self.weights) + self.bias
    
            exp_logits = np.exp(logits)
            softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True) #get a normal distribution -> exp / sum_exp to normalize
            
            dw = np.dot(X.T, softmax_probs - np.eye(num_classes)[y])
            db = np.sum(softmax_probs - np.eye(num_classes)[y])
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if np.linalg.norm(dw) < self.tolerance:
                break
    
    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        exp_logits = np.exp(logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        y_pred = np.argmax(softmax_probs, axis=1)
        return y_pred