import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import math
from copy import copy

def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()
    return loss

def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # To avoid log(0)
    y_true = OneHotEncoder().fit_transform(y_true.reshape(-1, 1)).toarray()
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    num_classes = len(np.unique(y_true))
    return -np.sum(y_true * np.log(y_pred + 10 ** -100))

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

    def predict_proba(self, X):
        y_prob = np.zeros(shape=(X.shape[0], len(self.classes)))
        
        for m in range(len(self.classes)):
            proba = self.models[m].predict(X)
            y_prob[:, m] = proba 
        
        return y_prob / np.sum(y_prob, axis=1, keepdims=True)
    
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
            
            batch_sample = np.random.choice(samples_n, batch_size, replace=False)
            batch_X = X[batch_sample] 
            batch_y = y[batch_sample]
            
            y_pred = sigmoid(np.dot(batch_X, self.weights) + self.bias)
            
            dw = (1 / batch_size) * np.dot(batch_X.T, (y_pred - batch_y)) 
            db = (1 / batch_size) * np.sum(y_pred - batch_y) 
            
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
        # Number of classes
        num_classes = len(np.unique(y))
        # Initialize weights randomly
        num_features = X.shape[1]
        self.weights = np.random.randn(num_features, num_classes)
        self.bias = 1
        # Training loop
        for _ in range(self.epochs):
            # Compute logits (linear combinations)
            logits = np.dot(X, self.weights) + self.bias
            
            # Apply softmax function to logits
            exp_logits = np.exp(logits)
            softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True) #get a normal distribution -> exp / sum_exp to normalize
            
            # Compute gradient of cross-entropy loss with respect to weights and bias
            dw = np.dot(X.T, softmax_probs - np.eye(num_classes)[y])
            db = np.sum(softmax_probs - np.eye(num_classes)[y])
            
            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        exp_logits = np.exp(logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        np.set_printoptions(precision=5, suppress=True)
        y_pred = np.argmax(softmax_probs, axis=1)
        return y_pred