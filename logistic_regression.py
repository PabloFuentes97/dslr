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

class OneVsRestClassifier:
    def __init__(self, classifier):
        self.classifier = classifier
        
    def fit(self, X, y, batch_size=None):       
        self.classes = np.unique(y)
        y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
        n_labels = y.shape[1]
        self.models = []
        for m in range(n_labels):
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
        print("PROBS:", softmax_probs)
        y_pred = np.argmax(softmax_probs, axis=1)
        print("PREDS:", y_pred)
        return y_pred
        
class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0, epochs=1000):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay
        self.epochs = epochs
        self.weights = 0
        self.bias = 0
        
    def fit(self, X, y, batch_size=32, stopping_threshold=1e-6, momentum=0.9):
        #init parameters
        samples_n, features_n = X.shape
        self.weights = np.zeros(features_n)
        self.bias = 0
        previous_loss = np.inf
        # Initialize first and second moment vectors
        #first moment -> It is an estimate of the mean (first moment) of gradients.
            #Calculated as an exponentially decaying average of past gradients.
            #Helps in estimating the direction of the gradient.
        #second moment -> It’s an estimate of the uncentered variance (second moment) of the gradients.
            #Also calculated as an exponentially decaying average, but of squared gradients.
            #Helps in adapting the learning rate for each parameter.
    
        t = 1  # Initialize timestep
        for _ in self.epochs:
            
            batch_sample = np.random.choice(samples_n, batch_size, replace=False)
            batch_X = X[batch_sample] 
            batch_y = y[batch_sample]
            
            y_pred = sigmoid(np.dot(batch_X, self.weights) + self.bias)
            
            dw = (1 / batch_size) * np.dot(batch_X.T, (y_pred - batch_y)) 
            db = (1 / batch_size) * np.sum(y_pred - batch_y) 
            dw = np.clip(dw, -1, 1)
            db = np.clip(db, -1, 1)
            
            # Update biased first moment estimate
            m_w = self.beta1 * m_w + (1 - self.beta1) * dw
            m_b = self.beta1 * m_b + (1 - self.beta1) * db
            # Update biased second raw moment estimate
            v_w = self.beta2 * v_w + (1 - self.beta2) * (dw ** 2)
            v_b = self.beta2 * v_b + (1 - self.beta2) * (db ** 2)
            # Compute bias-corrected first moment estimate
            m_m_hat = m_w / (1 - self.beta1 **t)
            m_b_hat = m_b / (1 - self.beta1 **t)
            # Compute bias-corrected second raw moment estimate
            v_m_hat = v_w / (1 - self.beta2 **t)
            v_b_hat = v_b / (1 - self.beta2 **t)
            # Update parameters
            self.weights -= self.lr * m_m_hat / (np.sqrt(v_m_hat) + self.epsilon)
            self.bias -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
            
            curr_loss = binary_cross_entropy(y_pred, y)
            if abs(previous_loss - curr_loss) < stopping_threshold:
                break
            previous_loss = curr_loss
            t += 1   
            
class NewtonMethod:
    def __init__(self, lr=0.01, epochs=1000):
        self.weights = 0
        self.bias = 0
        self.lr = lr
        self.epochs = epochs
      
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
            
            # Compute Hessian matrix
            hessian_b_b = np.mean(y_pred * (1 - y_pred))
            hessian_w_w = np.mean(X ** 2 * y_pred * (1 - y_pred))
            hessian_b_w = np.mean(X * y_pred * (1 - y_pred))
            
            hessian_inv = np.linalg.inv([[hessian_b_b , hessian_b_w],
                                 [hessian_b_w, hessian_w_w ]])
            
            # Update weights
            grad = np.array([db, dw])
            delta_w = np.dot(hessian_inv, grad)

            # Check convergence
            if np.linalg.norm(delta_w) < self.tolerance:
                break

            # Update weights
            self.bias -= delta_w[0]
            self.weights -= delta_w[1]

            # Check for improvement in convergence
            if np.linalg.norm(delta_w - prev_delta_w) < self.tolerance:
                break

            # Update previous delta_w for next iteration
            prev_delta_w = delta_w
        
    def predict(self, X):
        return sigmoid(np.dot(X, self.weights) + self.bias)   