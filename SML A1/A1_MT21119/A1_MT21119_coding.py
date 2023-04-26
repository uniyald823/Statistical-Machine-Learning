#!/usr/bin/env python
# coding: utf-8

# ### SML ASSIGNMENT 1

## DRISHYA UNIYAL---------------MT21119


# example of loading the mnist dataset
from keras.datasets import mnist
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


def Q1():
#     # load dataset
#     (X_train, y_train), (X_test, y_test) = mnist.load_data()
#     # summarize loaded dataset
#     print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
#     print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))
#     # plot first few images

    mnist_train=pd.read_csv("C:/Users/HP/Desktop/mnist_train.csv")
    mnist_test=pd.read_csv("C:/Users/HP/Desktop/mnist_test.csv")

    y_train=mnist_train.iloc[:,0]
    X_train=mnist_train.iloc[:,1:785]

    y_test=mnist_test.iloc[:,0]
    X_test=mnist_test.iloc[:,1:785]

    idx = np.argsort(y_train)
    x_train_sorted = X_train[idx]
    y_train_sorted = X_train[idx]

    for i in range(0,10):
        x_train_ones1 = X_train[y_train == i]
        for j in range(0,5):
            pyplot.subplot(330 + 1 + j)
            pyplot.imshow(x_train_ones1[j], cmap=pyplot.get_cmap('gray'))
        pyplot.show()
 
def sklearn_lib():
    
    mnist_train=pd.read_csv("C:/Users/HP/Desktop/mnist_train.csv")
    mnist_test=pd.read_csv("C:/Users/HP/Desktop/mnist_test.csv")

    y_train=mnist_train.iloc[:,0]
    X_train=mnist_train.iloc[:,1:785]

    y_test=mnist_test.iloc[:,0]
    X_test=mnist_test.iloc[:,1:785]
    
    lda = LinearDiscriminantAnalysis(n_components=9)
    X_train_r2 = lda.fit(X_train, y_train).transform(X_train)

    qda = QuadraticDiscriminantAnalysis()
    X_train_r2 = qda.fit(X_train, y_train)


    y_pred = lda.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    y_pred = qda.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    
    
    
   
class LDA:
    def __init__(self, reg_covar=0.01):
        self.reg_covar = reg_covar
        self.class_mean = None
        self.covariance = None
        self.priors = None
        self.classes = None
        
    def fit(self, X, y):
        """
        Fit LDA model to the training data
        """
        n_samples, self.D = X.shape
        self.classes = np.unique(y)
        n_classes = self.classes.shape[0]
        
        class_means = []
        class_covs = []
        class_covs_inv = []
        class_samples = []
        self.priors = np.zeros(n_classes)

       
        # calculate mean and covariance matrix
        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / float(n_samples) #calculate priors
            class_means.append(X_c.mean(axis=0))  #calculate means for the classes
            cov = np.zeros((self.D, self.D))  #initially the matrix is zero
            for i in range(X_c.shape[0]):
                row = X_c[i, :].reshape(self.D, 1)
                cov += np.dot(row, row.T)
            cov /= X_c.shape[0]
            cov += np.eye(self.D) * self.reg_covar
            class_covs.append(cov)
            class_covs_inv.append(np.linalg.inv(cov))
        
        self.class_mean = np.array(class_means)
        self.covariance = class_covs
        self.covariance_inv = class_covs_inv
        
    def predict(self, X):
        n_samples = X.shape[0]
        log_likelihood = np.zeros((n_samples, len(self.classes)))  #initally zero
        
        for i, c in enumerate(self.classes):
            mean = self.class_mean[i, :]
            cov_inv = self.covariance_inv[i]
            log_det = np.linalg.slogdet(self.covariance[i])[1]
            for j in range(n_samples):
                row = X[j, :].reshape(1, self.D)
                log_likelihood[j, i] = -0.5 * (np.dot(np.dot((row - mean), cov_inv), (row - mean).T) + log_det) + np.log(self.priors[i])
                #calculate the log likelihood        
        return self.classes[np.argmax(log_likelihood, axis=1)]

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
       
        
        
        
class QDA:
    def __init__(self, reg_param=0.0):
        self.reg_param = reg_param
        self.classes_ = None
        self.means_ = None
        self.covs_ = None
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean_vectors = []
        self.cov_matrices = []
        self.priors = []
        for c in self.classes:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            n_samples = X_c.shape[0]
            X_centered = X_c - mean_c
            cov_c = np.dot(X_centered.T, X_centered) / (n_samples - 1)
            self.mean_vectors.append(mean_c)
            cov_c_regularized = cov_c + self.reg_param * np.eye(X.shape[1])
            self.cov_matrices.append(cov_c_regularized)
            self.priors.append(X_c.shape[0] / X.shape[0])
        return self

    def predict(self, X):
        n_samples, n_features = X.shape
        log_probs = np.zeros((n_samples, len(self.classes)))
        for i, c in enumerate(self.classes):
            mean_c, cov_c = self.mean_vectors[i], self.cov_matrices[i]
            cov_c_det = np.linalg.det(cov_c)
            if cov_c_det == 0:
                cov_c_det = np.finfo(float).eps
            cov_c_inv = np.linalg.inv(cov_c)
            X_centered = X - mean_c
            log_probs[:, i] = -0.5 * np.sum(np.dot(X_centered, cov_c_inv) * X_centered, axis=1) -                               0.5 * np.log(cov_c_det) + np.log(self.priors[i])
        return self.classes[np.argmax(log_probs, axis=1)]

    def acc(self, X, y):
        return np.mean(self.predict(X) == y)
    
    


Q1()
train_data = np.loadtxt('C:/Users/HP/Desktop/mnist_train.csv', delimiter=',', dtype=int, skiprows=1)
train_labels = train_data[:, 0]
train_data = train_data[:, 1:]
test_data = np.loadtxt('C:/Users/HP/Desktop/mnist_test.csv', delimiter=',', dtype=int, skiprows=1)
test_labels = test_data[:, 0]
test_data = test_data[:, 1:]

print("----------------------------For LDA:----------------------------------")  
lda = LDA()
lda.fit(train_data, train_labels)
y_pred_lda = lda.predict(test_data)
acc = lda.accuracy(test_data, test_labels)
print("Accuracy for LDA",acc)
r = np.random.choice(len(test_data), size=5, replace=False)
# display the images and their predicted labels
for i in r:
    image = test_data[i].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title("True Label: {}\nPredicted Label: {}".format(test_labels[i], y_pred_lda[i]))
    plt.show()
        
        
print("----------------------------For QDA:----------------------------------")  
qda = QDA(reg_param=0.1)
qda.fit(train_data,train_labels)
y_pred_qda = qda.predict(test_data)
acc = qda.acc(test_data, test_labels)
print("Accuracy For QDA",acc)
r = np.random.choice(len(test_data), size=5, replace=False)
# display the images and their predicted labels
for i in r:
    image = test_data[i].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title("True Label: {}\nPredicted Label: {}".format(test_labels[i], y_pred_lda[i]))
    plt.show()
sklearn_lib()

