# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 16:02:15 2019

@author: rajui
"""
#importing packages
import os
import pandas as pd
import numpy as np 
import math
import matplotlib.pyplot as plt 

#This function is used to load CSV file from the 'data' directory 
#in the present working directly 
def loadCSV (fileName):
    scriptDirectory = os.path.dirname(__file__)
    dataDirectoryPath = "."
    dataDirectory = os.path.join(scriptDirectory, dataDirectoryPath)
    dataFilePath = os.path.join(dataDirectory, fileName)
    return pd.read_csv(dataFilePath)

#This function is to compute the Ridge regression cost
def costFunctionReg(X,y,theta,lamda = 10):
    '''Cost function for ridge regression (regularized L2)'''
    #Initialization
    m = len(y) 
    J = 0
    
    #Vectorized implementation
    h = X @ theta
    J_reg = (lamda / (2*m)) * np.sum(np.square(theta))
    J = float((1./(2*m)) * (h - y).T @ (h - y)) + J_reg;
    return(J) 

#This function performs ridge regression using gradient descent
def gradientDescentRidge(X, y, theta, alpha, tuningParameter, numberOfIterations):
    '''Gradient descent for ridge regression'''
    #Initialisation of useful values 
    m = np.size(y)
    J_history = np.zeros(numberOfIterations)


    for i in range(numberOfIterations):
        #Hypothesis function
        h = np.dot(X,theta)
        
        #Grad function in vectorized form
        theta = theta - alpha * (1/m)* (  (X.T @ (h-y)) + tuningParameter * theta )
           
        if i >= 1:
            previousCost = cost
            
        #Cost function in vectorized form       
        J_history[i] = cost = costFunctionReg(X,y,theta,tuningParameter)

        print ("i {}, theta {}, cost {}".format(i, theta, J_history[i]))
        #print ("i {}, theta0 {}, theta1 {}, cost {}".format(i, theta[0][0], theta[1][0], J_history[i]))

        if i >= 1:
            #Checking if the cost computed in the previous step and the 
            #current step are close enough so that we can descide whether to 
            #break from the iterations
            if math.isclose(previousCost, cost, rel_tol=1e-9, abs_tol=0.0):
                break
    return theta ,J_history
           

#Define file names and call loadCSV to load the CSV files
dataFile = "Advertising.csv"
dataSet = loadCSV(dataFile)
dataSet.drop(['Unnamed: 0'], axis=1, inplace=True)

#In this example we will be performing the Ridge regression to compute the model
#that will be used to predict the sales given the information about how much 
#marketing was performed using different channels like TV, Radio and Newspaper

#We are dropping the sales column from dataset which is a label
features = dataSet.drop(['sales','newspaper','radio'], axis=1)
label = dataSet['sales'].values.reshape(-1,1)
featuresArray = np.array(features)
labelArray = np.array(label)

#Defining initial weights
#theta = np.array([7.,10.]).reshape(-1,1)
theta = np.array([7.]).reshape(-1,1)

#Defining the learning rate
learningRate = 0.01

#Defining tuning parameter
tuningParameter = 20

#Defining number of iterations
numberOfIterations = 10

#Computing the gradient descent
theta_result_reg,J_history_reg = gradientDescentRidge(featuresArray, labelArray, theta, learningRate, tuningParameter, numberOfIterations)
