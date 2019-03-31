# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 16:02:15 2019

@author: rajui
"""
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def getOptimalWeightsUsingGradientDescent (x, y, mCurrent, bCurrent, 
                                           numberOfIterations, learningRate):
    n = len(x)
    
    for i in range(numberOfIterations):
        #The hypothesis function y = mx + b
        yPredicted = mCurrent * x + bCurrent
                                
        if i >= 1:
            previousCost = cost

        #The cost funtion, Mean Square Error function
        cost = (1/n) * sum([val**2 for val in (y-yPredicted)])        
        
        #Note: Partial derivative gives us the slop or direction
        #Calculate the patial derivative w.r.t 'm' to the cost function
        mDerivativeOfCostFunction = -(2/n) * sum(x*(y-yPredicted))
        #Calculate the patial derivative w.r.t 'm' to the cost function        
        bDerivativeOfCostFunction = -(2/n) * sum(y-yPredicted)
        #Computing the current 'm' value using the learning rate and 
        #the partial derivative. We subtract because the derivatives point 
        #in direction of steepest ascent
        mCurrent = mCurrent - learningRate * mDerivativeOfCostFunction
        #Computing the current 'b' value using the learning rate and 
        #the partial derivative. We subtract because the derivatives point 
        #in direction of steepest ascent
        bCurrent = bCurrent - learningRate * bDerivativeOfCostFunction
        print ("i {}, m {}, b {}, mDerivativeOfCostFunction {}, bDerivativeOfCostFunction {}, cost {}".format(i, mCurrent, bCurrent, mDerivativeOfCostFunction, bDerivativeOfCostFunction, cost))
        
        plt.plot(x, yPredicted, color='green', alpha=0.1)

        if i >= 1:
            #Checking if the cost computed in the previous step and the 
            #current step are close enough so that we can descide whether to 
            #break from the iterations
            if math.isclose(previousCost, cost, rel_tol=1e-09, abs_tol=0.0):
                plt.plot(x, yPredicted, color='red')
                return;

#This function is used to load CSV file from the 'data' directory 
#in the present working directly 
def loadCSV (fileName):
    scriptDirectory = os.path.dirname(__file__)
    dataDirectoryPath = "."
    dataDirectory = os.path.join(scriptDirectory, dataDirectoryPath)
    dataFilePath = os.path.join(dataDirectory, fileName)
    return pd.read_csv(dataFilePath)

#This funtion is used to preview the data in the given dataset
def previewData (dataSet):
    print(dataSet.head())
    print("\n")

#This function is used to check for missing values in a given dataSet
def checkForMissingValues (dataSet):
    print(dataSet.isnull().sum())
    print("\n")

#This function is used to check the statistics of a given dataSet
def getStatisticsOfData (dataSet):
    print("***** Datatype of each column in the data set: *****")
    dataSet.info()
    print("\n")
    print("***** Columns in the data set: *****")
    print(dataSet.columns.values)
    print("***** Details about the data set: *****")
    print(dataSet.describe())
    print("\n")
    print("***** Checking for any missing values in the data set: *****")
    checkForMissingValues(dataSet)
    print("\n")

#This funtion is used to handle the missing value in the features, in the 
#given examples
def handleMissingValues (feature):
    feature = np.array(feature).reshape(-1, 1)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
    imputer.fit(feature)
    feature_values = imputer.fit_transform(feature)
    return feature_values
    
#Define file names and call loadCSV to load the CSV files
dataFile = "Advertising.csv"
dataSet = loadCSV(dataFile)
dataSet.drop(['Unnamed: 0'], axis=1, inplace=True)

#Preview the dataSet and look at the statistics of the dataSet
#Check for any missing values 
#so that we will know whether to handle the missing values or not
print("** Preview the dataSet and look at the statistics of the dataSet **")
previewData(dataSet)
getStatisticsOfData(dataSet)

#In this example we will be performing the Ridge regression to compute the model
#that will be used to predict the sales given the information about how much 
#marketing was performed using different channels like TV, Radio and Newspaper

#We are dropping the sales column from dataset which is a label
features = dataSet.drop(['sales','newspaper','radio'], axis=1)
label = dataSet['sales'].values.reshape(-1,1)

featuresArray = np.array(features)
labelArray = np.array(label)

print ("featuresArray: ", featuresArray)
print ("labelArray: ", labelArray)

"""
#Initial values of m and b in the hypothesis function y = mx + b
mCurrent = bCurrent = 0

#Number of iterations
numberOfIterations = 15000

#Learning rate
learningRate = 0.02

#Invoke the getOptimalWeightsUsingGradientDescent function to perform the
#Gradient descent algorithm
getOptimalWeightsUsingGradientDescent (numberOfRoomsArray, priceOfHouseArray, 
                                       mCurrent, bCurrent, 
                                       numberOfIterations, learningRate)

print ("************************ Linear Regression *********************")
numberOfRooms = pd.DataFrame(np.c_[boston['RM']], columns = ['RM'])
priceOfHouse = boston['MEDV']

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(numberOfRooms, priceOfHouse)

#Predicting the prices
predictedPriceOfHouse = regressor.predict(numberOfRooms)

#The coefficients / the linear regression weights
print ('Coefficients: ', regressor.coef_)

#Calculating the Mean of the squared error
from sklearn.metrics import mean_squared_error
print ("Mean squared error: ", mean_squared_error(priceOfHouse, 
                                                  predictedPriceOfHouse))

#Finding out the accuracy of the model
from sklearn.metrics import r2_score
accuracyMeassure = r2_score(priceOfHouse, predictedPriceOfHouse)
print ("Accuracy of model is {} %".format(accuracyMeassure*100))
"""