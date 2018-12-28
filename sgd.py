# CS 451 HW 2
# based on an assignment by Joe Redmon
# Corinne & Lillie

from math import exp
import random

""" Takes in a value and returns the logistic function result of that value """
def logistic(x):
    return 1 / (1 + exp(-x))

""" Takes in two lists and returns the dot product of those lists """ 
def dot(x, y):
    dotProduct = 0
    for index in range(len(x)):
        dotProduct += x[index] * y[index]
    return dotProduct

""" Takes in a model, a list, and a point, a dictionary, and takes the dot 
    product of the model and the point's feautures, then returns the logistic
    function result of  the dot product, equivalent to 
    the hypothesis function's result of the model and the point's features """ 
def predict(model, point):
    result = dot(model, point['features'])
    return logistic(result)

""" Takes data and predictions, both lists, and returns the percentage of 
    correct predictions  """ 
def accuracy(data, predictions):
    correct = 0
    for index in range(len(data)):
        if predictions[index] >= 0.5 and data[index]['label'] == 1:
            correct +=1
        elif predictions[index] < 0.5 and data[index]['label'] == 0:
            correct +=1
    return float(correct)/len(data)

""" Takes in a model, point, alpha and lambd and updates the model using 
    regularized stochastic gradient descent """ 
def update(model, point, alpha, lambd):
    diff = (predict(model, point) - point['label']) # calculate difference 
    # outside of for loop so that we can update the model simaltaneously
    model[0] = model[0] - alpha * (diff * point['features'][0]) #update theta0
    # outside for loop so we don't regularize it
    for index in range(1,len(model)):
        model[index] = model[index] - alpha*(diff * point['features'][index] + 
        lambd*model[index])
    return model

""" Takes in an integer k, the number of features, and initializes the model 
    with k random integers """
def initialize_model(k):
    return [random.gauss(0, 1) for x in range(k)]

""" Takes in list of data, number of epochs, alpha and lambd and initializes 
    the model, and for each epoch it decreases alpha, and then for every point
    in data it updates the model using regularized stochastic gradient descent.
    It returns the model after all epochs """
def train(data, epochs, alpha, lambd):
    m = len(data) # number of training examples
    n = len(data[0]['features']) # number of features (+ 1)
    model = initialize_model(n)
    for epoch in (range(epochs)):
        alpha = 0.2* alpha # decreases alpha to reach convergence
        for index in range(m):
            update(model, data[index], alpha, lambd)
        predictions = [predict(model, p) for p in data]
        print(accuracy(data, predictions))
    return model
        
""" Extracts the defined features from the dataset to use for training the 
    model"""
def extract_features(raw):
    data = []
    for r in raw:
        features = []
        features.append(1.0)
        features.append(float(r['age'])/100)
        features.append(float(r['education_num'])/100)
        features.append(float(r['education_num'] ==  14))
        features.append(float(r['education_num']) > 12)
        features.append(float(r['marital'] == 'Married-civ-spouse'))
        features.append(float(r['marital'] == 'Divorced'))
        features.append(float(r['marital'] == 'Never-married'))
        features.append(float(r['relationship'] == 'Husband'))
        features.append(float(r['occupation'] == 'Machine-op-inspct'))
        features.append(float(r['occupation'] == 'Adm-clerical'))
        features.append(float(r['occupation'] == 'Protective-serv'))
        features.append(float(r['occupation'] == 'Self-emp-inc'))
        features.append(float(r['type_employer'] == 'State-gov'))
        features.append(float(r['race'] == 'White'))
        features.append(float(r['sex'] == 'Male'))
        features.append(float(r['hr_per_week'])/100)#divide by 100 to feature normalize
        features.append(float(r['relationship'] == 'Wife'))
        features.append(float(r['occupation'] == 'Exec-managerial'))
        features.append(float(r['occupation'] == 'Tech-support'))
        features.append(float(r['capital_gain']) > 7000)
        features.append(float(r['capital_loss']) > 0)
        features.append(float(r['age']) > 30 and float(r['age']) <50)
        features.append(float(r['age']) < 30)
        features.append(float(r['age']) > 50)
        point = {}
        point['features'] = features
        point['label'] = int(r['income'] == '>50K')
        data.append(point)
    return data

def submission(data):
    return train(data, epochs=9, alpha=1.5, lambd=0.00001)
