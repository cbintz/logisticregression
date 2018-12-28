# LILLIE AND CORINNE
import csv

""" Load 90% of the data into train, and 10% into validate from a file we read 
in """
def load_csv(filename):
    train = []
    validate = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0 # keeps track of when we hit line 27145
        for line in reader:
            if (count < 27145): # line 27146 is the 90% cut off
                train.append(line)
            else:
                validate.append(line)
            count+=1
    data = {'train' : train, 'validate': validate}
    return data

def load_adult_data():
    return load_csv("adult-data.csv")
    
""" Loads train data set """
def load_adult_train_data():
    data = load_adult_data()
    return data['train']
    
""" Loads validate data set """
def load_adult_valid_data():
    data = load_adult_data()
    return data['validate']

