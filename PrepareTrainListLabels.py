

import pandas as pd


# ** extract Class Labels List from the train CSV file:
def extractClassLabelsListFromCSVFile(address_train_csv_file):
    # read the Train CSV file:
    df = pd.read_csv(address_train_csv_file)

    Y = df['tags']
    #print("*****************",Y)
    numberOfTrainImages = len(Y) 
    # print(numberOfTrainImages)

    listResult = []

    # iterate over all Train elements:
    setClassLabels = set() # empty set for store class labels
    for i in range(numberOfTrainImages):
        # print(i)
        prevLen = len(setClassLabels)
        setClassLabels.add(Y[i])
        afterLen = len(setClassLabels)
        if (prevLen != afterLen):
            listResult.append(Y[i])

    return listResult


# ** extract Class Labels occurrences (label distributions) from the train CSV file:
def extractTrainLabelsDistribution(address_train_csv_file, listLabels):
    #listLabels = extractClassLabelsListFromCSVFile(address_train_csv_file)

    numberOfTotalLabels = len(listLabels)
    listDistributions = [0] * (numberOfTotalLabels)

    # read the Train CSV file:
    df = pd.read_csv(address_train_csv_file)
    Y = df['tags']

    numberOfTrainImages = len(Y)
    # print(numberOfTrainImages)

    # iterate over all Train elements:
    for i in range(numberOfTrainImages):
        # print(i)
        listDistributions[listLabels.index(Y[i])] += 1

    return listDistributions



