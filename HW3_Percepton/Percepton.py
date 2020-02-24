import sys
import os
import string
from nltk.stem import PorterStemmer 
from collections import Counter
import random


def ReadStopWordFile(stopFile):
    stopwords = []
    path = stopFile
    with open(path, 'r') as f:
        for line in f:
            line = line.translate(str.maketrans('', '', string.punctuation))
            for word in line.split():
                stopwords.append(word.lower())
    return stopwords

def trainPrediction(item, weight):
    weightedSum = 0
    wordList = list(item)
    for word in wordList:
        weightedSum = weightedSum + (weight[word]*item[word])
    if weightedSum >= 0:
        return 1
    else:
        return 0
    return None

def testprediction(item, weight):
    weightedSum = 0
    wordList = list(item)
    for word in wordList:
        if word in weight:
            weightedSum = weightedSum + (weight[word]*item[word])
    if weightedSum >= 0:
        return 1
    else:
        return 0
    return None

def trainPerceptron(data, eta, iteration, stopFile=None):
    with open('result.txt','a') as f:
        f.write("\nETA value is " + str(eta) + " Interation value is " + str(iteration))

    print("ETA value is " + str(eta) + " Interation value is " + str(iteration))
    if(stopFile != None):
        stopwords = ReadStopWordFile(stopFile)
        
    #preprocessing
    extractVocab = []
    docCountInClass = []
    weight = {}
    spamDict = []
    hamDict = []
    
    stemmer = PorterStemmer()
    for i in range (len(trainingData)):
        td = os.listdir(trainingData[i])
        docCountInClass.append(len(td))#[spamCount,hamCount]
        for j in range (docCountInClass[i]):
            txtFilePath = trainingData[i] + td[j] #fetching text file
            try:
                with open(txtFilePath, 'r', encoding='ascii', errors='ignore') as f:
                    for line in f:
                        line = line.translate(str.maketrans('', '', string.punctuation))
                        for word in line.split():
                            extractVocab.append(stemmer.stem(word.lower()))
            except:
                print("Error while encoding the text file name:", txtFilePath)
            

    extractVocab = list(set(extractVocab))
    extractVocab.sort()
    
    if(stopFile != None):
        extractVocab = [x for x in extractVocab if x not in stopwords]
    
    for word in extractVocab:
        weight[word] = 0
    
    i = 0
    for folder in trainingData:
        dirList = os.listdir(folder)
        for file in dirList:
           wordCount = Counter()
           txtfilePath = folder + file
           with open(txtfilePath,'r', encoding='ascii', errors='ignore') as f:
               for line in f:
                   line = line.translate(str.maketrans('', '', string.punctuation))
                   for word in line.split():
                       tempWord = word.lower()
                       if(tempWord in extractVocab):
                           wordCount[tempWord] += 1
               if(i == 0):
                   spamDict.append(wordCount)
               else:
                   hamDict.append(wordCount)       
        i += 1
        
    
    for i in range(iteration):
        target = 0
        for item in spamDict:
            output = trainPrediction(item, weight)
            wordList = list(item)
            for word in wordList:
                weight[word] = weight[word] + float(eta)*(target - output)*item[word]
        target = 1
        for item in hamDict:
            output = trainPrediction(item, weight)
            wordList = list(item)
            for word in wordList:
                weight[word] = weight[word] + float(eta)*(target - output)*item[word]
    return weight
    
def testPerceptron(Data, weight):
    spamDict = []
    hamDict = []
    i = 0
    for folder in Data:
        dirList = os.listdir(folder)
        for file in dirList:
           wordCount = Counter()
           filePath = folder + file
           with open(filePath,'r', encoding='ascii', errors='ignore') as f:
               for line in f:
                   line = line.translate(str.maketrans('', '', string.punctuation))
                   for word in line.split():
                       temp = word.lower()
                       wordCount[temp] += 1
               if(i == 0):
                   spamDict.append(wordCount)
               else:
                   hamDict.append(wordCount) 
        i += 1
    
    totalCount = len(spamDict) + len(hamDict)
    predict = 0
        
    for item in spamDict:
        output = testprediction(item, weight)
        if(output == 0):
            predict += 1
            
    for item in hamDict:
        output = testprediction(item, weight)
        if(output == 1):
            predict += 1
    
    with open('result.txt','a') as f:
        f.write("\nEmail Classified Correctly:" + str(predict) +"/" +str(totalCount)) 
    print("Email Classified Correctly:" + str(predict) +"/" +str(totalCount))
    return ((predict*100)/float(totalCount))
    
if __name__ == '__main__':
    if(len(sys.argv) != 4):
        print("Please enter:  python3 Percepton.py train test Stopwords.txt")
        sys.exit()
#    
    trainingData = [] #to store path of train data
    testData = [] #to store path of test data
    
    trainingData.append(sys.argv[1] + "/spam/")
    trainingData.append(sys.argv[1] + "/ham/")
    testData.append(sys.argv[2] + "/spam/")
    testData.append(sys.argv[2] + "/ham/")
    
#    trainingData.append("train/spam/")
#    trainingData.append("train/ham/")
#    testData.append("test/spam/")
#    testData.append("test/ham/")
    
    with open('result.txt','w') as f:
        f.write("Perceptron Output:")
        
    for i in range(20):
        with open('result.txt','a') as f:
            f.write("\n-------------------------------------------------------")
            f.write("\nWITHOUT STOPWORDS")
        print("-------------------------------------------------------")
        
        if(i == 0):
            weight = trainPerceptron(trainingData, 0.0272, 72, sys.argv[3])
        else:
            weight = trainPerceptron(trainingData, round(random.uniform(0.001,0.05),4), int(random.uniform(10,100)), sys.argv[3])
        
#        weight = trainPerceptron(trainingData, round(random.uniform(0.001,0.05),4), int(random.uniform(10,100)), "Stopwords.txt")
        accuracy = testPerceptron(testData, weight)
        with open('result.txt','a') as f:
            f.write("\nAccuracy without Stopwords:" + str(accuracy))
            f.write("\n")
        print("Accuracy without Stopwords:", accuracy)
        print("")
        with open('result.txt','a') as f:
            f.write("\nWITH STOPWORDS")
        weight = trainPerceptron(trainingData, round(random.uniform(0.001,0.05),4), int(random.uniform(10,100)))
        accuracy = testPerceptron(testData, weight)
        with open('result.txt','a') as f:
            f.write("\nAccuracy with Stopwords:" + str(accuracy))
        print("Accuracy with Stopwords:", accuracy)
    
    f.close()
    
    